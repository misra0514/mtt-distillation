"""Extended shape suite, paired timing, explicit failure/skip rows and stage diagnostics."""
import argparse,gc,hashlib,itertools,json,random,statistics,time
from pathlib import Path
import torch,triton
import test_script.stateless_conv_double_bwd.conv_double_bwd as kernel
from test_script.stateless_conv_double_bwd.conv_double_bwd import convolution_double_backward as custom,workspace_split


def scaling_from_g1(rows,max_cv=5):
    refs={(r['case'],r['F']):r for r in rows if r['G']==1 and r['status']=='ok'}
    for r in rows:
        ref=refs.get((r['case'],r['F']))
        if ref is None or r['status']!='ok':continue
        for name in ('triton','triton_tail','triton_v9','triton_v8','triton_implicit'):
            key=name+'_event_median'
            if key not in r or key not in ref:continue
            yield dict(case=r['case'],B=r['F'],G=r['G'],provider=name,
                       speedup_G_times_T1_over_TG=r['G']*ref[key]/r[key],
                       throughput_group_images_s=1000*r['G']*r['F']/r[key],
                       stable=max(ref[name+'_event_cv_pct'],r[name+'_event_cv_pct'])<=max_cv,
                       reference='measured single-group latency times G; not measured serial G calls')


BASELINES={}

def baseline_key(row):
    return tuple(row[k] for k in ('G','F','C','O','H','W','R','S','stride','padding','dilation','bias'))


def load_baseline(path,meta,a):
    env=None;rows=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith('ENV '):env=json.loads(line[4:])
        elif line.startswith('RESULT '):
            try:r=json.loads(line[7:])
            except json.JSONDecodeError:continue
            if r.get('status')=='ok':rows.append(r)
    if env is None:raise ValueError('ATen baseline has no ENV header')
    for key in ('torch','cuda','cudnn','gpu'):
        if env.get(key)!=meta.get(key):raise ValueError('ATen baseline environment mismatch: '+key)
    if a.dtype!='fp32' or a.precision!='ieee':raise ValueError('Saved baseline reuse currently requires FP32 IEEE')
    flags=env.get('args',{})
    if env.get('dtype',flags.get('dtype'))!='fp32' or env.get('precision',flags.get('precision'))!='ieee':raise ValueError('Baseline precision mismatch')
    if flags.get('dtype','fp32')!='fp32' or flags.get('precision','ieee')!='ieee':raise ValueError('Baseline precision mismatch')
    if env.get('tf32',False) or env.get('cudnn_benchmark',True)!=True:raise ValueError('Baseline backend settings mismatch')
    for row in rows:
        if 'grouped_aten_event_median' not in row:
            if 'aten_prepacked_event_median' not in row:continue
            row=dict(row)
            for old,new in [('grouped_aten','aten_prepacked'),('grouped_e2e','aten_e2e')]:
                for field in ('event_median','wall_median','event_cv_pct','peak_extra_bytes','task_peak_bytes'):
                    if new+'_'+field not in row:raise ValueError('Incomplete baseline memory/timing record: '+str(path))
                    row[old+'_'+field]=row[new+'_'+field]
            if 'serial_aten_baseline_ms' in row:
                row['serial_aten_event_median']=row['serial_aten_baseline_ms']
                row['serial_aten_event_cv_pct']=row['serial_aten_baseline_cv_pct']
        # Earlier files take precedence, preserving an original serial baseline.
        BASELINES.setdefault(baseline_key(row),row)
    print(f'Loaded ATen baseline {path}; {len(BASELINES)} matching-shape records available.',flush=True)


def cases(models=('conv3','resnet18'),datasets=('cifar10','imagenet'),imagenet_sizes=(128,224),stem='auto',width=128):
    result=[]
    for dataset in datasets:
        for size in ([32] if dataset=='cifar10' else [64] if dataset=='tinyimagenet' else imagenet_sizes):
            for model in models:
                variants=['small'] if model=='conv3' else (['small' if dataset=='cifar10' else 'standard'] if stem=='auto' else ['small','standard'] if stem=='both' else [stem])
                for variant in variants:
                    prefix=f'{model}_{dataset}_{size}_{variant}'
                    layers=[]
                    def add(name,c,o,h,k=3,stride=1,pad=1,bias=False):
                        layers.append(dict(name=name,C=c,O=o,H=h,W=h,R=k,S=k,stride=stride,padding=pad,dilation=1,bias=bias))
                        return (h+2*pad-k)//stride+1
                    if model=='conv3':
                        h=size
                        for n in range(3):
                            add(f'conv{n+1}',3 if n==0 else width,width,h,bias=True)
                            if n<2:h//=2
                    else:
                        h=add('stem',3,64,size,k=7 if variant=='standard' else 3,stride=2 if variant=='standard' else 1,pad=3 if variant=='standard' else 1)
                        if variant=='standard':h=(h+2-3)//2+1
                        c=64
                        for stage,o in enumerate((64,128,256,512)):
                            for block in range(2):
                                st=2 if stage>0 and block==0 else 1
                                old=h
                                h=add(f's{stage+1}b{block+1}_conv1',c,o,h,stride=st)
                                add(f's{stage+1}b{block+1}_conv2',o,o,h)
                                if st!=1 or c!=o:add(f's{stage+1}b{block+1}_shortcut',c,o,old,k=1,stride=st,pad=0)
                                c=o
                    # Deduplicate equivalent shapes, preserve layer names/count for weighted summaries.
                    unique={}
                    for layer in layers:
                        key=tuple(v for k,v in layer.items() if k!='name')
                        if key not in unique: unique[key]=dict(layer,layers=[layer['name']],multiplicity=1)
                        else:unique[key]['layers'].append(layer['name']);unique[key]['multiplicity']+=1
                    for j,layer in enumerate(unique.values()):
                        result.append(dict(case=f'{prefix}_{j}_{layer["name"]}',model=model,dataset=dataset,image_size=size,stem=variant,**layer))
    return result

def pack(t):
    if t is None: return None
    g,f,c,h,w=t.shape
    return t.permute(1,0,2,3,4).reshape(f,g*c,h,w).contiguous()


def unpack(t,g):
    if t is None: return None
    f,gc,h,w=t.shape
    return t.reshape(f,g,gc//g,h,w).permute(1,0,2,3,4).contiguous()


def prepare_aten(args):
    ai,aw,ab,d,w,x,st,pad,dil,trans,op,inner,mask=args
    g=x.shape[0]
    return (pack(ai),None if aw is None else aw.flatten(0,1),
            None if ab is None else ab.flatten(),pack(d),w.flatten(0,1),pack(x),
            st,pad,dil,trans,op,g,mask)


def aten_group_first(args):
    g=args[5].shape[0]
    a,b,c=torch.ops.aten._convolution_double_backward(*prepare_aten(args))
    return unpack(a,g),unpack(b,g),None if c is None else c.reshape(args[4].shape).contiguous()



def aten_serial_group_first(args):
    """Complete serial API: slice inputs, run G calls, assemble contiguous outputs.

    Output allocation/copies are timed. No prepared inputs or output tensors persist.
    Copy each group's results immediately to avoid retaining G temporary outputs.
    """
    g=args[5].shape[0]
    outputs=None
    for i in range(g):
        one=tuple(None if t is None else t[i] for t in args[:6])+tuple(args[6:])
        result=torch.ops.aten._convolution_double_backward(*one)
        if g==1:
            return tuple(None if t is None else t.unsqueeze(0).contiguous() for t in result)
        if outputs is None:
            outputs=[None if t is None else t.new_empty((g,)+tuple(t.shape)) for t in result]
        for out,t in zip(outputs,result):
            if out is not None:out[i].copy_(t)
        del result
    return tuple(outputs)


def e2e_providers(args,a):
    return {
        'triton_v18':lambda:custom(*args,precision=a.precision,workspace_limit_mb=a.workspace_mb,
                                  split_k=a.split_k,algorithm='stride2_packed_gi_tail'),
        'aten_grouped_e2e':lambda:aten_group_first(args),
        'aten_serial_e2e':lambda:aten_serial_group_first(args),
    }


def e2e_choice(row,max_cv):
    names=('triton_v18','aten_grouped_e2e','aten_serial_e2e')
    fast=min(names,key=lambda n:row[n+'_event_median'])
    wall=min(names,key=lambda n:row[n+'_wall_median'])
    near=[n for n in names if row[n+'_event_median']<=1.03*row[fast+'_event_median']]
    preferred=min(near,key=lambda n:(row[n+'_task_peak_bytes'],row[n+'_event_median']))
    stable=all(row[n+'_event_cv_pct']<=max_cv and row[n+'_wall_cv_pct']<=max_cv for n in names)
    agreement=row[preferred+'_wall_median']<=1.03*row[wall+'_wall_median']
    return dict(event_fastest=fast,wall_fastest=wall,near_tie_providers=near,
                suggested_provider=preferred if stable and agreement else None,
                status='candidate_only' if stable and agreement else 'inconclusive_noise_or_clock_disagreement',
                near_tie_pct=3,stable=stable,wall_agrees=agreement,
                rule='Within 3% of event fastest prefer lower allocated task peak; require stable event/wall and wall agreement. No dispatch is changed.')


def run_e2e(case,g,f,a):
    c,o,h,w,r,s=(case[k] for k in ('C','O','H','W','R','S'))
    st,pad,dil=(case[k] for k in ('stride','padding','dilation'))
    row=dict(case,G=g,F=f,B=f,status='pending',comparison='v21_same_run_e2e',
             policy='same_run_e2e_no_automatic_dispatch',aten_baseline_reused=False)
    p=(h+2*pad-dil*(r-1)-1)//st+1;q=(w+2*pad-dil*(s-1)-1)//st+1
    row.update(P=p,Q=q)
    nx=g*f*c*h*w;nd=g*f*o*p*q;nw=g*o*c*r*s
    if max(nx,nd,nw)>=2**30 or g*f>65535:return dict(row,status='skip_index_limit')
    estimate=4*(7*nx+6*nd+6*nw)+int(a.workspace_mb*1024**2)+32*1024**2
    free,_=torch.cuda.mem_get_info()
    if estimate>free*a.memory_fraction:return dict(row,status='skip_memory_estimate',estimated_bytes=estimate,free_bytes=free)
    seed=123+int(hashlib.sha256(str((g,f,c,o,h,w,r,s,st,pad,dil,case['bias'])).encode()).hexdigest()[:8],16)
    torch.manual_seed(seed);row['seed']=seed
    args=make_args((g,f,c,o,h,w,r,s),torch.float32,'cuda',(st,)*2,(pad,)*2,(dil,)*2,
                   opts=(True,True,case['bias']))
    funcs=e2e_providers(args,a)
    ref=aten_group_first(args)
    row['validation']={}
    for name,fn in funcs.items():
        got=fn()
        assert all(t is None or t.is_contiguous() for t in got),name
        row['validation'][name]=check(got,ref,torch.float32,a.precision,args=args)
        del got
    del ref
    gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
    # Only canonical inputs stay live during all providers' timing/memory passes.
    row.update(paired(funcs,a))
    row['input_bytes']=sum(t.numel()*t.element_size() for t in args[:6] if t is not None)
    for name,fn in funcs.items():
        gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
        base=torch.cuda.memory_allocated();reserved_base=torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        val=fn();torch.cuda.synchronize()
        peak=torch.cuda.max_memory_allocated();reserved_peak=torch.cuda.max_memory_reserved()
        row[name+'_base_allocated_bytes']=base
        row[name+'_peak_extra_bytes']=peak-base
        row[name+'_task_peak_bytes']=row['input_bytes']+peak-base
        row[name+'_process_peak_allocated_bytes']=peak
        row[name+'_reserved_base_bytes']=reserved_base
        row[name+'_process_peak_reserved_bytes']=reserved_peak
        del val
    row['memory_scope']='Warm compiled algorithms; empty allocator cache before each memory pass. Task peak = canonical input bytes + incremental allocated peak including output and layout copies. Reserved peak is process allocator state, not workspace.'
    row['timing_scope']='Same-run eager complete API: group-first contiguous inputs to group-first contiguous outputs; grouped pack/unpack and serial allocation/copies included; no per-group synchronization.'
    row['best_aten_e2e_ms']=min(row[n+'_event_median'] for n in ('aten_grouped_e2e','aten_serial_e2e'))
    row['best_aten_over_v18']=row['best_aten_e2e_ms']/row['triton_v18_event_median']
    row['selection']=e2e_choice(row,a.max_cv)
    return dict(row,status='ok')


def correctness21(dtype,precision):
    from types import SimpleNamespace
    a=SimpleNamespace(precision=precision,workspace_mb=1,split_k=None)
    count=0
    for shape in ((1,2,5,7,7,9,3,3),(3,2,5,7,8,8,3,3)):
        for opts in itertools.product((False,True),repeat=3):
            args=make_args(shape,dtype,'cuda',st=(2,2),opts=opts)
            ref=aten_group_first(args)
            for name,fn in e2e_providers(args,a).items():
                got=fn()
                assert all(t is None or t.is_contiguous() for t in got),name
                check(got,ref,dtype,precision,args=args)
                count+=1
    print('v21 complete API correctness passed:',count,'provider comparisons',flush=True)



def native_algorithms(case):
    if case['R']==case['S']==3 and case['stride']==1 and case['C']>3:
        return ['implicit','winograd_tail','winograd_exact','winograd_weight']
    if case['R']==case['S']==3 and case['stride']==2:
        return ['implicit','stride2_static_weight','stride2_packed_gi_tail']
    return ['implicit']


def native_providers(args,algorithms,a):
    # All conversion happens here, OUTSIDE timed callables.
    ready=prepare_aten(args)
    singles=[tuple(None if t is None else t[i] for t in args[:6])+tuple(args[6:])
             for i in range(args[5].shape[0])]
    funcs={alg:lambda alg=alg:custom(*args,algorithm=alg,precision=a.precision,
                  workspace_limit_mb=a.workspace_mb,split_k=a.split_k) for alg in algorithms}
    funcs['aten_grouped_native']=lambda:torch.ops.aten._convolution_double_backward(*ready)
    funcs['aten_serial_native']=lambda:[torch.ops.aten._convolution_double_backward(*one) for one in singles]
    return funcs


def native_to_group_first(name,val,g,weight_shape):
    # Correctness only; NEVER called by timing or memory measurement.
    if name=='aten_grouped_native':
        return unpack(val[0],g),unpack(val[1],g),None if val[2] is None else val[2].reshape(weight_shape)
    if name=='aten_serial_native':
        return tuple(None if val[0][j] is None else torch.stack([v[j] for v in val]) for j in range(3))
    return val


def native_verdict(t,ref,stable):
    if not stable:return 'noisy'
    if t<ref/1.03:return 'faster'
    if t>ref*1.03:return 'slower'
    return 'near_tie'


def run_native(case,g,f,a):
    c,o,h,w,r,s=(case[k] for k in ('C','O','H','W','R','S'))
    st,pad,dil=(case[k] for k in ('stride','padding','dilation'))
    p=(h+2*pad-dil*(r-1)-1)//st+1;q=(w+2*pad-dil*(s-1)-1)//st+1
    comparison=('v33_cross_generation_native' if a.compare_v33 else
                'v32_stride2_large_split_gw_native' if a.compare_v32 else
                'v31_stride2_joint_wide_gi_native' if a.compare_v31 else
                'v30_stride2_joint_onebuf_native' if a.compare_v30 else
                'v29_stem7_gi_ggo_native' if a.compare_v29 else
                'v28_stem7_native' if a.compare_v28 else
                'v27_wide_chunks_native' if a.compare_v27 else
                'v26_kernel_coverage_native' if a.compare_v26 else
                'v22_native_layout')
    row=dict(case,G=g,F=f,B=f,P=p,Q=q,status='pending',comparison=comparison,policy='native_layout',
             aten_baseline_reused=False)
    nx=g*f*c*h*w;nd=g*f*o*p*q;nw=g*o*c*r*s
    if max(nx,nd,nw)>=2**30 or g*f>65535:return dict(row,status='skip_index_limit')
    estimate=4*(8*nx+8*nd+8*nw)+int(a.workspace_mb*1024**2)+32*1024**2
    free,_=torch.cuda.mem_get_info()
    if estimate>free*a.memory_fraction:return dict(row,status='skip_memory_estimate',estimated_bytes=estimate,free_bytes=free)
    seed=123+int(hashlib.sha256(str((g,f,c,o,h,w,r,s,st,pad,dil,case['bias'])).encode()).hexdigest()[:8],16)
    torch.manual_seed(seed);row['seed']=seed
    args=make_args((g,f,c,o,h,w,r,s),torch.float32,'cuda',(st,)*2,(pad,)*2,(dil,)*2,opts=(True,True,case['bias']))
    if a.compare_v33:
        if r==s==1 and st==2:
            # v26 measured this only on CIFAR s3/s4 shortcuts.  Audit the
            # later sampled-lattice kernel on every earlier shortcut shape.
            algorithms=['implicit','pointwise_stride2']
        elif r==s==7 and st==2:
            # v29 measured only B=100,G=4.  Keep the gI-only intermediate so
            # small workloads can reveal whether packed ggO pays for itself.
            algorithms=['implicit','stem7_gi','stem7_gi_ggo']
        else:
            # Old v23 oracle candidates plus every later schedule that was
            # developed only on s3b1_conv1.  split8 is byte-for-byte the v31
            # wide path below its large-N guard, so do not time that duplicate.
            algorithms=['implicit','stride2_static_weight',
                        'stride2_packed_gi_tail','stride2_ggo_single',
                        'stride2_wide_chunks','stride2_joint_onebuf',
                        'stride2_joint_onebuf_wide_gi']
            if f*p*q>=16384:algorithms.append('stride2_joint_split_gw8')
    elif a.compare_v32:
        algorithms=['stride2_packed_gi_tail','stride2_wide_chunks',
                    'stride2_joint_onebuf','stride2_joint_onebuf_wide_gi',
                    'stride2_joint_split_gw4','stride2_joint_split_gw8']
    elif a.compare_v31:
        algorithms=['stride2_packed_gi_tail','stride2_ggo_tail',
                    'stride2_ggo_single','stride2_wide_chunks',
                    'stride2_joint_onebuf','stride2_joint_onebuf_wide_gi']
    elif a.compare_v30:
        algorithms=['stride2_packed_gi_tail','stride2_ggo_tail',
                    'stride2_ggo_single','stride2_wide_chunks',
                    'stride2_joint_onebuf']
    elif a.compare_v29:
        algorithms=['implicit','stem7_gi','stem7_gi_ggo']
    elif a.compare_v28:
        algorithms=['implicit','stem7_gi','stem7_gi_gw','stem7_full']
    elif a.compare_v27:
        algorithms=['stride2_packed_gi_tail','stride2_ggo_tail',
                    'stride2_ggo_single','stride2_wide_chunks']
    elif a.compare_v26:
        if r==s==1 and st==2:
            algorithms=['implicit','pointwise_stride2']
        else:
            algorithms=['implicit','stride2_static_weight','stride2_packed_gi_tail',
                        'stride2_ggo_tail','stride2_ggo_single','stride2_small_spatial']
    else:
        algorithms=['stride2_packed_gi_tail','stride2_ggo_tail','stride2_ggo_single'] if a.compare_v23 else native_algorithms(case)
    if (a.resnet_v23 or a.selector_sweep) and case['R']==case['S']==3 and case['stride']==2:
        algorithms+=['stride2_ggo_tail','stride2_ggo_single']
    funcs=native_providers(args,algorithms,a)
    ref=aten_group_first(args);row['validation']={}
    row['candidate_failures']={}
    for name,fn in list(funcs.items()):
        raw=fn();got=native_to_group_first(name,raw,g,args[4].shape)
        try:
            row['validation'][name]=check(got,ref,torch.float32,a.precision,args=args)
        except ValidationError as exc:
            if not (a.resnet_v23 or a.selector_sweep or a.compare_v27 or
                    a.compare_v28 or a.compare_v29 or a.compare_v30 or
                    a.compare_v31 or a.compare_v32 or a.compare_v33) or name not in algorithms:raise
            row['candidate_failures'][name]=str(exc)
            del funcs[name];algorithms.remove(name)
        finally:
            del raw,got
    if not algorithms:return dict(row,status='no_valid_candidate')
    del ref
    if a.compare_v23:
        row['candidate_plans23']={}
        for alg in algorithms:
            row['candidate_plans23'][alg]=kernel.winograd_plan(g,f,c,o,h,w,p,q,r,s,st,st,pad,pad,dil,dil,torch.float32,int(a.workspace_mb*1024**2),alg,a.split_k)
            stage=lambda alg=alg:custom(*args,algorithm=alg,precision=a.precision,workspace_limit_mb=a.workspace_mb,
                                        split_k=a.split_k,compute=(True,False,False))
            reference=aten_group_first(args)
            row['validation']['stage_ggO_'+alg]=check(stage(),(reference[0],None,None),torch.float32,a.precision,args=args)
            del reference
            funcs['stage_ggO_'+alg]=stage
    if (a.compare_v26 or a.compare_v27 or a.compare_v28 or a.compare_v29 or
            a.compare_v30 or a.compare_v31 or a.compare_v32 or a.compare_v33):
        plans_key=('candidate_plans33' if a.compare_v33 else
                   'candidate_plans32' if a.compare_v32 else
                   'candidate_plans31' if a.compare_v31 else
                   'candidate_plans30' if a.compare_v30 else
                   'candidate_plans29' if a.compare_v29 else
                   'candidate_plans28' if a.compare_v28 else
                   'candidate_plans27' if a.compare_v27 else 'candidate_plans26')
        row[plans_key]={}
        # v33 is a transfer audit, not another kernel-stage investigation.
        # Full-call timing and peak memory are the decision metrics, and every
        # full candidate was already validated above.  Avoid multiplying the
        # audit cost by another four or five isolated measurements.
        reference=None if a.compare_v33 else aten_group_first(args)
        for alg in algorithms:
            row[plans_key][alg]=kernel.winograd_plan(
                g,f,c,o,h,w,p,q,r,s,st,st,pad,pad,dil,dil,torch.float32,
                int(a.workspace_mb*1024**2),alg,a.split_k)
            if a.compare_v33:continue
            for idx,label in enumerate(('ggO','gI','gW')):
                flags=tuple(j==idx for j in range(3))
                stage=lambda alg=alg,flags=flags:custom(*args,algorithm=alg,
                    precision=a.precision,workspace_limit_mb=a.workspace_mb,
                    split_k=a.split_k,compute=flags)
                name='stage_'+label+'_'+alg
                expected=tuple(t if j==idx else None for j,t in enumerate(reference))
                row['validation'][name]=check(stage(),expected,torch.float32,
                                              a.precision,args=args)
                funcs[name]=stage
            if a.compare_v30 or a.compare_v31 or a.compare_v32:
                flags=(True,False,True)
                joint=lambda alg=alg,flags=flags:custom(*args,algorithm=alg,
                    precision=a.precision,workspace_limit_mb=a.workspace_mb,
                    split_k=a.split_k,compute=flags)
                name='stage_ggO_gW_'+alg
                expected=(reference[0],None,reference[2])
                row['validation'][name]=check(joint(),expected,torch.float32,
                                              a.precision,args=args)
                funcs[name]=joint
        if reference is not None:del reference,expected
    if a.diagnose_v24:
        reference=aten_group_first(args)
        for alg in algorithms:
            for idx,label in ((1,'gI'),(2,'gW')):
                flags=tuple(j==idx for j in range(3))
                stage=lambda alg=alg,flags=flags:custom(*args,algorithm=alg,precision=a.precision,
                    workspace_limit_mb=a.workspace_mb,split_k=a.split_k,compute=flags)
                name='stage_'+label+'_'+alg
                expected=tuple(t if j==idx else None for j,t in enumerate(reference))
                row['validation'][name]=check(stage(),expected,torch.float32,a.precision,args=args)
                funcs[name]=stage
        del reference,expected
    gc.collect();torch.cuda.synchronize()
    row.update(paired(funcs,a))
    # Both native input representations coexist for paired timing. Subtract that
    # shared baseline; report one logical native input set + incremental peak.
    input_bytes=sum(t.numel()*t.element_size() for t in args[:6] if t is not None)
    row['native_input_bytes']=input_bytes
    for name in funcs:row[name+'_task_peak_bytes']=input_bytes+row[name+'_peak_extra_bytes']
    row['memory_scope']='Native input bytes + allocated peak above shared live baseline. Excludes the other provider input representation, layout conversion and output assembly; includes all returned outputs. Not total process reserved memory.'
    row['timing_scope']='FP32 IEEE eager full double backward. Triton group-first; grouped ATen prepacked B-first; serial ATen pre-sliced NCHW, returning G output tuples without assembly. All layout conversion outside timing.'
    row['providers']=list(funcs);row['algorithms']=algorithms
    best_aten=min(('aten_grouped_native','aten_serial_native'),key=lambda n:row[n+'_event_median'])
    best=min(algorithms,key=lambda n:row[n+'_event_median'])
    # Explicitly separate pre-existing structural policy from measured best.
    policy='winograd_tail' if 'winograd_tail' in algorithms else 'stride2_packed_gi_tail' if 'stride2_packed_gi_tail' in algorithms else algorithms[0]
    row.update(best_aten=best_aten,best_triton=best,structural_policy=policy)
    baseline=row[best_aten+'_event_median'];row['comparisons']={}
    for alg in algorithms:
        t=row[alg+'_event_median']
        stable=all(row[n+'_'+clock+'_cv_pct']<=a.max_cv for n in (alg,'aten_grouped_native','aten_serial_native') for clock in ('event','wall'))
        wallbase=min(row[n+'_wall_median'] for n in ('aten_grouped_native','aten_serial_native'))
        verdict=native_verdict(t,baseline,stable)
        wallverdict=native_verdict(row[alg+'_wall_median'],wallbase,stable)
        if verdict!=wallverdict:verdict='clock_disagreement'
        row['comparisons'][alg]=dict(speedup=baseline/t,time_change_pct=100*(t/baseline-1),verdict=verdict)
    if a.compare_v23:
        row['vs_v18']={alg:dict(full_speedup=row['stride2_packed_gi_tail_event_median']/row[alg+'_event_median'],
            ggo_speedup=row['stage_ggO_stride2_packed_gi_tail_event_median']/row['stage_ggO_'+alg+'_event_median'],
            extra_memory_delta_bytes=row[alg+'_peak_extra_bytes']-row['stride2_packed_gi_tail_peak_extra_bytes']) for alg in algorithms[1:]}
    if a.compare_v26:
        old='implicit' if r==s==1 else 'stride2_ggo_single'
        new='pointwise_stride2' if r==s==1 else 'stride2_small_spatial'
        row['diagnostic26']={
            'scope':'Isolated stages include their own allocation/packing; stage sums are diagnostic and are not substituted for full-call timing.',
            'old_reference':old,'new_candidate':new,
            'full_speedup':row[old+'_event_median']/row[new+'_event_median'],
            'extra_memory_delta_bytes':row[new+'_peak_extra_bytes']-row[old+'_peak_extra_bytes'],
            'isolated_stages':{}}
        for alg in (old,new):
            times={label:row['stage_'+label+'_'+alg+'_event_median']
                   for label in ('ggO','gI','gW')}
            row['diagnostic26']['isolated_stages'][alg]=dict(
                times,stage_sum_ms=sum(times.values()),full_ms=row[alg+'_event_median'])
    if a.compare_v27 and all(name in algorithms for name in
                             ('stride2_ggo_tail','stride2_wide_chunks')):
        old='stride2_ggo_tail';new='stride2_wide_chunks'
        row['diagnostic27']={
            'scope':'Isolated stages include their own allocation/packing; stage sums are diagnostic and are not substituted for full-call timing.',
            'old_reference':old,'new_candidate':new,
            'full_speedup':row[old+'_event_median']/row[new+'_event_median'],
            'extra_memory_delta_bytes':row[new+'_peak_extra_bytes']-row[old+'_peak_extra_bytes'],
            'isolated_stages':{}}
        for alg in (old,new):
            times={label:row['stage_'+label+'_'+alg+'_event_median']
                   for label in ('ggO','gI','gW')}
            row['diagnostic27']['isolated_stages'][alg]=dict(
                times,stage_sum_ms=sum(times.values()),full_ms=row[alg+'_event_median'])
    elif a.compare_v27:
        row['diagnostic27']={
            'status':'candidate_invalid',
            'candidate_failures':row['candidate_failures'],
            'scope':'No speedup is reported unless both the v23 reference and v27 candidate validate.'}
    if a.compare_v28:
        row['diagnostic28']={
            'scope':'Cumulative ablation. Isolated stages include allocation/packing; only full-call medians are final performance results.',
            'reference':'implicit','isolated_stages':{},'full_speedups':{},
            'extra_memory_delta_bytes':{}}
        for alg in algorithms:
            times={label:row['stage_'+label+'_'+alg+'_event_median']
                   for label in ('ggO','gI','gW')}
            row['diagnostic28']['isolated_stages'][alg]=dict(
                times,stage_sum_ms=sum(times.values()),full_ms=row[alg+'_event_median'])
            if alg!='implicit':
                row['diagnostic28']['full_speedups'][alg]=(
                    row['implicit_event_median']/row[alg+'_event_median'])
                row['diagnostic28']['extra_memory_delta_bytes'][alg]=(
                    row[alg+'_peak_extra_bytes']-row['implicit_peak_extra_bytes'])
    if a.compare_v29:
        row['diagnostic29']={
            'scope':'Cumulative 7x7 ablation with implicit gW retained. Isolated stages include allocation/packing; only full-call medians are final performance results.',
            'reference':'implicit','isolated_stages':{},'full_speedups':{},
            'extra_memory_delta_bytes':{}}
        for alg in algorithms:
            times={label:row['stage_'+label+'_'+alg+'_event_median']
                   for label in ('ggO','gI','gW')}
            row['diagnostic29']['isolated_stages'][alg]=dict(
                times,stage_sum_ms=sum(times.values()),full_ms=row[alg+'_event_median'])
            if alg!='implicit':
                row['diagnostic29']['full_speedups'][alg]=(
                    row['implicit_event_median']/row[alg+'_event_median'])
                row['diagnostic29']['extra_memory_delta_bytes'][alg]=(
                    row[alg+'_peak_extra_bytes']-row['implicit_peak_extra_bytes'])
    if a.compare_v30:
        row['diagnostic30']={
            'scope':'Complete-call candidate. The joint ggO+gW stage exposes one-buffer reuse; isolated stages are fallback diagnostics only. Full-call medians are final.',
            'reference':'stride2_ggo_single','isolated_stages':{},
            'joint_ggO_gW':{},'full_speedups':{},
            'extra_memory_delta_bytes':{}}
        refalg='stride2_ggo_single'
        for alg in algorithms:
            times={label:row['stage_'+label+'_'+alg+'_event_median']
                   for label in ('ggO','gI','gW')}
            row['diagnostic30']['isolated_stages'][alg]=dict(
                times,stage_sum_ms=sum(times.values()),
                full_ms=row[alg+'_event_median'])
            row['diagnostic30']['joint_ggO_gW'][alg]=(
                row['stage_ggO_gW_'+alg+'_event_median'])
            if alg!=refalg:
                row['diagnostic30']['full_speedups'][alg]=(
                    row[refalg+'_event_median']/row[alg+'_event_median'])
                row['diagnostic30']['extra_memory_delta_bytes'][alg]=(
                    row[alg+'_peak_extra_bytes']-row[refalg+'_peak_extra_bytes'])
    if a.compare_v31 and all(name in algorithms for name in
                             ('stride2_joint_onebuf',
                              'stride2_joint_onebuf_wide_gi')):
        old='stride2_joint_onebuf';new='stride2_joint_onebuf_wide_gi'
        row['diagnostic31']={
            'scope':'v31 changes only the inherited gI schedule: v30 one-buffer joint ggO+gW plus v27 wide-chunks gI. Full-call medians are final.',
            'reference':old,'new_candidate':new,
            'full_speedup':row[old+'_event_median']/row[new+'_event_median'],
            'extra_memory_delta_bytes':(
                row[new+'_peak_extra_bytes']-row[old+'_peak_extra_bytes']),
            'isolated_stages':{},'joint_ggO_gW':{}}
        for alg in algorithms:
            times={label:row['stage_'+label+'_'+alg+'_event_median']
                   for label in ('ggO','gI','gW')}
            row['diagnostic31']['isolated_stages'][alg]=dict(
                times,stage_sum_ms=sum(times.values()),
                full_ms=row[alg+'_event_median'])
            row['diagnostic31']['joint_ggO_gW'][alg]=(
                row['stage_ggO_gW_'+alg+'_event_median'])
    elif a.compare_v31:
        row['diagnostic31']={
            'status':'candidate_invalid',
            'candidate_failures':row['candidate_failures'],
            'scope':'No v30-to-v31 speedup is reported unless both candidates validate.'}
    if a.compare_v32:
        prior=('stride2_wide_chunks','stride2_joint_onebuf',
               'stride2_joint_onebuf_wide_gi')
        new=('stride2_joint_split_gw4','stride2_joint_split_gw8')
        if all(name in algorithms for name in prior+new):
            prior_best=min(prior,key=lambda name:row[name+'_event_median'])
            row['diagnostic32']={
                'scope':'Large-N complete-call ablation. v32 keeps v31 wide gI and one-buffer reuse, then applies packed split-K gW and larger ggO/gW tiles. The split-K path requires the joint ggO+gW call, so isolated stage timings intentionally measure inherited fallbacks. Full-call medians and joint ggO+gW timings are final.',
                'prior_best':prior_best,'new_candidates':list(new),
                'full_speedups':{},'extra_memory_delta_bytes':{},
                'isolated_stages':{},'joint_ggO_gW':{}}
            for alg in prior+new:
                times={label:row['stage_'+label+'_'+alg+'_event_median']
                       for label in ('ggO','gI','gW')}
                row['diagnostic32']['isolated_stages'][alg]=dict(
                    times,stage_sum_ms=sum(times.values()),
                    full_ms=row[alg+'_event_median'])
                row['diagnostic32']['joint_ggO_gW'][alg]=(
                    row['stage_ggO_gW_'+alg+'_event_median'])
            for alg in new:
                row['diagnostic32']['full_speedups'][alg]=(
                    row[prior_best+'_event_median']/row[alg+'_event_median'])
                row['diagnostic32']['extra_memory_delta_bytes'][alg]=(
                    row[alg+'_peak_extra_bytes']-
                    row[prior_best+'_peak_extra_bytes'])
        else:
            row['diagnostic32']={
                'status':'candidate_invalid',
                'candidate_failures':row['candidate_failures'],
                'scope':'No v32 speedup is reported unless both split-K candidates and all three prior references validate.'}
    if a.compare_v33:
        if r==s==1:
            family='pointwise_stride2'
            old=('implicit',);new=('pointwise_stride2',)
        elif r==s==7:
            family='stem7_stride2'
            old=('implicit',);new=('stem7_gi','stem7_gi_ggo')
        else:
            family='dense3_stride2'
            old=('implicit','stride2_static_weight',
                 'stride2_packed_gi_tail','stride2_ggo_single')
            new=tuple(name for name in (
                'stride2_wide_chunks','stride2_joint_onebuf',
                'stride2_joint_onebuf_wide_gi','stride2_joint_split_gw8')
                if name in algorithms)
        old=tuple(name for name in old if name in algorithms)
        old_best=min(old,key=lambda name:row[name+'_event_median'])
        new_best=min(new,key=lambda name:row[name+'_event_median'])
        # Report the complete latency/memory Pareto frontier.  This makes a
        # later candidate useful even when it is only a near tie in latency but
        # eliminates workspace, which is the central question of this audit.
        frontier=[]
        for name in algorithms:
            t=row[name+'_event_median'];m=row[name+'_peak_extra_bytes']
            dominated=any(
                row[other+'_event_median']<=t and
                row[other+'_peak_extra_bytes']<=m and
                (row[other+'_event_median']<t or
                 row[other+'_peak_extra_bytes']<m)
                for other in algorithms if other!=name)
            if not dominated:frontier.append(name)
        fastest=min(algorithms,key=lambda name:row[name+'_event_median'])
        near=[name for name in algorithms
              if row[name+'_event_median']<=1.03*row[fastest+'_event_median']]
        memory_choice=min(near,key=lambda name:(
            row[name+'_peak_extra_bytes'],row[name+'_event_median']))
        row['diagnostic33']={
            'scope':'Cross-generation full-call native-layout audit. Old means the candidate family available in the v23 sweep; new means a later kernel originally measured on a narrower shape/BG region. No layout conversion or isolated-stage sum is used.',
            'family':family,'old_candidates':list(old),
            'new_candidates':list(new),'old_best':old_best,
            'new_best':new_best,
            'new_over_old_speedup':(
                row[old_best+'_event_median']/row[new_best+'_event_median']),
            'new_minus_old_peak_extra_bytes':(
                row[new_best+'_peak_extra_bytes']-
                row[old_best+'_peak_extra_bytes']),
            'all_candidate_fastest':fastest,
            'within_3pct_lower_memory_choice':memory_choice,
            'pareto_frontier':frontier,
            'per_candidate':{name:{
                'event_ms':row[name+'_event_median'],
                'peak_extra_bytes':row[name+'_peak_extra_bytes']}
                for name in algorithms}}
    row['best_is_measured_candidate_only']=True
    if a.diagnose_v24:
        row['diagnostic24']={'scope':'Isolated stages include their allocations/packing; their sum is NOT an in-call decomposition. Full-call CUDA traces below classify actual device activities; profiler times are not benchmark times.',
            'isolated_stages':{},'full_call_profiles':{}}
        for alg in algorithms:
            times={label:row['stage_'+label+'_'+alg+'_event_median'] for label in ('ggO','gI','gW')}
            row['diagnostic24']['isolated_stages'][alg]=dict(times,stage_sum_ms=sum(times.values()),
                full_ms=row[alg+'_event_median'],stage_sum_minus_full_ms=sum(times.values())-row[alg+'_event_median'])
        # Trace AFTER all normal timing/memory measurements. No instrumented
        # kernels, extra barriers, or record_function scopes in production code.
        if a.diagnostic_profile:
            for name in algorithms+['aten_grouped_native','aten_serial_native']:
                print('  profiling full API',name,flush=True)
                try:
                    trace=profile_callable(funcs[name],a.profile_iters)
                    if trace.get('status')=='ok' and name in algorithms:
                        trace['components24']=diagnostic_components24(trace['kernels'])
                except Exception as exc:
                    trace=dict(status='unavailable',reason=type(exc).__name__+': '+str(exc))
                row['diagnostic24']['full_call_profiles'][name]=trace
    return dict(row,status='ok')


def diagnostic_components24(kernels):
    # Exact known kernel prefixes, longest first. Unknown CUDA activities stay
    # explicit rather than being attributed to GEMM/packing speculatively.
    mapping={'_ggo_pack14':'ggO_pack','_ggo_mm14':'ggO_compute','_ggo_single23':'ggO_compute',
        '_gi_pack_d17':'gI_pack_activation','_gi_pack_w17':'gI_pack_weight','_gi_mm17':'gI_compute',
        '_gw':'gW_compute','_reduce':'gW_reduce','_ggo':'ggO_compute','_gi':'gI_compute'}
    out={}
    for k in kernels:
        name=k['name']
        prefix=next((p for p in sorted(mapping,key=len,reverse=True) if name==p or name.startswith(p+'_')),None)
        label=mapping[prefix] if prefix else 'unclassified_cuda'
        rec=out.setdefault(label,dict(ms_per_api=0.,calls_per_api=0.))
        rec['ms_per_api']+=k['ms_per_api'];rec['calls_per_api']+=k['calls_per_api']
    return out


def native_model_summaries(rows,tasks,a):
    by_key={(r['case'],r['G'],r['F']):r for r in rows}
    groups=sorted(set((c['model'],c['dataset'],c['image_size'],c['stem'],g,b) for c,g,b in tasks))
    for model,dataset,size,stem,g,b in groups:
        full=cases([model],[dataset],[size],stem,a.width)
        matched=[by_key.get((c['case'],g,b)) for c in full]
        good=[r for r in matched if r and r['status']=='ok']
        rec=dict(model=model,dataset=dataset,image_size=size,stem=stem,G=g,B=b,
            scope='Layer-count weighted sum of isolated convolution double backward medians; NOT network end-to-end. Best candidate is same-run oracle, NOT a deployed selector.',
            complete=len(good)==len(full),unique_shapes_ok=len(good),unique_shapes_expected=len(full),
            layers_ok=sum(r['multiplicity'] for r in good),layers_expected=sum(c['multiplicity'] for c in full),
            missing=[c['case'] for c,r in zip(full,matched) if not r or r['status']!='ok'])
        if good:
            def total(provider):return sum(r['multiplicity']*r[provider(r)+'_event_median'] for r in good)
            t=total(lambda r:r['best_triton']);ag=total(lambda r:'aten_grouped_native');ase=total(lambda r:'aten_serial_native');ab=total(lambda r:r['best_aten'])
            rec.update(covered_best_candidate_ms=t,covered_grouped_aten_ms=ag,covered_serial_aten_ms=ase,covered_per_layer_best_aten_ms=ab,
                candidate_failures={r['case']:r['candidate_failures'] for r in good if r.get('candidate_failures')},
                selected_candidates={r['case']:r['best_triton'] for r in good},
                stable=all(r[n+'_'+clock+'_cv_pct']<=a.max_cv for r in good for n in (r['best_triton'],'aten_grouped_native','aten_serial_native') for clock in ('event','wall')))
            if rec['complete']:rec.update(conv_sum_vs_grouped=ag/t,conv_sum_vs_serial=ase/t,conv_sum_vs_per_layer_best_aten=ab/t)
        yield rec


def native_summary(rows):
    good=[r for r in rows if r.get('comparison')=='v22_native_layout' and r['status']=='ok']
    result=dict(scope='isolated convolution double backward only, not full network; best candidate is selected on this same measurement',
                completed=len(good),not_completed=len(rows)-len(good))
    for label,key in [('structural_policy','structural_policy'),('best_candidate','best_triton')]:
        counts={};exceptions=[]
        for r in good:
            comp=r['comparisons'][r[key]];v=comp['verdict'];counts[v]=counts.get(v,0)+1
            if v!='faster':exceptions.append(dict(case=r['case'],B=r['B'],G=r['G'],algorithm=r[key],**comp))
        result[label]=dict(counts=counts,not_clear_wins=exceptions)
    return result


def transfer33_summary(rows):
    """Aggregate only the old-family versus later-family transfer question."""
    good=[r for r in rows if r.get('comparison')=='v33_cross_generation_native'
          and r.get('status')=='ok' and 'diagnostic33' in r]
    result=dict(
        scope='Same-run native-layout full APIs. Later-family best is compared with the best pre-v26/v23-family candidate measured in the same row. Within-3% memory choice is selected only among measured Triton candidates.',
        completed=len(good),families={})
    for family in sorted(set(r['diagnostic33']['family'] for r in good)):
        subset=[r for r in good if r['diagnostic33']['family']==family]
        counts={'later_faster':0,'near_tie':0,'older_faster':0}
        replacements=[]
        for r in subset:
            d=r['diagnostic33'];speed=d['new_over_old_speedup']
            verdict=('later_faster' if speed>1.03 else
                     'older_faster' if speed<1/1.03 else 'near_tie')
            counts[verdict]+=1
            if verdict!='older_faster' or d['new_minus_old_peak_extra_bytes']<0:
                replacements.append(dict(
                    case=r['case'],B=r['B'],G=r['G'],
                    old=d['old_best'],new=d['new_best'],speedup=speed,
                    peak_extra_delta_bytes=d['new_minus_old_peak_extra_bytes'],
                    fastest=d['all_candidate_fastest'],
                    within_3pct_lower_memory=d['within_3pct_lower_memory_choice'],
                    pareto=d['pareto_frontier']))
        result['families'][family]=dict(
            completed=len(subset),counts=counts,
            old_best_counts=dict(__import__('collections').Counter(
                r['diagnostic33']['old_best'] for r in subset)),
            new_best_counts=dict(__import__('collections').Counter(
                r['diagnostic33']['new_best'] for r in subset)),
            replacement_candidates=replacements)
    return result


def correctness22(dtype,precision):
    from types import SimpleNamespace
    a=SimpleNamespace(precision=precision,workspace_mb=2,split_k=None)
    count=0
    for shape in ((1,2,3,8,9,9,3,3),(2,3,8,8,8,8,3,3),(2,2,8,16,5,7,3,3)):
        for opts in ((True,True,True),(True,True,False),(True,False,False),(False,True,True),(False,False,False)):
            args=make_args(shape,dtype,'cuda',opts=opts)
            algorithms=['implicit'] if shape[2]==3 else ['implicit','winograd_tail','winograd_exact','winograd_weight']
            ref=aten_group_first(args)
            for name,fn in native_providers(args,algorithms,a).items():
                got=native_to_group_first(name,fn(),shape[0],args[4].shape)
                check(got,ref,dtype,precision,args=args);count+=1
    print('v22 native layout correctness passed:',count,'provider comparisons',flush=True)


def correctness23(dtype,precision):
    count=0
    for shape,pad in (((2,3,5,7,9,11,3,3),(0,2)),((3,2,8,16,8,8,3,3),(1,1)),((1,1,3,5,2,3,3,3),(1,1))):
        for opts in itertools.product((False,True),repeat=3):
            args=make_args(shape,dtype,'cuda',st=(2,2),pad=pad,opts=opts)
            for alg in ('stride2_ggo_tail','stride2_ggo_single'):
                compare(args,precision,budget=.02,algorithm=alg);count+=1
    for budget in (0,.001,.02,2):
        args=make_args((3,5,8,16,9,11,3,3),dtype,'cuda',st=(2,2))
        for alg in ('stride2_ggo_tail','stride2_ggo_single'):
            compare(args,precision,budget=budget,algorithm=alg);count+=1
    # Target channel reductions, important for the changed accumulation order.
    for c,o,h in ((64,128,32),(128,256,16),(256,512,8)):
        args=make_args((2,3,c,o,h,h,3,3),dtype,'cuda',st=(2,2),opts=(True,True,False))
        for alg in ('stride2_ggo_tail','stride2_ggo_single'):
            compare(args,precision,budget=32,algorithm=alg);count+=1
    print('v23 correctness passed:',count,'full API comparisons',flush=True)


def correctness26(dtype,precision):
    count=0
    pointwise=((2,3,64,128,32,32,1,1),
               (2,3,128,256,16,16,1,1),
               (1,2,5,7,7,9,1,1))
    downsample=((2,3,128,256,16,16,3,3),
                (2,3,256,512,8,8,3,3),
                (1,2,5,7,7,9,3,3))
    for algorithm,shapes,pad in (
            ('pointwise_stride2',pointwise,(0,0)),
            ('stride2_small_spatial',downsample,(1,1))):
        for shape in shapes:
            for opts in itertools.product((False,True),repeat=3):
                args=make_args(shape,dtype,'cuda',st=(2,2),pad=pad,opts=opts)
                compare(args,precision,budget=32,algorithm=algorithm);count+=1
    # Exercise explicit split-K and the sparse-zero dgrad output.
    for algorithm,shape,pad in (
            ('pointwise_stride2',(3,5,17,33,9,11,1,1),(0,0)),
            ('stride2_small_spatial',(3,5,17,33,9,11,3,3),(1,1))):
        args=make_args(shape,dtype,'cuda',st=(2,2),pad=pad,
                       opts=(True,True,False))
        for split in (1,4):
            compare(args,precision,split=split,budget=32,algorithm=algorithm);count+=1
    print('v26 correctness passed:',count,'full API comparisons',flush=True)


def correctness27(dtype,precision):
    count=0
    # The normal complete FP32/IEEE 1x1 stride-2 call must now route through
    # pointwise_stride2 without an external table.
    for shape in ((2,3,64,128,32,32,1,1),
                  (2,3,128,256,16,16,1,1),
                  (1,2,5,7,7,9,1,1)):
        args=make_args(shape,dtype,'cuda',st=(2,2),pad=(0,0),
                       opts=(True,True,False))
        sig=kernel.convolution_double_backward_signature(
            *args,precision=precision,workspace_limit_mb=128)
        selected,source=kernel._select_algorithm_cached(sig)
        if selected!='pointwise_stride2' or source!='measured_pointwise_v26':
            raise AssertionError((selected,source))
        compare(args,precision,budget=128,algorithm='auto');count+=1

    # Wide chunks retain the already-validated math and vary only packed NT
    # and group-wave sizes. Exercise every optional-input combination, odd
    # boundaries, and a target-like shape that is genuinely multi-chunk under
    # the historical 32 MiB cap.
    for shape,pad in (((2,3,128,256,16,16,3,3),(1,1)),
                      ((2,3,256,512,8,8,3,3),(1,1)),
                      ((1,2,5,7,7,9,3,3),(1,1))):
        for opts in itertools.product((False,True),repeat=3):
            args=make_args(shape,dtype,'cuda',st=(2,2),pad=pad,opts=opts)
            compare(args,precision,budget=128,algorithm='stride2_wide_chunks');count+=1
    args=make_args((4,25,128,256,16,16,3,3),dtype,'cuda',st=(2,2),pad=(1,1),
                   opts=(True,True,False))
    for split in (1,4):
        compare(args,precision,split=split,budget=128,
                algorithm='stride2_wide_chunks');count+=1
    print('v27 correctness passed:',count,'full API comparisons',flush=True)


def correctness28(dtype,precision):
    count=0
    algorithms=('stem7_gi','stem7_gi_gw','stem7_full')

    # Odd sizes and multiple outer groups expose phase tails, padding and every
    # optional-input route without making first-use compilation unnecessarily
    # expensive. All three candidates are cumulative, so this also verifies
    # that stages left on the implicit path remain unchanged.
    for opts in itertools.product((False,True),repeat=3):
        args=make_args((2,3,3,19,17,19,7,7),dtype,'cuda',st=(2,2),pad=(3,3),opts=opts)
        for algorithm in algorithms:
            compare(args,precision,budget=128,algorithm=algorithm);count+=1

    # Representative Tiny-ImageNet and ImageNet10 standard-stem shapes.
    for shape in ((2,2,3,64,64,64,7,7),
                  (1,1,3,64,128,128,7,7)):
        args=make_args(shape,dtype,'cuda',st=(2,2),pad=(3,3),
                       opts=(True,True,True))
        for algorithm in algorithms:
            compare(args,precision,budget=128,algorithm=algorithm);count+=1

    # Explicit split-K validates the 49-filter-position gW temporary layout and
    # reduction independently of the automatic workspace heuristic.
    args=make_args((3,5,3,17,15,21,7,7),dtype,'cuda',st=(2,2),pad=(3,3),
                   opts=(True,True,False))
    for algorithm in ('stem7_gi_gw','stem7_full'):
        compare(args,precision,split=4,budget=128,algorithm=algorithm);count+=1
    print('v28 correctness passed:',count,'full API comparisons',flush=True)


def correctness29(dtype,precision):
    count=0
    plan=kernel.winograd_plan(2,3,3,19,17,19,9,10,7,7,2,2,3,3,1,1,
                              dtype,128*1024**2,'stem7_gi_ggo',None)
    if not plan.get('stem7_gi28') or not plan.get('stem7_ggo28') or plan.get('stem7_gw28'):
        raise AssertionError('v29 must select specialized gI/ggO and retain implicit gW')
    # The underlying gI and ggO kernels were independently covered in v28;
    # exercise their new composition across every optional-input route.
    for opts in itertools.product((False,True),repeat=3):
        args=make_args((2,3,3,19,17,19,7,7),dtype,'cuda',st=(2,2),pad=(3,3),opts=opts)
        compare(args,precision,budget=128,algorithm='stem7_gi_ggo');count+=1
    for shape in ((2,2,3,64,64,64,7,7),
                  (1,1,3,64,128,128,7,7)):
        args=make_args(shape,dtype,'cuda',st=(2,2),pad=(3,3),
                       opts=(True,True,True))
        compare(args,precision,budget=128,algorithm='stem7_gi_ggo');count+=1
    print('v29 correctness passed:',count,'full API comparisons',flush=True)


def correctness30(dtype,precision):
    count=0
    target=kernel.winograd_plan(
        4,100,128,256,16,16,8,8,3,3,2,2,1,1,1,1,dtype,
        128*1024**2,'stride2_joint_onebuf',None)
    if not target.get('joint_onebuf30'):
        raise AssertionError('v30 target did not enable the joint path')
    if (target['onebuf_nt30'],target['onebuf_groups30'],
            target['onebuf_chunks30'])!=(6400,4,1):
        raise AssertionError(('unexpected target schedule',target))
    if target['onebuf_scratch30']>128*1024**2:
        raise AssertionError(('workspace overflow',target))

    # Odd tails plus every optional-input route. Calls that cannot form the
    # complete ggO+gW pair intentionally exercise the inherited v23 fallback.
    for opts in itertools.product((False,True),repeat=3):
        args=make_args((2,3,17,33,9,11,3,3),dtype,'cuda',st=(2,2),
                       pad=(1,1),opts=opts)
        compare(args,precision,budget=128,
                algorithm='stride2_joint_onebuf');count+=1

    # The exact high-impact stage-3 shape must validate with all four groups
    # and all 6400 columns resident in the single reusable packed buffer.
    args=make_args((4,100,128,256,16,16,3,3),dtype,'cuda',st=(2,2),
                   pad=(1,1),opts=(True,True,False))
    compare(args,precision,budget=128,
            algorithm='stride2_joint_onebuf');count+=1

    # Force multiple one-buffer chunks, and separately verify that an explicit
    # split-K request takes the documented v23 fallback without changing math.
    args=make_args((2,5,5,7,9,11,3,3),dtype,'cuda',st=(2,2),
                   pad=(1,1),opts=(True,True,True))
    chunked=kernel.winograd_plan(
        2,5,5,7,9,11,5,6,3,3,2,2,1,1,1,1,dtype,
        int(.02*1024**2),'stride2_joint_onebuf',None)
    if not chunked.get('joint_onebuf30') or chunked['onebuf_chunks30']<=1:
        raise AssertionError(('small-budget test did not form chunks',chunked))
    compare(args,precision,budget=.02,
            algorithm='stride2_joint_onebuf');count+=1
    fallback=kernel.winograd_plan(
        2,5,5,7,9,11,5,6,3,3,2,2,1,1,1,1,dtype,
        128*1024**2,'stride2_joint_onebuf',4)
    if fallback.get('joint_onebuf30'):
        raise AssertionError(('explicit split-K must use fallback',fallback))
    compare(args,precision,split=4,budget=128,
            algorithm='stride2_joint_onebuf');count+=1
    print('v30 correctness passed:',count,'full API comparisons',flush=True)


def correctness31(dtype,precision):
    count=0
    # The v31 plan must preserve v30's joint schedule while taking every gI
    # scheduling key from v27. B=10/G=64 is the largest observed gI regression.
    dims=(64,10,128,256,16,16,8,8,3,3,2,2,1,1,1,1)
    plan=kernel.winograd_plan(*dims,dtype,128*1024**2,
                              'stride2_joint_onebuf_wide_gi',None)
    old=kernel.winograd_plan(*dims,dtype,128*1024**2,
                             'stride2_joint_onebuf',None)
    wide=kernel.winograd_plan(*dims,dtype,128*1024**2,
                              'stride2_wide_chunks',None)
    if not (plan.get('joint_onebuf30') and plan.get('joint_wide_gi31') and
            plan.get('wide_chunks27')):
        raise AssertionError(('v31 composition not enabled',plan))
    for key in ('onebuf_nt30','onebuf_groups30','onebuf_chunks30'):
        if plan[key]!=old[key]:raise AssertionError((key,plan[key],old[key]))
    for key in ('gi_nt17','gi_groups17','gi_scratch17'):
        if plan[key]!=wide[key]:raise AssertionError((key,plan[key],wide[key]))

    # Odd tails and every optional-input route. Incomplete calls deliberately
    # exercise the inherited wide fallback rather than the complete-call join.
    for opts in itertools.product((False,True),repeat=3):
        args=make_args((2,3,17,33,9,11,3,3),dtype,'cuda',st=(2,2),
                       pad=(1,1),opts=opts)
        compare(args,precision,budget=128,
                algorithm='stride2_joint_onebuf_wide_gi');count+=1

    # Validate both the high-G regression point and the original B=100/G=4
    # target whose v30 gain must be retained.
    for shape in ((64,10,128,256,16,16,3,3),
                  (4,100,128,256,16,16,3,3)):
        args=make_args(shape,dtype,'cuda',st=(2,2),pad=(1,1),
                       opts=(True,True,False))
        compare(args,precision,budget=128,
                algorithm='stride2_joint_onebuf_wide_gi');count+=1

    # Explicit split-K remains a documented fallback and must not enter the
    # joint schedule accidentally.
    args=make_args((2,5,5,7,9,11,3,3),dtype,'cuda',st=(2,2),
                   pad=(1,1),opts=(True,True,True))
    fallback=kernel.winograd_plan(
        2,5,5,7,9,11,5,6,3,3,2,2,1,1,1,1,dtype,
        128*1024**2,'stride2_joint_onebuf_wide_gi',4)
    if fallback.get('joint_onebuf30'):
        raise AssertionError(('explicit split-K must use fallback',fallback))
    compare(args,precision,split=4,budget=128,
            algorithm='stride2_joint_onebuf_wide_gi');count+=1
    print('v31 correctness passed:',count,'full API comparisons',flush=True)


def correctness32(dtype,precision):
    count=0
    candidates=('stride2_joint_split_gw4','stride2_joint_split_gw8')
    # Both experimental schedules must enter the large-N path and remain
    # within the complete 128 MiB live-workspace calculation.
    for groups in (1,4):
        dims=(groups,1000,128,256,16,16,8,8,3,3,2,2,1,1,1,1)
        for alg,split in zip(candidates,(4,8)):
            plan=kernel.winograd_plan(*dims,dtype,128*1024**2,alg,None)
            if not plan.get('joint_split_gw32') or plan.get('gw_split32')!=split:
                raise AssertionError(('v32 large-N path not enabled',groups,
                                      alg,plan))
            if plan['split_gw_scratch32']>128*1024**2:
                raise AssertionError(('v32 workspace overflow',plan))

    # The new experiment is deliberately absent below the long-reduction
    # cutoff; this protects every B<=100 result from the completed v31 scan.
    fallback=kernel.winograd_plan(
        4,100,128,256,16,16,8,8,3,3,2,2,1,1,1,1,dtype,
        128*1024**2,'stride2_joint_split_gw8',None)
    if fallback.get('joint_split_gw32') or not fallback.get('joint_onebuf30'):
        raise AssertionError(('short-N call did not preserve v31',fallback))

    # Exercise every optional-input combination on a cheap shape whose
    # B*P*Q still crosses the v32 threshold.
    for alg in candidates:
        for opts in itertools.product((False,True),repeat=3):
            args=make_args((2,600,5,7,9,11,3,3),dtype,'cuda',st=(2,2),
                           pad=(1,1),opts=opts)
            compare(args,precision,budget=128,algorithm=alg);count+=1

    # Validate the exact performance target with both split schedules. The
    # timed benchmark will independently validate G=2 and G=4 before timing.
    args=make_args((1,1000,128,256,16,16,3,3),dtype,'cuda',st=(2,2),
                   pad=(1,1),opts=(True,True,False))
    for alg in candidates:
        compare(args,precision,budget=128,algorithm=alg);count+=1

    # Explicit user split-K never composes with the internal packed split.
    args=make_args((2,600,5,7,9,11,3,3),dtype,'cuda',st=(2,2),
                   pad=(1,1),opts=(True,True,True))
    fallback=kernel.winograd_plan(
        2,600,5,7,9,11,5,6,3,3,2,2,1,1,1,1,dtype,
        128*1024**2,'stride2_joint_split_gw4',4)
    if fallback.get('joint_split_gw32'):
        raise AssertionError(('explicit split-K entered v32 path',fallback))
    compare(args,precision,split=4,budget=128,
            algorithm='stride2_joint_split_gw4');count+=1
    print('v32 correctness passed:',count,'full API comparisons',flush=True)


def make_args(shape,dtype,device,st=(1,1),pad=(1,1),dil=(1,1),opts=(True,True,True),mask=(True,True,True)):
    g,f,c,o,h,w,r,s=shape
    p=(h+2*pad[0]-dil[0]*(r-1)-1)//st[0]+1
    q=(w+2*pad[1]-dil[1]*(s-1)-1)//st[1]+1
    def rand(*a): return torch.randn(*a,device=device,dtype=dtype)
    x=rand(g,f,c,h,w); wt=rand(g,o,c,r,s); d=rand(g,f,o,p,q)
    ai=rand(*x.shape) if opts[0] else None
    aw=rand(*wt.shape) if opts[1] else None
    ab=rand(g,o) if opts[2] else None
    return (ai,aw,ab,d,wt,x,list(st),list(pad),list(dil),False,[0,0],1,list(mask))


def compare(args,precision,split=None,budget=32,algorithm="auto"):
    got=custom(*args,precision=precision,split_k=split,workspace_limit_mb=budget,algorithm=algorithm)
    ref=aten_group_first(args)
    return check(got,ref,args[5].dtype,precision,args=args)


def correctness(dtype,precision):
    count=0
    cases=[((2,2,17,33,7,9,3,3),(1,1),(1,1),(1,1)),
           ((2,2,32,64,9,11,3,3),(2,2),(1,1),(1,1)),
           ((1,1,3,64,17,17,7,7),(2,2),(3,3),(1,1)),
           ((2,2,3,17,9,11,3,3),(1,1),(1,1),(1,1)),
           ((3,2,5,7,10,11,3,2),(2,1),(2,1),(2,1)),
           ((2,1,33,65,4,5,1,1),(1,1),(0,0),(1,1)),
           ((2,1000,2,3,4,4,3,3),(1,1),(1,1),(1,1))]
    for shape,st,pad,dil in cases:
        args=make_args(shape,dtype,'cuda',st,pad,dil)
        for sk in (1,4):
            print('check',shape,'split',sk,compare(args,precision,sk),flush=True); count+=1
    # All optional-input combinations and a false mask: mirrors ATen 2.5.1.
    for opts in itertools.product((False,True),repeat=3):
        args=make_args((2,2,3,7,5,6,3,3),dtype,'cuda',opts=opts)
        print('optional',opts,compare(args,precision,1),flush=True); count+=1
    args=make_args((2,2,3,7,5,6,3,3),dtype,'cuda',mask=(False,False,False))
    print('ATen output_mask compatibility',compare(args,precision,1),flush=True); count+=1
    if dtype==torch.float32:
        for shape,pad in [((2,2,17,33,7,9,3,3),(1,1)),((2,3,32,64,6,7,3,3),(0,2)),((2,10,128,128,16,16,3,3),(1,1))]:
            args=make_args(shape,dtype,'cuda',pad=pad)
            print('Packed Winograd forced',shape,compare(args,precision,algorithm='winograd'),flush=True);count+=1
        for shape,budget in [((5,2,3,7,5,6,3,3),.06),((2,35,3,7,4,5,3,3),.02)]:
            args=make_args(shape,dtype,'cuda')
            print('Packed group/tile tail',shape,compare(args,precision,budget=budget,algorithm='winograd'),flush=True);count+=1
        for opts in itertools.product((False,True),repeat=3):
            args=make_args((2,2,3,7,5,6,3,3),dtype,'cuda',opts=opts)
            print('Packed Winograd optional',opts,compare(args,precision,algorithm='winograd'),flush=True);count+=1
    if dtype==torch.float32:
        for sk in (1,2,4):
            args=make_args((3,35,17,33,7,9,3,3),dtype,'cuda')
            print('V8 joint split',sk,compare(args,precision,split=sk,budget=1,algorithm='winograd_joint'),flush=True);count+=1
        args=make_args((2,3,17,33,7,9,3,3),dtype,'cuda')
        ref=aten_group_first(args)
        for i in range(3):
            flags=tuple(j==i for j in range(3))
            got=custom(*args,precision=precision,algorithm='winograd',compute=flags)
            print('V8 isolated stage',i,check(got,tuple(ref[j] if j==i else None for j in range(3)),dtype,precision,args=args),flush=True);count+=1
    if dtype==torch.float32:
        for alg in ('winograd_separate','winograd_joint','winograd_v6'):
            args=make_args((3,5,17,33,7,9,3,3),dtype,'cuda',pad=(0,2))
            print('Schedule check',alg,compare(args,precision,budget=1,algorithm=alg),flush=True);count+=1
    if dtype==torch.float32:
        args=make_args((2,1000,2,3,4,4,3,3),dtype,'cuda')
        print('V8 long reduction',compare(args,precision,budget=.1,split=2,algorithm='winograd_joint'),flush=True);count+=1
    if dtype==torch.float32:
        for shape in ((5,3,33,17,7,9,3,3),(8,1,17,33,7,9,3,3),(2,3,128,128,8,8,3,3)):
            args=make_args(shape,dtype,'cuda')
            print('V8 fused ggO',shape,compare(args,precision,budget=4,algorithm='winograd'),flush=True);count+=1
        args=make_args((3,2,5,7,5,6,3,3),dtype,'cuda')
        providers,singles=serial_providers(args,dict(precision=precision,algorithm='winograd',workspace_limit_mb=1))
        validate_serial(providers,singles,dtype,precision)
        print('PASS serial baselines use distinct group slices',flush=True);count+=1
    correctness_large(dtype,precision)
    print(f'PASS: {count} GPU cases; compared directly with aten._convolution_double_backward',flush=True)


def _correctness10(dtype,precision,algorithm):
    if dtype!=torch.float32:raise ValueError('Large Winograd path requires fp32')
    count=0
    if dtype==torch.float32:
        for shape,budget,sk in [((3,17,5,7,5,7,3,3),.25,16),((3,137,17,33,7,9,3,3),4,None),
                                ((2,1000,2,3,4,4,3,3),1,None),
                                ((3,35,33,17,7,9,3,3),.1,2),
                                ((2,100,128,128,16,16,3,3),128,None),
                                ((2,100,64,64,32,32,3,3),128,None)]:
            args=make_args(shape,dtype,'cuda')
            print('V10 large/tail',shape,compare(args,precision,budget=budget,split=sk,algorithm=algorithm),flush=True);count+=1
        for opts in itertools.product((False,True),repeat=3):
            args=make_args((2,3,5,7,5,6,3,3),dtype,'cuda',opts=opts)
            print('V10 optional',opts,compare(args,precision,budget=1,algorithm=algorithm),flush=True);count+=1
        args=make_args((2,5,17,33,7,9,3,3),dtype,'cuda',pad=(0,2))
        ref=aten_group_first(args)
        for i in range(3):
            flags=tuple(j==i for j in range(3))
            got=custom(*args,precision=precision,algorithm=algorithm,workspace_limit_mb=1,compute=flags)
            print('V10 isolated stage',i,check(got,tuple(ref[j] if j==i else None for j in range(3)),dtype,precision,args=args),flush=True);count+=1
    print(f'PASS: {count} V10 GPU cases; compared directly with ATen',flush=True)


def correctness_large(dtype,precision):
    for alg in ('winograd_tail','winograd_group'):
        print('CHECK',alg,flush=True)
        _correctness10(dtype,precision,alg)


class ValidationError(AssertionError):
    def __init__(self,details):
        self.details=details
        super().__init__(json.dumps(details,ensure_ascii=False))


def fp64_gw_samples(args,indices,got,ref):
    # Vectorized independent direct FP64 sums. Scratch is bounded by 16*4096 entries.
    ai,_,_,d,wt,x,st,pad,dil,*_=args
    g,f,c,h,w=x.shape; _,o,_,r,s=wt.shape; p,q=d.shape[-2:]
    out=[];aa=got.reshape(-1);bb=ref.reshape(-1)
    for off in range(0,len(indices),16):
        ids=torch.tensor(indices[off:off+16],device=x.device,dtype=torch.long)
        j=ids;ss=j%s;j=j//s;rr=j%r;j=j//r;ci=j%c;j=j//c;oc=j%o;gg=j//o
        total=torch.zeros(len(ids),device=x.device,dtype=torch.float64)
        for begin in range(0,f*p*q,4096):
            k=torch.arange(begin,min(begin+4096,f*p*q),device=x.device)[None,:]
            ff=k//(p*q);py=k//q%p;qx=k%q
            ih=py*st[0]-pad[0]+rr[:,None]*dil[0];iw=qx*st[1]-pad[1]+ss[:,None]*dil[1]
            valid=(ih>=0)&(ih<h)&(iw>=0)&(iw<w)
            v=ai[gg[:,None],ff,ci[:,None],ih.clamp(0,h-1),iw.clamp(0,w-1)].double()
            total+=(v*d[gg[:,None],ff,oc[:,None],py,qx].double()*valid).sum(dim=1)
        for index,oracle,custom_val,aten_val in zip(ids.tolist(),total.tolist(),aa[ids].tolist(),bb[ids].tolist()):
            out.append(dict(index=index,fp64=oracle,triton=custom_val,aten=aten_val,
                            triton_abs_error=abs(custom_val-oracle),aten_abs_error=abs(aten_val-oracle)))
    return out


def fp64_spatial_samples(args,indices,got,ref,name):
    """Direct FP64 oracle at all mismatching positions plus random controls."""
    ai,aw,ab,d,wt,x,st,pad,dil,*_=args
    g,f,c,h,w=x.shape;_,o,_,r,s=wt.shape;p,q=d.shape[-2:]
    out=[];aa=got.reshape(-1);bb=ref.reshape(-1)
    for off in range(0,len(indices),16):
        ids=torch.tensor(indices[off:off+16],device=x.device,dtype=torch.long)
        j=ids
        if name=='gI':
            ix=j%w;j=j//w;iy=j%h;j=j//h;ci=j%c;j=j//c;ff=j%f;gg=j//f
            length=o*r*s
        else:
            qx=j%q;j=j//q;py=j%p;j=j//p;oc=j%o;j=j//o;ff=j%f;gg=j//f
            length=c*r*s
        total=torch.zeros(len(ids),device=x.device,dtype=torch.float64)
        for begin in range(0,length,4096):
            k=torch.arange(begin,min(begin+4096,length),device=x.device)[None,:]
            rr=k//s%r;ss=k%s;ch=k//(r*s)
            if name=='gI':
                hp=iy[:,None]+pad[0]-rr*dil[0];wp=ix[:,None]+pad[1]-ss*dil[1]
                yy=hp//st[0];xx=wp//st[1]
                valid=(hp%st[0]==0)&(wp%st[1]==0)&(yy>=0)&(yy<p)&(xx>=0)&(xx<q)
                v=d[gg[:,None],ff[:,None],ch,yy.clamp(0,p-1),xx.clamp(0,q-1)].double()
                weight=aw[gg[:,None],ch,ci[:,None],rr,ss].double()
                total+=(v*weight*valid).sum(dim=1)
            else:
                yy=py[:,None]*st[0]-pad[0]+rr*dil[0];xx=qx[:,None]*st[1]-pad[1]+ss*dil[1]
                valid=(yy>=0)&(yy<h)&(xx>=0)&(xx<w)
                if ai is not None:
                    v=ai[gg[:,None],ff[:,None],ch,yy.clamp(0,h-1),xx.clamp(0,w-1)].double()
                    weight=wt[gg[:,None],oc[:,None],ch,rr,ss].double()
                    total+=(v*weight*valid).sum(dim=1)
                if aw is not None:
                    v=x[gg[:,None],ff[:,None],ch,yy.clamp(0,h-1),xx.clamp(0,w-1)].double()
                    weight=aw[gg[:,None],oc[:,None],ch,rr,ss].double()
                    total+=(v*weight*valid).sum(dim=1)
        if name=='ggO' and ab is not None:total+=ab[gg,oc].double()
        for index,oracle,custom_val,aten_val in zip(ids.tolist(),total.tolist(),aa[ids].tolist(),bb[ids].tolist()):
            out.append(dict(index=index,fp64=oracle,triton=custom_val,aten=aten_val,
                            triton_abs_error=abs(custom_val-oracle),aten_abs_error=abs(aten_val-oracle)))
    return out


def check(got,ref,dtype,precision,args=None):
    atol,rtol,lim={torch.float32:(.003,.002,2e-4),torch.float16:(.08,.02,.003),torch.bfloat16:(.6,.08,.02)}[dtype]
    if dtype==torch.float32 and precision=='tf32':atol,rtol,lim=.1,.03,.004
    out={};failures=[]
    for name,x,y in zip(('ggO','gI','gW'),got,ref):
        if x is None or y is None:
            assert x is y is None,(name,'None mismatch');continue
        assert x.shape==y.shape,(name,'shape mismatch')
        xx=x.reshape(-1); yy=y.reshape(-1)
        ds=ys=mx=0.;mismatches=0;bad_indices=[]
        for begin in range(0,x.numel(),262144):
            a=xx[begin:begin+262144]; b=yy[begin:begin+262144]
            d=a.double()-b.double(); ds+=d.square().sum().item(); ys+=b.double().square().sum().item()
            mx=max(mx,d.abs().max().item())
            bad=~torch.isclose(a,b,atol=atol,rtol=rtol)
            mismatches+=bad.sum().item()
            if len(bad_indices)<4097:
                bad_indices.extend((bad.nonzero().flatten()[:4097-len(bad_indices)]+begin).tolist())
        rel=(ds/max(ys,1e-60))**.5
        out[name+'_max_abs']=mx;out[name+'_rel_l2']=rel
        if mismatches or not rel<=lim:
            failure=dict(output=name,shape=list(x.shape),mismatches=mismatches,numel=x.numel(),
                         max_abs=mx,rel_l2=rel,atol=atol,rtol=rtol,rel_l2_limit=lim,first_bad_indices=bad_indices[:8])
            if dtype==torch.float32 and args is not None and 0<mismatches<=4096 and rel<=lim:
                rng=random.Random(73021)
                indices=sorted(set(bad_indices+rng.sample(range(x.numel()),min(64,x.numel()))))
                samples=fp64_gw_samples(args,indices,x,y) if name=='gW' else fp64_spatial_samples(args,indices,x,y,name)
                oracle_ok=all(v['triton_abs_error']<=atol+rtol*abs(v['fp64']) for v in samples)
                failure['fp64_checked']=len(samples);failure['fp64_samples']=samples[:8]
                if oracle_ok:
                    out[name+'_validation']='fp64_resolved'
                    out[name+'_aten_mismatches']=mismatches
                    out[name+'_fp64_checked']=len(samples)
                    out[name+'_fp64_max_abs']=max(v['triton_abs_error'] for v in samples)
                    out[name+'_aten_fp64_max_abs']=max(v['aten_abs_error'] for v in samples)
                    continue
            failures.append(failure)
    if failures:raise ValidationError(dict(metrics=out,failures=failures))
    return out


def once(fn,n):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); t=time.perf_counter(); start.record()
    for _ in range(n):
        result=fn()
        del result
    end.record();end.synchronize()
    return start.elapsed_time(end)/n,(time.perf_counter()-t)*1000/n


def stats(values):
    mean=statistics.mean(values)
    return dict(mean=mean,median=statistics.median(values),min=min(values),max=max(values),
                cv_pct=100*statistics.stdev(values)/mean if len(values)>1 and mean else 0)


def paired(funcs,a):
    counts={}; raw={k:dict(event=[],wall=[]) for k in funcs}
    for name,fn in funcs.items():
        print('  preparing',name,flush=True)
        for _ in range(a.warmup):fn()
        torch.cuda.synchronize(); t=time.perf_counter()
        while (time.perf_counter()-t)*1000<a.warmup_ms:
            fn();torch.cuda.synchronize()
        probe,_=once(fn,3)
        counts[name]=max(2,min(a.iters,int(a.target_ms/max(probe,.001))))
    rng=random.Random(123)
    for repeat in range(a.rounds):
        order=list(funcs);rng.shuffle(order)
        for name in order:
            ev,wall=once(funcs[name],counts[name]);raw[name]['event'].append(ev);raw[name]['wall'].append(wall)
    out={}
    for name,fn in funcs.items():
        out[name+'_iters']=counts[name]
        for mode in ('event','wall'):
            out[name+'_'+mode+'_rounds']=json.dumps(raw[name][mode])
            for k,val in stats(raw[name][mode]).items():out[name+'_'+mode+'_'+k]=val
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
        val=fn();torch.cuda.synchronize();out[name+'_peak_extra_bytes']=torch.cuda.max_memory_allocated()-base;del val
    return out


def summarize_cuda_events(events,repeats):
    """Use only device events: CPU-parent CUDA totals would double-count kernels.
    Union time is observed CUDA activity coverage, NOT SM utilization. Profiler
    overhead can enlarge gaps, so these timings are not benchmark speedups.
    """
    groups={};intervals=[];families={}
    known=('_win_weight_all14','_gi_weight_pack14','_gi_reordered14','_ggo_pack14','_ggo_mm14','_gi_phase12','_pack_pair9','_weight_pair9','_packed_dual8','_pack_dout7','_packed_gw7','_packed_conv7','_pack_tiles_all','_packed_gw_acc','_packed_conv','_packed_gw','_pack_tiles','_win_weight_pack',
           '_win_gw_out','_win_out','_accumulate','_ggo','_gi','_gw','_reduce')
    for e in events:
        if str(e.device_type).split('.')[-1]!='CUDA':continue
        start=float(e.time_range.start);end=float(e.time_range.end)
        if end<start:continue
        name=e.name;duration=end-start
        intervals.append((start,end))
        rec=groups.setdefault(name,dict(name=name,calls=0,total_us=0.))
        rec['calls']+=1;rec['total_us']+=duration
        family=next((k for k in known if name==k or name.startswith(k+'_')), 'other_cuda')
        fam=families.setdefault(family,dict(name=family,calls=0,total_us=0.))
        fam['calls']+=1;fam['total_us']+=duration
    if not intervals:
        return dict(status='unavailable',reason='No CUDA device events: CUPTI/Kineto tracing unavailable; normal benchmark is still valid.')
    intervals.sort();left,right=intervals[0];union=0.
    for start,end in intervals[1:]:
        if start>right:union+=right-left;left,right=start,end
        else:right=max(right,end)
    union+=right-left;span=max(b for _,b in intervals)-intervals[0][0]
    total=sum(r['total_us'] for r in groups.values())
    def finish(records):
        result=[]
        for r in sorted(records.values(),key=lambda r:r['total_us'],reverse=True):
            result.append(dict(name=r['name'],calls_per_api=r['calls']/repeats,
                ms_per_api=r['total_us']/repeats/1000,mean_us=r['total_us']/r['calls'],
                activity_share_pct=100*r['total_us']/total if total else 0.))
        return result
    return dict(status='ok',repeats=repeats,activity_ms_per_api=total/repeats/1000,
                timeline_span_ms_per_api=span/repeats/1000,
                uncovered_span_pct=100*(span-union)/span if span else 0.,
                families=finish(families),kernels=finish(groups))


def profile_callable(fn,repeats):
    # All compilation/autotuning and regular timing have finished before entry.
    from torch.profiler import profile,ProfilerActivity
    for _ in range(3):fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
                 record_shapes=False,profile_memory=False,with_stack=False) as prof:
        for _ in range(repeats):
            val=fn()
            del val
        torch.cuda.synchronize()
    result=summarize_cuda_events(prof.events(),repeats)
    result['note']='Instrumented trace only; CUDA activity span gaps are not SM utilization or a pure CPU-overhead measurement.'
    return result


def graph_probe(fn,a):
    """Diagnostic only: capture/allocation excluded, static outputs retained.
    Apply equally to Triton and ATen. Does not change production dispatch.
    """
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):held=fn()
    graph.replay();torch.cuda.synchronize()
    reference=fn();torch.cuda.synchronize()
    for actual,expected in zip(held,reference):
        if actual is None or expected is None:
            if actual is not expected:raise AssertionError('Graph optional output mismatch')
        else:torch.testing.assert_close(actual,expected,rtol=.002,atol=.003)
    del reference
    for _ in range(a.warmup):graph.replay()
    probe,_=once(graph.replay,3)
    n=max(2,min(a.iters,int(a.target_ms/max(probe,.001))))
    values=[once(graph.replay,n)[0] for _ in range(a.rounds)]
    return dict(status='ok',event_ms=stats(values),iters=n,
                note='Replay only; static outputs, capture and allocations excluded. Not an eager API replacement.')


def split_group_args(args):
    # Views prepared outside timing. Each group receives its own original values.
    return [tuple(None if t is None else t[i:i+1] for t in args[:6])+tuple(args[6:])
            for i in range(args[5].shape[0])]


def serial_providers(args,kw):
    singles=split_group_args(args)
    ready=[prepare_aten(one) for one in singles]
    def serial_triton():
        # Retain every group's outputs until the full G-group task is complete.
        return [custom(*one,**kw) for one in singles]
    def serial_aten():
        return [torch.ops.aten._convolution_double_backward(*one) for one in ready]
    return dict(serial_triton=serial_triton,serial_aten=serial_aten),singles


def validate_serial(providers,singles,dtype,precision):
    # No concatenation or validation is included in timed serial work.
    for name,fn in providers.items():
        results=fn()
        for one,got in zip(singles,results):
            if name=='serial_aten':got=tuple(None if t is None else t.unsqueeze(0) for t in got)
            check(got,aten_group_first(one),dtype,precision,args=one)
        del results


def scaling_metrics(row,g,f):
    t=row['triton_event_median'];sa=row['serial_aten_event_median'];st=row['serial_triton_event_median']
    return dict(serial_aten_over_fused_triton=sa/t,serial_triton_over_fused_triton=st/t,
                serial_aten_over_grouped_aten=sa/row['aten_prepacked_event_median'],
                serial_aten_over_serial_triton=sa/st,
                fusion_efficiency_pct=100*st/t/g,
                group_images_per_second=g*f*1000/t,
                serial_triton_group_images_per_second=g*f*1000/st,
                scaling_stable=max(row[k+'_event_cv_pct'] for k in ('triton','serial_aten','serial_triton','aten_prepacked'))<=5)


def run(case,g,f,a):
    c,o,h,w,r,s=(case[k] for k in ('C','O','H','W','R','S'))
    st,pad,dil=(case[k] for k in ('stride','padding','dilation'))
    p=(h+2*pad-dil*(r-1)-1)//st+1;q=(w+2*pad-dil*(s-1)-1)//st+1
    nx=g*f*c*h*w;nd=g*f*o*p*q;nw=g*o*c*r*s
    dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[a.dtype]
    item=4 if dtype==torch.float32 else 2
    row=dict(case,G=g,F=f,B=f,P=p,Q=q,status='pending')
    cached=BASELINES.get(baseline_key(row)) if a.aten_baseline else None
    if a.aten_baseline and cached is None and a.missing_baseline=='skip':
        return dict(row,status='skip_missing_baseline',error='No measured matching baseline; no automatic retiming')
    # Conservative estimate, not a promise of fitting cuDNN workspaces.
    estimate=item*(7*nx+6*nd+6*nw)+int(a.workspace_mb*1024**2)+32*1024**2
    row['estimated_bytes']=estimate
    free,_=torch.cuda.mem_get_info()
    if max(nx,nd,nw)>=2**30 or g*f>65535:
        return dict(row,status='skip_index_limit')
    if estimate>free*a.memory_fraction:
        return dict(row,status='skip_memory_estimate',free_bytes=free)
    seed=123+int(hashlib.sha256(str((g,f,c,o,h,w,r,s,st,pad,dil,case['bias'])).encode()).hexdigest()[:8],16)
    torch.manual_seed(seed);row['seed']=seed
    args=make_args((g,f,c,o,h,w,r,s),dtype,'cuda',(st,)*2,(pad,)*2,(dil,)*2,opts=(True,True,case['bias']))
    kw=dict(precision=a.precision,workspace_limit_mb=a.workspace_mb,split_k=a.split_k,algorithm=a.algorithm)
    primary_kw=kw
    selected_algorithm=a.algorithm
    if a.algorithm=='auto':
        signature=kernel.convolution_double_backward_signature(
            *args,precision=a.precision,workspace_limit_mb=a.workspace_mb,
            split_k=a.split_k)
        selected_algorithm,selector_source=kernel._select_algorithm_cached(signature)
        row.update(selector_algorithm=selected_algorithm,selector_source=selector_source,
                   selector_cached_during_warmup=True,
                   selector_bound_during_warmup=False)
    plan=kernel.winograd_plan(g,f,c,o,h,w,p,q,r,s,st,st,pad,pad,dil,dil,dtype,int(a.workspace_mb*1024**2),selected_algorithm,a.split_k)
    if plan.get('gw_candidate19'):
        sk19=a.split_k or kernel.workspace_split(g,f,c,o,r,s,p,q,torch.cuda.get_device_properties(args[5].device).multi_processor_count,int(a.workspace_mb*1024**2))
        plan.update(kernel._gw_plan19(g,f,c,o,p,q,sk19,int(a.workspace_mb*1024**2)))
        plan['scratch_bytes']=max(plan['scratch_bytes'],plan['gw_scratch19'])
        plan['dispatch_route']='stride2_packed_gw' if plan['gw_groups19'] else 'stride2_packed_gi'
        plan['dispatch_reason']=plan['gw_reason19']
    ref=aten_group_first(args); got=kernel.convolution_double_backward(*args,**primary_kw)
    row.update(check(got,ref,dtype,a.precision,args=args));del got
    if plan.get('version') in (8,9,10) and plan.get('joint_nt'):
        row['joint_ggO_config']=str(getattr(kernel._packed_dual8,'best_config','unavailable'))
    if a.stages:
        for i,name in enumerate(('ggO','gI','gW')):
            flags=tuple(j==i for j in range(3))
            val=kernel.convolution_double_backward(*args,**kw,compute=flags)
            check(val,tuple(ref[j] if j==i else None for j in range(3)),dtype,a.precision,args=args);del val
            active=(plan['ggO_nt'],plan['gI_nt'],plan['gW_split'])[i]
            selected=((kernel._packed_conv7 if i<2 else kernel._packed_gw7) if plan.get('version') in (7,8,9,10) else (kernel._packed_conv if i<2 else (kernel._packed_gw_acc if plan.get('version')==6 else kernel._packed_gw))) if active else (kernel._ggo,kernel._gi,kernel._gw)[i]
            if i==1 and plan.get('phase_gI'):selected=kernel._gi_phase12
            if i==1 and plan.get('gi_weight14'):selected=kernel._gi_reordered14
            if i==1 and plan.get('gi_static15'):selected=kernel._gi_static15
            if i==1 and plan.get('gi_static_weight16'):selected=kernel._gi_static_weight16
            if i==1 and plan.get('gi_packed17'):selected=kernel._gi_mm17
            if i==2 and plan.get('gw_static20'):selected=kernel._gw_static20
            if i==2 and plan.get('gw_groups19',0):selected=kernel._gw_mm19
            if i==0 and plan.get('ggo_pack14'):selected=kernel._ggo_mm14
            row[name+'_config']=str(getattr(selected,'best_config','unavailable'))
    del ref
    ready=prepare_aten(args) if cached is None else None
    funcs={'triton':lambda:kernel.convolution_double_backward(*args,**primary_kw),
           'aten_prepacked':lambda:torch.ops.aten._convolution_double_backward(*ready),
           'aten_e2e':lambda:aten_group_first(args)}
    if a.compare_v20:
        funcs['triton_v18']=lambda:kernel.convolution_double_backward(*args,**dict(kw,algorithm='stride2_packed_gi_tail'))
        row['candidate_validation20']=check(funcs['triton_v18'](),aten_group_first(args),dtype,a.precision,args=args)
    if a.compare_v19:
        funcs['triton_v17']=lambda:kernel.convolution_double_backward(*args,**dict(kw,algorithm='stride2_packed_gi'))
        row['candidate_validation19']=check(funcs['triton_v17'](),aten_group_first(args),dtype,a.precision,args=args)
    if a.compare_v18:
        funcs['triton_v17']=lambda:kernel.convolution_double_backward(*args,**dict(kw,algorithm='stride2_packed_gi'))
        row['candidate_validation18']=check(funcs['triton_v17'](),aten_group_first(args),dtype,a.precision,args=args)
    if a.compare_v17:
        funcs['triton_v16']=lambda:kernel.convolution_double_backward(*args,**dict(kw,algorithm='stride2_static_weight'))
        row['candidate_validation17']=check(funcs['triton_v16'](),aten_group_first(args),dtype,a.precision,args=args)
    if a.compare_v16:
        row['candidate_plans16']={};row['candidate_validation16']={}
        for label,alg in [('triton_v14','adaptive14'),('triton_static15','stride2_static')]:
            funcs[label]=lambda alg=alg:kernel.convolution_double_backward(*args,**dict(kw,algorithm=alg))
            row['candidate_validation16'][label]=check(funcs[label](),aten_group_first(args),dtype,a.precision,args=args)
            row['candidate_plans16'][label]=kernel.winograd_plan(g,f,c,o,h,w,p,q,r,s,st,st,pad,pad,dil,dil,dtype,int(a.workspace_mb*1024**2),alg,a.split_k)
    if a.compare_v15:
        row['candidate_plans15']={};row['candidate_validation15']={}
        for label,alg in [('triton_v14','adaptive14'),('triton_shared15','stride2_shared')]:
            funcs[label]=lambda alg=alg:kernel.convolution_double_backward(*args,**dict(kw,algorithm=alg))
            row['candidate_validation15'][label]=check(funcs[label](),aten_group_first(args),dtype,a.precision,args=args)
            row['candidate_plans15'][label]=kernel.winograd_plan(g,f,c,o,h,w,p,q,r,s,st,st,pad,pad,dil,dil,dtype,int(a.workspace_mb*1024**2),alg,a.split_k)
    if a.compare_v13:
        funcs['triton_v13']=lambda:kernel.convolution_double_backward(*args,**dict(kw,algorithm='adaptive13'))
        check(funcs['triton_v13'](),aten_group_first(args),dtype,a.precision,args=args)
    if a.compare_weight_only:
        funcs['triton_weight_only']=lambda:kernel.convolution_double_backward(*args,**dict(kw,algorithm='stride2_weight'))
        check(funcs['triton_weight_only'](),aten_group_first(args),dtype,a.precision,args=args)
    if a.compare_v12:
        funcs['triton_v12']=lambda:kernel.convolution_double_backward(*args,**dict(kw,algorithm='adaptive12'))
        check(funcs['triton_v12'](),aten_group_first(args),dtype,a.precision,args=args)
    if a.compare_v11:
        funcs['triton_v11']=lambda:kernel.convolution_double_backward(*args,**dict(kw,algorithm='shape_safe'))
        check(funcs['triton_v11'](),aten_group_first(args),dtype,a.precision,args=args)
    if a.compare_schedules:
        for alg in ('winograd_joint','winograd_separate'):
            alt=dict(kw,algorithm=alg)
            funcs[alg]=lambda alt=alt:kernel.convolution_double_backward(*args,**alt)
            check(funcs[alg](),aten_group_first(args),dtype,a.precision,args=args)
    for enabled,label,algorithm in [(a.compare_tail,'triton_tail','winograd_tail'),(a.compare_v9,'triton_v9','winograd_v9')]:
        if enabled:
            funcs[label]=lambda algorithm=algorithm:kernel.convolution_double_backward(*args,**dict(kw,algorithm=algorithm))
            check(funcs[label](),aten_group_first(args),dtype,a.precision,args=args)
    if a.compare_v8:
        funcs['triton_v8']=lambda:kernel.convolution_double_backward(*args,**dict(kw,algorithm='winograd_v8'))
    if a.compare_v7:
        funcs['triton_v7']=lambda:kernel.convolution_double_backward(*args,**dict(kw,algorithm='winograd_v7'))
    if a.compare_v6:
        funcs['triton_v6']=lambda:kernel.convolution_double_backward(*args,**dict(kw,algorithm='winograd_v6'))
    if a.compare_v5:
        funcs['triton_v5']=lambda:kernel.convolution_double_backward(*args,**dict(kw,algorithm='winograd_v5'))
    if a.compare_implicit and selected_algorithm!='implicit' and plan.get('dispatch_route')!='implicit':
        implicit_kw=dict(kw,algorithm='implicit')
        funcs['triton_implicit']=lambda:kernel.convolution_double_backward(*args,**implicit_kw)
    if a.stages:
        for i,name in enumerate(('ggO','gI','gW')):
            flags=tuple(j==i for j in range(3))
            funcs['stage_'+name]=lambda flags=flags:kernel.convolution_double_backward(*args,**kw,compute=flags)
    if a.stages and a.compare_v20:
        funcs['stage_gW_v18']=lambda:kernel.convolution_double_backward(*args,**dict(kw,algorithm='stride2_packed_gi_tail'),compute=(False,False,True))
    if a.stages and a.compare_v19:
        for label,alg in [('stage_gI_v17','stride2_packed_gi'),('stage_gI_v18','stride2_packed_gi_tail'),('stage_gW_v17','stride2_packed_gi')]:
            flags=(False,True,False) if 'gI' in label else (False,False,True)
            funcs[label]=lambda alg=alg,flags=flags:kernel.convolution_double_backward(*args,**dict(kw,algorithm=alg),compute=flags)
    if a.scaling:
        serial,singles=serial_providers(args,kw)
        validate_serial(serial,singles,dtype,a.precision)
        funcs.update(serial)
    if a.serial_check:
        singles=split_group_args(args)
        serial={}
        for label,alg in [('serial_candidate',a.algorithm),('serial_implicit','implicit')]:
            serial[label]=lambda alg=alg:[kernel.convolution_double_backward(*one,**dict(kw,algorithm=alg)) for one in singles]
        validate_serial(serial,singles,dtype,a.precision)
        funcs.update(serial)
    if cached is not None:
        # ATen was called above only to validate outputs, not for timing.
        funcs.pop('aten_prepacked');funcs.pop('aten_e2e')
    row.update(paired(funcs,a))
    if a.compare_implicit and (selected_algorithm=='implicit' or plan.get('dispatch_route')=='implicit'):
        row['implicit_comparison']='same implementation; no redundant timing'
        for key,value in list(row.items()):
            if key.startswith('triton_'):row['triton_implicit_'+key[len('triton_'):]]=value
    if cached:
        for old,new in [('grouped_aten','aten_prepacked'),('grouped_e2e','aten_e2e')]:
            for field in ('event_median','wall_median','event_cv_pct','peak_extra_bytes','task_peak_bytes'):
                row[new+'_'+field]=cached[old+'_'+field]
        row['aten_baseline_reused']=True
        row['aten_baseline_source']=str(a.aten_baseline)
        if 'serial_aten_event_median' in cached:
            row['serial_aten_baseline_ms']=cached['serial_aten_event_median']
            row['serial_aten_baseline_cv_pct']=cached['serial_aten_event_cv_pct']
        row['best_aten_baseline_ms']=min(cached.get('serial_aten_event_median',float('inf')),cached['grouped_aten_event_median'])
        row['best_aten_speedup']=row['best_aten_baseline_ms']/row['triton_event_median']
        stores={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in args[:6] if t is not None}
        row['triton_task_peak_bytes']=sum(stores.values())+row['triton_peak_extra_bytes']

    if not cached:
        stores={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in args[:6] if t is not None}
        row['triton_task_peak_bytes']=sum(stores.values())+row['triton_peak_extra_bytes']
        row['aten_e2e_task_peak_bytes']=sum(stores.values())+row['aten_e2e_peak_extra_bytes']
        for t in ready[:6]:
            if t is not None:stores[t.untyped_storage().data_ptr()]=t.untyped_storage().nbytes()
        row['aten_prepacked_task_peak_bytes']=sum(stores.values())+row['aten_prepacked_peak_extra_bytes']

    if a.serial_check:
        serial_ms=min(row['serial_candidate_event_median'],row['serial_implicit_event_median'])
        row['measured_best_serial_ms']=serial_ms
        row['measured_serial_speedup']=serial_ms/row['triton_event_median']
        row['serial_check_stable']=max(row[n+'_event_cv_pct'] for n in ('triton','serial_candidate','serial_implicit'))<=a.max_cv
        row['serial_guard']='unstable' if not row['serial_check_stable'] else 'slower_than_serial' if row['measured_serial_speedup']<.97 else 'near_tie' if row['measured_serial_speedup']<=1.03 else 'fused_faster'
        row['serial_contract']='distinct group views, one stream, no inner synchronization, retain all outputs; output assembly excluded; diagnostic only, no automatic fallback'
    if a.scaling:
        row.update(scaling_metrics(row,g,f))
        row['scaling_stable']=max(row[k+'_event_cv_pct'] for k in ('triton','serial_aten','serial_triton','aten_prepacked'))<=a.max_cv
        fusion=row['serial_triton_over_fused_triton']
        row['fusion_policy']='unstable' if not row['scaling_stable'] else ('regression' if fusion<.97 else ('near_tie' if fusion<=1.03 else 'improvement'))
        row['serial_contract']='G distinct group views; same stream; no per-group synchronize; retain outputs; no concatenation'
        row['serial_algorithm']=a.algorithm
    sk=plan['gW_split'] or a.split_k or kernel.workspace_split(g,f,c,o,r,s,p,q,torch.cuda.get_device_properties(args[5].device).multi_processor_count,int(a.workspace_mb*1024**2))
    if plan.get('shared15'):sk=1
    row.update(split_k=sk,scratch_bytes=max(plan['scratch_bytes'],0 if plan['gW_split'] or sk==1 else 4*g*sk*o*c*r*s))
    for name,active,win,implicit in [('ggO',plan['ggO_nt'],kernel._packed_conv,kernel._ggo),('gI',plan['gI_nt'],kernel._packed_conv,kernel._gi),('gW',plan['gW_split'],kernel._packed_gw,kernel._gw)]:
        row[name+'_algorithm']=('winograd_v'+str(plan.get('version',5))) if active else 'implicit'
        # _packed_conv is shared by ggO/gI; capture each stage configuration immediately.
        row.setdefault(name+'_config',str(getattr(implicit,'best_config','unavailable')) if not active else 'not captured: enable stages for per-output configuration')
    if plan.get('phase_gI'):row['gI_algorithm']='stride2_phase'
    if plan.get('gi_weight14'):row['gI_algorithm']='stride2_weight_reorder'
    if plan.get('ggo_pack14'):row['ggO_algorithm']='stride2_bounded_pack'
    if plan.get('gi_static_weight16'):row['gI_algorithm']='stride2_static_reordered_weights'
    if plan.get('gi_static15'):row['gI_algorithm']='stride2_static_parity'
    if plan.get('shared15'):row['gW_algorithm']='shared_pack_compensated';row['shared_stage_note']='isolated ggO/gW use separate v14 paths; only full API shares packing'
    if plan.get('weight_all14'):row['weight_transform']='all16_reuse_loads'
    row['winograd_plan']=plan
    if plan.get('gi_packed17'):row['gI_algorithm']='stride2_parity_packed_gemm'
    if plan.get('gi_tail18'):row['gI_algorithm']='stride2_parity_packed_gemm_tail'
    if plan.get('wide_chunks27'):
        row['ggO_algorithm']='stride2_exact_tail_wide_chunks'
        row['gI_algorithm']='stride2_parity_packed_gemm_wide_chunks'
    if plan.get('stem7_gi28'):row['gI_algorithm']='stem7_parity_packed_gemm'
    if plan.get('stem7_gw28'):row['gW_algorithm']='stem7_static_cropped_incremental'
    if plan.get('stem7_ggo28'):row['ggO_algorithm']='stem7_bounded_im2col_gemm'
    if plan.get('gw_groups19',0):row['gW_algorithm']='stride2_packed_parallel_splitk'
    if plan.get('gw_static20'):row['gW_algorithm']='stride2_static_cropped_incremental'
    row['experimental']=a.algorithm.startswith('winograd') or a.algorithm in ('stem7_gi','stem7_gi_ggo','stem7_gi_gw','stem7_full','pointwise_stride2','stride2_wide_chunks','stride2_small_spatial','stride2_implicit_gw','stride2_packed_gw','stride2_packed_gi_tail','stride2_packed_gi','stride2_static_weight','stride2_static','stride2_shared','shape_safe','adaptive12','adaptive13','adaptive14','stride2_weight','stride2_packed','stride2_phase')
    row['joint_ggO_gW']=bool(plan.get('joint_nt') or plan.get('shared15'))
    for label in ('aten_prepacked','aten_e2e'):
        if label+'_event_median' not in row:continue
        row[label+'_speedup']=row[label+'_event_median']/row['triton_event_median']
    torch_ms=row.get('best_aten_baseline_ms',row['aten_prepacked_event_median'])
    speed=row.get('best_aten_speedup',row['aten_prepacked_speedup'])
    row['policy_reference']='best available cached ATen (serial if measured, grouped)' if cached else 'same-run grouped ATen'
    noisy=max(row['triton_event_cv_pct'],row['aten_prepacked_event_cv_pct'],row.get('serial_aten_baseline_cv_pct',0))>a.max_cv
    if noisy:row['policy']='unstable'
    elif g==1 or torch_ms<a.small_ms:
        bound=max(torch_ms*(1+a.small_slowdown_pct/100),torch_ms+a.small_extra_us/1000)
        row['policy']='allowed_small' if row['triton_event_median']<=bound else 'small_regression'
    elif .97<=speed<=1.03:row['policy']='inconclusive_near_tie'
    else:row['policy']='pass_candidate' if speed>1.03 else 'large_regression'
    if a.compare_v20:
        ratio=row['triton_v18_event_median']/row['triton_event_median']
        noisy=max(row['triton_v18_event_cv_pct'],row['triton_event_cv_pct'])>a.max_cv
        row['v18_guard']=dict(speedup=ratio,status='unstable' if noisy else 'regression' if ratio<.97 else 'near_tie' if ratio<=1.03 else 'faster',extra_memory_delta_bytes=row['triton_peak_extra_bytes']-row['triton_v18_peak_extra_bytes'])
    if a.compare_v18 or a.compare_v19:
        ratio=row['triton_v17_event_median']/row['triton_event_median']
        noisy=max(row['triton_v17_event_cv_pct'],row['triton_event_cv_pct'])>a.max_cv
        row['v17_guard']=dict(speedup=ratio,status='unstable' if noisy else 'regression' if ratio<.97 else 'near_tie' if ratio<=1.03 else 'faster',extra_memory_delta_bytes=row['triton_peak_extra_bytes']-row['triton_v17_peak_extra_bytes'])
    if a.compare_v17:
        ratio=row['triton_v16_event_median']/row['triton_event_median']
        noisy=max(row['triton_v16_event_cv_pct'],row['triton_event_cv_pct'])>a.max_cv
        row['v16_guard']=dict(speedup=ratio,status='unstable' if noisy else 'regression' if ratio<.97 else 'near_tie' if ratio<=1.03 else 'faster',extra_memory_delta_bytes=row['triton_peak_extra_bytes']-row['triton_v16_peak_extra_bytes'])
    if a.compare_v15 or a.compare_v16:
        ref_ms=row['triton_v14_event_median'];ref_mem=row['triton_v14_peak_extra_bytes']
        row['v14_guard']={}
        for label in (('triton','triton_static15') if a.compare_v16 else ('triton','triton_shared15')):
            ratio=ref_ms/row[label+'_event_median']
            noisy=max(row['triton_v14_event_cv_pct'],row[label+'_event_cv_pct'])>a.max_cv
            status='unstable' if noisy else 'regression' if ratio<.97 else 'near_tie' if ratio<=1.03 else 'faster'
            row['v14_guard'][label]=dict(status=status,speedup=ratio,extra_memory_delta_bytes=row[label+'_peak_extra_bytes']-ref_mem)
    if a.compare_graph:
        row['graph_diagnostic']={}
        for name in ('triton','aten_prepacked','aten_e2e'):
            if name not in funcs:continue
            print('  graph diagnostic',name,flush=True)
            try:row['graph_diagnostic'][name]=graph_probe(funcs[name],a)
            except Exception as e:row['graph_diagnostic'][name]=dict(status='error',error=repr(e))
    if a.profile_kernels:
        row['cuda_profile']={}
        for name in ('triton','triton_v18','triton_v17','triton_v16','triton_v14','triton_static15','triton_shared15','triton_v13','triton_weight_only','triton_v12','aten_prepacked','stage_ggO','stage_gI','stage_gW','stage_gI_v17','stage_gI_v18','stage_gW_v17','stage_gW_v18'):
            if name not in funcs:continue
            print('  CUDA trace',name,flush=True)
            try:row['cuda_profile'][name]=profile_callable(funcs[name],a.profile_iters)
            except Exception as e:row['cuda_profile'][name]=dict(status='error',error=repr(e))
    return dict(row,status='ok')


def summary(row):
    title=f"{row['case']} | G={row['G']} batch={row['F']} | {row['status']}"
    if row['status']!='ok':
        return title+' | '+row.get('error','')
    if row.get('policy')=='native_layout':
        lines=[title+' | native layouts; conversion excluded']
        for n in row['providers']:
            lines.append(f"  {n}: event={row[n+'_event_median']:.4f} ms CV={row[n+'_event_cv_pct']:.2f}% wall={row[n+'_wall_median']:.4f} ms native_task_peak={row[n+'_task_peak_bytes']/1048576:.2f} MiB")
        for alg,comp in row['comparisons'].items():
            lines.append(f"  {alg} vs best ATen: {comp['speedup']:.3f}x time_change={comp['time_change_pct']:+.1f}% {comp['verdict']}")
        if row.get('candidate_failures'):lines.append('  excluded invalid candidates: '+json.dumps(row['candidate_failures']))
        if 'vs_v18' in row:lines.append('  vs_v18: '+json.dumps(row['vs_v18']))
        if 'diagnostic26' in row:
            diag=row['diagnostic26']
            lines.append('  v26 old/new: '+diag['old_reference']+' -> '+diag['new_candidate']+
                         f" full_speedup={diag['full_speedup']:.3f}x memory_delta={diag['extra_memory_delta_bytes']/1048576:+.2f} MiB")
            for alg,times in diag['isolated_stages'].items():
                lines.append('  isolated '+alg+': '+json.dumps(times))
        if 'diagnostic27' in row:
            diag=row['diagnostic27']
            if diag.get('status')=='candidate_invalid':
                lines.append('  v27 candidate invalid: '+json.dumps(diag['candidate_failures']))
            else:
                lines.append('  v27 old/new: '+diag['old_reference']+' -> '+diag['new_candidate']+
                             f" full_speedup={diag['full_speedup']:.3f}x memory_delta={diag['extra_memory_delta_bytes']/1048576:+.2f} MiB")
                for alg,times in diag['isolated_stages'].items():
                    lines.append('  isolated '+alg+': '+json.dumps(times))
        if 'diagnostic28' in row:
            diag=row['diagnostic28']
            lines.append('  v28 cumulative 7x7 ablation vs '+diag['reference']+':')
            for alg,times in diag['isolated_stages'].items():
                extra=''
                if alg!=diag['reference']:
                    extra=(f" full_speedup={diag['full_speedups'][alg]:.3f}x"
                           f" memory_delta={diag['extra_memory_delta_bytes'][alg]/1048576:+.2f} MiB")
                lines.append('  isolated '+alg+': '+json.dumps(times)+extra)
        if 'diagnostic29' in row:
            diag=row['diagnostic29']
            lines.append('  v29 7x7 gI+ggO ablation; implicit gW retained:')
            for alg,times in diag['isolated_stages'].items():
                extra=''
                if alg!=diag['reference']:
                    extra=(f" full_speedup={diag['full_speedups'][alg]:.3f}x"
                           f" memory_delta={diag['extra_memory_delta_bytes'][alg]/1048576:+.2f} MiB")
                lines.append('  isolated '+alg+': '+json.dumps(times)+extra)
        if 'diagnostic30' in row:
            diag=row['diagnostic30']
            lines.append('  v30 one-buffer joint ggO+gW; reference '+diag['reference']+':')
            for alg,times in diag['isolated_stages'].items():
                extra=f" joint_ggO_gW={diag['joint_ggO_gW'][alg]:.4f} ms"
                if alg!=diag['reference']:
                    extra+=(f" full_speedup={diag['full_speedups'][alg]:.3f}x"
                            f" memory_delta={diag['extra_memory_delta_bytes'][alg]/1048576:+.2f} MiB")
                lines.append('  isolated '+alg+': '+json.dumps(times)+extra)
        if 'diagnostic31' in row:
            diag=row['diagnostic31']
            if diag.get('status')=='candidate_invalid':
                lines.append('  v31 candidate invalid: '+
                             json.dumps(diag['candidate_failures']))
            else:
                old=diag['reference'];new=diag['new_candidate']
                lines.append('  v31 hybrid: '+old+' -> '+new+
                             f" full_speedup={diag['full_speedup']:.3f}x"
                             f" memory_delta={diag['extra_memory_delta_bytes']/1048576:+.2f} MiB")
                for alg in (old,new):
                    lines.append('  hybrid stages '+alg+': '+
                        json.dumps(diag['isolated_stages'][alg])+
                        f" joint_ggO_gW={diag['joint_ggO_gW'][alg]:.4f} ms")
        if 'diagnostic32' in row:
            diag=row['diagnostic32']
            if diag.get('status')=='candidate_invalid':
                lines.append('  v32 candidate invalid: '+
                             json.dumps(diag['candidate_failures']))
            else:
                old=diag['prior_best']
                lines.append('  v32 large-N packed split-K; prior best '+old+':')
                for alg in diag['new_candidates']:
                    lines.append('  '+alg+
                        f" full_speedup={diag['full_speedups'][alg]:.3f}x"+
                        f" memory_delta={diag['extra_memory_delta_bytes'][alg]/1048576:+.2f} MiB"+
                        f" joint_ggO_gW={diag['joint_ggO_gW'][alg]:.4f} ms"+
                        ' isolated='+json.dumps(diag['isolated_stages'][alg]))
        if 'diagnostic33' in row:
            diag=row['diagnostic33']
            lines.append('  v33 cross-generation '+diag['family']+': '+
                         diag['old_best']+' -> '+diag['new_best']+
                         f" speedup={diag['new_over_old_speedup']:.3f}x"+
                         f" memory_delta={diag['new_minus_old_peak_extra_bytes']/1048576:+.2f} MiB")
            lines.append('  within-3% lower-memory choice='+
                         diag['within_3pct_lower_memory_choice']+
                         '; Pareto='+','.join(diag['pareto_frontier']))
        if 'diagnostic24' in row:
            diag=row['diagnostic24']
            lines.append('  diagnostic scope: '+diag['scope'])
            for alg,times in diag['isolated_stages'].items():
                lines.append('  isolated '+alg+': '+json.dumps(times))
            for name,trace in diag['full_call_profiles'].items():
                if trace.get('status')!='ok':
                    lines.append('  profile '+name+': '+json.dumps(trace));continue
                lines.append('  profile '+name+': activity_ms_per_api='+format(trace['activity_ms_per_api'],'.6f'))
                if 'components24' in trace:lines.append('    components: '+json.dumps(trace['components24']))
                for k in trace['kernels'][:12]:
                    lines.append(f"    {k['ms_per_api']:.6f} ms/API calls={k['calls_per_api']:.2f} {k['name']}")
        lines.append('  structural policy='+row['structural_policy']+'; measured best candidate='+row['best_triton']+'; best ATen='+row['best_aten'])
        return '\n'.join(lines)
    if row.get('comparison')=='v21_same_run_e2e':
        lines=[title+' | same-run complete API (layout conversion included)']
        for n in ('triton_v18','aten_grouped_e2e','aten_serial_e2e'):
            lines.append(f"  {n}: event={row[n+'_event_median']:.4f} ms CV={row[n+'_event_cv_pct']:.2f}% wall={row[n+'_wall_median']:.4f} ms wallCV={row[n+'_wall_cv_pct']:.2f}% extra={row[n+'_peak_extra_bytes']/1048576:.2f} MiB task_peak={row[n+'_task_peak_bytes']/1048576:.2f} MiB reserved_peak={row[n+'_process_peak_reserved_bytes']/1048576:.2f} MiB")
        lines.append(f"  best ATen / v18: {row['best_aten_over_v18']:.3f}x")
        lines.append('  selection: '+json.dumps(row['selection']))
        return '\n'.join(lines)
    lines=[title+f" | {row['policy']}"]
    if 'v18_guard' in row:
        guard=row['v18_guard']
        lines.append(f"  v18 guard: {guard['status']} speedup={guard['speedup']:.3f}x extra_memory_delta={guard['extra_memory_delta_bytes']/1048576:+.2f} MiB; no automatic promotion")
    if 'v17_guard' in row:
        guard=row['v17_guard']
        lines.append(f"  v17 guard: {guard['status']} speedup={guard['speedup']:.3f}x extra_memory_delta={guard['extra_memory_delta_bytes']/1048576:+.2f} MiB; no automatic promotion")
    if 'v16_guard' in row:
        guard=row['v16_guard']
        lines.append(f"  v16 guard: {guard['status']} speedup={guard['speedup']:.3f}x extra_memory_delta={guard['extra_memory_delta_bytes']/1048576:+.2f} MiB; no automatic promotion")
    for label,guard in row.get('v14_guard',{}).items():
        lines.append(f"  v14 guard {label}: {guard['status']} speedup={guard['speedup']:.3f}x extra_memory_delta={guard['extra_memory_delta_bytes']/1048576:+.2f} MiB; no automatic promotion")
    lines.append('  policy reference: '+row.get('policy_reference','grouped ATen'))
    if row.get('aten_baseline_reused'):
        lines.append(f"  cached ATen baseline (different run): best available={row['best_aten_baseline_ms']:.6f} ms speedup={row['best_aten_speedup']:.3f}x; near ties need same-run confirmation")
    if 'triton_task_peak_bytes' in row:
        lines.append('  task_peak(MiB): '+ ' '.join(n+'='+format(row[n+'_task_peak_bytes']/1048576,'.2f') for n in ('triton','aten_prepacked','aten_e2e'))+' (inputs + layout copies + peak extra allocations)')
    if 'dispatch_route' in row.get('winograd_plan',{}):lines.append('  route: '+row['winograd_plan']['dispatch_route']+' ('+row['winograd_plan']['dispatch_reason']+')')
    if 'schedule' in row.get('winograd_plan',{}):lines.append('  schedule: '+row['winograd_plan']['schedule']+' (provisional shape/workspace policy)')
    lines.append('  algorithms: '+' '.join(n+'='+row.get(n+'_algorithm','implicit') for n in ('ggO','gI','gW')))
    if row.get('gW_validation')=='fp64_resolved':
        lines.append(f"  validation: gW FP64-resolved; ATen mismatches={row['gW_aten_mismatches']}, checked={row['gW_fp64_checked']}, Triton max_abs={row['gW_fp64_max_abs']:.3e}, ATen max_abs={row['gW_aten_fp64_max_abs']:.3e} (checked positions only)")
    lines.append(f"  time(ms): Triton={row['triton_event_median']:.6f} ATen={row['aten_prepacked_event_median']:.6f} ATen+layout={row['aten_e2e_event_median']:.6f}")
    lines.append(f"  speedup: ATen={row['aten_prepacked_speedup']:.3f}x E2E={row['aten_e2e_speedup']:.3f}x | CV: Triton={row['triton_event_cv_pct']:.2f}% ATen={row['aten_prepacked_event_cv_pct']:.2f}%")
    if 'triton_implicit_event_median' in row:
        lines.append(f"  previous implicit(ms)={row['triton_implicit_event_median']:.6f} | speedup vs implicit={row['triton_implicit_event_median']/row['triton_event_median']:.3f}x")
    for label in ('triton_tail','triton_v9','triton_v11','triton_v12','triton_v18','triton_v17','triton_v16','triton_v14','triton_static15','triton_shared15','triton_v13','triton_weight_only'):
        if label+'_event_median' in row:
            lines.append(f"  {label}(ms)={row[label+'_event_median']:.6f} speedup_vs_{label}={row[label+'_event_median']/row['triton_event_median']:.3f}x CV={row[label+'_event_cv_pct']:.2f}%")
    for label in ('triton_v18','triton_v17','triton_v16','triton_v14','triton_static15','triton_shared15','triton_v13','triton_weight_only'):
        if label+'_peak_extra_bytes' in row:
            lines.append(f"  {label} peak_extra(MiB)={row[label+'_peak_extra_bytes']/1048576:.2f}")
    if 'triton_v8_event_median' in row:
        lines.append(f"  v8(ms)={row['triton_v8_event_median']:.6f} speedup_vs_v8={row['triton_v8_event_median']/row['triton_event_median']:.3f}x (same workspace budget)")
    if 'triton_v7_event_median' in row:
        lines.append(f"  v7(ms)={row['triton_v7_event_median']:.6f} speedup_vs_v7={row['triton_v7_event_median']/row['triton_event_median']:.3f}x")
    if 'triton_v6_event_median' in row:
        lines.append(f"  v6(ms)={row['triton_v6_event_median']:.6f} speedup_vs_v6={row['triton_v6_event_median']/row['triton_event_median']:.3f}x")
    if 'triton_v5_event_median' in row:
        lines.append(f"  v5(ms)={row['triton_v5_event_median']:.6f} speedup_vs_v5={row['triton_v5_event_median']/row['triton_event_median']:.3f}x")
    for alg in ('winograd_joint','winograd_separate'):
        if alg+'_event_median' in row:
            lines.append(f"  {alg}(ms)={row[alg+'_event_median']:.6f} CV={row[alg+'_event_cv_pct']:.2f}%")
    if 'measured_serial_speedup' in row:
        lines.append(f"  measured serial: best={row['measured_best_serial_ms']:.6f} ms fused_speedup={row['measured_serial_speedup']:.3f}x guard={row['serial_guard']} (assembly excluded)")
    for name in ('ggO','gI'):
        if row.get(name+'_validation')=='fp64_resolved':
            lines.append(f"  validation: {name} FP64-resolved at {row[name+'_fp64_checked']} positions; not full FP64 validation")
    if 'joint_ggO_config' in row:lines.append('  joint ggO config: '+row['joint_ggO_config'])
    if 'serial_triton_event_median' in row:
        lines.append(f"  serial G calls(ms): ATen={row['serial_aten_event_median']:.6f} Triton={row['serial_triton_event_median']:.6f}")
        lines.append(f"  scaling: serial_ATen/fused_Triton={row['serial_aten_over_fused_triton']:.3f}x serial_Triton/fused_Triton={row['serial_triton_over_fused_triton']:.3f}x serial_ATen/grouped_ATen={row['serial_aten_over_grouped_aten']:.3f}x")
        lines.append(f"  throughput={row['group_images_per_second']:.1f} group-images/s | fusion_efficiency={row['fusion_efficiency_pct']:.1f}% | stable={row['scaling_stable']} | fusion={row.get('fusion_policy','unclassified')}")
        lines.append(f"  serial CV: ATen={row['serial_aten_event_cv_pct']:.2f}% Triton={row['serial_triton_event_cv_pct']:.2f}%")
    if row.get('joint_ggO_gW'):lines.append('  ggO+gW share transformed ggI in full API; isolated stage times do not include this reuse.')
    if 'stage_ggO_event_median' in row:
        lines.append('  stages(ms): '+' '.join(n+'='+format(row['stage_'+n+'_event_median'],'.6f') for n in ('ggO','gI','gW')))
    if 'stage_gI_v17_event_median' in row:
        lines.append('  same-run stages(ms): '+' '.join(n+'='+format(row['stage_'+n+'_event_median'],'.6f') for n in ('gI_v17','gI_v18','gW_v17')))
    if 'stage_gW_v18_event_median' in row:
        lines.append(f"  same-run gW(ms): v18={row['stage_gW_v18_event_median']:.6f} candidate={row['stage_gW_event_median']:.6f}")
    lines.append('  error(rel_L2): '+' '.join(n+'='+format(row[n+'_rel_l2'],'.3e') for n in ('ggO','gI','gW')))
    lines.append('  peak_extra(MiB): '+' '.join(n+'='+format(row[n+'_peak_extra_bytes']/1048576,'.2f') for n in ('triton','aten_prepacked','aten_e2e'))+f" | split_k={row['split_k']}")
    for name,v in row.get('graph_diagnostic',{}).items():
        if v['status']=='ok':
            ms=v['event_ms']['median'];eager=row[name+'_event_median']
            lines.append(f"  GRAPH {name}: replay_ms={ms:.6f} eager/replay={eager/ms:.3f} CV={v['event_ms']['cv_pct']:.2f}% (diagnostic only)")
        else:lines.append('  GRAPH '+name+': '+str(v))
    for name,v in row.get('cuda_profile',{}).items():
        if v['status']!='ok':lines.append('  PROFILE '+name+': '+str(v));continue
        lines.append(f"  PROFILE {name}: activity_ms/API={v['activity_ms_per_api']:.6f} uncovered_span={v['uncovered_span_pct']:.2f}% (instrumented; not SM utilization)")
        records=v['families'] if name!='aten_prepacked' else v['kernels'][:12]
        for r in records:
            lines.append(f"    {r['name']} calls/API={r['calls_per_api']:.1f} ms/API={r['ms_per_api']:.6f} mean_us={r['mean_us']:.3f} share={r['activity_share_pct']:.1f}%")
    return '\n'.join(lines)



def matrix_points(a,case):
    if a.native_layout:
        pairs=([(b,g) for b in (1,10) for g in (1,4,16,64)]+[(b,g) for b in (100,1000) for g in (1,2,4)]
               if a.native_sweep=='balanced' else list(itertools.product(a.batch,a.groups)))
        return [(b,g) for b,g in pairs if b in a.batch and g in a.groups]
    """B is images PER GROUP, not IPC. Presets are pairs, not Cartesian grids."""
    if a.matrix=='none':return [(b,g) for g in a.groups for b in a.batch]
    if a.matrix=='cifar-core':
        pairs=[(10,1),(10,64),(100,1),(100,4),(1000,1),(1000,4)]
    elif a.matrix=='cifar-scale':
        pairs=[(10,g) for g in (1,2,4,8,16,32,64)]
        pairs += [(b,g) for b in (100,1000) for g in (1,2,4,8)]
        # B=1 is a per-class/microbatch diagnostic, NOT CIFAR10 IPC=1.
        pairs += [(1,g) for g in (1,8,64)]
    else:
        pairs=[(10,1),(10,8),(100,1),(100,4),(1000,1),(1000,2)]
        if case['dataset']=='tinyimagenet':pairs += [(200,1),(200,4)]
    return [(b,g) for b,g in pairs if b in a.batch and g in a.groups]


def selector_task_weight(task):
    """Deterministic runtime proxy used only to balance independent shards.

    Adaptive timing makes provider count the main cost.  A capped log-FLOP
    correction keeps large spatial cases from collecting on one machine
    without letting likely OOM cases dominate the partition.
    """
    case,g,b=task
    p=(case['H']+2*case['padding']-case['dilation']*(case['R']-1)-1)//case['stride']+1
    q=(case['W']+2*case['padding']-case['dilation']*(case['S']-1)-1)//case['stride']+1
    candidates=len(native_algorithms(case))
    if case['R']==case['S']==3 and case['stride']==2:candidates+=2
    providers=candidates+2  # grouped and serial native-layout ATen
    ops=max(1,g*b*case['C']*case['O']*case['R']*case['S']*p*q)
    scale=max(0.,min(6.,__import__('math').log2(ops/1e8)))
    return providers*(1.+.15*scale)


def coverage26_task_weight(task):
    case,g,b=task
    p=(case['H']+2*case['padding']-case['dilation']*(case['R']-1)-1)//case['stride']+1
    q=(case['W']+2*case['padding']-case['dilation']*(case['S']-1)-1)//case['stride']+1
    candidates=2 if case['R']==1 else 6
    providers=2+4*candidates  # full candidates, three isolated stages, two ATen
    ops=max(1,g*b*case['C']*case['O']*case['R']*case['S']*p*q)
    scale=max(0.,min(6.,__import__('math').log2(ops/1e8)))
    return providers*(1.+.15*scale)


def coverage27_task_weight(task):
    case,g,b=task
    p=(case['H']+2*case['padding']-case['dilation']*(case['R']-1)-1)//case['stride']+1
    q=(case['W']+2*case['padding']-case['dilation']*(case['S']-1)-1)//case['stride']+1
    candidates=4
    providers=2+4*candidates  # full candidates, three isolated stages, two ATen
    ops=max(1,g*b*case['C']*case['O']*case['R']*case['S']*p*q)
    scale=max(0.,min(6.,__import__('math').log2(ops/1e8)))
    return providers*(1.+.15*scale)


def coverage28_task_weight(task):
    # Four cumulative candidates, each measured as a full call and as three
    # isolated stages, plus the two native-layout ATen references.
    return coverage27_task_weight(task)


def coverage29_task_weight(task):
    case,g,b=task
    p=(case['H']+2*case['padding']-case['dilation']*(case['R']-1)-1)//case['stride']+1
    q=(case['W']+2*case['padding']-case['dilation']*(case['S']-1)-1)//case['stride']+1
    candidates=3
    providers=2+4*candidates
    ops=max(1,g*b*case['C']*case['O']*case['R']*case['S']*p*q)
    scale=max(0.,min(6.,__import__('math').log2(ops/1e8)))
    return providers*(1.+.15*scale)


def coverage30_task_weight(task):
    case,g,b=task
    p=(case['H']+2*case['padding']-case['dilation']*(case['R']-1)-1)//case['stride']+1
    q=(case['W']+2*case['padding']-case['dilation']*(case['S']-1)-1)//case['stride']+1
    candidates=5
    # Full call, three isolated fallbacks and joint ggO+gW for each candidate,
    # plus both native-layout ATen baselines.
    providers=2+5*candidates
    ops=max(1,g*b*case['C']*case['O']*case['R']*case['S']*p*q)
    scale=max(0.,min(6.,__import__('math').log2(ops/1e8)))
    return providers*(1.+.15*scale)


def coverage31_task_weight(task):
    case,g,b=task
    p=(case['H']+2*case['padding']-case['dilation']*(case['R']-1)-1)//case['stride']+1
    q=(case['W']+2*case['padding']-case['dilation']*(case['S']-1)-1)//case['stride']+1
    candidates=6
    providers=2+5*candidates
    ops=max(1,g*b*case['C']*case['O']*case['R']*case['S']*p*q)
    scale=max(0.,min(6.,__import__('math').log2(ops/1e8)))
    return providers*(1.+.15*scale)


def coverage32_task_weight(task):
    case,g,b=task
    p=(case['H']+2*case['padding']-case['dilation']*(case['R']-1)-1)//case['stride']+1
    q=(case['W']+2*case['padding']-case['dilation']*(case['S']-1)-1)//case['stride']+1
    candidates=6
    providers=2+5*candidates
    ops=max(1,g*b*case['C']*case['O']*case['R']*case['S']*p*q)
    scale=max(0.,min(6.,__import__('math').log2(ops/1e8)))
    return providers*(1.+.15*scale)


def coverage33_task_weight(task):
    case,g,b=task
    p=(case['H']+2*case['padding']-case['dilation']*(case['R']-1)-1)//case['stride']+1
    q=(case['W']+2*case['padding']-case['dilation']*(case['S']-1)-1)//case['stride']+1
    if case['R']==1:candidates=2
    elif case['R']==7:candidates=3
    else:
        candidates=7+(b*p*q>=16384)  # split8 is omitted below its v32 guard
    # v33 deliberately measures only complete APIs: candidates plus grouped
    # and serial ATen.  Weight by approximate arithmetic so its shards remain
    # balanced despite the very different spatial sizes.
    providers=2+candidates
    ops=max(1,g*b*case['C']*case['O']*case['R']*case['S']*p*q)
    scale=max(0.,min(6.,__import__('math').log2(ops/1e8)))
    return providers*(1.+.15*scale)


def shard_selector_tasks(tasks,count,index,weight_fn=selector_task_weight):
    """LPT partition: disjoint, deterministic and more even than slicing."""
    if count==1:return tasks,[sum(weight_fn(t) for t in tasks)]
    ranked=sorted(enumerate(tasks),key=lambda x:(-weight_fn(x[1]),x[0]))
    bins=[[] for _ in range(count)];loads=[0.]*count
    for original,task in ranked:
        dst=min(range(count),key=lambda j:(loads[j],j))
        bins[dst].append((original,task));loads[dst]+=weight_fn(task)
    return [task for _,task in sorted(bins[index])],loads


def model_summaries(rows,tasks,a):
    """Weighted convolution-only estimates, never model end-to-end timings."""
    by_key={(r['case'],r['G'],r['F']):r for r in rows}
    groups={}
    for case,g,b in tasks:
        k=(case['model'],case['dataset'],case['image_size'],case['stem'],g,b)
        groups.setdefault(k,[]).append(case)
    for (model,dataset,size,stem,g,b),selected in groups.items():
        # Compare against the unfiltered architecture to expose --filter omissions.
        full=[c for c in cases([model],[dataset],[size],stem,a.width)]
        matched=[by_key.get((c['case'],g,b)) for c in full]
        good=[r for r in matched if r and r['status']=='ok']
        complete=len(good)==len(full)
        rec=dict(model=model,dataset=dataset,image_size=size,stem=stem,G=g,B=b,
                 scope='sum of isolated convolution double-bwd medians weighted by layer count; NOT end-to-end model time',
                 complete=complete,unique_shapes_ok=len(good),unique_shapes_expected=len(full),
                 layers_ok=sum(r['multiplicity'] for r in good),layers_expected=sum(c['multiplicity'] for c in full),
                 missing=[c['case'] for c,r in zip(full,matched) if not r or r['status']!='ok'])
        if good:
            def total(key):return sum(r['multiplicity']*r[key] for r in good)
            t=total('triton_event_median')
            rec['covered_triton_ms']=t
            rec['covered_grouped_aten_ms']=total('aten_prepacked_event_median')
            rec['covered_e2e_aten_ms']=total('aten_e2e_event_median')
            rec['cached_aten_rows']=sum(bool(r.get('aten_baseline_reused')) for r in good)
            rec['worst_layer_grouped_speedup']=min(r['aten_prepacked_event_median']/r['triton_event_median'] for r in good)
            rec['regressed_shapes']=[r['case'] for r in good if r['triton_event_median']>r['aten_prepacked_event_median']*1.03]
            rec['stable']=all(max(r['triton_event_cv_pct'],r['aten_prepacked_event_cv_pct'])<=a.max_cv for r in good)
            if complete:
                rec['conv_sum_grouped_speedup']=rec['covered_grouped_aten_ms']/t
                rec['conv_sum_e2e_speedup']=rec['covered_e2e_aten_ms']/t
            if all('triton_implicit_event_median' in r for r in good):
                rec['covered_implicit_ms']=total('triton_implicit_event_median')
                rec['regressions_vs_implicit']=[r['case'] for r in good if r['triton_event_median']>r['triton_implicit_event_median']*1.03]
        yield rec


def correctness_structural(dtype,precision):
    count=0
    # RGB stem, dense regular/tail, downsample, shortcut, 7x7, dilation, low budget.
    examples=[((2,2,3,17,9,11,3,3),(1,1),(1,1),(1,1),4),
              ((2,3,17,33,7,9,3,3),(1,1),(1,1),(1,1),4),
              ((2,2,17,33,9,11,3,3),(2,2),(1,1),(1,1),4),
              ((2,2,17,33,9,11,1,1),(2,2),(0,0),(1,1),4),
              ((1,2,3,17,17,19,7,7),(2,2),(3,3),(1,1),4),
              ((2,2,17,33,9,11,3,3),(1,1),(2,2),(2,2),4),
              ((3,17,17,33,5,7,3,3),(1,1),(1,1),(1,1),.1),
              ((2,3,17,33,8,8,3,3),(2,2),(1,1),(1,1),4),
              ((2,3,17,33,7,9,3,3),(2,2),(0,2),(1,1),4),
              ((2,25,17,33,7,9,3,3),(1,1),(1,1),(1,1),4)]
    for shape,st,pad,dil,budget in examples:
        args=make_args(shape,dtype,'cuda',st,pad,dil)
        print('structural',shape,compare(args,precision,budget=budget,algorithm='adaptive14'),flush=True);count+=1
    for opts in itertools.product((False,True),repeat=3):
        args=make_args((2,3,17,33,5,7,3,3),dtype,'cuda',opts=opts)
        print('structural optional',opts,compare(args,precision,budget=4,algorithm='adaptive14'),flush=True);count+=1
    for opts in itertools.product((False,True),repeat=3):
        args=make_args((2,3,17,33,5,7,3,3),dtype,'cuda',st=(2,2),opts=opts)
        print('phase optional',opts,compare(args,precision,budget=4,algorithm='adaptive14'),flush=True);count+=1
    for shape,budget in [((3,10,17,33,16,16,3,3),4),((2,17,17,33,7,9,3,3),.2)]:
        args=make_args(shape,dtype,'cuda')
        print('aligned',shape,compare(args,precision,budget=budget,algorithm='winograd_weight'),flush=True);count+=1
    for opts in itertools.product((False,True),repeat=3):
        args=make_args((2,3,17,33,5,7,3,3),dtype,'cuda',opts=opts)
        print('aligned optional',opts,compare(args,precision,budget=4,algorithm='winograd_weight'),flush=True);count+=1
    for shape,pad,budget in [((3,7,17,33,9,11,3,3),(0,2),.05),
                             ((2,2,17,33,6,7,3,3),(1,1),0)]:
        args=make_args(shape,dtype,'cuda',st=(2,2),pad=pad)
        print('packed budget',shape,compare(args,precision,budget=budget,algorithm='stride2_packed'),flush=True);count+=1
    for opts in itertools.product((False,True),repeat=3):
        args=make_args((2,2,17,33,7,9,3,3),dtype,'cuda',st=(2,2),opts=opts)
        print('weight optional',opts,compare(args,precision,budget=4,algorithm='stride2_weight'),flush=True);count+=1
    for alg in ('stride2_static','stride2_shared'):
        for shape,pad,budget in [((3,7,17,33,9,11,3,3),(0,2),.08),
                                 ((2,2,17,33,6,7,3,3),(1,1),0),
                                 ((2,100,32,64,8,8,3,3),(1,1),4)]:
            args=make_args(shape,dtype,'cuda',st=(2,2),pad=pad)
            print('v15 boundary',alg,shape,compare(args,precision,budget=budget,algorithm=alg),flush=True);count+=1
        for opts in itertools.product((False,True),repeat=3):
            args=make_args((2,3,17,33,7,9,3,3),dtype,'cuda',st=(2,2),opts=opts)
            print('v15 optional',alg,opts,compare(args,precision,budget=4,algorithm=alg),flush=True);count+=1
    print(f'PASS: {count} structural GPU cases; direct ATen comparison',flush=True)


def correctness16(dtype,precision):
    if dtype!=torch.float32:raise ValueError('v16 candidate requires fp32')
    count=0
    # Group chunks, spatial/channel tails, minimum H/W, and zero-budget fallback.
    shapes=[((3,7,17,33,9,11,3,3),(0,2),.04),
            ((2,2,17,33,6,7,3,3),(1,1),0),
            ((2,3,5,7,2,3,3,3),(1,1),1),
            ((2,100,32,64,8,8,3,3),(1,1),4)]
    for shape,pad,budget in shapes:
        args=make_args(shape,dtype,'cuda',st=(2,2),pad=pad)
        for split in (1,4) if budget>=1 else (1,):
            print('v16 boundary',shape,'split',split,compare(args,precision,split=split,budget=budget,algorithm='stride2_static_weight'),flush=True);count+=1
    for opts in itertools.product((False,True),repeat=3):
        args=make_args((2,3,17,33,7,9,3,3),dtype,'cuda',st=(2,2),opts=opts)
        print('v16 optional',opts,compare(args,precision,budget=4,algorithm='stride2_static_weight'),flush=True);count+=1
    args=make_args((3,3,17,33,7,9,3,3),dtype,'cuda',st=(2,2))
    ref=aten_group_first(args)
    for i in range(3):
        flags=tuple(j==i for j in range(3))
        got=custom(*args,precision=precision,workspace_limit_mb=4,algorithm='stride2_static_weight',compute=flags)
        print('v16 isolated',i,check(got,tuple(v if j==i else None for j,v in enumerate(ref)),dtype,precision,args=args),flush=True);count+=1
    print(f'PASS: {count} v16 GPU cases; compared directly with ATen',flush=True)


def correctness17(dtype,precision):
    if dtype!=torch.float32:raise ValueError('v17 candidate requires fp32')
    count=0
    # Group chunks, spatial/channel tails, minimum H/W, and zero-budget fallback.
    shapes=[((3,7,17,33,9,11,3,3),(0,2),.04),
            ((2,2,17,33,6,7,3,3),(1,1),0),
            ((2,3,5,7,2,3,3,3),(1,1),1),
            ((2,100,32,64,8,8,3,3),(1,1),4),
            ((2,1000,5,7,4,5,3,3),(1,1),.02)]
    for shape,pad,budget in shapes:
        args=make_args(shape,dtype,'cuda',st=(2,2),pad=pad)
        for split in (1,4) if budget>=1 else (1,):
            print('v17 boundary',shape,'split',split,compare(args,precision,split=split,budget=budget,algorithm='stride2_packed_gi'),flush=True);count+=1
    for opts in itertools.product((False,True),repeat=3):
        args=make_args((2,3,17,33,7,9,3,3),dtype,'cuda',st=(2,2),opts=opts)
        print('v17 optional',opts,compare(args,precision,budget=4,algorithm='stride2_packed_gi'),flush=True);count+=1
    args=make_args((3,3,17,33,7,9,3,3),dtype,'cuda',st=(2,2))
    ref=aten_group_first(args)
    for i in range(3):
        flags=tuple(j==i for j in range(3))
        got=custom(*args,precision=precision,workspace_limit_mb=4,algorithm='stride2_packed_gi',compute=flags)
        print('v17 isolated',i,check(got,tuple(v if j==i else None for j,v in enumerate(ref)),dtype,precision,args=args),flush=True);count+=1
    print(f'PASS: {count} v17 GPU cases; compared directly with ATen',flush=True)


def correctness19(dtype,precision):
    if dtype!=torch.float32:raise ValueError('v19 candidate requires fp32')
    count=0
    # Group chunks, spatial/channel tails, minimum H/W, and zero-budget fallback.
    shapes=[((3,7,17,33,9,11,3,3),(0,2),.04),
            ((2,2,17,33,6,7,3,3),(1,1),0),
            ((2,3,5,7,2,3,3,3),(1,1),1),
            ((2,100,32,64,8,8,3,3),(1,1),4),
            ((2,1000,5,7,4,5,3,3),(1,1),.02)]
    shapes += [((3,3,5,7,7,9,3,3),(1,1),.02),
               ((2,7,17,33,8,8,3,3),(1,1),1)]
    for shape,pad,budget in shapes:
        args=make_args(shape,dtype,'cuda',st=(2,2),pad=pad)
        for split in (1,4) if budget>=1 else (1,):
            print('v19 boundary',shape,'split',split,compare(args,precision,split=split,budget=budget,algorithm='stride2_packed_gw'),flush=True);count+=1
    for opts in itertools.product((False,True),repeat=3):
        args=make_args((2,3,17,33,7,9,3,3),dtype,'cuda',st=(2,2),opts=opts)
        print('v19 optional',opts,compare(args,precision,budget=4,algorithm='stride2_packed_gw'),flush=True);count+=1
    args=make_args((3,3,17,33,7,9,3,3),dtype,'cuda',st=(2,2))
    ref=aten_group_first(args)
    for i in range(3):
        flags=tuple(j==i for j in range(3))
        got=custom(*args,precision=precision,workspace_limit_mb=4,algorithm='stride2_packed_gw',compute=flags)
        print('v19 isolated',i,check(got,tuple(v if j==i else None for j,v in enumerate(ref)),dtype,precision,args=args),flush=True);count+=1
    print(f'PASS: {count} v19 GPU cases; compared directly with ATen',flush=True)


def correctness20(dtype,precision):
    if dtype!=torch.float32:raise ValueError('v20 candidate requires fp32')
    count=0
    # Group chunks, spatial/channel tails, minimum H/W, and zero-budget fallback.
    shapes=[((3,7,17,33,9,11,3,3),(0,2),.04),
            ((2,2,17,33,6,7,3,3),(1,1),0),
            ((2,3,5,7,2,3,3,3),(1,1),1),
            ((2,100,32,64,8,8,3,3),(1,1),4),
            ((2,1000,5,7,4,5,3,3),(1,1),.02)]
    shapes += [((3,3,5,7,7,9,3,3),(1,1),.02),
               ((2,7,17,33,8,8,3,3),(1,1),1)]
    # Rectangular crops, empty filter domains, long compensated reduction.
    shapes += [((2,3,5,7,2,2,3,3),(3,2),1),
               ((2,3,5,7,5,7,3,3),(0,0),1),
               ((2,1000,2,3,8,8,3,3),(1,1),1)]
    for shape,pad,budget in shapes:
        args=make_args(shape,dtype,'cuda',st=(2,2),pad=pad)
        for split in (1,4) if budget>=1 else (1,):
            print('v20 boundary',shape,'split',split,compare(args,precision,split=split,budget=budget,algorithm='stride2_implicit_gw'),flush=True);count+=1
    for opts in itertools.product((False,True),repeat=3):
        args=make_args((2,3,17,33,7,9,3,3),dtype,'cuda',st=(2,2),opts=opts)
        print('v20 optional',opts,compare(args,precision,budget=4,algorithm='stride2_implicit_gw'),flush=True);count+=1
    args=make_args((3,3,17,33,7,9,3,3),dtype,'cuda',st=(2,2))
    ref=aten_group_first(args)
    for i in range(3):
        flags=tuple(j==i for j in range(3))
        got=custom(*args,precision=precision,workspace_limit_mb=4,algorithm='stride2_implicit_gw',compute=flags)
        print('v20 isolated',i,check(got,tuple(v if j==i else None for j,v in enumerate(ref)),dtype,precision,args=args),flush=True);count+=1
    print(f'PASS: {count} v20 GPU cases; compared directly with ATen',flush=True)


def correctness18(dtype,precision):
    if dtype!=torch.float32:raise ValueError('v18 candidate requires fp32')
    count=0
    # Group chunks, spatial/channel tails, minimum H/W, and zero-budget fallback.
    shapes=[((3,7,17,33,9,11,3,3),(0,2),.04),
            ((2,2,17,33,6,7,3,3),(1,1),0),
            ((2,3,5,7,2,3,3,3),(1,1),1),
            ((2,100,32,64,8,8,3,3),(1,1),4),
            ((2,1000,5,7,4,5,3,3),(1,1),.02)]
    # NT=64 with this budget: exact chunks, 1-column tail, and 63-column tail.
    shapes += [((3,b,5,7,2,2,3,3),(1,1),.023) for b in (64,65,127)]
    for shape,pad,budget in shapes:
        args=make_args(shape,dtype,'cuda',st=(2,2),pad=pad)
        for split in (1,4) if budget>=1 else (1,):
            print('v18 boundary',shape,'split',split,compare(args,precision,split=split,budget=budget,algorithm='stride2_packed_gi_tail'),flush=True);count+=1
    for opts in itertools.product((False,True),repeat=3):
        args=make_args((2,3,17,33,7,9,3,3),dtype,'cuda',st=(2,2),opts=opts)
        print('v18 optional',opts,compare(args,precision,budget=4,algorithm='stride2_packed_gi_tail'),flush=True);count+=1
    args=make_args((3,3,17,33,7,9,3,3),dtype,'cuda',st=(2,2))
    ref=aten_group_first(args)
    for i in range(3):
        flags=tuple(j==i for j in range(3))
        got=custom(*args,precision=precision,workspace_limit_mb=4,algorithm='stride2_packed_gi_tail',compute=flags)
        print('v18 isolated',i,check(got,tuple(v if j==i else None for j,v in enumerate(ref)),dtype,precision,args=args),flush=True);count+=1
    print(f'PASS: {count} v18 GPU cases; compared directly with ATen',flush=True)


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['correctness','bench','list'])
    ap.add_argument('--models',nargs='+',choices=['conv3','resnet18'],default=None)
    ap.add_argument('--datasets',nargs='+',choices=['cifar10','tinyimagenet','imagenet','imagenet10'],default=None)
    ap.add_argument('--imagenet-sizes',nargs='+',type=int,default=None)
    ap.add_argument('--stem',choices=['auto','small','standard','both'],default='auto')
    ap.add_argument('--width',type=int,default=128)
    ap.add_argument('--groups',nargs='+',type=int,default=None)
    ap.add_argument('--batch',nargs='+',type=int,default=None)
    ap.add_argument('--matrix',choices=['none','cifar-core','cifar-scale','transfer'],default='none',help='Paired B/G presets; --batch/--groups filter preset points rather than forming a Cartesian product')
    ap.add_argument('--missing-baseline',choices=['skip','measure'],default='skip',help='When using a saved ATen baseline, optionally measure only missing shapes')
    ap.add_argument('--correctness-structural',action='store_true',help='Small route/stride/tail/optional-input regression suite')
    ap.add_argument('--family',choices=['all','rgb','dense3','downsample3','pointwise'],default='all',help='Structural shape filter; keeps original --filter available')
    ap.add_argument('--filter',default='',help='Substring of case name')
    ap.add_argument('--dtype',choices=['fp32','fp16','bf16'],default='fp32')
    ap.add_argument('--profile-kernels',action='store_true',help='After normal timing, summarize CUDA traces in the same TXT; no hardware counters')
    ap.add_argument('--profile-iters',type=int,default=3)
    ap.add_argument('--compare-graph',action='store_true',help='Diagnostic CUDA graph replay for both Triton and ATen; eager measurements remain primary')
    ap.add_argument('--compare-v5',action='store_true',help='Time the unchanged v5 packed path on identical inputs')
    ap.add_argument('--compare-implicit',action='store_true',help='Also time the previous implicit implementation on identical inputs')
    ap.add_argument('--compare-schedules',action='store_true',help='Compare joint/separate v8 on identical inputs')
    ap.add_argument('--focus-large',action='store_true',help='Twelve CIFAR10 cases: Conv3 conv2 and ResNet18 stage1; B=100/1000, G=1/2/8')
    ap.add_argument('--aten-baseline',nargs='+',help='Reuse benchmark_aten.py TXT; no ATen performance remeasurement')
    ap.add_argument('--serial-check',action='store_true',help='Time G single-group calls for adaptive12 and implicit; compatible with cached ATen')
    ap.add_argument('--probe',choices=['none','small','downsample','downsample15','downsample16'],default='none',help='Targeted CIFAR diagnosis: 6 small or 3 stride2 cases; cached ATen supported')
    ap.add_argument('--compare-v13',action='store_true',help='Time unchanged adaptive13 on identical inputs')
    ap.add_argument('--compare-weight-only',action='store_true',help='Stride2 ablation: weight reorder only, original ggO/gW')
    ap.add_argument('--native-layout',action='store_true',help='Compare native layouts; no timed conversion; default Conv3 CIFAR balanced sweep')
    ap.add_argument('--native-sweep',choices=['balanced','cartesian'],default='balanced',help='Balanced: B=1/10 G=1/4/16/64 and B=100/1000 G=1/2/4; batch/groups filter pairs. Cartesian: explicit batch x groups.')
    ap.add_argument('--resnet-v23',action='store_true',help='Full ResNet convolution candidate sweep, including v23 stride2 kernels; native layouts')
    ap.add_argument('--selector-sweep',action='store_true',help='Trace-driven v23 selector dataset: Conv3 + ResNet18 on CIFAR10, Tiny ImageNet and ImageNet10-128; all legal candidates, native layouts')
    ap.add_argument('--shard-count',type=int,default=1,help='Deterministically divide selector-sweep or v26-v33 tasks into balanced, disjoint shards')
    ap.add_argument('--shard-index',type=int,default=1,help='One-based selector-sweep/v26-v33 shard to execute')
    ap.add_argument('--compare-v23',action='store_true',help='Native stride2 comparison: v18, ggO tail, ggO single accumulator; same-run ggO stages')
    ap.add_argument('--compare-v26',action='store_true',help='Native ResNet coverage test: pointwise shortcut direct GEMMs and late-stage 3x3 stride2 candidate versus all prior candidates and ATen')
    ap.add_argument('--compare-v27',action='store_true',help='Native s3b1_conv1 test: 128-MiB wide ggO/gI chunks versus v18/v23 candidates and ATen')
    ap.add_argument('--compare-v28',action='store_true',help='Native RGB 7x7 stride-2 stem test: cumulative specialized gI, gW and ggO candidates versus implicit and ATen')
    ap.add_argument('--compare-v29',action='store_true',help='Native RGB 7x7 stride-2 stem test: measured gI plus packed ggO while retaining implicit gW')
    ap.add_argument('--compare-v30',action='store_true',help='Native s3b1_conv1 test: one reusable im2col buffer jointly schedules ggO and gW versus v18/v23/v27 and ATen')
    ap.add_argument('--compare-v31',action='store_true',help='Native s3b1_conv1 test: v30 joint ggO+gW plus v27 wide gI versus every prior candidate and ATen')
    ap.add_argument('--compare-v32',action='store_true',help='Native B=1000 s3b1_conv1 test: v31 one-buffer path with packed gW split-K 4/8 and larger GEMM tiles versus prior candidates and ATen')
    ap.add_argument('--compare-v33',action='store_true',help='Cross-generation native-layout audit: apply later pointwise, stem7 and stride2 schedules to every earlier ResNet family shape and compare with the old family oracle')
    ap.add_argument('--v33-family',choices=['all','pointwise','stride2','stem7'],default='all',help='With --compare-v33, restrict the cross-generation audit to one convolution family')
    ap.add_argument('--diagnose-v24',action='store_true',help='v23 candidates + independent ggO/gI/gW timings + full-call CUDA activity breakdown; no kernel changes')
    ap.add_argument('--diagnostic-profile',action=argparse.BooleanOptionalAction,default=True,help='With --diagnose-v24, trace full APIs after normal timing; disable if CUPTI unavailable')
    ap.add_argument('--correctness-v23',action='store_true')
    ap.add_argument('--correctness-v26',action='store_true')
    ap.add_argument('--correctness-v27',action='store_true')
    ap.add_argument('--correctness-v28',action='store_true')
    ap.add_argument('--correctness-v29',action='store_true')
    ap.add_argument('--correctness-v30',action='store_true')
    ap.add_argument('--correctness-v31',action='store_true')
    ap.add_argument('--correctness-v32',action='store_true')
    ap.add_argument('--correctness-v22',action='store_true')
    ap.add_argument('--compare-aten-e2e',action='store_true',help='v21: same-run complete v18/grouped ATen/serial ATen, including all layout conversion')
    ap.add_argument('--correctness-v21',action='store_true',help='Validate all three complete APIs and optional input combinations')
    ap.add_argument('--compare-v20',action='store_true',help='Same-run v18 full API and implicit gW stage reference')
    ap.add_argument('--correctness-v20',action='store_true',help='Static cropped incremental gW validation')
    ap.add_argument('--compare-v19',action='store_true',help='Same-run v17 full API plus v17/v18 gI and v17 gW stages')
    ap.add_argument('--correctness-v19',action='store_true',help='Packed parallel split-K weight-gradient validation')
    ap.add_argument('--compare-v18',action='store_true',help='Same-run unchanged v17 full-tail reference')
    ap.add_argument('--correctness-v18',action='store_true',help='Compact parity tail and group stride validation')
    ap.add_argument('--compare-v17',action='store_true',help='Same-run unchanged v16 reference')
    ap.add_argument('--correctness-v17',action='store_true',help='Bounded parity packing validation')
    ap.add_argument('--compare-v16',action='store_true',help='Same-run v14 and v15 static-parity references')
    ap.add_argument('--correctness-v16',action='store_true',help='Targeted static-parity reordered-weight validation')
    ap.add_argument('--compare-v15',action='store_true',help='Same-run adaptive14 and shared-packing candidate comparison')
    ap.add_argument('--compare-v12',action='store_true',help='Paired unchanged adaptive12 reference')
    ap.add_argument('--compare-v11',action='store_true',help='Time unchanged shape_safe dispatch')
    ap.add_argument('--compare-tail',action='store_true',help='Also time v9 scheduling with only tail correction')
    ap.add_argument('--compare-v9',action='store_true',help='Also time unchanged v9 with the same budget')
    ap.add_argument('--compare-v8',action='store_true',help='Also time v8 with the SAME workspace budget')
    ap.add_argument('--compare-v7',action='store_true',help='Time unchanged v7 on the same inputs')
    ap.add_argument('--scaling',action='store_true',help='Measure G independent calls without per-group synchronization')
    ap.add_argument('--compare-v6',action='store_true',help='Time unchanged v6 on the same inputs')
    ap.add_argument('--algorithm',choices=['auto','implicit','stem7_gi','stem7_gi_ggo','stem7_gi_gw','stem7_full','pointwise_stride2','stride2_joint_split_gw4','stride2_joint_split_gw8','stride2_joint_onebuf_wide_gi','stride2_joint_onebuf','stride2_wide_chunks','stride2_small_spatial','stride2_ggo_tail','stride2_ggo_single','stride2_implicit_gw','stride2_packed_gw','stride2_packed_gi_tail','stride2_packed_gi','stride2_static_weight','stride2_static','stride2_shared','adaptive14','stride2_weight','stride2_packed','winograd_weight','adaptive13','winograd_exact','adaptive12','stride2_phase','shape_safe','winograd','winograd_v5','winograd_v6','winograd_v7','winograd_v8','winograd_v9','winograd_tail','winograd_group','winograd_large','winograd_joint','winograd_separate'],default='auto',help='auto: optional exact table, built-in measured rules, then fitted fallback; implicit: generic path; other values force one candidate')
    ap.add_argument('--precision',choices=['ieee','tf32','tf32x3'],default='ieee')
    ap.add_argument('--rounds',type=int,default=5);ap.add_argument('--warmup',type=int,default=20)
    ap.add_argument('--warmup-ms',type=float,default=200);ap.add_argument('--target-ms',type=float,default=100)
    ap.add_argument('--iters',type=int,default=100,help='Maximum calls per round; adaptive minimum 2')
    ap.add_argument('--workspace-mb',type=float,default=128)
    ap.add_argument('--split-k',type=int,choices=[1,2,4,8,16])
    ap.add_argument('--memory-fraction',type=float,default=.65)
    ap.add_argument('--small-ms',type=float,default=.05,help='Provisional fast-case cutoff; 0.05ms=50us')
    ap.add_argument('--max-cv',type=float,default=5)
    ap.add_argument('--small-slowdown-pct',type=float,default=20)
    ap.add_argument('--small-extra-us',type=float,default=10)
    ap.add_argument('--stages',action=argparse.BooleanOptionalAction,default=False)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--output',default=None,help='Optional single TXT report; otherwise console only')
    a=ap.parse_args()
    if a.selector_sweep:
        if a.resnet_v23 or a.compare_v23 or a.compare_v26 or a.compare_v27 or a.compare_v28 or a.compare_v29 or a.compare_v30 or a.compare_v31 or a.compare_v32 or a.compare_v33 or a.diagnose_v24:ap.error('--selector-sweep is standalone; omit other comparison modes')
        a.native_layout=True
        a.models=['conv3','resnet18']
        a.datasets=['cifar10','tinyimagenet','imagenet10']
        a.imagenet_sizes=[128]
        a.stem='auto'
    if a.resnet_v23:
        if a.compare_v23 or a.compare_v26 or a.compare_v27 or a.compare_v28 or a.compare_v29 or a.compare_v30 or a.compare_v31 or a.compare_v32 or a.compare_v33 or a.diagnose_v24:ap.error('--resnet-v23 is a full-layer sweep; omit other comparison modes')
        a.native_layout=True;a.models=['resnet18']
    if a.diagnose_v24:
        if a.compare_v26 or a.compare_v27 or a.compare_v28 or a.compare_v29 or a.compare_v30 or a.compare_v31 or a.compare_v32 or a.compare_v33:ap.error('--diagnose-v24 and --compare-v26 through --compare-v33 are standalone modes')
        a.compare_v23=True
        if a.profile_iters<1:ap.error('--profile-iters must be positive')
    if a.compare_v26:
        if a.compare_v23 or a.compare_v27 or a.compare_v28 or a.compare_v29 or a.compare_v30 or a.compare_v31 or a.compare_v32 or a.compare_v33:ap.error('--compare-v26 cannot be combined with other version comparisons')
        a.native_layout=True;a.models=['resnet18'];a.datasets=a.datasets or ['cifar10']
        a.stem='auto';a.family='all';a.batch=a.batch or [100];a.groups=a.groups or [4]
    if a.compare_v27:
        if a.compare_v23 or a.compare_v26 or a.compare_v28 or a.compare_v29 or a.compare_v30 or a.compare_v31 or a.compare_v32 or a.compare_v33:ap.error('--compare-v27 cannot be combined with other version comparisons')
        a.native_layout=True;a.models=['resnet18'];a.datasets=a.datasets or ['cifar10']
        a.stem='auto';a.family='all';a.batch=a.batch or [100];a.groups=a.groups or [4]
    if a.compare_v28:
        if a.compare_v23 or a.compare_v26 or a.compare_v27 or a.compare_v29 or a.compare_v30 or a.compare_v31 or a.compare_v32 or a.compare_v33:ap.error('--compare-v28 cannot be combined with other version comparisons')
        a.native_layout=True;a.models=['resnet18']
        a.datasets=a.datasets or ['tinyimagenet','imagenet10']
        a.imagenet_sizes=[128];a.stem='standard';a.family='all'
        a.batch=a.batch or [100];a.groups=a.groups or [4]
    if a.compare_v29:
        if a.compare_v23 or a.compare_v26 or a.compare_v27 or a.compare_v28 or a.compare_v30 or a.compare_v31 or a.compare_v32 or a.compare_v33:ap.error('--compare-v29 cannot be combined with other version comparisons')
        a.native_layout=True;a.models=['resnet18']
        a.datasets=a.datasets or ['tinyimagenet','imagenet10']
        a.imagenet_sizes=[128];a.stem='standard';a.family='all'
        a.batch=a.batch or [100];a.groups=a.groups or [4]
    if a.compare_v30:
        if a.compare_v23 or a.compare_v26 or a.compare_v27 or a.compare_v28 or a.compare_v29 or a.compare_v31 or a.compare_v32 or a.compare_v33:ap.error('--compare-v30 cannot be combined with other version comparisons')
        a.native_layout=True;a.models=['resnet18'];a.datasets=['cifar10']
        a.stem='small';a.family='downsample3'
        a.batch=a.batch or [1,10,100,1000]
        a.groups=a.groups or [1,2,4,16,64]
    if a.compare_v31:
        if a.compare_v23 or a.compare_v26 or a.compare_v27 or a.compare_v28 or a.compare_v29 or a.compare_v30 or a.compare_v32 or a.compare_v33:ap.error('--compare-v31 cannot be combined with other version comparisons')
        a.native_layout=True;a.models=['resnet18'];a.datasets=['cifar10']
        a.stem='small';a.family='downsample3'
        a.batch=a.batch or [1,10,100,1000]
        a.groups=a.groups or [1,2,4,16,64]
    if a.compare_v32:
        if a.compare_v23 or a.compare_v26 or a.compare_v27 or a.compare_v28 or a.compare_v29 or a.compare_v30 or a.compare_v31 or a.compare_v33:ap.error('--compare-v32 cannot be combined with other version comparisons')
        a.native_layout=True;a.models=['resnet18'];a.datasets=['cifar10']
        a.stem='small';a.family='downsample3'
        a.batch=a.batch or [1000];a.groups=a.groups or [1,2,4]
    if a.compare_v33:
        if a.compare_v23 or a.compare_v26 or a.compare_v27 or a.compare_v28 or a.compare_v29 or a.compare_v30 or a.compare_v31 or a.compare_v32:ap.error('--compare-v33 cannot be combined with other version comparisons')
        a.native_layout=True;a.models=['resnet18']
        a.datasets=a.datasets or ['cifar10','tinyimagenet','imagenet10']
        a.imagenet_sizes=[128];a.stem='auto';a.family='all'
        a.batch=a.batch or [1,10,100,1000]
        a.groups=a.groups or [1,2,4,16,64]
    if a.compare_v23:
        a.native_layout=True;a.models=a.models or ['resnet18'];a.datasets=a.datasets or ['cifar10']
        a.family='downsample3';a.batch=a.batch or [100];a.groups=a.groups or [4]
    if a.native_layout:
        conflicts=[k for k,v in vars(a).items() if k.startswith('compare_') and k not in ('compare_v23','compare_v26','compare_v27','compare_v28','compare_v29','compare_v30','compare_v31','compare_v32','compare_v33') and v]
        if conflicts or a.aten_baseline or a.matrix!='none' or a.probe!='none' or a.focus_large or a.scaling or a.serial_check or a.stages or a.profile_kernels:
            ap.error('--native-layout is standalone; no cached baseline/stages/other comparisons')
        if a.algorithm!='auto':ap.error('--native-layout compares a fixed structural candidate set; omit --algorithm')
        a.models=a.models or ['conv3'];a.datasets=a.datasets or ['cifar10']
        a.batch=a.batch or [1,10,100,1000];a.groups=a.groups or [1,2,4,16,64]
    if a.shard_count<1 or not 1<=a.shard_index<=a.shard_count:ap.error('--shard-index must be in [1, --shard-count]')
    if not (a.selector_sweep or a.compare_v26 or a.compare_v27 or a.compare_v28 or a.compare_v29 or a.compare_v30 or a.compare_v31 or a.compare_v32 or a.compare_v33) and (a.shard_count!=1 or a.shard_index!=1):ap.error('--shard-count/--shard-index require --selector-sweep or --compare-v26 through --compare-v33')
    a.models=a.models or ['conv3','resnet18']
    a.imagenet_sizes=a.imagenet_sizes or ([128] if a.native_layout else [128,224])
    if a.native_layout or a.correctness_v32 or a.correctness_v31 or a.correctness_v30 or a.correctness_v29 or a.correctness_v28 or a.correctness_v27 or a.correctness_v26 or a.correctness_v23 or a.correctness_v22 or a.compare_aten_e2e or a.correctness_v21:
        if a.dtype!='fp32' or a.precision!='ieee':ap.error('native-layout and v21-v33 validation modes require FP32 IEEE')
    if a.compare_aten_e2e:
        conflicts=[k for k,v in vars(a).items() if k.startswith('compare_') and k!='compare_aten_e2e' and v]
        if a.aten_baseline or a.matrix!='none' or a.focus_large or a.probe!='none' or a.scaling or a.serial_check or a.stages or a.profile_kernels or conflicts:
            ap.error('--compare-aten-e2e is standalone; do not combine with saved baselines, stages or other comparison modes')
        if a.algorithm not in ('auto','stride2_packed_gi_tail'):ap.error('v21 comparison fixes Triton to v18')
        a.algorithm='stride2_packed_gi_tail'
        a.models=['resnet18'];a.datasets=a.datasets or ['cifar10'];a.family='downsample3'
        a.batch=a.batch or [100];a.groups=a.groups or [4]
    if a.probe!='none':
        if a.matrix!='none' or a.focus_large:ap.error('--probe cannot combine with matrix/focus-large')
        a.models=['conv3'] if a.probe=='small' else ['resnet18']
        a.datasets=['cifar10'];a.stem='small';a.width=128
        a.batch=[10] if a.probe=='small' else [100]
        a.groups=[1,4,64] if a.probe=='small' else [4]
        a.family='dense3' if a.probe=='small' else 'downsample3'
        a.algorithm='adaptive14';a.stages=True;a.profile_kernels=True
        a.compare_v13=True;a.compare_weight_only=a.probe=='downsample'
        a.serial_check=a.probe=='small'
        if a.probe=='downsample15':
            a.algorithm='stride2_static';a.compare_v15=True
            a.compare_v13=False;a.compare_weight_only=False
            a.stages=False;a.profile_kernels=False
        if a.probe=='downsample16':
            a.algorithm='stride2_static_weight';a.compare_v16=True
            a.compare_v13=False;a.compare_v15=False;a.compare_weight_only=False
            a.stages=False;a.profile_kernels=False
    if a.matrix!='none' and a.focus_large:ap.error('--matrix and --focus-large are mutually exclusive')
    a.datasets=a.datasets or (['cifar10'] if a.matrix.startswith('cifar') else ['tinyimagenet','imagenet'] if a.matrix=='transfer' else ['cifar10','imagenet'])
    if a.matrix.startswith('cifar') and a.datasets!=['cifar10']:ap.error('CIFAR matrix requires --datasets cifar10')
    a.groups=a.groups or ([1,2,4,8,16,32,64] if a.matrix!='none' else [1,2,4,8,16,32])
    a.batch=a.batch or ([1,10,100,200,1000] if a.matrix!='none' else [1,10,32,100,256,1000])
    if a.matrix=='transfer' and a.stem=='auto':a.stem='both'
    if a.focus_large:
        a.groups=[1,2,8];a.batch=[100,1000];a.datasets=['cifar10'];a.models=['conv3','resnet18'];a.width=128;a.stem='small'
    if a.aten_baseline and (a.scaling or a.compare_graph):ap.error('--aten-baseline cannot combine with --scaling/--compare-graph')
    if min(a.groups+a.batch+a.imagenet_sizes+[a.width,a.warmup,a.rounds,a.iters,a.profile_iters])<1 or a.rounds<2 or a.iters<2:ap.error('Positive dimensions/warmup; rounds and iters >=2')
    if not 0<a.memory_fraction<=1 or min(a.workspace_mb,a.warmup_ms,a.target_ms,a.small_ms,a.max_cv,a.small_slowdown_pct,a.small_extra_us)<0:ap.error('Invalid budget/timing parameters')
    suite=[c for c in cases(a.models,a.datasets,a.imagenet_sizes,a.stem,a.width) if a.filter in c['case']]
    if a.family!='all':
        suite=[c for c in suite if (c['C']==3 if a.family=='rgb' else c['R']==c['S']==3 and c['stride']==1 and c['C']>3 if a.family=='dense3' else c['R']==c['S']==3 and c['stride']==2 if a.family=='downsample3' else c['R']==c['S']==1)]
    if a.compare_v26:
        suite=[c for c in suite if c['model']=='resnet18' and c['stride']==2 and
               ((c['R']==c['S']==3 and c['name'] in ('s3b1_conv1','s4b1_conv1')) or
                (c['R']==c['S']==1 and c['name'] in ('s3b1_shortcut','s4b1_shortcut')))]
    if a.compare_v27:
        suite=[c for c in suite if c['model']=='resnet18' and
               c['name']=='s3b1_conv1' and c['R']==c['S']==3 and c['stride']==2]
    if a.compare_v28:
        suite=[c for c in suite if c['model']=='resnet18' and c['name']=='stem' and
               c['C']==3 and c['R']==c['S']==7 and c['stride']==2 and c['padding']==3]
    if a.compare_v29:
        suite=[c for c in suite if c['model']=='resnet18' and c['name']=='stem' and
               c['C']==3 and c['R']==c['S']==7 and c['stride']==2 and c['padding']==3]
    if a.compare_v30:
        suite=[c for c in suite if c['model']=='resnet18' and
               c['name']=='s3b1_conv1' and c['R']==c['S']==3 and
               c['stride']==2 and c['padding']==1]
    if a.compare_v31:
        suite=[c for c in suite if c['model']=='resnet18' and
               c['name']=='s3b1_conv1' and c['R']==c['S']==3 and
               c['stride']==2 and c['padding']==1]
    if a.compare_v32:
        suite=[c for c in suite if c['model']=='resnet18' and
               c['name']=='s3b1_conv1' and c['R']==c['S']==3 and
               c['stride']==2 and c['padding']==1]
    if a.compare_v33:
        def transfer_family(c):
            if (c['R'],c['S'],c['stride'],c['padding'])==(1,1,2,0):
                return 'pointwise'
            if (c['R'],c['S'],c['stride'],c['padding'],c['C'])==(7,7,2,3,3):
                return 'stem7'
            if (c['R'],c['S'],c['stride'],c['padding'])==(3,3,2,1):
                return 'stride2'
            return None
        # Retain only unique physical families: CIFAR supplies the standard
        # downsample shapes, Tiny supplies the smaller transfer shapes, and
        # ImageNet10 contributes only its distinct 128x128 7x7 stem.  This
        # avoids rerunning identical downsample tensors under dataset aliases.
        suite=[c for c in suite if transfer_family(c) is not None and
               (c['dataset']!='imagenet10' or transfer_family(c)=='stem7') and
               (a.v33_family=='all' or transfer_family(c)==a.v33_family)]
    if a.focus_large:
        suite=[c for c in suite if (c['model']=='conv3' and c['name']=='conv2') or (c['model']=='resnet18' and (c['C'],c['O'],c['H'],c['R'],c['stride'])==(64,64,32,3,1))]
    if not suite:ap.error('No matching cases')
    tasks=[(case,g,b) for case in suite for b,g in matrix_points(a,case)]
    if not tasks:ap.error('No matching B/G pairs in matrix; check --batch/--groups filters')
    if a.selector_sweep or a.compare_v26 or a.compare_v27 or a.compare_v28 or a.compare_v29 or a.compare_v30 or a.compare_v31 or a.compare_v32 or a.compare_v33:
        all_task_count=len(tasks)
        weight_fn=(coverage33_task_weight if a.compare_v33 else
                   coverage32_task_weight if a.compare_v32 else
                   coverage31_task_weight if a.compare_v31 else
                   coverage30_task_weight if a.compare_v30 else
                   coverage29_task_weight if a.compare_v29 else
                   coverage28_task_weight if a.compare_v28 else
                   coverage27_task_weight if a.compare_v27 else
                   coverage26_task_weight if a.compare_v26 else selector_task_weight)
        tasks,shard_loads=shard_selector_tasks(tasks,a.shard_count,a.shard_index-1,weight_fn)
        prefix='V33_SHARD ' if a.compare_v33 else 'V32_SHARD ' if a.compare_v32 else 'V31_SHARD ' if a.compare_v31 else 'V30_SHARD ' if a.compare_v30 else 'V29_SHARD ' if a.compare_v29 else 'V28_SHARD ' if a.compare_v28 else 'V27_SHARD ' if a.compare_v27 else 'V26_SHARD ' if a.compare_v26 else 'SELECTOR_SHARD '
        print(prefix+json.dumps(dict(index=a.shard_index,count=a.shard_count,
              selected_tasks=len(tasks),all_tasks=all_task_count,estimated_loads=shard_loads),ensure_ascii=False),flush=True)
    if a.mode=='list':
        for case,g,b in tasks:
            print(json.dumps(dict(case,G=g,B=b),ensure_ascii=False))
        print('Total configurations:',len(tasks));return
    if not torch.cuda.is_available():raise SystemExit('CUDA GPU required')
    torch.manual_seed(123);torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.allow_tf32=a.precision=='tf32';torch.backends.cuda.matmul.allow_tf32=a.precision=='tf32'
    dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[a.dtype]
    if a.mode=='correctness':
        (correctness32 if a.correctness_v32 else correctness31 if a.correctness_v31 else correctness30 if a.correctness_v30 else correctness29 if a.correctness_v29 else correctness28 if a.correctness_v28 else correctness27 if a.correctness_v27 else correctness26 if a.correctness_v26 else correctness23 if a.correctness_v23 else correctness22 if a.correctness_v22 else correctness21 if a.correctness_v21 else correctness20 if a.correctness_v20 else correctness19 if a.correctness_v19 else correctness18 if a.correctness_v18 else correctness17 if a.correctness_v17 else correctness16 if a.correctness_v16 else correctness_structural if a.correctness_structural else correctness_large if a.focus_large else correctness)(dtype,a.precision);return
    hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('benchmark.py','conv_double_bwd.py')}
    revision=('v33-cross-generation-transfer-audit-1' if a.compare_v33 else
              'v32.1-stride2-large-split-gw-compile-fix-1' if a.compare_v32 else
              'v31-stride2-joint-wide-gi-1' if a.compare_v31 else
              'v30-stride2-joint-onebuf-1' if a.compare_v30 else
              'v29-stem7-gi-ggo-1' if a.compare_v29 else
              'v28-stem7-1' if a.compare_v28 else
              'v27-wide-chunks-1' if a.compare_v27 else
              'v26-kernel-coverage-1' if a.compare_v26 else
              'v23-selector-sweep-1' if a.selector_sweep else
              'v34-measured-family-auto-selector-1')
    meta=dict(benchmark_revision=revision,source_sha256=hashes,torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(),args=vars(a))
    if a.aten_baseline:
        BASELINES.clear()
        meta['baseline_sha256']={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in a.aten_baseline}
        for path in a.aten_baseline:load_baseline(path,meta,a)
    print('ENV '+json.dumps(meta,ensure_ascii=False),flush=True)
    file=Path(a.output) if a.output else None
    if file and file.suffix.lower()!='.txt':ap.error('--output must end with .txt')
    if a.resume and file is None:ap.error('--resume requires --output')
    done=set();records=[]
    if file and a.resume and file.exists():
        old=None
        with file.open(encoding='utf-8') as fh:
            for line in fh:
                if line.startswith('ENV '):old=json.loads(line[4:])
                elif line.startswith('RESULT '):
                    try:row=json.loads(line[7:])
                    except json.JSONDecodeError:continue  # interrupted final line
                    records.append(row)
                    if row['status'] in ('ok','skip_index_limit','skip_memory_estimate','skip_missing_baseline'):
                        done.add((row['case'],row['G'],row['F']))
        if old is None:raise SystemExit('Missing report environment header')
        old['args']['resume']=True
        if old!=meta:raise SystemExit('Resume metadata mismatch; use a new output name')
    elif file and file.exists():raise SystemExit('Output exists; use --resume or another filename')
    fh=file.open('a',encoding='utf-8') if file else None
    try:
        if fh:
            if file.stat().st_size==0:fh.write('ENV '+json.dumps(meta,ensure_ascii=False)+'\n')
            else:fh.write('\n')
            fh.flush()
        totals={}
        for case,g,f in tasks:
            if (case['case'],g,f) in done:continue
            print('Testing',case['case'],'G',g,'batch',f,flush=True)
            fatal=False
            try:row=(run_native if a.native_layout else run_e2e if a.compare_aten_e2e else run)(case,g,f,a)
            except torch.cuda.OutOfMemoryError as e:row=dict(case,G=g,F=f,status='oom',error=str(e))
            except Exception as e:
                row=dict(case,G=g,F=f,status='error',error=repr(e));fatal=True
                if isinstance(e,ValidationError):row['validation']=e.details
            records.append(row)
            msg=summary(row);print(msg,flush=True)
            totals[row.get('policy',row['status'])]=totals.get(row.get('policy',row['status']),0)+1
            if fh:
                fh.write(msg+'\nRESULT '+json.dumps(row,ensure_ascii=False)+'\n');fh.flush()
            gc.collect();torch.cuda.empty_cache()
            if fatal:raise SystemExit('Stopped after recorded error')
        for rec in (() if a.native_layout or a.compare_aten_e2e else scaling_from_g1(records,a.max_cv)):
            line='SCALING '+json.dumps(rec,ensure_ascii=False)
            print(line,flush=True)
            if fh:fh.write(line+'\n')
        for rec in (() if a.native_layout or a.compare_aten_e2e else model_summaries(records,tasks,a)):
            line='MODEL_SUM '+json.dumps(rec,ensure_ascii=False)
            print(line,flush=True)
            if fh:fh.write(line+'\n')
        if a.native_layout:
            for rec in native_model_summaries(records,tasks,a):
                line='NATIVE_MODEL_SUM '+json.dumps(rec,ensure_ascii=False)
                print(line,flush=True)
                if fh:fh.write(line+'\n')
            if a.compare_v33:
                line='V33_SUMMARY '+json.dumps(
                    transfer33_summary(records),ensure_ascii=False)
                print(line,flush=True)
                if fh:fh.write(line+'\n')
            else:
                line='NATIVE_SUMMARY '+json.dumps(
                    native_summary(records),ensure_ascii=False)
                print(line,flush=True)
                if fh:fh.write(line+'\n')
        msg='SUMMARY '+json.dumps(totals,ensure_ascii=False)
        print(msg,flush=True)
        if fh:fh.write(msg+'\n');fh.flush()
    finally:
        if fh:fh.close()

if __name__=='__main__':main()
