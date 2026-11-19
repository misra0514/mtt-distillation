"""Standalone ATen convolution double-backward baseline. PyTorch only; no dataset downloads."""
import argparse,gc,hashlib,json,random,statistics,time
from pathlib import Path
import torch

def cases(models=('conv3','resnet18'),datasets=('cifar10','imagenet'),imagenet_sizes=(128,224),stem='auto',width=128):
    result=[]
    for dataset in datasets:
        for size in ([32] if dataset=='cifar10' else imagenet_sizes):
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



def sweep_pairs(batch,groups=None):
    for f in batch:
        gs=groups if groups is not None else ([1,2,4,8,16,32,64] if f==1 else ([1,2,4,8,16,32] if f<=10 else [1,2,4,8]))
        for g in gs:yield g,f


def unique_cases(a):
    # Time equivalent convolutions only once across models/datasets/image sizes.
    unique={}
    for c in cases(a.models,a.datasets,a.imagenet_sizes,'auto',128):
        key=tuple(c[k] for k in ('C','O','H','W','R','S','stride','padding','dilation','bias'))
        if a.filter and a.filter not in c['case']:continue
        occurrence=dict(model=c['model'],dataset=c['dataset'],image_size=c['image_size'],layers=c['layers'],multiplicity=c['multiplicity'])
        if key not in unique:unique[key]=dict(c,occurrences=[occurrence])
        else:unique[key]['occurrences'].append(occurrence)
    return list(unique.values())


def split_args(args):
    return [tuple(None if t is None else t[g:g+1] for t in args[:6])+tuple(args[6:]) for g in range(args[5].shape[0])]


def storage_bytes(ts):
    storages={}
    for t in ts:
        if isinstance(t,torch.Tensor):
            st=t.untyped_storage();storages[(str(t.device),st.data_ptr())]=st.nbytes()
    return sum(storages.values())


def providers(args):
    ready=prepare_aten(args)
    singles=[prepare_aten(one) for one in split_args(args)]
    def serial():
        # No synchronization between groups. Retain all outputs as in the fused API.
        return [torch.ops.aten._convolution_double_backward(*one) for one in singles]
    return dict(serial_aten=serial,grouped_aten=lambda:torch.ops.aten._convolution_double_backward(*ready),
                grouped_e2e=lambda:aten_group_first(args)),ready,singles


def sampled_layout_check(args,ready,singles):
    # Diagnostic only: grouped/ungrouped cuDNN may choose different algorithms.
    # No tolerance relaxation or sampled check is presented as full correctness proof.
    fused=torch.ops.aten._convolution_double_backward(*ready)
    g,f,c,h,w=args[5].shape;o=args[4].shape[1];p,q=args[3].shape[-2:]
    views=(None if fused[0] is None else fused[0].reshape(f,g,o,p,q).permute(1,0,2,3,4),
           None if fused[1] is None else fused[1].reshape(f,g,c,h,w).permute(1,0,2,3,4),
           None if fused[2] is None else fused[2].reshape(args[4].shape))
    report={n:dict(max_abs=0.,diff_sq=0.,ref_sq=0.,sampled=0,mismatches=0) for n in ('ggO','gI','gW')}
    for i,one in enumerate(singles):
        refs=torch.ops.aten._convolution_double_backward(*one)
        for n,x,y in zip(report,views,refs):
            if x is None or y is None:
                if x is not y:raise RuntimeError('None-output mismatch')
                continue
            x=x[i]
            if x.shape!=y.shape:raise RuntimeError('Output layout/shape mismatch')
            # Use multidimensional indexing to avoid a full contiguous layout copy.
            count=min(4096,y.numel());flat=torch.linspace(0,y.numel()-1,count,device=y.device).long()
            indices=[];remainder=flat
            for size in reversed(y.shape):indices.append(remainder%size);remainder=remainder//size
            idx=tuple(reversed(indices));xx=x[idx].double();yy=y[idx].double();d=xx-yy
            if not torch.isfinite(xx).all() or not torch.isfinite(yy).all():raise RuntimeError('Nonfinite ATen output')
            r=report[n];r['max_abs']=max(r['max_abs'],d.abs().max().item())
            r['diff_sq']+=d.square().sum().item();r['ref_sq']+=yy.square().sum().item();r['sampled']+=count
            r['mismatches']+=(~torch.isclose(xx,yy,atol=.003,rtol=.002)).sum().item()
        del refs
    for r in report.values():r['rel_l2']=(r.pop('diff_sq')/max(r.pop('ref_sq'),1e-60))**.5
    return report


def run(case,g,f,a):
    c,o,h,w,r,s=[case[k] for k in ('C','O','H','W','R','S')]
    st,pad,dil=[case[k] for k in ('stride','padding','dilation')]
    p=(h+2*pad-dil*(r-1)-1)//st+1;q=(w+2*pad-dil*(s-1)-1)//st+1
    row=dict(case,G=g,F=f,P=p,Q=q,status='pending')
    nx=g*f*c*h*w;nd=g*f*o*p*q;nw=g*o*c*r*s
    # Includes resident inputs, prepacked copies, retained outputs and a workspace allowance.
    estimate=4*(7*nx+6*nd+6*nw)+256*1024**2
    free,_=torch.cuda.mem_get_info();row['estimated_bytes']=estimate
    if estimate>free*a.memory_fraction:return dict(row,status='skip_memory_estimate',free_bytes=free)
    if max(nx,nd,nw)>=2**30:return dict(row,status='skip_index_limit')
    seed=123+int(hashlib.sha256(str((g,f,c,o,h,w,r,s,st,pad,dil,case['bias'])).encode()).hexdigest()[:8],16)
    torch.manual_seed(seed);row['seed']=seed
    args=make_args((g,f,c,o,h,w,r,s),torch.float32,'cuda',(st,)*2,(pad,)*2,(dil,)*2,opts=(True,True,case['bias']))
    funcs,ready,singles=providers(args)
    row['input_storage_bytes']=storage_bytes(args[:6])
    row['prepacked_extra_storage_bytes']=storage_bytes(list(args[:6])+list(ready[:6]))-row['input_storage_bytes']
    row['sampled_grouped_vs_serial']=sampled_layout_check(args,ready,singles)
    row.update(paired(funcs,a))
    for mode in ('event','wall'):
        serial=row['serial_aten_'+mode+'_median'];grouped=row['grouped_aten_'+mode+'_median'];e2e=row['grouped_e2e_'+mode+'_median']
        row['grouped_'+mode+'_speedup']=serial/grouped;row['e2e_'+mode+'_speedup']=serial/e2e
    row['group_images_per_second']=g*f*1000/row['grouped_aten_event_median']
    row['e2e_group_images_per_second']=g*f*1000/row['grouped_e2e_event_median']
    row['stable']=max(row[k+'_event_cv_pct'] for k in funcs)<=a.max_cv
    sp=row['grouped_event_speedup']
    row['scaling']='single_group_control' if g==1 else ('unstable' if not row['stable'] else ('improvement' if sp>1.03 else ('regression' if sp<.97 else 'near_tie')))
    # Comparable live storage accounting; not allocator-reserved memory or nvidia-smi usage.
    for name in funcs:
        row[name+'_task_peak_bytes']=row['input_storage_bytes']+row[name+'_peak_extra_bytes']+(row['prepacked_extra_storage_bytes'] if name=='grouped_aten' else 0)
    return dict(row,status='ok')


def summary(r):
    title=f"{r['case']} | G={r['G']} batch={r['F']} | {r['status']}"
    if r['status']!='ok':return title+' | '+str(r.get('error',r.get('estimated_bytes','')))
    a=r['serial_aten_event_median'];b=r['grouped_aten_event_median'];c=r['grouped_e2e_event_median']
    lines=[title+' | '+r['scaling'],f'  ms: serial_ATen={a:.6f} grouped_ATen={b:.6f} grouped_ATen+layout={c:.6f}',
           f"  speedup: grouped={r['grouped_event_speedup']:.3f}x E2E={r['e2e_event_speedup']:.3f}x | throughput={r['group_images_per_second']:.1f} group-images/s"]
    lines.append('  CV(%): '+' '.join(k+'='+format(r[k+'_event_cv_pct'],'.2f') for k in ('serial_aten','grouped_aten','grouped_e2e')))
    lines.append('  peak_extra(MiB): '+' '.join(k+'='+format(r[k+'_peak_extra_bytes']/1048576,'.2f') for k in ('serial_aten','grouped_aten','grouped_e2e')))
    lines.append('  task_peak(MiB): '+' '.join(k+'='+format(r[k+'_task_peak_bytes']/1048576,'.2f') for k in ('serial_aten','grouped_aten','grouped_e2e')))
    lines.append('  sampled rel_L2: '+' '.join(k+'='+format(v['rel_l2'],'.3e') for k,v in r['sampled_grouped_vs_serial'].items()))
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models',nargs='+',choices=['conv3','resnet18'],default=['conv3','resnet18'])
    ap.add_argument('--datasets',nargs='+',choices=['cifar10','imagenet'],default=['cifar10','imagenet'])
    ap.add_argument('--imagenet-sizes',nargs='+',type=int,default=[128,224])
    ap.add_argument('--batch',nargs='+',type=int,default=[1,10,100,1000])
    ap.add_argument('--groups',nargs='+',type=int,default=None,help='Override the batch-dependent Fuse sweep')
    ap.add_argument('--filter',default='')
    ap.add_argument('--rounds',type=int,default=5);ap.add_argument('--warmup',type=int,default=20)
    ap.add_argument('--warmup-ms',type=float,default=200);ap.add_argument('--target-ms',type=float,default=100)
    ap.add_argument('--iters',type=int,default=100);ap.add_argument('--memory-fraction',type=float,default=.65)
    ap.add_argument('--max-cv',type=float,default=5)
    ap.add_argument('--output',default='aten_baseline.txt');ap.add_argument('--resume',action='store_true');ap.add_argument('--list',action='store_true')
    a=ap.parse_args()
    if min(a.batch+a.imagenet_sizes+(a.groups or [1]))<1:ap.error('Dimensions must be positive')
    if a.rounds<2 or a.iters<2 or a.warmup<1 or min(a.warmup_ms,a.target_ms,a.max_cv)<0 or not 0<a.memory_fraction<=1:ap.error('Invalid timing/budget arguments')
    suite=unique_cases(a);pairs=list(sweep_pairs(a.batch,a.groups))
    if not suite:ap.error('No matching shapes')
    if a.list:
        print(json.dumps(dict(shapes=suite,pairs=pairs,configurations=len(suite)*len(pairs)),indent=2));return
    if not torch.cuda.is_available():raise SystemExit('CUDA GPU required; this script only imports PyTorch')
    torch.manual_seed(123);torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    meta=dict(revision='aten-baseline-1',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),dtype='fp32',precision='ieee',cudnn_benchmark=True,tf32=False,args=vars(a),pairs=pairs,unique_shapes=len(suite),configurations=len(suite)*len(pairs),serial_contract='G independent groups on the current stream; no per-group synchronize; all outputs retained; no concatenation',memory_contract='peak_extra is allocated bytes above resident harness; task_peak adds group-first input storage and required prepacking; excludes allocator reservation; first-use tuning excluded',validation='up to 4096 positions per output per group; grouped vs serial numeric diagnostics, not a full correctness proof')
    dest=Path(a.output)
    if dest.suffix.lower()!='.txt':ap.error('--output must end in .txt')
    done=set()
    if dest.exists():
        if not a.resume:ap.error('Output exists; choose a new filename or use --resume')
        old=None
        for l in dest.read_text().splitlines():
            if l.startswith('ENV '):old=json.loads(l[4:])
            if l.startswith('RESULT '):
                row=json.loads(l[7:])
                if row['status'] in ('ok','skip_memory_estimate','skip_index_limit'):done.add((row['case'],row['G'],row['F']))
        def comparable(m):
            z=dict(m);z['args']={k:v for k,v in m['args'].items() if k not in ('resume','output')};return z
        if old is None or comparable(old)!=comparable(meta):ap.error('Resume requires matching source, environment and arguments')
    elif a.resume:ap.error('Resume file does not exist')
    with dest.open('a' if a.resume else 'w',encoding='utf-8') as out:
        def emit(text):print(text,flush=True);out.write(text+'\n');out.flush()
        if not a.resume:emit('ENV '+json.dumps(meta,ensure_ascii=False))
        totals={}
        for case in suite:
            for g,f in pairs:
                if (case['case'],g,f) in done:continue
                print('Testing',case['case'],'G',g,'batch',f,flush=True)
                try:
                    with torch.no_grad():row=run(case,g,f,a)
                except torch.OutOfMemoryError as e:row=dict(case,G=g,F=f,status='oom',error=str(e)[:1000])
                except Exception as e:row=dict(case,G=g,F=f,status='error',error=repr(e))
                emit(summary(row));emit('RESULT '+json.dumps(row,ensure_ascii=False))
                totals[row['status']]=totals.get(row['status'],0)+1
                gc.collect();torch.cuda.empty_cache()
                if row['status']=='error':emit('Stopped after recorded error; resume after resolving it.');return
        emit('SUMMARY '+json.dumps(totals))


if __name__=='__main__':main()
