#!/usr/bin/env python3
"""Standalone ConvNet-only interface Fuse benchmark; no training script."""
import argparse
import csv
import gc
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version', action='version', version='ConvNet standalone: single CSV, Fuse or IPC sweep')
    p.add_argument('--scan', choices=['fuse','ipc'], default='fuse')
    p.add_argument('--ipc', type=int, default=1, help='Fixed IPC in Fuse sweep')
    p.add_argument('--fuse', type=int, default=1, help='Fixed Fuse in IPC sweep')
    p.add_argument('--ipcs', type=int, nargs='+', help='Explicit IPC sweep points')
    p.add_argument('--ipc-start', type=int, default=10)
    p.add_argument('--ipc-max', type=int, default=640, help='Double IPC from ipc-start up to this limit')
    p.add_argument('--fuses', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument('--module', default='networks.networks_stateless_basicblock')
    p.add_argument('--repo', default='.')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--classes', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=None, help='Default = IPC * classes, fixed across Fuse')
    p.add_argument('--image-size', type=int, default=32)
    p.add_argument('--channels', type=int, default=3)
    p.add_argument('--conv-width', type=int, default=128)
    p.add_argument('--conv-depth', type=int, default=3)
    p.add_argument('--conv-pool', choices=['all', 'except-last'], default='all')
    p.add_argument('--v-fuse', choices=['true', 'false', 'both'], default='true')
    p.add_argument('--only', default='', help='Comma-separated interface names')
    p.add_argument('--case-contains', default='', help='Substring filter on case ID')
    p.add_argument('--shape-config', help='JSON array of layer descriptions; replaces built-in layer templates')
    p.add_argument('--list-cases', action='store_true', help='Print planned cases without importing torch')
    p.add_argument('--out', default=None, help='ONE output CSV file, not a directory')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-reset-inputs', action='store_true', help='Reuse inputs without restoring; less memory')
    p.add_argument('--tf32', choices=['off', 'on'], default='off')
    p.add_argument('--cudnn-benchmark', action='store_true')
    a = p.parse_args()
    if min(a.ipc, a.fuse, a.ipc_start, a.ipc_max, a.classes, a.image_size, a.channels, a.conv_width, a.conv_depth) < 1:
        p.error('Dimensions and IPC must be positive')
    if a.warmup < 1 or a.repeats < 2 or any(k < 1 for k in a.fuses):
        p.error('warmup >=1, repeats >=2, fuse >=1 required')
    if a.batch_size is not None and a.batch_size < 1:
        p.error('batch size must be positive')
    if a.ipcs is not None and any(v<1 for v in a.ipcs):
        p.error('IPC sweep values must be positive')
    if a.scan=='ipc' and a.batch_size is not None:
        p.error('--batch-size would prevent IPC scaling; omit it for --scan ipc')
    if a.scan=='ipc' and a.ipcs is None and a.ipc_max<a.ipc_start:
        p.error('--ipc-max must be >= --ipc-start')
    a.repo = str(Path(a.repo).resolve())
    if a.shape_config:
        a.shape_config = str(Path(a.shape_config).resolve())
    label='fuse_ipc%d'%a.ipc if a.scan=='fuse' else 'ipc_fuse%d'%a.fuse
    a.out=str(Path(a.out or ('conv_kernel_%s_%s_%d.csv'%(label,time.strftime('%Y%m%d_%H%M%S'),os.getpid()))).resolve())
    return a


def layers(a):
    """ConvNet layer shapes only; input channels/groups grow with Fuse."""
    if a.shape_config:
        data=json.loads(Path(a.shape_config).read_text())
        if not isinstance(data,list):
            raise ValueError('Shape config must be a JSON array')
        for item in data:
            if item['kind'] not in ['conv','norm','pool','linear_fused','ce']:
                raise ValueError('Only ConvNet layer kinds are supported')
            if item['kind']=='norm' and not item.get('relu',True):
                raise ValueError('ConvNet uses IN+ReLU, not a standalone IN test')
        return data
    result=[]
    def add(layer,kind,**kw):
        result.append(dict(model='conv',layer=layer,kind=kind,**kw))
    h=a.image_size; ci=a.channels
    for i in range(a.conv_depth):
        co=a.conv_width
        add('conv%d'%(i+1),'conv',ci=ci,co=co,h=h,w=h,kernel=3,stride=1,padding=1,bias=True)
        add('norm%d_relu'%(i+1),'norm',c=co,h=h,w=h)
        if a.conv_pool=='all' or i<a.conv_depth-1:
            add('pool%d'%(i+1),'pool',c=co,h=h,w=h)
            h//=2
        ci=co
    if h<1: raise ValueError('Too many pooling layers for image size')
    add('classifier','linear_fused',i=co*h*h,o=a.classes,h=h,w=h)
    add('cross_entropy','ce')
    return result


def cases(a):
    result=[]
    flags=[True,False] if a.v_fuse=='both' else [a.v_fuse=='true']
    names={
        'conv':['conv_bwd','conv_double_bwd'],
        'norm':['insNormNRelu_bwd','insNormNRelu_double_bwd'],
        'pool':['avgPool_bwd','avgPool_double_bwd'],
        'linear_fused':['linerFused_bwd','linearFused_double_bwd'],
        'ce':['crossEntropy_bwd','crossEntropy_double_bwd'],
    }
    for spec in layers(a):
        for name in names[spec['kind']]:
            for flag in flags if spec['kind']=='norm' else [None]:
                item=dict(spec,interface=name)
                suffix=''
                if flag is not None:
                    item['vfuse']=flag; suffix='.vfuse%d'%int(flag)
                item['case_id']='conv.%s.%s%s'%(spec['layer'],name,suffix)
                result.append(item)
    if a.only:
        requested=set(a.only.split(','))
        allowed={name for values in names.values() for name in values}
        if requested-allowed:
            raise ValueError('Not a selected ConvNet interface: '+','.join(sorted(requested-allowed)))
        result=[c for c in result if c['interface'] in requested]
    if a.case_contains: result=[c for c in result if a.case_contains in c['case_id']]
    ids=[c['case_id'] for c in result]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate case IDs')
    if not result: raise ValueError('No cases selected')
    return result


def make_inputs(torch,kernels,a,spec,k):
    b=a.batch_size or a.ipc*a.classes
    device='cuda:%d'%a.device
    tensors=[]
    def rand(*shape):
        t=torch.randn(*shape,device=device,dtype=torch.float32); tensors.append(t); return t
    def ones(*shape):
        t=torch.ones(*shape,device=device,dtype=torch.float32); tensors.append(t); return t
    name=spec['interface']; kind=spec['kind']; fn=getattr(kernels,name); kw={}
    if kind=='conv':
        ci,co,h,w=spec['ci'],spec['co'],spec['h'],spec['w']
        stride,pad,q=spec.get('stride',1),spec.get('padding',1),spec.get('kernel',3)
        oh,ow=(h+2*pad-q)//stride+1,(w+2*pad-q)//stride+1
        x=rand(b,k*ci,h,w); weight=rand(k*co,ci,q,q); go=rand(b,k*co,oh,ow)
        if name=='conv_bwd':
            args=(x,weight,go); kw=dict(stride=stride,padding=pad,groups=k)
        else:
            args=(rand(*x.shape),rand(*weight.shape),rand(k*co) if spec.get('bias',True) else None,go,weight,x)
            kw=dict(stride_=[stride,stride],padding_=[pad,pad],groups_=k)
    elif kind=='norm':
        c,h,w=k*spec['c'],spec['h'],spec['w']
        x=rand(b,c,h,w); weight=ones(c); go=rand(b,c,h,w)
        out=torch.relu(torch.nn.functional.instance_norm(x,weight=weight,eps=1e-5)); tensors.append(out)
        if name=='insNormNRelu_bwd': args=(x,weight,out,go)
        else: args=(rand(*x.shape),rand(c),rand(c),go,out,weight,x)
        kw=dict(v_fuse=spec['vfuse'])
    elif kind=='pool':
        c,h,w=k*spec['c'],spec['h'],spec['w']; x=rand(b,c,h,w)
        args=(x,rand(b,c,h//2,w//2)) if name=='avgPool_bwd' else (x,)
    elif kind=='linear_fused':
        i,o=spec['i'],spec['o']; h,w=spec.get('h',1),spec.get('w',1)
        if i%(h*w): raise ValueError('Linear features must be divisible by h*w')
        x=rand(b,k*(i//(h*w)),h,w); weight=rand(k*o,i); go=rand(k*b,o)
        args=(x,weight,go); kw=dict(Fuse=k)
        if name=='linearFused_double_bwd': args+=(rand(*x.shape),rand(*weight.shape),rand(k*o))
    elif kind=='ce':
        logits=rand(k*b,a.classes)
        if name=='crossEntropy_bwd':
            target=torch.randint(a.classes,(k*b,),device=device); tensors.append(target)
            args=(logits,target,k)
        else: args=(logits,rand(*logits.shape),k)
    else: raise ValueError('Only ConvNet interfaces are supported')
    def meta(v):
        if isinstance(v,torch.Tensor): return dict(shape=list(v.shape),stride=list(v.stride()),dtype=str(v.dtype))
        if isinstance(v,(tuple,list)): return [meta(t) for t in v]
        if isinstance(v,dict): return {n:meta(t) for n,t in v.items()}
        return v
    return fn,args,kw,tensors,meta(dict(args=args,kwargs=kw))


def measure(torch, fn, args, kwargs, tensors, a):
    snapshots=[] if a.no_reset_inputs else [x.clone() for x in tensors]
    def reset():
        for dst,src in zip(tensors,snapshots): dst.copy_(src)
    samples=[]
    wall=[]
    # Initialize event objects outside measured calls.
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record(); end.record(); end.synchronize()
    torch.cuda.synchronize(a.device)
    for _ in range(a.warmup):
        reset(); torch.cuda.synchronize(a.device)
        output=fn(*args,**kwargs)
        torch.cuda.synchronize(a.device)
        del output
    torch.cuda.reset_peak_memory_stats(a.device)
    for _ in range(a.repeats):
        reset(); torch.cuda.synchronize(a.device)
        t=time.perf_counter()
        start.record(); output=fn(*args,**kwargs); end.record(); end.synchronize()
        wall.append((time.perf_counter()-t)*1000)
        samples.append(start.elapsed_time(end))
        del output
    mean=statistics.mean(samples)
    return dict(mean_ms=mean,median_ms=statistics.median(samples),std_ms=statistics.stdev(samples),
                min_ms=min(samples),max_ms=max(samples),samples_ms=samples,
                cv_percent=100*statistics.stdev(samples)/mean if mean else 0,
                wall_mean_ms=statistics.mean(wall),
                peak_allocated_mb=torch.cuda.max_memory_allocated(a.device)/2**20)


def sweep_points(a):
    if a.scan=='fuse':
        return [(a.ipc,k) for k in sorted(set([1]+a.fuses))]
    if a.ipcs is not None:
        values=sorted(set(a.ipcs))
    else:
        values=[]; value=a.ipc_start
        while value<=a.ipc_max:
            values.append(value); value*=2
    return [(ipc,a.fuse) for ipc in values]


CSV_FIELDS=[
    'scan','scan_value','ipc','fuse','batch','layer','interface','vfuse','status',
    'mean_ms','std_ms','cv_percent','median_ms','min_ms','max_ms',
    'baseline_ipc','baseline_fuse','baseline_mean_ms','work_ratio','time_growth',
    'throughput_speedup','parallel_efficiency_percent','extra_latency_ms',
    'wall_mean_ms','peak_allocated_mb','warmup','repeats','gpu','torch','cuda',
    'tf32','cudnn_benchmark','reset_inputs','inputs','samples_ms','error'
]


def csv_row(writer,row):
    writer.writerow({key:json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else value
                     for key,value in row.items() if key in CSV_FIELDS})


def run_sweep(torch,kernels,a,selected,output):
    # One process, one output file, no worker configs, JSON files, or log files.
    points=sweep_points(a)
    baseline={}
    failed=False; successes=0
    env=dict(gpu=torch.cuda.get_device_name(a.device),torch=torch.__version__,cuda=torch.version.cuda,
             tf32=a.tf32,cudnn_benchmark=a.cudnn_benchmark,reset_inputs=not a.no_reset_inputs,
             warmup=a.warmup,repeats=a.repeats)
    writer=csv.DictWriter(output,fieldnames=CSV_FIELDS)
    writer.writeheader(); output.flush()
    with torch.no_grad():
        for ipc,k in points:
            a.ipc=ipc
            print('[RUN] IPC=%d Fuse=%d cases=%d'%(ipc,k,len(selected)),flush=True)
            for spec in selected:
                row=dict(env,scan=a.scan,scan_value=k if a.scan=='fuse' else ipc,
                         ipc=ipc,fuse=k,batch=a.batch_size or ipc*a.classes,
                         layer=spec['layer'],interface=spec['interface'],vfuse=spec.get('vfuse',''))
                try:
                    if not callable(getattr(kernels,spec['interface'],None)):
                        raise AttributeError('Missing interface '+a.module+'.'+spec['interface'])
                    def run_one():
                        fn,args,kwargs,tensors,meta=make_inputs(torch,kernels,a,spec,k)
                        row['inputs']=meta
                        return measure(torch,fn,args,kwargs,tensors,a)
                    row.update(run_one()); row['status']='ok'; successes+=1
                    # Always compare against the FIRST requested point, never
                    # silently change the baseline after an error or OOM.
                    key=spec['case_id']
                    if (ipc,k)==points[0]: baseline[key]=dict(row)
                    base=baseline.get(key)
                    if base and base['mean_ms']>0 and row['mean_ms']>0:
                        ratio=(ipc*k)/(base['ipc']*base['fuse'])
                        growth=row['mean_ms']/base['mean_ms']
                        row.update(baseline_ipc=base['ipc'],baseline_fuse=base['fuse'],
                                   baseline_mean_ms=base['mean_ms'],work_ratio=ratio,time_growth=growth,
                                   throughput_speedup=ratio/growth,
                                   extra_latency_ms=row['mean_ms']-base['mean_ms'])
                        if a.scan=='fuse': row['parallel_efficiency_percent']=100/growth
                    print('[OK] IPC=%d Fuse=%d %s %.6f ms CV=%.2f%%'%
                          (ipc,k,spec['case_id'],row['mean_ms'],row['cv_percent']),flush=True)
                except torch.cuda.OutOfMemoryError:
                    row.update(status='oom',error=traceback.format_exc())
                    print('[OOM] IPC=%d Fuse=%d %s'%(ipc,k,spec['case_id']),flush=True)
                except Exception:
                    row.update(status='error',error=traceback.format_exc()); failed=True
                    print('[ERROR] IPC=%d Fuse=%d %s\n%s'%(ipc,k,spec['case_id'],row['error']),flush=True)
                finally:
                    # Flush each result immediately: an interrupted run leaves
                    # all completed rows in the SAME CSV.
                    csv_row(writer,row); output.flush()
                    gc.collect(); torch.cuda.empty_cache()
    return failed or successes==0


def main():
    a=parse_args()
    selected=cases(a)
    if a.list_cases:
        print(json.dumps(selected,indent=2,ensure_ascii=False)); return
    os.chdir(a.repo); sys.path.insert(0,a.repo)
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32=(a.tf32=='on')
    torch.backends.cudnn.allow_tf32=(a.tf32=='on')
    torch.backends.cudnn.benchmark=a.cudnn_benchmark
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    kernels=importlib.import_module(a.module)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation protects an existing run from accidental overwrite.
    with open(a.out,'x',newline='',encoding='utf-8-sig') as output:
        try:
            failed=run_sweep(torch,kernels,a,selected,output)
        finally:
            print('[OUTPUT] '+a.out,flush=True)
    if failed: raise SystemExit(1)


if __name__=='__main__': main()
