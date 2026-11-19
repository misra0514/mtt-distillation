"""Two modes: correctness against ATen, or shape-sweep performance."""
import argparse,csv,itertools,json,statistics,time
from pathlib import Path
import torch
import triton
from test_script.stateless_conv_double_bwd.conv_double_bwd import convolution_double_backward as custom,workspace_split


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


def compare(args,precision,split=None,budget=32):
    got=custom(*args,precision=precision,split_k=split,workspace_limit_mb=budget)
    ref=aten_group_first(args)
    # Explicit thresholds: report relative L2 as well as assert_close.
    dtype=args[5].dtype
    atol,rtol,relmax={torch.float32:(.003,.002,2e-4),torch.float16:(.08,.02,.003),
                     torch.bfloat16:(.6,.08,.02)}[dtype]
    if dtype==torch.float32 and precision=='tf32': atol,rtol,relmax=.1,.03,.004
    metrics={}
    for name,a,b in zip(('ggO','gI','gW'),got,ref):
        if a is None or b is None:
            assert a is None and b is None,(name,'None mismatch')
            continue
        torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
        diff=a.double()-b.double()
        rel=(diff.norm()/b.double().norm().clamp_min(1e-30)).item()
        assert rel<=relmax,(name,rel,relmax)
        metrics[name+'_max_abs']=diff.abs().max().item(); metrics[name+'_rel_l2']=rel
    return metrics


def correctness(dtype,precision):
    count=0
    cases=[((2,2,3,17,9,11,3,3),(1,1),(1,1),(1,1)),
           ((3,2,5,7,10,11,3,2),(2,1),(2,1),(2,1)),
           ((2,1,33,65,4,5,1,1),(1,1),(0,0),(1,1))]
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
    print(f'PASS: {count} GPU cases; compared directly with aten._convolution_double_backward',flush=True)


def timed(fn,iters):
    for _ in range(10): result=fn()
    del result
    torch.cuda.synchronize()
    wall=[]; event=[]
    for _ in range(3):
        start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); t=time.perf_counter(); start.record()
        for _ in range(iters): result=fn()
        end.record(); end.synchronize()
        wall.append((time.perf_counter()-t)*1000/iters); event.append(start.elapsed_time(end)/iters)
        del result
    # Baseline input tensors already allocated: incremental peak includes outputs/workspace.
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base=torch.cuda.memory_allocated(); result=fn(); torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated()-base; del result
    return statistics.mean(event),statistics.mean(wall),peak


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['correctness','bench'])
    ap.add_argument('--groups',nargs='+',type=int,default=[1,2,4,8,16,32])
    ap.add_argument('--batch',nargs='+',type=int,default=[10])
    ap.add_argument('--case',choices=['conv1','conv2','conv3','all'],default='all')
    ap.add_argument('--shape',nargs=6,type=int,help='C O H W R S; overrides --case')
    ap.add_argument('--stride',type=int,default=1); ap.add_argument('--padding',type=int,default=1)
    ap.add_argument('--dilation',type=int,default=1)
    ap.add_argument('--dtype',choices=['fp32','fp16','bf16'],default='fp32')
    ap.add_argument('--precision',choices=['ieee','tf32','tf32x3'],default='ieee')
    ap.add_argument('--iters',type=int,default=100)
    ap.add_argument('--workspace-mb',type=float,default=32)
    ap.add_argument('--split-k',type=int,choices=[1,2,4,8,16])
    ap.add_argument('--output',default='double_bwd_times.csv')
    a=ap.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA GPU required')
    if min(a.groups+a.batch+[a.iters])<1: ap.error('Positive groups/batch/iters required')
    torch.manual_seed(123)
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.allow_tf32=a.precision=='tf32'
    torch.backends.cuda.matmul.allow_tf32=a.precision=='tf32'
    dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[a.dtype]
    meta=dict(torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,
              cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(),
              capability=torch.cuda.get_device_capability(),args=vars(a),
              note='ATen 2.5.1 ignores output_mask. TF32x3 vs strict ATen is not identical arithmetic.')
    print(json.dumps(meta,indent=2),flush=True)
    if a.mode=='correctness': correctness(dtype,a.precision); return
    Path(a.output+'.json').write_text(json.dumps(meta,indent=2))
    shapes={'conv1':(3,128,32,32,3,3),'conv2':(128,128,16,16,3,3),'conv3':(128,128,8,8,3,3)}
    if a.shape: shapes={'custom':tuple(a.shape)}
    elif a.case!='all': shapes={a.case:shapes[a.case]}
    with open(a.output,'w',newline='') as file:
        writer=None
        for name,shape in shapes.items():
            c,o,h,w,r,s=shape
            for g in a.groups:
                for f in a.batch:
                    full=(g,f,c,o,h,w,r,s)
                    args=make_args(full,dtype,'cuda',(a.stride,)*2,(a.padding,)*2,(a.dilation,)*2)
                    # Every benchmark shape checked before recording any performance result.
                    print('Checking/tuning',name,full,flush=True)
                    errs=compare(args,a.precision,a.split_k,a.workspace_mb)
                    ready=prepare_aten(args)
                    fn=lambda:custom(*args,precision=a.precision,workspace_limit_mb=a.workspace_mb,split_k=a.split_k)
                    funcs={'triton':fn,'aten_prepacked':lambda:torch.ops.aten._convolution_double_backward(*ready),
                           'aten_group_first_e2e':lambda:aten_group_first(args)}
                    row=dict(case=name,G=g,F=f,C=c,O=o,H=h,W=w,R=r,S=s,**errs)
                    p,q=args[3].shape[-2:]
                    sk=a.split_k or workspace_split(g,f,c,o,r,s,p,q,torch.cuda.get_device_properties(0).multi_processor_count,int(a.workspace_mb*1024**2))
                    row['split_k']=sk; row['triton_scratch_bytes']=g*sk*o*c*r*s*4 if sk>1 else 0
                    for label,func in funcs.items():
                        ev,wall,peak=timed(func,a.iters)
                        row[label+'_event_ms']=ev; row[label+'_wall_ms']=wall; row[label+'_peak_extra_bytes']=peak
                    for label in ('aten_prepacked','aten_group_first_e2e'):
                        row[label+'_event_speedup']=row[label+'_event_ms']/row['triton_event_ms']
                        row[label+'_wall_speedup']=row[label+'_wall_ms']/row['triton_wall_ms']
                    if writer is None: writer=csv.DictWriter(file,fieldnames=list(row)); writer.writeheader()
                    writer.writerow(row); file.flush(); print(json.dumps(row),flush=True)
                    del args,ready,funcs,fn,func
    print('Saved',a.output,flush=True)

if __name__=='__main__': main()
