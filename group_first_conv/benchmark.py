"""Correctness-gated, group-first vs cuDNN benchmark. Run on the target GPU."""
import argparse,csv,json,time,statistics
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
from group_first_conv import conv2d_group_first,_conv


def packed_input(x,channels_last=False):
    g,b,c,h,w=x.shape
    v=x.permute(1,0,2,3,4).reshape(b,g*c,h,w)
    return v.contiguous(memory_format=torch.channels_last if channels_last else torch.contiguous_format)


def grouped(x,w,bias,stride,padding,dilation,channels_last=False):
    g,b,c,h,wi=x.shape
    o=w.shape[1]
    v=packed_input(x,channels_last)
    ww=w.reshape(g*o,c,*w.shape[-2:])
    if channels_last:
        ww=ww.contiguous(memory_format=torch.channels_last)
    y=F.conv2d(v,ww,None if bias is None else bias.flatten(),stride,padding,dilation,g)
    return y.reshape(b,g,o,*y.shape[-2:]).permute(1,0,2,3,4).contiguous()


def serial(x,w,bias,stride,padding,dilation):
    return torch.stack([F.conv2d(x[i],w[i],None if bias is None else bias[i],
                               stride,padding,dilation) for i in range(x.shape[0])])


def error(y,ref):
    d=y.double()-ref.double()
    return float(d.abs().max()),float(d.norm()/ref.double().norm().clamp_min(1e-30))


def wall_ms(fn,iters):
    values=[]
    for _ in range(3):
        torch.cuda.synchronize(); t=time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter()-t)*1000/iters)
    return statistics.median(values)


def verify(x,w,bias,stride,padding,dilation,precision):
    # FP64 cuDNN-independent reference also exposes TF32 approximation error.
    ref=serial(x.double(),w.double(),None if bias is None else bias.double(),stride,padding,dilation)
    y=conv2d_group_first(x,w,bias,stride,padding,dilation,precision)
    atol,rtol={torch.float16:(.02,.01),torch.bfloat16:(.15,.04),torch.float32:(.002,.0005)}[x.dtype]
    if x.dtype==torch.float32 and precision=='tf32': atol,rtol=.05,.01
    torch.testing.assert_close(y.double(),ref,atol=atol,rtol=rtol)
    for cl in (False,True):
        torch.testing.assert_close(grouped(x,w,bias,stride,padding,dilation,cl).double(),ref,atol=atol,rtol=rtol)
    return error(y,ref)


def edge_tests(dtype,precision):
    # Ragged C/O, multiple batches/groups, borders, non-square, stride/dilation, no bias.
    for g,b,c,o,h,wi,r,s,st,pad,dil,has_bias in [
        (3,2,5,7,9,11,3,2,(2,1),(2,1),(2,1),True),
        (2,3,3,17,5,7,1,1,1,0,1,False),
        (1,1,1,1,3,3,3,3,1,1,1,True),
    ]:
        x=torch.randn(g,b,c,h,wi,device='cuda',dtype=dtype)
        w=torch.randn(g,o,c,r,s,device='cuda',dtype=dtype)
        bias=torch.randn(g,o,device='cuda',dtype=dtype) if has_bias else None
        verify(x,w,bias,st,pad,dil,precision)
    print('Edge tests passed',flush=True)


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--groups',nargs='+',type=int,default=[1,2,4,8])
    ap.add_argument('--batch',nargs='+',type=int,default=[10])
    ap.add_argument('--dtype',choices=['fp32','fp16','bf16'],default='fp32')
    ap.add_argument('--precision',choices=['ieee','tf32','tf32x3'],default='ieee')
    ap.add_argument('--case',choices=['conv1','conv2','conv3','pointwise','all'],default='all')
    ap.add_argument('--shape',nargs=6,type=int,metavar=('C','O','H','W','R','S'))
    ap.add_argument('--stride',type=int,default=1)
    ap.add_argument('--padding',type=int,default=1)
    ap.add_argument('--dilation',type=int,default=1)
    ap.add_argument('--iters',type=int,default=100)
    ap.add_argument('--output',default='results.csv')
    ap.add_argument('--skip-edge-tests',action='store_true')
    ap.add_argument('--cuda-graph',action='store_true',help='Also measure graph replay, excluding Python launch gaps')
    args=ap.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA GPU is required')
    if min(args.groups+args.batch+[args.iters])<=0: ap.error('groups/batch/iters must be positive')
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic=False
    # tf32x3 has no matching cuDNN mode: compare to strict FP32 and report error.
    torch.backends.cudnn.allow_tf32=args.precision=='tf32'
    torch.backends.cuda.matmul.allow_tf32=args.precision=='tf32'
    dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[args.dtype]
    metadata=dict(args=vars(args),torch=torch.__version__,triton=triton.__version__,
                  gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
                  cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                  cudnn_tf32=torch.backends.cudnn.allow_tf32)
    print(json.dumps(metadata,indent=2),flush=True)
    Path(args.output+'.json').write_text(json.dumps(metadata,indent=2))
    if not args.skip_edge_tests: edge_tests(dtype,args.precision)
    cases={'conv1':(3,128,32,32,3,3),'conv2':(128,128,16,16,3,3),
           'conv3':(128,128,8,8,3,3),'pointwise':(128,128,16,16,1,1)}
    if args.shape: cases={'custom':tuple(args.shape)}
    elif args.case!='all': cases={args.case:cases[args.case]}
    with open(args.output,'w',newline='') as f:
        writer=None
        for name,(c,o,h,wi,r,s) in cases.items():
            pad=0 if name=='pointwise' else args.padding
            for g in args.groups:
                for b in args.batch:
                    print(f'Checking/tuning {name} G={g} B={b}',flush=True)
                    x=torch.randn(g,b,c,h,wi,device='cuda',dtype=dtype)
                    w=torch.randn(g,o,c,r,s,device='cuda',dtype=dtype)
                    bias=torch.randn(g,o,device='cuda',dtype=dtype)
                    convargs=(args.stride,pad,args.dilation)
                    maxabs,rel=verify(x,w,bias,*convargs,args.precision)
                    funcs={'triton':lambda:conv2d_group_first(x,w,bias,*convargs,args.precision),
                           'serial':lambda:serial(x,w,bias,*convargs),
                           'grouped_e2e':lambda:grouped(x,w,bias,*convargs),
                           'channels_last_e2e':lambda:grouped(x,w,bias,*convargs,True)}
                    packed=packed_input(x); packed_cl=packed_input(x,True)
                    ww=w.reshape(g*o,c,r,s); ww_cl=ww.contiguous(memory_format=torch.channels_last)
                    bb=bias.flatten()
                    funcs['grouped_prepacked']=lambda:F.conv2d(packed,ww,bb,*convargs,groups=g)
                    funcs['channels_last_prepacked']=lambda:F.conv2d(packed_cl,ww_cl,bb,*convargs,groups=g)
                    row=dict(case=name,G=g,B=b,C=c,O=o,H=h,W=wi,R=r,S=s,
                             dtype=args.dtype,precision=args.precision,max_abs=maxabs,rel_l2=rel,
                             triton_config=str(_conv.best_config))
                    for label,fn in funcs.items():
                        for _ in range(10): fn()
                        torch.cuda.synchronize()
                        row[label+'_event_ms']=triton.testing.do_bench(fn,warmup=100,rep=300,return_mode='median')
                        row[label+'_wall_ms']=wall_ms(fn,args.iters)
                        if args.cuda_graph:
                            row[label+'_graph_ms']=triton.testing.do_bench_cudagraph(fn,rep=300)
                    for label in funcs:
                        if label!='triton':
                            for mode in (('event','wall','graph') if args.cuda_graph else ('event','wall')):
                                row[label+'_'+mode+'_speedup']=row[label+'_'+mode+'_ms']/row['triton_'+mode+'_ms']
                    if writer is None: writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader()
                    writer.writerow(row); f.flush()
                    print(json.dumps(row),flush=True)
    print('Saved '+args.output,flush=True)


if __name__=='__main__': main()
