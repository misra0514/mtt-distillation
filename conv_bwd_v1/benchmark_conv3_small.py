"""Fixed direct-kernel Conv3 small-batch benchmark, with stage timings."""
import argparse
import contextlib
import hashlib
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch
import triton
import conv_bwd as kernel
from benchmark import Tee, aten, compare, original, pack, paired, tensors


def run_case(dataset,layer,g,b,a):
    h=(32 if dataset=="cifar10" else 64)//(2**(layer-1))
    c=3 if layer==1 else 128
    o=128
    x,w,d=tensors((g,b,c,h,h,o,3,3),(1,1),(1,1),(1,1))
    xp,wp,dp=pack(x),w.flatten(0,1),pack(d)
    dims=(b,c,o,h,h,h,h,3,3,1,1,1,1,1,1,1)
    parts=kernel._parts(g,b,c,o,3,3,h,h,torch.cuda.get_device_properties(x.device).multi_processor_count)
    dx,dw,db=torch.empty_like(x),torch.empty_like(w),x.new_empty(g,o)
    wp_tmp=dw if parts==1 else x.new_empty(g,parts,o,c,3,3)
    bp_tmp=db if parts==1 else x.new_empty(g,parts,o)
    bm=16 if c<32 else 32
    tiles=triton.cdiv(c*9,32) if c<16 else triton.cdiv(c,32)*9

    def dx_kernel():
        kernel._dx[(triton.cdiv(c,bm)*triton.cdiv(b*h*h,32),g,1)](
            d,w,dx,*dims,False,bm,32,16,num_warps=4)
        return dx

    def dw_db_core():
        kernel._dw_db[(triton.cdiv(o,32)*tiles,g,parts)](
            x,d,wp_tmp,bp_tmp,*dims,parts,32,32,32,num_warps=4)
        return wp_tmp,bp_tmp

    def finish():
        nw=o*c*9
        kernel._finish[(triton.cdiv(nw,64)+triton.cdiv(o,64),g)](
            wp_tmp,bp_tmp,dw,db,nw,o,parts,64)
        return dw,db

    def dw_db_stage():
        dw_db_core()
        if parts>1:
            finish()
        return dw,db

    def direct_api():
        return kernel._backward(x,w,d,1,1,1,1,"direct")

    def aten_subset(mask):
        return torch.ops.aten.convolution_backward.default(
            dp,xp,wp,[g*o],[1,1],[1,1],[1,1],False,[0,0],g,mask)

    def torch_given_dw_db():
        return (torch.nn.grad.conv2d_weight(xp,wp.shape,dp,1,1,1,g),dp.sum((0,2,3)))

    dx_kernel()
    dw_db_stage()
    api=direct_api()
    ref=original(xp,wp,dp,1,1,1,g)
    expected=(ref[0].unflatten(1,(g,c)).transpose(0,1),ref[1].view_as(w),ref[2].view_as(db))
    errors={}
    for name,aa,bb,target in zip(("grad_x","grad_weight","grad_bias"),(dx,dw,db),api,expected):
        try:
            errors[name]=dict(stage_rel_l2=compare(aa,target),api_rel_l2=compare(bb,target))
        except AssertionError as error:
            raise AssertionError(f"{dataset} conv{layer} G={g} B={b} direct {name}: {error}") from error
    del api,ref,expected

    common=dict(dataset=dataset,layer=f"conv{layer}",G=g,B=b,x_shape=list(x.shape),
                weight_shape=list(w.shape),grad_output_shape=list(d.shape),split_k=parts)
    print("CASE",json.dumps(common),flush=True)
    print("CORRECTNESS",json.dumps(errors),flush=True)
    # Full APIs include output allocation. Stage kernels write to persistent
    # output buffers so device time can expose the arithmetic bottleneck.
    groups={
        "full_api":{
            "triton_direct":direct_api,
            "torch_given":lambda:original(xp,wp,dp,1,1,1,g),
            "aten_grouped":lambda:aten(xp,wp,dp,1,1,1,g),
            "aten_serial":lambda:[aten(x[j],w[j],d[j],1,1,1,1) for j in range(g)]},
        "grad_x":{
            "triton_dx":dx_kernel,
            "torch_given_dx":lambda:torch.nn.grad.conv2d_input(xp.shape,wp,dp,1,1,1,g),
            "aten_dx":lambda:aten_subset([True,False,False])},
        "grad_weight_bias":{
            "triton_dw_db":dw_db_stage,
            "torch_given_dw_db":torch_given_dw_db,
            "aten_dw_db":lambda:aten_subset([False,True,True])},
        "triton_internal":{"dw_db_core":dw_db_core},
    }
    if parts>1:
        groups["triton_internal"]["split_reduce"]=finish
    for stage,funcs in groups.items():
        results=paired(funcs,a)
        for provider,(event,wall,cv,wcv,peak) in results.items():
            print("STAGE_RESULT",json.dumps(dict(common,stage=stage,provider=provider,
                event_ms=event,wall_ms=wall,event_cv_pct=cv,wall_cv_pct=wcv,peak_extra_MiB=peak)),
                flush=True)
    print("CASE_FINISHED",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets",nargs="+",choices=["cifar10","tinyimagenet"],default=["cifar10"])
    p.add_argument("--layers",nargs="+",type=int,choices=[1,2,3],default=[1,2,3])
    p.add_argument("--groups",nargs="+",type=int,default=[1,2,4,8,16,32])
    p.add_argument("--batch",nargs="+",type=int,choices=[1,4],default=[1])
    p.add_argument("--warmup",type=int,default=10)
    p.add_argument("--rounds",type=int,default=5)
    p.add_argument("--iters",type=int,default=100)
    p.add_argument("--target-ms",type=float,default=50.)
    p.add_argument("--output",type=Path,default=Path("conv3_small_stages.txt"))
    a=p.parse_args()
    if min(a.groups)<1 or a.warmup<1 or a.rounds<2 or a.iters<2 or a.target_ms<=0:
        p.error("expected positive sizes and warmup, rounds/iters >=2")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",encoding="utf-8",buffering=1) as f:
        with contextlib.redirect_stdout(Tee(sys.stdout,f)),contextlib.redirect_stderr(Tee(sys.stderr,f)):
            try:
                print("START",datetime.now().astimezone().isoformat(),flush=True)
                print("ARGS",json.dumps(vars(a),default=str),flush=True)
                print("SOURCE_SHA256",json.dumps({n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                      for n in ("benchmark_conv3_small.py","conv_bwd.py","benchmark.py")}),flush=True)
                torch.manual_seed(17)
                torch.backends.cuda.matmul.allow_tf32=False
                torch.backends.cudnn.allow_tf32=False
                torch.backends.cudnn.benchmark=True
                print("ENV",torch.__version__,triton.__version__,torch.version.cuda,
                      torch.backends.cudnn.version(),torch.cuda.get_device_name(),"FP32 IEEE",flush=True)
                print("ALGORITHM direct; full_api includes allocations; Triton stages reuse buffers; "
                      "stage medians are isolated and need not sum to full_api.",flush=True)
                with torch.no_grad():
                    for dataset in a.datasets:
                        for layer in a.layers:
                            for g in a.groups:
                                for b in a.batch:
                                    print("CASE_BEGIN",dataset,f"conv{layer}",f"G={g}",f"B={b}",flush=True)
                                    run_case(dataset,layer,g,b,a)
                print("FINISHED",datetime.now().astimezone().isoformat(),flush=True)
            except Exception:
                traceback.print_exc()
                raise SystemExit(1)


if __name__=="__main__":
    main()
