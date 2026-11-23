# 11.19 test 完之后固定下来的正式版本。
# 目前只包含d-bwd： convolution_double_backward
# 调用的时候：from conv_double_bwd import convolution_double_backward

"""Stateless convolution double backward v32.1; target: NVIDIA sm80+.
No autograd, ATen convolution calls, cached tensors or global scratch buffers.
"""
from dataclasses import dataclass
from functools import lru_cache,partial
import torch
import triton
import triton.language as tl

_SHAPE_KEY=['G','F','C','H','W','O','R','S','P','Q','SH','SW','PH','PW','DH','DW','PREC']

def _configs():
    return [triton.Config(dict(BM=m,BN=n,BK=k),num_warps=warps,num_stages=stages)
            for m,n,k,warps,stages in [(16,64,32,4,2),(32,64,32,4,3),
                (32,128,32,4,3),(64,64,32,4,3),(32,64,64,4,3)]]

def _spatial_configs():
    # Keep every v1 candidate. Add fixed-(r,s) iterators with spatial-contiguous loads.
    return [triton.Config(dict(**cfg.kwargs, RS_LOOP=rs, BATCH_FOLD=fold),
                          num_warps=cfg.num_warps,num_stages=cfg.num_stages)
            for fold in (False,True) for rs in (False,True) for cfg in (_configs()+[
                triton.Config(dict(BM=16,BN=32,BK=32),num_warps=2,num_stages=2),
                triton.Config(dict(BM=32,BN=32,BK=32),num_warps=4,num_stages=2),
                triton.Config(dict(BM=32,BN=64,BK=16),num_warps=4,num_stages=2)])]

def _prune_spatial(configs,named_args,**kwargs):
    a=dict(named_args,**kwargs)
    use_rs=1<a['R']*a['S']<=9 and min(a['C'],a['O'])>=16
    use_fold=a['F']>1 and min(a['H']*a['W'],a['P']*a['Q'])<=64
    return [cfg for cfg in configs if (a['PREC']=='ieee' or (cfg.kwargs['BN']!=32 and cfg.kwargs['BK']!=16)) and (use_rs or not cfg.kwargs['RS_LOOP']) and (use_fold or not cfg.kwargs['BATCH_FOLD'])]

@triton.autotune(configs=_spatial_configs(),key=_SHAPE_KEY+['HI','HW','HB'],prune_configs_by={'early_config_prune':_prune_spatial})
@triton.jit
def _ggo(X,Wt,AI,AW,AB,Y,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
         H:tl.constexpr,W:tl.constexpr,O:tl.constexpr,R:tl.constexpr,S:tl.constexpr,
         P:tl.constexpr,Q:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
         PH:tl.constexpr,PW:tl.constexpr,DH:tl.constexpr,DW:tl.constexpr,
         PREC:tl.constexpr,HI:tl.constexpr,HW:tl.constexpr,HB:tl.constexpr,
         BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,RS_LOOP:tl.constexpr,BATCH_FOLD:tl.constexpr):
    pid=tl.program_id(0); gf=tl.program_id(1)
    g=gf if BATCH_FOLD else gf//F
    nn=tl.cdiv(F*P*Q if BATCH_FOLD else P*Q,BN); nm=tl.cdiv(O,BM)
    first=pid//(4*nn)*4; count=tl.minimum(nm-first,4)
    mi=first+(pid%(4*nn))%count; ni=(pid%(4*nn))//count
    oc=mi*BM+tl.arange(0,BM); n=ni*BN+tl.arange(0,BN)
    spatial=n%(P*Q) if BATCH_FOLD else n
    batch=n//(P*Q) if BATCH_FOLD else tl.full((BN,),gf%F,tl.int32)
    nlimit:tl.constexpr=F*P*Q if BATCH_FOLD else P*Q
    kk=tl.arange(0,BK)
    acc1=tl.zeros((BM,BN),tl.float32); acc2=tl.zeros((BM,BN),tl.float32)
    if HI or HW:
        if RS_LOOP:
            # Keep filter loops rolled to bound compile time/register live ranges.
            # Address arithmetic for spatial coordinates is invariant across channel blocks.
            for rr in range(R):
                for ss in range(S):
                    ih=spatial//Q*SH-PH+rr*DH; iw=spatial%Q*SW-PW+ss*DW
                    valid=(n<nlimit)&(ih>=0)&(ih<H)&(iw>=0)&(iw<W)
                    for cb in range(tl.cdiv(C,BK)):
                        ci=cb*BK+kk
                        xo=(g*F+batch[None,:])*C*H*W+ci[:,None]*H*W+ih[None,:]*W+iw[None,:]
                        wo=g*O*C*R*S+oc[:,None]*C*R*S+ci[None,:]*R*S+rr*S+ss
                        xm=(ci[:,None]<C)&valid[None,:]
                        wm=(oc[:,None]<O)&(ci[None,:]<C)
                        if HI:
                            a=tl.load(Wt+wo,wm,0); b=tl.load(AI+xo,xm,0)
                            acc1=tl.dot(a,b,acc1,input_precision=PREC)
                        if HW:
                            a=tl.load(AW+wo,wm,0); b=tl.load(X+xo,xm,0)
                            acc2=tl.dot(a,b,acc2,input_precision=PREC)
        else:
            for kb in range(tl.cdiv(C*R*S,BK)):
                k=kb*BK+kk
                ih=spatial[None,:]//Q*SH-PH+(k[:,None]//S%R)*DH
                iw=spatial[None,:]%Q*SW-PW+(k[:,None]%S)*DW
                xo=(g*F+batch[None,:])*C*H*W+(k[:,None]//(R*S))*H*W+ih*W+iw
                wo=g*O*C*R*S+oc[:,None]*C*R*S+k[None,:]
                xm=(k[:,None]<C*R*S)&(n[None,:]<nlimit)&(ih>=0)&(ih<H)&(iw>=0)&(iw<W)
                wm=(oc[:,None]<O)&(k[None,:]<C*R*S)
                if HI:
                    a=tl.load(Wt+wo,wm,0); b=tl.load(AI+xo,xm,0)
                    acc1=tl.dot(a,b,acc1,input_precision=PREC)
                if HW:
                    a=tl.load(AW+wo,wm,0); b=tl.load(X+xo,xm,0)
                    acc2=tl.dot(a,b,acc2,input_precision=PREC)
    # Match the intermediate dtype rounding of the two ATen conv outputs.
    if HI and HW:
        val=(acc1.to(Y.dtype.element_ty).to(tl.float32)+acc2.to(Y.dtype.element_ty).to(tl.float32)).to(Y.dtype.element_ty).to(tl.float32)
    elif HI:
        val=acc1.to(Y.dtype.element_ty).to(tl.float32)
    else:
        val=acc2.to(Y.dtype.element_ty).to(tl.float32)
    if HB:
        val+=tl.load(AB+g*O+oc,oc<O,0)[:,None]
    tl.store(Y+((g*F+batch[None,:])*O+oc[:,None])*P*Q+spatial[None,:],val,(oc[:,None]<O)&(n[None,:]<nlimit))

@triton.autotune(configs=_spatial_configs(),key=_SHAPE_KEY,prune_configs_by={'early_config_prune':_prune_spatial})
@triton.jit
def _gi(D,AW,Y,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
        H:tl.constexpr,W:tl.constexpr,O:tl.constexpr,R:tl.constexpr,S:tl.constexpr,
        P:tl.constexpr,Q:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
        PH:tl.constexpr,PW:tl.constexpr,DH:tl.constexpr,DW:tl.constexpr,
        PREC:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,RS_LOOP:tl.constexpr,BATCH_FOLD:tl.constexpr):
    pid=tl.program_id(0); gf=tl.program_id(1)
    g=gf if BATCH_FOLD else gf//F
    nn=tl.cdiv(F*H*W if BATCH_FOLD else H*W,BN)
    ci=pid//nn*BM+tl.arange(0,BM); n=pid%nn*BN+tl.arange(0,BN)
    spatial=n%(H*W) if BATCH_FOLD else n
    batch=n//(H*W) if BATCH_FOLD else tl.full((BN,),gf%F,tl.int32)
    nlimit:tl.constexpr=F*H*W if BATCH_FOLD else H*W
    kk=tl.arange(0,BK); acc=tl.zeros((BM,BN),tl.float32)
    # Gather dgrad: no expanded dY and no col2im/atomic scatter.
    if RS_LOOP:
        for rr in range(R):
            for ss in range(S):
                hp=spatial//W+PH-rr*DH; wp=spatial%W+PW-ss*DW
                py=hp//SH; qx=wp//SW
                valid=(n<nlimit)&(hp%SH==0)&(wp%SW==0)&(py>=0)&(py<P)&(qx>=0)&(qx<Q)
                for cb in range(tl.cdiv(O,BK)):
                    oc=cb*BK+kk
                    a=tl.load(AW+g*O*C*R*S+oc[None,:]*C*R*S+ci[:,None]*R*S+rr*S+ss,
                              (ci[:,None]<C)&(oc[None,:]<O),0)
                    b=tl.load(D+(g*F+batch[None,:])*O*P*Q+oc[:,None]*P*Q+py[None,:]*Q+qx[None,:],
                              (oc[:,None]<O)&valid[None,:],0)
                    acc=tl.dot(a,b,acc,input_precision=PREC)
    else:
        for kb in range(tl.cdiv(O*R*S,BK)):
            k=kb*BK+kk; oc=k//(R*S); r=k//S%R; s=k%S
            hp=spatial[None,:]//W+PH-r[:,None]*DH
            wp=spatial[None,:]%W+PW-s[:,None]*DW
            py=hp//SH; qx=wp//SW
            a=tl.load(AW+g*O*C*R*S+oc[None,:]*C*R*S+ci[:,None]*R*S+r[None,:]*S+s[None,:],
                      (ci[:,None]<C)&(oc[None,:]<O),0)
            b=tl.load(D+(g*F+batch[None,:])*O*P*Q+oc[:,None]*P*Q+py*Q+qx,
                      (oc[:,None]<O)&(n[None,:]<nlimit)&(hp%SH==0)&(wp%SW==0)&
                      (py>=0)&(py<P)&(qx>=0)&(qx<Q),0)
            acc=tl.dot(a,b,acc,input_precision=PREC)
    tl.store(Y+((g*F+batch[None,:])*C+ci[:,None])*H*W+spatial[None,:],acc,(ci[:,None]<C)&(n[None,:]<nlimit))

def _wgrad_configs():
    flat=[triton.Config(dict(**c.kwargs,FIXED_RS=False),num_warps=c.num_warps,num_stages=c.num_stages) for c in _configs()]
    fixed=[triton.Config(dict(BM=m,BN=n,BK=k,FIXED_RS=True),num_warps=w,num_stages=st)
           for m,n,k,w,st in [(32,32,32,4,2),(32,64,32,4,3),(64,32,32,4,3),(32,32,64,4,3),(16,32,64,4,2)]]
    return flat+fixed

@triton.autotune(configs=_wgrad_configs(),key=_SHAPE_KEY+['SPLIT'])
@triton.jit
def _gw(AI,D,Y,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
        H:tl.constexpr,W:tl.constexpr,O:tl.constexpr,R:tl.constexpr,S:tl.constexpr,
        P:tl.constexpr,Q:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
        PH:tl.constexpr,PW:tl.constexpr,DH:tl.constexpr,DW:tl.constexpr,
        PREC:tl.constexpr,SPLIT:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,FIXED_RS:tl.constexpr):
    pid=tl.program_id(0); gs=tl.program_id(1); g=gs//SPLIT; part=gs%SPLIT
    if FIXED_RS:
        # One filter position per CTA. Both operands load contiguous spatial K.
        # Compute [C,O] and transpose on store, avoiding the strided CRS gather.
        nn=tl.cdiv(O,BN); tiles=tl.cdiv(C,BM)*nn
        rs=pid//tiles; tile=pid%tiles
        ci=tile//nn*BM+tl.arange(0,BM)
        oc=tile%nn*BN+tl.arange(0,BN)
        rr=rs//S; ss=rs%S
    else:
        nn=tl.cdiv(C*R*S,BN)
        oc=pid//nn*BM+tl.arange(0,BM); crs=pid%nn*BN+tl.arange(0,BN)
        ci=crs//(R*S); rr=crs//S%R; ss=crs%S
    kk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32); correction=tl.zeros((BM,BN),tl.float32)
    # Long reductions use 512-term partial sums plus compensated outer summation.
    # No extra global workspace, and the original validation tolerances stay fixed.
    CHUNKED:tl.constexpr=F*P*Q>=SPLIT*8192
    INNER:tl.constexpr=512//BK if CHUNKED else 1
    for chunk in range(tl.cdiv(F*P*Q,SPLIT*BK*INNER)):
        partial=tl.zeros((BM,BN),tl.float32)
        for block in range(INNER):
            k=((chunk*INNER+block)*SPLIT+part)*BK+kk
            f=k//(P*Q); py=k//Q%P; qx=k%Q
            if FIXED_RS:
                ih=py*SH-PH+rr*DH; iw=qx*SW-PW+ss*DW
                a=tl.load(AI+(g*F+f[None,:])*C*H*W+ci[:,None]*H*W+ih[None,:]*W+iw[None,:],
                          (ci[:,None]<C)&(k[None,:]<F*P*Q)&(ih[None,:]>=0)&(ih[None,:]<H)&(iw[None,:]>=0)&(iw[None,:]<W),0)
                b=tl.load(D+(g*F+f[:,None])*O*P*Q+oc[None,:]*P*Q+py[:,None]*Q+qx[:,None],
                          (oc[None,:]<O)&(k[:,None]<F*P*Q),0)
            else:
                ih=py[:,None]*SH-PH+rr[None,:]*DH
                iw=qx[:,None]*SW-PW+ss[None,:]*DW
                a=tl.load(D+g*F*O*P*Q+f[None,:]*O*P*Q+oc[:,None]*P*Q+py[None,:]*Q+qx[None,:],
                          (oc[:,None]<O)&(k[None,:]<F*P*Q),0)
                b=tl.load(AI+g*F*C*H*W+f[:,None]*C*H*W+ci[None,:]*H*W+ih*W+iw,
                          (k[:,None]<F*P*Q)&(crs[None,:]<C*R*S)&(ih>=0)&(ih<H)&(iw>=0)&(iw<W),0)
            if CHUNKED:
                partial=tl.dot(a,b,partial,input_precision=PREC)
            else:
                acc=tl.dot(a,b,acc,input_precision=PREC)
        if CHUNKED:
            delta=partial-correction
            updated=acc+delta
            correction=(updated-acc)-delta
            acc=updated
    if FIXED_RS:
        offsets=gs*O*C*R*S+oc[None,:]*C*R*S+ci[:,None]*R*S+rs
        tl.store(Y+offsets,acc,(ci[:,None]<C)&(oc[None,:]<O))
    else:
        tl.store(Y+gs*O*C*R*S+oc[:,None]*C*R*S+crs[None,:],acc,
                 (oc[:,None]<O)&(crs[None,:]<C*R*S))

@triton.jit
def _reduce(T,Y,N:tl.constexpr,PER:tl.constexpr,SPLIT:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    g=i//PER; j=i%PER
    acc=tl.full((BLOCK,),0,tl.float32)
    for s in tl.static_range(SPLIT):
        acc+=tl.load(T+(g*SPLIT+s)*PER+j,i<N,0)
    tl.store(Y+i,acc,i<N)


# v28 RGB 7x7 stride-2 ResNet stem.  The generic iterator pays the full
# coordinate/mask cost for every 7x7 tap.  These candidates instead use a
# bounded im2col+GEMM for ggO, parity-reduced GEMMs for gI, and a cropped
# valid-lattice reduction for every gW filter position.
def _stem7_mm_configs28():
    return [triton.Config(dict(BM=m,BN=n,BK=k),num_warps=w,num_stages=st)
            for m,n,k,w,st in [(16,64,32,4,2),(16,128,32,4,3),
                               (32,64,32,4,3),(32,128,32,8,3),
                               (16,64,64,4,3)]]


@triton.jit
def _stem7_ggo_pack28(X,AI,V0,V1,START,G:tl.constexpr,F:tl.constexpr,
        C:tl.constexpr,H:tl.constexpr,W:tl.constexpr,P:tl.constexpr,
        Q:tl.constexpr,NT:tl.constexpr,HI:tl.constexpr,HW:tl.constexpr,
        BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);g=tl.program_id(1)
    k=i//NT;n=i%NT;t=START+n
    batch=t//(P*Q);pos=t%(P*Q)
    ci=k//49;rs=k%49;rr=rs//7;ss=rs%7
    iy=pos//Q*2-3+rr;ix=pos%Q*2-3+ss
    off=((g*F+batch)*C+ci)*H*W+iy*W+ix
    valid=(k<C*49)&(t<F*P*Q)&(iy>=0)&(iy<H)&(ix>=0)&(ix<W)
    dst=(g*C*49+k)*NT+n
    if HI and HW:
        if tl.program_id(2)==0:
            tl.store(V0+dst,tl.load(AI+off,valid,0),k<C*49)
        else:
            tl.store(V1+dst,tl.load(X+off,valid,0),k<C*49)
    elif HI:
        tl.store(V0+dst,tl.load(AI+off,valid,0),k<C*49)
    else:
        tl.store(V1+dst,tl.load(X+off,valid,0),k<C*49)


@triton.autotune(configs=_stem7_mm_configs28(),
                 key=['G','F','C','O','P','Q','NT','HI','HW','HB','PREC'])
@triton.jit
def _stem7_ggo_mm28(Wt,AW,V0,V1,Bias,Y,START,G:tl.constexpr,F:tl.constexpr,
        C:tl.constexpr,O:tl.constexpr,P:tl.constexpr,Q:tl.constexpr,
        NT:tl.constexpr,HI:tl.constexpr,HW:tl.constexpr,HB:tl.constexpr,
        PREC:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0);g=tl.program_id(1);nn:tl.constexpr=tl.cdiv(NT,BN)
    oc=pid//nn*BM+tl.arange(0,BM);n=pid%nn*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    acc_i=tl.zeros((BM,BN),tl.float32);acc_w=tl.zeros((BM,BN),tl.float32)
    for kb in range(tl.cdiv(C*49,BK)):
        k=kb*BK+kk
        wm=(oc[:,None]<O)&(k[None,:]<C*49)
        vm=(k[:,None]<C*49)&(n[None,:]<NT)
        if HI:
            a=tl.load(Wt+(g*O+oc[:,None])*C*49+k[None,:],wm,0)
            b=tl.load(V0+(g*C*49+k[:,None])*NT+n[None,:],vm,0)
            acc_i=tl.dot(a,b,acc_i,input_precision=PREC)
        if HW:
            a=tl.load(AW+(g*O+oc[:,None])*C*49+k[None,:],wm,0)
            b=tl.load(V1+(g*C*49+k[:,None])*NT+n[None,:],vm,0)
            acc_w=tl.dot(a,b,acc_w,input_precision=PREC)
    if HI and HW:
        val=(acc_i.to(Y.dtype.element_ty).to(tl.float32)+
             acc_w.to(Y.dtype.element_ty).to(tl.float32)).to(Y.dtype.element_ty)
    elif HI:
        val=acc_i.to(Y.dtype.element_ty)
    else:
        val=acc_w.to(Y.dtype.element_ty)
    if HB:val+=tl.load(Bias+g*O+oc,oc<O,0)[:,None]
    t=START+n;batch=t//(P*Q);pos=t%(P*Q)
    tl.store(Y+((g*F+batch[None,:])*O+oc[:,None])*P*Q+pos[None,:],val,
             (oc[:,None]<O)&(n[None,:]<NT)&(t[None,:]<F*P*Q))


def _stem7_ggo28(x,wt,ai,aw,bias,out,plan,precision):
    g,f,c,h,w=x.shape;_,_,o,p,q=out.shape
    hi=ai is not None;hw=aw is not None;hb=bias is not None
    nt=plan['stem7_ggo_nt28'];group_batch=plan['stem7_ggo_groups28']
    for base in range(0,g,group_batch):
        end=min(g,base+group_batch);gb=end-base
        v0=torch.empty((gb,c*49,nt),device=x.device,dtype=torch.float32)
        v1=torch.empty_like(v0) if hi and hw else v0
        xx=x[base:end];aa=ai[base:end] if hi else xx
        ww=wt[base:end];aw1=aw[base:end] if hw else ww
        bb=bias[base:end] if hb else xx;yy=out[base:end]
        for start in range(0,f*p*q,nt):
            active_nt=min(nt,triton.cdiv(f*p*q-start,32)*32)
            _stem7_ggo_pack28[(triton.cdiv(c*49*active_nt,256),gb,
                                  2 if hi and hw else 1)](
                xx,aa,v0,v1,start,gb,f,c,h,w,p,q,active_nt,hi,hw,256)
            _stem7_ggo_mm28[lambda m:(triton.cdiv(o,m['BM'])*
                triton.cdiv(active_nt,m['BN']),gb)](
                ww,aw1,v0,v1,bb,yy,start,gb,f,c,o,p,q,active_nt,
                hi,hw,hb,precision)
        del v0,v1


@triton.jit
def _stem7_gi_pack_d28(D,V,F:tl.constexpr,O:tl.constexpr,H:tl.constexpr,
        W:tl.constexpr,P:tl.constexpr,Q:tl.constexpr,RH:tl.constexpr,
        RW:tl.constexpr,NT:tl.constexpr,START,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);g=tl.program_id(1)
    HH:tl.constexpr=(H+1-RH)//2;WW:tl.constexpr=(W+1-RW)//2
    R0:tl.constexpr=(RH+3)%2;S0:tl.constexpr=(RW+3)%2
    NR:tl.constexpr=(7-R0+1)//2;NS:tl.constexpr=(7-S0+1)//2
    K:tl.constexpr=O*NR*NS
    k=i//NT;n=START+i%NT;oc=k%O;rs=k//O
    rr=R0+2*(rs//NS);ss=S0+2*(rs%NS)
    b=n//(HH*WW);iy=(n//WW%HH)*2+RH;ix=(n%WW)*2+RW
    py=(iy+3-rr)//2;qx=(ix+3-ss)//2
    valid=(i<K*NT)&(b<F)&(py>=0)&(py<P)&(qx>=0)&(qx<Q)
    val=tl.load(D+((g*F+b)*O+oc)*P*Q+py*Q+qx,valid,0)
    tl.store(V+g*K*NT+i,val,i<K*NT)


@triton.jit
def _stem7_gi_pack_w28(AW,A,C:tl.constexpr,O:tl.constexpr,
        RH:tl.constexpr,RW:tl.constexpr,BC:tl.constexpr,BO:tl.constexpr):
    R0:tl.constexpr=(RH+3)%2;S0:tl.constexpr=(RW+3)%2
    NR:tl.constexpr=(7-R0+1)//2;NS:tl.constexpr=(7-S0+1)//2
    K:tl.constexpr=O*NR*NS
    g=tl.program_id(1);rs=tl.program_id(2)
    ci=tl.program_id(0)//tl.cdiv(O,BO)*BC+tl.arange(0,BC)
    oc=tl.program_id(0)%tl.cdiv(O,BO)*BO+tl.arange(0,BO)
    rr=R0+2*(rs//NS);ss=S0+2*(rs%NS)
    val=tl.load(AW+((g*O+oc[None,:])*C+ci[:,None])*49+rr*7+ss,
                (ci[:,None]<C)&(oc[None,:]<O),0)
    tl.store(A+(g*C+ci[:,None])*K+rs*O+oc[None,:],val,
             (ci[:,None]<C)&(oc[None,:]<O))


@triton.autotune(configs=_stem7_mm_configs28(),
                 key=['G','F','C','O','H','W','RH','RW','NT','PREC'])
@triton.jit
def _stem7_gi_mm28(A,V,Y,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
        O:tl.constexpr,H:tl.constexpr,W:tl.constexpr,RH:tl.constexpr,
        RW:tl.constexpr,NT:tl.constexpr,START,PREC:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    HH:tl.constexpr=(H+1-RH)//2;WW:tl.constexpr=(W+1-RW)//2
    R0:tl.constexpr=(RH+3)%2;S0:tl.constexpr=(RW+3)%2
    K:tl.constexpr=O*((7-R0+1)//2)*((7-S0+1)//2)
    g=tl.program_id(1);pid=tl.program_id(0)
    ci=pid//tl.cdiv(NT,BN)*BM+tl.arange(0,BM)
    ni=pid%tl.cdiv(NT,BN)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for base in range(tl.cdiv(K,BK)):
        k=base*BK+kk
        a=tl.load(A+(g*C+ci[:,None])*K+k[None,:],
                  (ci[:,None]<C)&(k[None,:]<K),0)
        v=tl.load(V+(g*K+k[:,None])*NT+ni[None,:],
                  (k[:,None]<K)&(ni[None,:]<NT),0)
        acc=tl.dot(a,v,acc,input_precision=PREC)
    n=START+ni;b=n//(HH*WW);iy=(n//WW%HH)*2+RH;ix=(n%WW)*2+RW
    tl.store(Y+((g*F+b[None,:])*C+ci[:,None])*H*W+iy[None,:]*W+ix[None,:],acc,
             (ci[:,None]<C)&(ni[None,:]<NT)&(b[None,:]<F))


def _stem7_gi28(d,aw,out,plan,dims):
    g,f,c,h,w,o,r,s,p,q,sh,sw,ph,pw,dh,dw,prec=dims
    group_batch=plan['stem7_gi_groups28'];nt=plan['stem7_gi_nt28']
    for start_group in range(0,g,group_batch):
        end=min(g,start_group+group_batch);gb=end-start_group
        for rh in range(2):
            for rw in range(2):
                nr=(7-(rh+3)%2+1)//2;ns=(7-(rw+3)%2+1)//2;k=o*nr*ns
                total=f*((h+1-rh)//2)*((w+1-rw)//2)
                a=torch.empty((gb,c,k),device=d.device,dtype=torch.float32)
                v=torch.empty((gb,k,nt),device=d.device,dtype=torch.float32)
                _stem7_gi_pack_w28[(triton.cdiv(c,16)*triton.cdiv(o,32),gb,nr*ns)](
                    aw[start_group:end],a,c,o,rh,rw,16,32)
                for pos in range(0,total,nt):
                    active_nt=min(nt,triton.cdiv(total-pos,32)*32)
                    _stem7_gi_pack_d28[(triton.cdiv(k*active_nt,256),gb)](
                        d[start_group:end],v,f,o,h,w,p,q,rh,rw,active_nt,pos,256)
                    _stem7_gi_mm28[lambda m:(triton.cdiv(c,m['BM'])*
                        triton.cdiv(active_nt,m['BN']),gb)](
                        a,v,out[start_group:end],gb,f,c,o,h,w,rh,rw,
                        active_nt,pos,prec)
                del a,v


@triton.jit
def _stem7_gw_filter28(AI,D,Y,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
        H:tl.constexpr,W:tl.constexpr,O:tl.constexpr,R:tl.constexpr,
        S:tl.constexpr,P:tl.constexpr,Q:tl.constexpr,SH:tl.constexpr,
        SW:tl.constexpr,PH:tl.constexpr,PW:tl.constexpr,DH:tl.constexpr,
        DW:tl.constexpr,PREC:tl.constexpr,SPLIT:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,RS:tl.constexpr):
    rr:tl.constexpr=RS//7;ss:tl.constexpr=RS%7
    ylo:tl.constexpr=(PH-rr+1)//2 if PH>rr else 0
    xlo:tl.constexpr=(PW-ss+1)//2 if PW>ss else 0
    ye:tl.constexpr=(H-1+PH-rr)//2+1;yhi:tl.constexpr=ye if ye<P else P
    xe:tl.constexpr=(W-1+PW-ss)//2+1;xhi:tl.constexpr=xe if xe<Q else Q
    vh:tl.constexpr=yhi-ylo if yhi>ylo else 0
    vw:tl.constexpr=xhi-xlo if xhi>xlo else 0
    pid=tl.program_id(0);gs=tl.program_id(1);g=gs//SPLIT;part=gs%SPLIT
    ci=pid//tl.cdiv(O,BN)*BM+tl.arange(0,BM)
    oc=pid%tl.cdiv(O,BN)*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),tl.float32)
    if vh>0 and vw>0:
        k=part*BK+tl.arange(0,BK)
        f=k//(vh*vw);yy=k//vw%vh;xx=k%vw
        aoff=(g*F+f)*C*H*W+((yy+ylo)*2-PH+rr)*W+(xx+xlo)*2-PW+ss
        doff=(g*F+f)*O*P*Q+(yy+ylo)*Q+xx+xlo
        step:tl.constexpr=SPLIT*BK
        dx:tl.constexpr=step%vw;dy:tl.constexpr=(step//vw)%vh
        df:tl.constexpr=step//(vh*vw)
        CHUNKED:tl.constexpr=F*vh*vw>=SPLIT*8192
        INNER:tl.constexpr=512//BK if CHUNKED else 1
        correction=tl.zeros((BM,BN),tl.float32)
        for chunk in range((F*vh*vw+SPLIT*BK*INNER-1)//(SPLIT*BK*INNER)):
            partial=tl.zeros((BM,BN),tl.float32)
            for block in range(INNER):
                a=tl.load(AI+ci[:,None]*H*W+aoff[None,:],
                          (ci[:,None]<C)&(f[None,:]<F),0)
                d=tl.load(D+oc[None,:]*P*Q+doff[:,None],
                          (oc[None,:]<O)&(f[:,None]<F),0)
                if CHUNKED:partial=tl.dot(a,d,partial,input_precision=PREC)
                else:acc=tl.dot(a,d,acc,input_precision=PREC)
                tx=xx+dx;cx=tx>=vw;nx=tx-tl.where(cx,vw,0)
                ty=yy+dy+cx.to(tl.int32);cy=ty>=vh;ny=ty-tl.where(cy,vh,0)
                fd=df+cy.to(tl.int32);xd=nx-xx;yd=ny-yy
                aoff+=fd*C*H*W+yd*2*W+xd*2
                doff+=fd*O*P*Q+yd*Q+xd
                f+=fd;xx=nx;yy=ny
            if CHUNKED:
                delta=partial-correction;updated=acc+delta
                correction=(updated-acc)-delta;acc=updated
    tl.store(Y+(gs*O+oc[None,:])*C*49+ci[:,None]*49+RS,acc,
             (ci[:,None]<C)&(oc[None,:]<O))


@triton.jit
def _stem7_gw_static28(AI,D,Y,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
        H:tl.constexpr,W:tl.constexpr,O:tl.constexpr,R:tl.constexpr,
        S:tl.constexpr,P:tl.constexpr,Q:tl.constexpr,SH:tl.constexpr,
        SW:tl.constexpr,PH:tl.constexpr,PW:tl.constexpr,DH:tl.constexpr,
        DW:tl.constexpr,PREC:tl.constexpr,SPLIT:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    for rs in tl.static_range(49):
        if tl.program_id(2)==rs:
            _stem7_gw_filter28(AI,D,Y,G,F,C,H,W,O,R,S,P,Q,SH,SW,PH,PW,
                               DH,DW,PREC,SPLIT,BM,BN,BK,rs)


# v26 pointwise stride-2 path. A 1x1 downsample is three dense matrix
# products over the sampled input lattice; keeping it on the generic
# convolution iterator wastes index/mask work and computes dgrad zeros.
def _pointwise_configs():
    return [triton.Config(dict(BM=m,BN=n,BK=k),num_warps=w,num_stages=st)
            for m,n,k,w,st in [(32,32,32,4,2),(32,64,32,4,3),
                (32,128,32,8,3),(64,32,32,4,3),(64,64,32,4,3),
                (128,32,32,8,3),(64,64,64,8,3)]]


@triton.autotune(configs=_pointwise_configs(),
                 key=['G','F','C','H','W','O','P','Q','SH','SW','HI','HW','HB','PREC'])
@triton.jit
def _pointwise_ggo26(X,Wt,AI,AW,Bias,Y,G:tl.constexpr,F:tl.constexpr,
        C:tl.constexpr,H:tl.constexpr,W:tl.constexpr,O:tl.constexpr,
        P:tl.constexpr,Q:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
        HI:tl.constexpr,HW:tl.constexpr,HB:tl.constexpr,PREC:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0);g=tl.program_id(1)
    nn:tl.constexpr=tl.cdiv(F*P*Q,BN)
    oc=pid//nn*BM+tl.arange(0,BM)
    n=pid%nn*BN+tl.arange(0,BN)
    batch=n//(P*Q);pos=n%(P*Q)
    ih=pos//Q*SH;iw=pos%Q*SW
    kk=tl.arange(0,BK)
    acc_i=tl.zeros((BM,BN),tl.float32)
    acc_w=tl.zeros((BM,BN),tl.float32)
    for cb in range(tl.cdiv(C,BK)):
        ci=cb*BK+kk
        mask_w=(oc[:,None]<O)&(ci[None,:]<C)
        mask_x=(ci[:,None]<C)&(n[None,:]<F*P*Q)
        if HI:
            a=tl.load(Wt+(g*O+oc[:,None])*C+ci[None,:],mask_w,0)
            b=tl.load(AI+((g*F+batch[None,:])*C+ci[:,None])*H*W+
                      ih[None,:]*W+iw[None,:],mask_x,0)
            acc_i=tl.dot(a,b,acc_i,input_precision=PREC)
        if HW:
            a=tl.load(AW+(g*O+oc[:,None])*C+ci[None,:],mask_w,0)
            b=tl.load(X+((g*F+batch[None,:])*C+ci[:,None])*H*W+
                      ih[None,:]*W+iw[None,:],mask_x,0)
            acc_w=tl.dot(a,b,acc_w,input_precision=PREC)
    # Preserve the two convolution-result roundings used by ATen/generic v23.
    if HI and HW:
        val=(acc_i.to(Y.dtype.element_ty).to(tl.float32)+
             acc_w.to(Y.dtype.element_ty).to(tl.float32)).to(Y.dtype.element_ty)
    elif HI:
        val=acc_i.to(Y.dtype.element_ty)
    elif HW:
        val=acc_w.to(Y.dtype.element_ty)
    else:
        val=tl.zeros((BM,BN),tl.float32).to(Y.dtype.element_ty)
    if HB:
        val+=tl.load(Bias+g*O+oc,oc<O,0)[:,None]
    tl.store(Y+((g*F+batch[None,:])*O+oc[:,None])*P*Q+pos[None,:],val,
             (oc[:,None]<O)&(n[None,:]<F*P*Q))


@triton.autotune(configs=_pointwise_configs(),
                 key=['G','F','C','H','W','O','P','Q','SH','SW','PREC'])
@triton.jit
def _pointwise_gi26(D,AW,Y,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
        H:tl.constexpr,W:tl.constexpr,O:tl.constexpr,P:tl.constexpr,
        Q:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,PREC:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0);g=tl.program_id(1)
    nn:tl.constexpr=tl.cdiv(F*P*Q,BN)
    ci=pid//nn*BM+tl.arange(0,BM)
    n=pid%nn*BN+tl.arange(0,BN)
    batch=n//(P*Q);pos=n%(P*Q)
    ih=pos//Q*SH;iw=pos%Q*SW
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for ob in range(tl.cdiv(O,BK)):
        oc=ob*BK+kk
        a=tl.load(AW+(g*O+oc[None,:])*C+ci[:,None],
                  (ci[:,None]<C)&(oc[None,:]<O),0)
        b=tl.load(D+((g*F+batch[None,:])*O+oc[:,None])*P*Q+pos[None,:],
                  (oc[:,None]<O)&(n[None,:]<F*P*Q),0)
        acc=tl.dot(a,b,acc,input_precision=PREC)
    # Y is zero-initialized by the caller; only the sampled lattice is written.
    tl.store(Y+((g*F+batch[None,:])*C+ci[:,None])*H*W+
             ih[None,:]*W+iw[None,:],acc,
             (ci[:,None]<C)&(n[None,:]<F*P*Q)&(ih[None,:]<H)&(iw[None,:]<W))


@triton.autotune(configs=_pointwise_configs(),
                 key=['G','F','C','H','W','O','P','Q','SH','SW','SPLIT','PREC'])
@triton.jit
def _pointwise_gw26(AI,D,Y,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
        H:tl.constexpr,W:tl.constexpr,O:tl.constexpr,P:tl.constexpr,
        Q:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,PREC:tl.constexpr,
        SPLIT:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0);gs=tl.program_id(1);g=gs//SPLIT;part=gs%SPLIT
    nn:tl.constexpr=tl.cdiv(C,BN)
    oc=pid//nn*BM+tl.arange(0,BM)
    ci=pid%nn*BN+tl.arange(0,BN)
    TOTAL:tl.constexpr=F*P*Q
    CHUNK:tl.constexpr=((TOTAL+SPLIT*BK-1)//(SPLIT*BK))*BK
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for kb in range(CHUNK//BK):
        k=part*CHUNK+kb*BK+kk
        batch=k//(P*Q);pos=k%(P*Q)
        ih=pos//Q*SH;iw=pos%Q*SW
        a=tl.load(D+((g*F+batch[None,:])*O+oc[:,None])*P*Q+pos[None,:],
                  (oc[:,None]<O)&(k[None,:]<TOTAL),0)
        b=tl.load(AI+((g*F+batch[:,None])*C+ci[None,:])*H*W+
                  ih[:,None]*W+iw[:,None],
                  (ci[None,:]<C)&(k[:,None]<TOTAL),0)
        acc=tl.dot(a,b,acc,input_precision=PREC)
    tl.store(Y+(gs*O+oc[:,None])*C+ci[None,:],acc,
             (oc[:,None]<O)&(ci[None,:]<C))

# Winograd F(2x2,3x3): BT rows (1,0,-1,0),(0,1,1,0),
# (0,-1,1,0),(0,-1,0,1); AT rows (1,1,1,0),(0,1,-1,1).
# Transform equations: https://github.com/andravin/wincnn (F(2,3)).
@triton.jit
def _win_input(X,g,ci,t,tr,tc,F:tl.constexpr,C:tl.constexpr,H:tl.constexpr,W:tl.constexpr,
               TH:tl.constexpr,TW:tl.constexpr,PH:tl.constexpr,PW:tl.constexpr):
    f=t//(TH*TW); y=t//TW%TH*2-PH; x=t%TW*2-PW
    r0=tl.where(tr==0,0,1);r1=tl.where(tr==3,3,2)
    c0=tl.where(tc==0,0,1);c1=tl.where(tc==3,3,2)
    sr0=tl.where(tr>=2,-1.,1.);sr1=tl.where(tr==0,-1.,1.)
    sc0=tl.where(tc>=2,-1.,1.);sc1=tl.where(tc==0,-1.,1.)
    base=((g*F+f)*C+ci)*H*W
    valid=(f<F)&(ci<C)
    a=tl.load(X+base+(y+r0)*W+x+c0,valid&(y+r0>=0)&(y+r0<H)&(x+c0>=0)&(x+c0<W),0)
    b=tl.load(X+base+(y+r0)*W+x+c1,valid&(y+r0>=0)&(y+r0<H)&(x+c1>=0)&(x+c1<W),0)
    c=tl.load(X+base+(y+r1)*W+x+c0,valid&(y+r1>=0)&(y+r1<H)&(x+c0>=0)&(x+c0<W),0)
    d=tl.load(X+base+(y+r1)*W+x+c1,valid&(y+r1>=0)&(y+r1<H)&(x+c1>=0)&(x+c1<W),0)
    return sr0*(sc0*a+sc1*b)+sr1*(sc0*c+sc1*d)

@triton.jit
def _win_g(tr,k:tl.constexpr):
    return tl.where(tr==0,1. if k==0 else 0.,tl.where(tr==3,1. if k==2 else 0.,tl.where((tr==2)&(k==1),-.5,.5)))

@triton.jit
def _win_weight(Wt,g,oc,ci,tr,tc,C:tl.constexpr,O:tl.constexpr,FLIP:tl.constexpr):
    v=tl.full((oc.shape[0],ci.shape[1]),0,tl.float32)
    for r in tl.static_range(3):
        for s in tl.static_range(3):
            coef=_win_g(tr,r)*_win_g(tc,s)
            if FLIP:off=g*O*C*9+ci*O*9+oc*9+(2-r)*3+2-s
            else:off=g*O*C*9+oc*C*9+ci*9+r*3+s
            w=tl.load(Wt+off,(oc<O)&(ci<C),0)
            v+=coef*w
    return v

@triton.jit
def _win_weight_pack(Wt,U,G:tl.constexpr,C:tl.constexpr,O:tl.constexpr,FLIP:tl.constexpr):
    pid=tl.program_id(0);g=tl.program_id(1);z=tl.program_id(2)
    oc=pid//tl.cdiv(C,32)*16+tl.arange(0,16)
    ci=pid%tl.cdiv(C,32)*32+tl.arange(0,32)
    v=_win_weight(Wt,g,oc[:,None],ci[None,:],z//4,z%4,C,O,FLIP)
    tl.store(U+((z*G+g)*O+oc[:,None])*C+ci[None,:],v,(oc[:,None]<O)&(ci[None,:]<C))

@triton.jit
def _win_out(M,Bias,Y,start,G:tl.constexpr,F:tl.constexpr,O:tl.constexpr,P:tl.constexpr,Q:tl.constexpr,
             NT:tl.constexpr,HB:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    n=i%NT;oc=i//NT%O;g=i//(NT*O);t=start+n
    f=t//(((P+1)//2)*((Q+1)//2));y=t//((Q+1)//2)%((P+1)//2)*2;x=t%((Q+1)//2)*2
    valid=(i<G*O*NT)&(f<F)
    base=(g*O+oc)*NT+n;step=G*O*NT
    m00=tl.load(M+0*step+base,valid,0)
    m01=tl.load(M+1*step+base,valid,0)
    m02=tl.load(M+2*step+base,valid,0)
    m03=tl.load(M+3*step+base,valid,0)
    m10=tl.load(M+4*step+base,valid,0)
    m11=tl.load(M+5*step+base,valid,0)
    m12=tl.load(M+6*step+base,valid,0)
    m13=tl.load(M+7*step+base,valid,0)
    m20=tl.load(M+8*step+base,valid,0)
    m21=tl.load(M+9*step+base,valid,0)
    m22=tl.load(M+10*step+base,valid,0)
    m23=tl.load(M+11*step+base,valid,0)
    m30=tl.load(M+12*step+base,valid,0)
    m31=tl.load(M+13*step+base,valid,0)
    m32=tl.load(M+14*step+base,valid,0)
    m33=tl.load(M+15*step+base,valid,0)
    v00=m00+m10+m20
    v10=m10-m20+m30
    v01=m01+m11+m21
    v11=m11-m21+m31
    v02=m02+m12+m22
    v12=m12-m22+m32
    v03=m03+m13+m23
    v13=m13-m23+m33
    bias=tl.load(Bias+g*O+oc,valid,0) if HB else tl.full((BLOCK,),0,tl.float32)
    tl.store(Y+((g*F+f)*O+oc)*P*Q+(y+0)*Q+x+0,v00+v01+v02+bias,valid&(y+0<P)&(x+0<Q))
    tl.store(Y+((g*F+f)*O+oc)*P*Q+(y+0)*Q+x+1,v01-v02+v03+bias,valid&(y+0<P)&(x+1<Q))
    tl.store(Y+((g*F+f)*O+oc)*P*Q+(y+1)*Q+x+0,v10+v11+v12+bias,valid&(y+1<P)&(x+0<Q))
    tl.store(Y+((g*F+f)*O+oc)*P*Q+(y+1)*Q+x+1,v11-v12+v13+bias,valid&(y+1<P)&(x+1<Q))

@triton.jit
def _win_dout(D,g,oc,t,tr,tc,F:tl.constexpr,O:tl.constexpr,P:tl.constexpr,Q:tl.constexpr):
    TH:tl.constexpr=((P+1)//2);TW:tl.constexpr=((Q+1)//2)
    f=t//(TH*TW);y=t//TW%TH*2;x=t%TW*2
    base=((g*F+f)*O+oc)*P*Q+y*Q+x
    valid=(f<F)&(oc<O)
    a=tl.load(D+base,valid&(y<P)&(x<Q)&(tr!=3)&(tc!=3),0)
    b=tl.load(D+base+1,valid&(y<P)&(x+1<Q)&(tr!=3)&(tc!=0),0)
    c=tl.load(D+base+Q,valid&(y+1<P)&(x<Q)&(tr!=0)&(tc!=3),0)
    d=tl.load(D+base+Q+1,valid&(y+1<P)&(x+1<Q)&(tr!=0)&(tc!=0),0)
    sr=tl.where(tr==2,-1.,1.);sc=tl.where(tc==2,-1.,1.)
    return a+sc*b+sr*c+sr*sc*d

@triton.jit
def _win_gw_out(U,Y,G:tl.constexpr,O:tl.constexpr,C:tl.constexpr,SK:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);g=i//(O*C);j=i%(O*C)
    valid=i<G*O*C
    u0=tl.full((BLOCK,),0,tl.float32)
    for part in range(SK):
        u0+=tl.load(U+((g*SK+part)*16+0)*O*C+j,valid,0)
    u1=tl.full((BLOCK,),0,tl.float32)
    for part in range(SK):
        u1+=tl.load(U+((g*SK+part)*16+1)*O*C+j,valid,0)
    u2=tl.full((BLOCK,),0,tl.float32)
    for part in range(SK):
        u2+=tl.load(U+((g*SK+part)*16+2)*O*C+j,valid,0)
    u3=tl.full((BLOCK,),0,tl.float32)
    for part in range(SK):
        u3+=tl.load(U+((g*SK+part)*16+3)*O*C+j,valid,0)
    u4=tl.full((BLOCK,),0,tl.float32)
    for part in range(SK):
        u4+=tl.load(U+((g*SK+part)*16+4)*O*C+j,valid,0)
    u5=tl.full((BLOCK,),0,tl.float32)
    for part in range(SK):
        u5+=tl.load(U+((g*SK+part)*16+5)*O*C+j,valid,0)
    u6=tl.full((BLOCK,),0,tl.float32)
    for part in range(SK):
        u6+=tl.load(U+((g*SK+part)*16+6)*O*C+j,valid,0)
    u7=tl.full((BLOCK,),0,tl.float32)
    for part in range(SK):
        u7+=tl.load(U+((g*SK+part)*16+7)*O*C+j,valid,0)
    u8=tl.full((BLOCK,),0,tl.float32)
    for part in range(SK):
        u8+=tl.load(U+((g*SK+part)*16+8)*O*C+j,valid,0)
    u9=tl.full((BLOCK,),0,tl.float32)
    for part in range(SK):
        u9+=tl.load(U+((g*SK+part)*16+9)*O*C+j,valid,0)
    u10=tl.full((BLOCK,),0,tl.float32)
    for part in range(SK):
        u10+=tl.load(U+((g*SK+part)*16+10)*O*C+j,valid,0)
    u11=tl.full((BLOCK,),0,tl.float32)
    for part in range(SK):
        u11+=tl.load(U+((g*SK+part)*16+11)*O*C+j,valid,0)
    u12=tl.full((BLOCK,),0,tl.float32)
    for part in range(SK):
        u12+=tl.load(U+((g*SK+part)*16+12)*O*C+j,valid,0)
    u13=tl.full((BLOCK,),0,tl.float32)
    for part in range(SK):
        u13+=tl.load(U+((g*SK+part)*16+13)*O*C+j,valid,0)
    u14=tl.full((BLOCK,),0,tl.float32)
    for part in range(SK):
        u14+=tl.load(U+((g*SK+part)*16+14)*O*C+j,valid,0)
    u15=tl.full((BLOCK,),0,tl.float32)
    for part in range(SK):
        u15+=tl.load(U+((g*SK+part)*16+15)*O*C+j,valid,0)
    v0=u0+0.5*(u4+u8)
    v4=0.5*(u4-u8)
    v8=0.5*(u4+u8)+u12
    v1=u1+0.5*(u5+u9)
    v5=0.5*(u5-u9)
    v9=0.5*(u5+u9)+u13
    v2=u2+0.5*(u6+u10)
    v6=0.5*(u6-u10)
    v10=0.5*(u6+u10)+u14
    v3=u3+0.5*(u7+u11)
    v7=0.5*(u7-u11)
    v11=0.5*(u7+u11)+u15
    tl.store(Y+i*9+0,v0+0.5*(v1+v2),valid)
    tl.store(Y+i*9+1,0.5*(v1-v2),valid)
    tl.store(Y+i*9+2,0.5*(v1+v2)+v3,valid)
    tl.store(Y+i*9+3,v4+0.5*(v5+v6),valid)
    tl.store(Y+i*9+4,0.5*(v5-v6),valid)
    tl.store(Y+i*9+5,0.5*(v5+v6)+v7,valid)
    tl.store(Y+i*9+6,v8+0.5*(v9+v10),valid)
    tl.store(Y+i*9+7,0.5*(v9-v10),valid)
    tl.store(Y+i*9+8,0.5*(v9+v10)+v11,valid)


def _packed_configs():
    # Small SIMT tiles; resource-checked offline for sm86. No transformed gathers
    # or two simultaneous accumulators in the GEMM mainloop.
    return [triton.Config(dict(BM=m,BN=n,BK=k),num_warps=w,num_stages=2)
            for m,n,k,w in [(16,32,16,4),(32,32,16,4),(32,64,16,4),(32,64,32,4),(64,32,16,4),(64,64,16,4),(64,64,32,4),(16,64,32,4)]]

@triton.jit
def _pack_tiles(X,V,start,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,H:tl.constexpr,W:tl.constexpr,
                P:tl.constexpr,Q:tl.constexpr,PH:tl.constexpr,PW:tl.constexpr,NT:tl.constexpr,
                DOUT:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    g=tl.program_id(1);z=tl.program_id(2);ci=i//NT;n=i%NT;t=start+n
    if DOUT:
        v=_win_dout(X,g,ci,t,z//4,z%4,F,C,H,W)
    else:
        v=_win_input(X,g,ci,t,z//4,z%4,F,C,H,W,(P+1)//2,(Q+1)//2,PH,PW)
    tl.store(V+((z*G+g)*C+ci)*NT+n,v,ci<C)

@triton.autotune(configs=_packed_configs(),key=['G','C','O','NT','ADD','PREC'])
@triton.jit
def _packed_conv(U,V,M,G:tl.constexpr,C:tl.constexpr,O:tl.constexpr,NT:tl.constexpr,
                 ADD:tl.constexpr,PREC:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0);g=tl.program_id(1);z=tl.program_id(2)
    nn=tl.cdiv(NT,BN);nm=tl.cdiv(O,BM)
    first=pid//(4*nn)*4;count=tl.minimum(nm-first,4)
    oc=(first+pid%(4*nn)%count)*BM+tl.arange(0,BM)
    n=(pid%(4*nn)//count)*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for cb in range(tl.cdiv(C,BK)):
        ci=cb*BK+kk
        a=tl.load(U+((z*G+g)*O+oc[:,None])*C+ci[None,:],(oc[:,None]<O)&(ci[None,:]<C),0)
        b=tl.load(V+((z*G+g)*C+ci[:,None])*NT+n[None,:],(ci[:,None]<C)&(n[None,:]<NT),0)
        acc=tl.dot(a,b,acc,input_precision=PREC)
    off=((z*G+g)*O+oc[:,None])*NT+n[None,:];mask=(oc[:,None]<O)&(n[None,:]<NT)
    if ADD:acc+=tl.load(M+off,mask,0)
    tl.store(M+off,acc,mask)

@triton.autotune(configs=_packed_configs(),key=['G','C','O','NT','PREC'])
@triton.jit
def _packed_gw(V,D,U,G:tl.constexpr,C:tl.constexpr,O:tl.constexpr,NT:tl.constexpr,
               PREC:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0);g=tl.program_id(1);z=tl.program_id(2)
    oc=pid//tl.cdiv(C,BN)*BM+tl.arange(0,BM);ci=pid%tl.cdiv(C,BN)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for kb in range(tl.cdiv(NT,BK)):
        k=kb*BK+kk
        a=tl.load(D+((z*G+g)*O+oc[:,None])*NT+k[None,:],(oc[:,None]<O)&(k[None,:]<NT),0)
        b=tl.load(V+((z*G+g)*C+ci[None,:])*NT+k[:,None],(ci[None,:]<C)&(k[:,None]<NT),0)
        acc=tl.dot(a,b,acc,input_precision=PREC)
    tl.store(U+((g*16+z)*O+oc[:,None])*C+ci[None,:],acc,(oc[:,None]<O)&(ci[None,:]<C))

@triton.jit
def _accumulate(U,Sum,Corr,N:tl.constexpr,FIRST:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N
    v=tl.load(U+i,valid,0)
    if FIRST:
        updated=v;corr=tl.full((BLOCK,),0,tl.float32)
    else:
        old=tl.load(Sum+i,valid,0);prev=tl.load(Corr+i,valid,0)
        delta=v-prev;updated=old+delta;corr=(updated-old)-delta
    tl.store(Sum+i,updated,valid);tl.store(Corr+i,corr,valid)


def _v5_winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget,algorithm='auto',split_k=None):
    """Explicit experimental path only. auto stays on measured implicit kernels.
    Workspace counts all simultaneously live packed tensors, excluding outputs.
    Outer group batching prevents workspace growth with G. split_k applies only
    to implicit fallback; packed gW uses streamed tiles and compensated summation.
    """
    supported=dtype==torch.float32 and (r,s,sh,sw,dh,dw)==(3,3,1,1,1,1) and 0<=ph<=2 and 0<=pw<=2
    if algorithm=='winograd' and not supported:raise ValueError('Winograd requires FP32, 3x3, stride=dilation=1, padding<=2')
    use=supported and algorithm=='winograd'
    budget=min(budget,(2**30-1)*4)
    # Bounded K=NT also limits each FP32 gW reduction before compensated summation.
    def choose(cin,cout,total,weight_copies):
        if not use:return (0,0,0)
        cap=min(1024,max(32,triton.next_power_of_2(total)))
        choices=[]
        for nt in [cap//(2**i) for i in range(cap.bit_length()-5)]:
            per=64*(weight_copies*cin*cout+(cin+cout)*nt)
            if per<=budget:
                gb=min(g,max(1,budget//per))
                launches=triton.cdiv(g,gb)*triton.cdiv(total,nt)
                choices.append((launches,-gb,-nt,nt,gb,per*gb))
        return min(choices)[3:] if choices else (0,0,0)
    nt0,gb0,b0=choose(c,o,f*triton.cdiv(p,2)*triton.cdiv(q,2),2)
    nt1,gb1,b1=choose(o,c,f*triton.cdiv(h,2)*triton.cdiv(w,2),1)
    ntw,gbw,bw=choose(c,o,f*triton.cdiv(p,2)*triton.cdiv(q,2),3)
    return dict(ggO_nt=nt0,gI_nt=nt1,gW_split=1 if ntw else 0,gW_nt=ntw,
                ggO_groups=gb0,gI_groups=gb1,gW_groups=gbw,
                ggO_pack=bool(nt0),gI_pack=bool(nt1),scratch_bytes=max(b0,b1,bw))


def _packed_forward(x,wt,ai,aw,bias,out,nt,group_batch,ph,pw,hi,hw,flip,precision):
    g,f,c,h,w=x.shape;_,_,o,p,q=out.shape
    total=f*triton.cdiv(p,2)*triton.cdiv(q,2)
    for base in range(0,g,group_batch):
        end=min(g,base+group_batch);gb=end-base
        weights=[];sources=[]
        if hi:weights.append(wt[base:end]);sources.append(ai[base:end])
        if hw:weights.append(aw[base:end]);sources.append(x[base:end])
        us=[]
        for weight in weights:
            u=torch.empty((16,gb,o,c),device=x.device,dtype=torch.float32)
            _win_weight_pack[(triton.cdiv(o,16)*triton.cdiv(c,32),gb,16)](weight,u,gb,c,o,flip)
            us.append(u)
        v=torch.empty((16,gb,c,nt),device=x.device,dtype=torch.float32)
        m=torch.empty((16,gb,o,nt),device=x.device,dtype=torch.float32)
        for start in range(0,total,nt):
            for term,(source,u) in enumerate(zip(sources,us)):
                _pack_tiles[(triton.cdiv(c*nt,256),gb,16)](source,v,start,gb,f,c,h,w,p,q,ph,pw,nt,False,256)
                # Separate output buffers are unnecessary: the second term adds to M.
                # Tune overwrite mode only; reuse its config for in-place addition.
                _launch_packed_conv(u,v,m,gb,c,o,nt,term!=0,precision)
            _win_out[(triton.cdiv(gb*o*nt,128),)](m,bias[base:end] if bias is not None else x,out[base:end],start,gb,f,o,p,q,nt,bias is not None,128)
        # Do not retain the previous group's workspaces during next allocation.
        del us,u,v,m


def _launch_packed_conv(u,v,m,g,c,o,nt,add,precision):
    grid=lambda meta:(triton.cdiv(o,meta['BM'])*triton.cdiv(nt,meta['BN']),g,16)
    if not add:
        _packed_conv[grid](u,v,m,g,c,o,nt,False,precision)
    else:
        # Avoid in-place autotune corruption: use the already measured overwrite
        # configuration. No input/output values are retained between API calls.
        cfg=_packed_conv.best_config
        _packed_conv.fn[grid](u,v,m,g,c,o,nt,True,precision,**cfg.kwargs,num_warps=cfg.num_warps,num_stages=cfg.num_stages)


def _packed_weight_grad(ai,d,out,nt,group_batch,ph,pw,precision):
    g,f,c,h,w=ai.shape;_,_,o,p,q=d.shape
    total=f*triton.cdiv(p,2)*triton.cdiv(q,2)
    for base in range(0,g,group_batch):
        end=min(g,base+group_batch);gb=end-base
        v=torch.empty((16,gb,c,nt),device=ai.device,dtype=torch.float32)
        dv=torch.empty((16,gb,o,nt),device=ai.device,dtype=torch.float32)
        u=torch.empty((gb,16,o,c),device=ai.device,dtype=torch.float32)
        acc=torch.empty_like(u);corr=torch.empty_like(u)
        ne=gb*16*o*c
        for start in range(0,total,nt):
            _pack_tiles[(triton.cdiv(c*nt,256),gb,16)](ai[base:end],v,start,gb,f,c,h,w,p,q,ph,pw,nt,False,256)
            _pack_tiles[(triton.cdiv(o*nt,256),gb,16)](d[base:end],dv,start,gb,f,o,p,q,p,q,0,0,nt,True,256)
            _packed_gw[lambda meta:(triton.cdiv(o,meta['BM'])*triton.cdiv(c,meta['BN']),gb,16)](v,dv,u,gb,c,o,nt,precision)
            _accumulate[(triton.cdiv(ne,256),)](u,acc,corr,ne,start==0,256,enable_fp_fusion=False)
        _win_gw_out[(triton.cdiv(gb*o*c,64),)](acc,out[base:end],gb,o,c,1,64)
        del v,dv,u,acc,corr

def _pair(x):
    if isinstance(x,int): return (x,x)
    x=tuple(x)
    if len(x)==1: x=x*2
    if len(x)!=2 or any(not isinstance(v,int) for v in x): raise ValueError('Expected int or 1/2 integer sequence')
    return x


def workspace_split(g,f,c,o,r,s,p,q,num_sms,budget_bytes):
    """Bounded heuristic, not a measured optimum. No workspace for split=1."""
    tiles=g*triton.cdiv(o,32)*triton.cdiv(c*r*s,64)
    occupancy=max(1,triton.cdiv(2*num_sms,tiles))
    # Also bound long per-CTA reductions, even when G already supplies many CTAs.
    length=max(1,triton.cdiv(f*p*q,2048))
    want=min(16,max(occupancy,length),max(1,(f*p*q)//256))
    split=1
    while split*2<=want: split*=2
    per=g*o*c*r*s*4
    while split>1 and per*split>budget_bytes: split//=2
    return split


_ALGORITHMS=('auto','implicit','stem7_gi','stem7_gi_ggo','stem7_gi_gw','stem7_full',
    'pointwise_stride2','stride2_joint_split_gw4','stride2_joint_split_gw8',
    'stride2_joint_onebuf_wide_gi',
    'stride2_joint_onebuf','stride2_wide_chunks',
    'stride2_small_spatial',
    'stride2_ggo_tail','stride2_ggo_single',
    'stride2_implicit_gw','stride2_packed_gw','stride2_packed_gi_tail',
    'stride2_packed_gi','stride2_static_weight','stride2_static',
    'stride2_shared','adaptive14','stride2_weight','stride2_packed',
    'winograd_weight','adaptive13','winograd_exact','adaptive12',
    'stride2_phase','shape_safe','winograd','winograd_v5','winograd_v6',
    'winograd_v7','winograd_v8','winograd_v9','winograd_tail',
    'winograd_group','winograd_large','winograd_joint','winograd_separate')


@dataclass(frozen=True)
class ConvDoubleBackwardSignature:
    """Hashable selector key. 4D and equivalent G=1 5D calls share a key."""
    device: tuple
    dtype: str
    runtime: tuple
    shape: tuple
    convolution: tuple
    optional_inputs: tuple
    compute: tuple
    precision: str
    workspace_bytes: int
    split_k: object


_EXACT_ALGORITHM_TABLE={}

# Stable >3% misses of the fitted family rules in the completed RTX A5000
# FP32/IEEE v23 selector sweep. These are deliberately sparse exceptions, not
# a copy of every measured point. Key = (shape, convolution, optional_inputs).
# shape: (G,F,C,H,W,O,R,S,P,Q)
# convolution: (SH,SW,PH,PW,DH,DW)
_BUILTIN_EXACT_ALGORITHM_TABLE={
    # Dense 3x3 stride=1. The first physical shape was measured both without
    # bias (ResNet) and with bias (Conv3), and has the same winner.
    ((1,10,128,16,16,128,3,3,16,16),(1,1,1,1,1,1),(True,True,False)):'winograd_exact',
    ((1,10,128,16,16,128,3,3,16,16),(1,1,1,1,1,1),(True,True,True)):'winograd_exact',
    ((4,10,128,8,8,128,3,3,8,8),(1,1,1,1,1,1),(True,True,True)):'winograd_weight',
    ((4,1,64,32,32,64,3,3,32,32),(1,1,1,1,1,1),(True,True,False)):'implicit',
    ((1,1,512,4,4,512,3,3,4,4),(1,1,1,1,1,1),(True,True,False)):'winograd_weight',
    # Dense ResNet-like 3x3 stride=2 cases where the v34 family rule was more
    # than 3% behind the measured candidate oracle.  These seven entries are
    # intentionally sparse; all other completed v33 points use family rules.
    ((16,1,64,32,32,128,3,3,16,16),(2,2,1,1,1,1),(True,True,False)):'stride2_joint_onebuf',
    ((64,1,64,32,32,128,3,3,16,16),(2,2,1,1,1,1),(True,True,False)):'stride2_joint_onebuf_wide_gi',
    ((64,1,128,16,16,256,3,3,8,8),(2,2,1,1,1,1),(True,True,False)):'stride2_joint_onebuf_wide_gi',
    ((1,10,64,32,32,128,3,3,16,16),(2,2,1,1,1,1),(True,True,False)):'stride2_wide_chunks',
    ((4,10,64,32,32,128,3,3,16,16),(2,2,1,1,1,1),(True,True,False)):'stride2_joint_onebuf',
    ((4,10,128,16,16,256,3,3,8,8),(2,2,1,1,1,1),(True,True,False)):'stride2_joint_onebuf_wide_gi',
    ((4,10,256,8,8,512,3,3,4,4),(2,2,1,1,1,1),(True,True,False)):'stride2_joint_onebuf',
}


def convolution_double_backward_signature(ggI_opt,ggW_r_opt,ggb_opt,gO_r,
        weight_r,input,stride_,padding_,dilation_,transposed_,output_padding_,
        groups_,output_mask,*,precision='ieee',workspace_limit_mb=128,
        split_k=None,compute=(True,True,True)):
    """Build the normalized key used by the exact table and warmup cache."""
    if transposed_ or groups_!=1 or _pair(output_padding_)!=(0,0):
        raise NotImplementedError('Only transposed=False, output_padding=0, inner groups=1')
    if input.ndim==4 and weight_r.ndim==4 and gO_r.ndim==4:
        f,c,h,w=input.shape;o,cw,r,s=weight_r.shape
        g=1
        if c!=cw: raise ValueError('Input/weight channel mismatch')
    elif input.ndim==5 and weight_r.ndim==5 and gO_r.ndim==5:
        g,f,c,h,w=input.shape;gw,o,cw,r,s=weight_r.shape
        if (g,c)!=(gw,cw): raise ValueError('Input/weight group or channel mismatch')
    else:
        raise ValueError('Expected all convolution tensors to use either 4D ATen or 5D group-first layout')
    sh,sw=_pair(stride_);ph,pw=_pair(padding_);dh,dw=_pair(dilation_)
    p=(h+2*ph-dh*(r-1)-1)//sh+1;q=(w+2*pw-dw*(s-1)-1)//sw+1
    expected=(f,o,p,q) if input.ndim==4 else (g,f,o,p,q)
    if tuple(gO_r.shape)!=expected: raise ValueError('Invalid gO shape')
    if len(output_mask)!=3 or len(compute)!=3: raise ValueError('Expected three output/compute flags')
    if input.is_cuda:
        props=torch.cuda.get_device_properties(input.device)
        device=(props.name,props.major,props.minor)
    else:
        device=(input.device.type,None,None)
    return ConvDoubleBackwardSignature(
        device=device,dtype=str(input.dtype),
        runtime=(str(torch.__version__),str(triton.__version__)),
        shape=(int(g),int(f),int(c),int(h),int(w),int(o),int(r),int(s),int(p),int(q)),
        convolution=(int(sh),int(sw),int(ph),int(pw),int(dh),int(dw)),
        optional_inputs=(ggI_opt is not None,ggW_r_opt is not None,ggb_opt is not None),
        compute=tuple(bool(x) for x in compute),precision=str(precision),
        workspace_bytes=int(workspace_limit_mb*1024**2),split_k=split_k)


def _legacy_stride2_algorithm(g,f,c,o,p,q):
    """Pre-v27 fitted fallback for unmeasured/non-ResNet stride-2 shapes."""
    pq=p*q;k=f*pq;total=g*k
    if c*o>5200*k:
        return 'stride2_packed_gi_tail' if g<=8 else 'implicit'
    if total<=400:
        if c*o<=650*k:
            return 'implicit' if o<=128 else 'stride2_static_weight'
        return 'stride2_ggo_single'
    if total<=1280 and 10*c<=14*pq:
        return 'stride2_static_weight'
    return 'stride2_ggo_single'


def _resnet_stride2_family_algorithm(g,f,c,p,q):
    """v34 family rule fitted to the complete v33 native-layout audit.

    The rule uses per-group batch F, outer independent groups G, channel
    regime, output spatial work P*Q, and the gW reduction length F*P*Q.  Seven
    stable >3% misses are handled by the built-in exact table before this
    function.  It deliberately does not inspect model or dataset names.
    """
    pq=p*q
    reduction=f*pq

    # Very small per-group batches are launch/packing dominated.  Prefer the
    # low-work legacy schedules unless enough independent/spatial work exists.
    if f<=1:
        if g<=1:
            return 'implicit' if c<=64 else 'stride2_static_weight'
        if g<=4:
            if c<=64:
                return 'stride2_packed_gi_tail' if pq>=256 else 'implicit'
            if c<256:
                return 'stride2_static_weight'
            return 'stride2_packed_gi_tail' if pq>=16 else 'implicit'
        if g<=16:
            if c<=64:return 'stride2_static_weight'
            if c<256:return 'stride2_packed_gi_tail'
            return 'implicit'
        if c<=64:return 'stride2_packed_gi_tail'
        return 'implicit'

    # Medium batches benefit from reusable one-buffer packing once G exposes
    # enough parallel work.  Low-G cases retain compact legacy schedules.
    if f<64:
        if g<=1:
            if c<256 and c>64 and pq>=64:return 'stride2_ggo_single'
            if c>=256 and pq>=16:return 'stride2_packed_gi_tail'
            return 'stride2_static_weight'
        if g<=4:
            return 'stride2_ggo_single' if c<=64 else 'stride2_packed_gi_tail'
        if g<=16:
            return ('stride2_joint_onebuf_wide_gi' if pq<=16 else
                    'stride2_joint_onebuf')
        return ('stride2_joint_onebuf_wide_gi' if c>64 else
                'stride2_joint_onebuf')

    # Large batches: deep/small-spatial layers favor one-buffer reuse; long
    # gW reductions need split-K; early layers below that threshold use wide
    # chunks for G<=2 and the compact ggo schedule otherwise.
    if c>=256:
        return 'stride2_joint_onebuf'
    if reduction>=16384:
        return 'stride2_joint_split_gw8'
    if c<=64:
        return 'stride2_wide_chunks' if g<=2 else 'stride2_ggo_single'
    if f>=512:
        if g<=1:return 'stride2_packed_gi_tail'
        if g<=2:return 'stride2_ggo_single'
        return 'stride2_joint_onebuf'
    return ('stride2_joint_onebuf_wide_gi' if g<=1 else
            'stride2_joint_onebuf')


def _fallback_algorithm(sig):
    g,f,c,h,w,o,r,s,p,q=sig.shape
    sh,sw,ph,pw,dh,dw=sig.convolution
    hi,hw,_=sig.optional_inputs
    # v26's sampled-lattice 1x1 path won over the previous group-first kernel
    # on every measured shortcut point and is now the v27 default for the
    # normal complete FP32/IEEE call.  Registered/per-call exact tables still
    # take precedence because they are consulted before this fallback.
    pointwise=(r,s,sh,sw,ph,pw,dh,dw)==(1,1,2,2,0,0,1,1)
    if (sig.dtype=='torch.float32' and sig.precision=='ieee' and pointwise and
            sig.compute==(True,True,True) and hi and hw):
        return 'pointwise_stride2','measured_pointwise_v26'
    # The fitted 3x3 rules cover the complete three-output call with both
    # tensor double-gradient terms present. Preserve the generic route outside it.
    if (sig.dtype!='torch.float32' or sig.precision!='ieee' or
            sig.compute!=(True,True,True) or not hi or not hw):
        return 'implicit','shape_fallback'
    exception=_BUILTIN_EXACT_ALGORITHM_TABLE.get(
        (sig.shape,sig.convolution,sig.optional_inputs))
    if exception is not None:
        return exception,'builtin_exact_table'
    pq=p*q;k=f*pq;total=g*k
    stem7=(c==3 and (r,s,sh,sw,ph,pw,dh,dw)==
           (7,7,2,2,3,3,1,1))
    if stem7:
        if total<=1024:
            algorithm='implicit'
        elif f==1 and total<=4096:
            algorithm='stem7_gi'
        else:
            algorithm='stem7_gi_ggo'
        return algorithm,'stem7_family_v34'
    dense3=(r,s,sh,sw,dh,dw)==(3,3,1,1,1,1) and c>3 and 0<=ph<=2 and 0<=pw<=2
    if dense3:
        if f<=3:
            algorithm='winograd_weight' if total>2048 and k>128 else 'implicit'
        else:
            algorithm='implicit' if total<=2850 and o<=128 else 'winograd_weight'
        return algorithm,'shape_fallback'
    stride2=(r,s,sh,sw,dh,dw)==(3,3,2,2,1,1) and min(h,w)>=2
    if stride2:
        # The broad v33 audit covers the ResNet downsample family O=2*C,
        # padding=1.  Do not extrapolate its aggressive joint/split schedules
        # to other channel ratios or padding modes without measurements.
        if (ph,pw)==(1,1) and o==2*c:
            return _resnet_stride2_family_algorithm(g,f,c,p,q),'stride2_family_v34'
        return _legacy_stride2_algorithm(g,f,c,o,p,q),'shape_fallback'
    return 'implicit','shape_fallback'


@lru_cache(maxsize=4096)
def _select_algorithm_cached(signature):
    exact=_EXACT_ALGORITHM_TABLE.get(signature)
    if exact is not None:return exact,'exact_table'
    return _fallback_algorithm(signature)


def select_convolution_double_backward_algorithm(signature):
    """Return the selected explicit algorithm for a normalized signature."""
    if not isinstance(signature,ConvDoubleBackwardSignature):
        raise TypeError('signature must be ConvDoubleBackwardSignature')
    return _select_algorithm_cached(signature)[0]


def _lookup_call_exact_table(signature,table):
    if table is None:return None
    if not hasattr(table,'get'):
        raise TypeError('exact_algorithm_table must be a mapping or None')
    algorithm=table.get(signature)
    if algorithm is None:return None
    if algorithm not in _ALGORITHMS or algorithm=='auto':
        raise ValueError('External exact-table values must be explicit supported algorithms')
    return algorithm


def register_convolution_double_backward_algorithms(entries,*,replace=False):
    """Register {ConvDoubleBackwardSignature: algorithm} exact-table entries."""
    global _EXACT_ALGORITHM_TABLE
    new={} if replace else dict(_EXACT_ALGORITHM_TABLE)
    for signature,algorithm in dict(entries).items():
        if not isinstance(signature,ConvDoubleBackwardSignature):
            raise TypeError('Exact-table keys must be ConvDoubleBackwardSignature')
        if algorithm not in _ALGORITHMS or algorithm=='auto':
            raise ValueError('Exact-table values must be explicit supported algorithms')
        new[signature]=algorithm
    _EXACT_ALGORITHM_TABLE=new
    _select_algorithm_cached.cache_clear()


def clear_convolution_double_backward_selector_cache():
    """Clear only selector metadata; Triton compilation caches are untouched."""
    _select_algorithm_cached.cache_clear()


# TODO: D- bwd 调用接口

def convolution_double_backward(ggI_opt,ggW_r_opt,ggb_opt,gO_r,weight_r,input,
        stride_,padding_,dilation_,transposed_,output_padding_,groups_,output_mask,
        *,precision='ieee',workspace_limit_mb=128,split_k=None,compute=(True,True,True),
        algorithm='auto',exact_algorithm_table=None):
    """ATen argument order, supporting ordinary 4D and group-first 5D tensors.

    4D uses ATen shapes: input/ggI [B,C,H,W], weight/ggW [O,C,R,S],
    gO [B,O,P,Q], ggb [O]. 5D adds an independent leading group G:
    input/ggI [G,F,C,H,W], weight/ggW [G,O,C,R,S], gO [G,F,O,P,Q].
    groups_=1 means dense convolution INSIDE each independent outer group.
    Returns (ggO,gI,gW), missing terms are None. Matches PyTorch 2.5.1's
    ignored output_mask; use optional ggI/ggW/ggb=None to omit terms.
    Non-transposed 2D only. No autograd graph; no retained tensor state.
    compute selects (ggO,gI,gW) explicitly for profiling; default computes all.
    adaptive14 is an opt-in FP32 candidate: stride2_weight repacks only dgrad
    weights; stride2_packed also uses bounded im2col for ggO. Both retain gW.
    Small dense Winograd routes reuse each weight load across all 16 transform
    components. winograd_weight explicitly selects this transform experiment.
    New stride2 packing uses at most min(workspace_limit_mb,32) MiB.
    Execution workspaces are bounded, sequential, and freed within the call.
    No speedup guarantee; compare adaptive13 on the target GPU.
    adaptive13 retains adaptive12 routes, but coalesces Winograd remainders
    for <=1024 tiles/group. winograd_exact forces this experimental scheduling
    for supported Winograd shapes. Both require target-GPU performance validation.
    adaptive12 is an opt-in workload dispatcher: <=256 tiles/group uses implicit;
    larger dense cases retain the tail-corrected path. Stride-2 3x3 uses parity
    dgrad. The cutoff is provisional, not a universal performance guarantee.
    stem7_gi and stem7_gi_ggo are v29 candidates for the RGB 7x7
    stride-2 padding-3 ResNet stem. The latter combines the measured parity
    dgrad and packed ggO paths while deliberately retaining implicit wgrad.
    The earlier stem7_gi_gw/stem7_full ablations remain available only for
    reproducibility. Auto selects implicit/gi/gi+ggo from measured total work.
    pointwise_stride2 is selected automatically for the measured complete
    FP32/IEEE 1x1 stride-2 padding-0 family.
    stride2_joint_split_gw4/8 are v32 large-N experiments. They retain v31's
    wide parity-packed gI and reusable ggI buffer, but split the packed gW
    reduction across four or eight CTAs and use larger ggO/gW GEMM tiles.
    Shapes below the large-N threshold deliberately execute the unchanged v31
    path. stride2_joint_onebuf_wide_gi is the v31 opt-in composition: v30's reusable
    one-buffer ggO+gW path plus v27's larger-workspace parity-packed gI
    schedule. stride2_joint_onebuf is the unchanged v30
    opt-in 3x3 stride-2 complete-call experiment: it packs ggI into one
    workspace, reuses that matrix for ggO and gW, then overwrites it with the
    input packing for ggO's second term. It keeps the measured v18 gI route.
    stride2_wide_chunks is a v27 3x3 stride-2 schedule that spends up to
    128 MiB on larger ggO/gI packing chunks. The v34 auto selector uses it,
    the one-buffer variants and split8 only in measured workload regimes.
    stride2_small_spatial remains the unsuccessful v26 opt-in experiment.
    auto first consults an optional per-call exact_algorithm_table, then the
    registered table, built-in measured exceptions, and fitted shape fallback.
    shape_safe selects tail-corrected Winograd for FP32 dense 3x3 stride=1
    with C>3, and preserves implicit for other shapes. This structural candidate
    is not a measured optimal dispatcher. implicit always retains the generic path.
    winograd_large selects
    the v9 large-batch candidate (also winograd_v9); winograd_v8 is unchanged v8.
    winograd_tail fixes tail padding with v9 scheduling; winograd_group also
    prefers processing all outer groups within the same workspace budget.
    These v10 candidates are explicit opt-ins until measured on the target GPU.
    Explicit winograd
    selects the experimental v8 path; winograd_v7/v6 preserve prior versions, v5 the
    old packed implementation. winograd_joint/separate force v8 scheduling.
    split_k controls gW partial sums. Force algorithm='implicit' to bypass auto.
    Workspace bounds apply to execution buffers; first-use autotuning may clone
    accumulator buffers to restore their values between candidate trials. It keeps
    FP32 arithmetic but changes summation order; it is not bitwise ATen-equivalent.
    """
    if transposed_ or groups_!=1 or _pair(output_padding_)!=(0,0):
        raise NotImplementedError('Only transposed=False, output_padding=0, inner groups=1; outer G inferred from input')
    if len(output_mask)!=3: raise ValueError('output_mask must have 3 entries')
    if input.ndim==4:
        if weight_r.ndim!=4 or gO_r.ndim!=4:
            raise ValueError('4D input requires 4D weight and gO')
        lift=lambda x:None if x is None else x.unsqueeze(0)
        out=convolution_double_backward(lift(ggI_opt),lift(ggW_r_opt),lift(ggb_opt),
            lift(gO_r),lift(weight_r),lift(input),stride_,padding_,dilation_,
            transposed_,output_padding_,groups_,output_mask,precision=precision,
            workspace_limit_mb=workspace_limit_mb,split_k=split_k,compute=compute,
            algorithm=algorithm,exact_algorithm_table=exact_algorithm_table)
        return tuple(None if x is None else x.squeeze(0) for x in out)
    if input.ndim!=5 or weight_r.ndim!=5: raise ValueError('Expected 5D group-first input/weight')
    g,f,c,h,w=input.shape; gw,o,cw,r,s=weight_r.shape
    sh,sw=_pair(stride_); ph,pw=_pair(padding_); dh,dw=_pair(dilation_)
    if min(*input.shape,*weight_r.shape,sh,sw,dh,dw)<=0 or min(ph,pw)<0 or (g,c)!=(gw,cw):
        raise ValueError('Invalid dimensions or convolution parameters')
    p=(h+2*ph-dh*(r-1)-1)//sh+1; q=(w+2*pw-dw*(s-1)-1)//sw+1
    if min(p,q)<=0 or tuple(gO_r.shape)!=(g,f,o,p,q): raise ValueError('Invalid gO shape')
    for t,shape in [(ggI_opt,input.shape),(ggW_r_opt,weight_r.shape),(ggb_opt,(g,o))]:
        if t is not None and tuple(t.shape)!=tuple(shape): raise ValueError('Optional tensor shape mismatch')
    ts=[input,weight_r,gO_r]+[t for t in (ggI_opt,ggW_r_opt,ggb_opt) if t is not None]
    if input.dtype not in (torch.float32,torch.float16,torch.bfloat16): raise TypeError('FP32/FP16/BF16 required')
    for t in ts:
        if not t.is_cuda or t.device!=input.device or t.dtype!=input.dtype or not t.is_contiguous():
            raise ValueError('All inputs must be contiguous, same CUDA device/dtype')
        if torch.is_grad_enabled() and t.requires_grad: raise RuntimeError('Stateless manual derivative: call under torch.no_grad()')
    if max(t.numel() for t in ts)>=2**30 or g*f>65535: raise ValueError('Shape exceeds conservative indexing/grid limits')
    if precision not in ('ieee','tf32','tf32x3'): raise ValueError('Unsupported precision')
    if workspace_limit_mb<0: raise ValueError('Negative workspace budget')
    if len(compute)!=3: raise ValueError('compute must contain three flags')
    if algorithm not in _ALGORITHMS:raise ValueError('Invalid algorithm')
    if algorithm=='auto':
        signature=convolution_double_backward_signature(
            ggI_opt,ggW_r_opt,ggb_opt,gO_r,weight_r,input,stride_,padding_,dilation_,
            transposed_,output_padding_,groups_,output_mask,precision=precision,
            workspace_limit_mb=workspace_limit_mb,split_k=split_k,compute=compute)
        algorithm=_lookup_call_exact_table(signature,exact_algorithm_table)
        if algorithm is None:
            algorithm=_select_algorithm_cached(signature)[0]
    plan=winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,input.dtype,int(workspace_limit_mb*1024**2),algorithm,split_k)
    ggO=gI=gW=None
    with torch.cuda.device(input.device):
        dims=(g,f,c,h,w,o,r,s,p,q,sh,sw,ph,pw,dh,dw,precision)
        hi=ggI_opt is not None; hw=ggW_r_opt is not None; hb=ggb_opt is not None
        v6=plan.get('version') in (6,7,8,9,10)
        v7=plan.get('version') in (7,8,9,10)
        forward=_forward10 if plan.get("version")==10 else (_forward7 if v7 else (_forward6 if v6 else _packed_forward))
        joint=_joint10 if plan.get("version")==10 else (_joint9 if plan.get('version')==9 else (_joint8 if plan.get('version')==8 else (_joint7 if v7 else _joint6)))
        weight_grad=_weight_grad10 if plan.get("version")==10 else (_weight_grad7 if v7 else _weight_grad6)
        if plan.get('exact_chunks'):
            forward=partial(forward,chunker=_tile_chunks13)
            joint=partial(joint,chunker=_tile_chunks13)
            weight_grad=partial(weight_grad,chunker=_tile_chunks13)
        if plan.get('weight_all14'):
            forward=partial(forward,weight_all=True)
            joint=partial(joint,weight_all=True)
        if plan.get('joint_split_gw32') and hi and compute[0] and compute[2]:
            ggO=torch.empty_like(gO_r);gW=torch.empty_like(weight_r)
            _joint_split_gw32(input,weight_r,ggI_opt,ggW_r_opt,ggb_opt,
                              gO_r,ggO,gW,plan,ph,pw,precision)
            compute=(False,compute[1],False)
        if plan.get('joint_onebuf30') and hi and compute[0] and compute[2]:
            ggO=torch.empty_like(gO_r);gW=torch.empty_like(weight_r)
            _joint_onebuf30(input,weight_r,ggI_opt,ggW_r_opt,ggb_opt,gO_r,
                            ggO,gW,plan,ph,pw,precision)
            compute=(False,compute[1],False)
        if plan.get('shared15') and hi and compute[0] and compute[2]:
            ggO=torch.empty_like(gO_r);gW=torch.empty_like(weight_r)
            _shared15(input,weight_r,ggI_opt,ggW_r_opt,ggb_opt,gO_r,ggO,gW,plan,ph,pw,precision)
            compute=(False,compute[1],False)
        if v6 and hi and compute[0] and compute[2] and plan['joint_nt']:
            ggO=torch.empty_like(gO_r);gW=torch.empty_like(weight_r)
            joint(input,weight_r,ggI_opt,ggW_r_opt,ggb_opt,gO_r,ggO,gW,plan['joint_nt'],plan['joint_groups'],plan['gW_split'],ph,pw,precision)
            compute=(False,compute[1],False)
        if compute[0] and (hi or hw or hb):
            ggO=torch.empty_like(gO_r)
            if plan.get('stem7_ggo28') and (hi or hw):
                _stem7_ggo28(input,weight_r,ggI_opt,ggW_r_opt,ggb_opt,ggO,
                             plan,precision)
            elif plan.get('pointwise26'):
                _pointwise_ggo26[lambda m:(triton.cdiv(o,m['BM'])*triton.cdiv(f*p*q,m['BN']),g)](
                    input,weight_r,ggI_opt if hi else input,ggW_r_opt if hw else weight_r,
                    ggb_opt if hb else input,ggO,g,f,c,h,w,o,p,q,sh,sw,hi,hw,hb,precision)
            elif plan.get('ggo_tail23') and (hi or hw):
                _ggo_packed23(input,weight_r,ggI_opt,ggW_r_opt,ggb_opt,ggO,plan,ph,pw,precision)
            elif plan.get('ggo_pack14') and (hi or hw):
                _ggo_packed14(input,weight_r,ggI_opt,ggW_r_opt,ggb_opt,ggO,plan,ph,pw,precision)
            elif plan['ggO_nt'] and (hi or hw):
                forward(input,weight_r,ggI_opt if hi else input,ggW_r_opt if hw else weight_r,ggb_opt,ggO,plan['ggO_nt'],plan['ggO_groups'],ph,pw,hi,hw,False,precision)
            else:
                _ggo[lambda m:(triton.cdiv(o,m['BM'])*triton.cdiv(f*p*q if m['BATCH_FOLD'] else p*q,m['BN']),g if m['BATCH_FOLD'] else g*f)](
                input,weight_r,ggI_opt if hi else input,ggW_r_opt if hw else weight_r,
                ggb_opt if hb else input,ggO,*dims,hi,hw,hb)
        if compute[1] and hw:
            gI=torch.zeros_like(input) if plan.get('pointwise26') else torch.empty_like(input)
            if plan.get('stem7_gi28'):
                _stem7_gi28(gO_r,ggW_r_opt,gI,plan,dims)
            elif plan.get('pointwise26'):
                _pointwise_gi26[lambda m:(triton.cdiv(c,m['BM'])*triton.cdiv(f*p*q,m['BN']),g)](
                    gO_r,ggW_r_opt,gI,g,f,c,h,w,o,p,q,sh,sw,precision)
            elif plan.get('gi_tail18'):
                _gi_packed18(gO_r,ggW_r_opt,gI,plan,dims)
            elif plan.get('gi_packed17'):
                _gi_packed17(gO_r,ggW_r_opt,gI,plan,dims)
            elif plan.get('gi_static_weight16'):
                _gi_weight16(gO_r,ggW_r_opt,gI,plan['gi_groups14'],dims)
            elif plan.get('gi_static15'):
                _gi_static15[lambda m:(triton.cdiv(c,m['BM'])*triton.cdiv(f*triton.cdiv(h,2)*triton.cdiv(w,2),m['BN']),g,4)](gO_r,ggW_r_opt,gI,*dims)
            elif plan.get('gi_weight14'):
                _gi_weight14(gO_r,ggW_r_opt,gI,plan['gi_groups14'],dims)
            elif plan.get('phase_gI'):
                _gi_phase12[lambda m:(triton.cdiv(c,m['BM'])*triton.cdiv(f*triton.cdiv(h,2)*triton.cdiv(w,2),m['BN']),g,4)](gO_r,ggW_r_opt,gI,*dims)
            elif plan['gI_nt']:
                forward(gO_r,ggW_r_opt,gO_r,ggW_r_opt,None,gI,plan['gI_nt'],plan['gI_groups'],2-ph,2-pw,True,False,True,precision)
            else:
                _gi[lambda m:(triton.cdiv(c,m['BM'])*triton.cdiv(f*h*w if m['BATCH_FOLD'] else h*w,m['BN']),g if m['BATCH_FOLD'] else g*f)](gO_r,ggW_r_opt,gI,*dims)
        if compute[2] and hi:
            if plan.get('stem7_gw28'):
                budget=int(workspace_limit_mb*1024**2)
                sk=workspace_split(g,f,c,o,r,s,p,q,torch.cuda.get_device_properties(input.device).multi_processor_count,budget) if split_k is None else split_k
                if sk not in (1,2,4,8,16) or g*sk>65535: raise ValueError('Invalid split_k')
                ne=g*o*c*r*s
                if sk>1 and (ne*sk*4>budget or ne*sk>=2**30): raise ValueError('Split-K exceeds workspace/indexing budget')
                gW=torch.empty_like(weight_r)
                tmp=torch.empty((g,sk,o,c,r,s),device=input.device,dtype=torch.float32) if sk>1 else gW
                _stem7_gw_static28[(triton.cdiv(c,16)*triton.cdiv(o,32),g*sk,49)](
                    ggI_opt,gO_r,tmp,*dims,sk,16,32,32,
                    num_warps=4,num_stages=2,enable_fp_fusion=False)
                if sk>1:_reduce[(triton.cdiv(ne,256),)](tmp,gW,ne,o*c*r*s,sk,256)
            elif plan.get('pointwise26'):
                budget=int(workspace_limit_mb*1024**2)
                sk=workspace_split(g,f,c,o,r,s,p,q,torch.cuda.get_device_properties(input.device).multi_processor_count,budget) if split_k is None else split_k
                if sk not in (1,2,4,8,16) or g*sk>65535: raise ValueError('Invalid split_k')
                ne=g*o*c
                if sk>1 and (ne*sk*4>budget or ne*sk>=2**30): raise ValueError('Split-K exceeds workspace/indexing budget')
                gW=torch.empty_like(weight_r)
                tmp=torch.empty((g,sk,o,c),device=input.device,dtype=torch.float32) if sk>1 else gW
                _pointwise_gw26[lambda m:(triton.cdiv(o,m['BM'])*triton.cdiv(c,m['BN']),g*sk)](
                    ggI_opt,gO_r,tmp,g,f,c,h,w,o,p,q,sh,sw,precision,sk)
                if sk>1:_reduce[(triton.cdiv(ne,256),)](tmp,gW,ne,o*c,sk,256)
            elif plan.get('gw_small26'):
                budget=int(workspace_limit_mb*1024**2)
                sk=workspace_split(g,f,c,o,r,s,p,q,torch.cuda.get_device_properties(input.device).multi_processor_count,budget) if split_k is None else split_k
                if sk not in (1,2,4,8,16) or g*sk>65535: raise ValueError('Invalid split_k')
                ne=g*o*c*r*s
                if sk>1 and (ne*sk*4>budget or ne*sk>=2**30): raise ValueError('Split-K exceeds workspace/indexing budget')
                gW=torch.empty_like(weight_r)
                tmp=torch.empty((g,sk,o,c,r,s),device=input.device,dtype=torch.float32) if sk>1 else gW
                _gw_small26[lambda m:(triton.cdiv(c,m['BM'])*triton.cdiv(o,m['BN']),g*sk,9)](
                    ggI_opt,gO_r,tmp,*dims,sk,enable_fp_fusion=False)
                if sk>1:_reduce[(triton.cdiv(ne,256),)](tmp,gW,ne,o*c*r*s,sk,256)
            elif plan['gW_split']:
                gW=torch.empty_like(weight_r)
                if v6:weight_grad(ggI_opt,gO_r,gW,plan['gW_nt'],plan['gW_groups'],plan['gW_split'],ph,pw,precision)
                else:_packed_weight_grad(ggI_opt,gO_r,gW,plan['gW_nt'],plan['gW_groups'],ph,pw,precision)
            else:
                budget=int(workspace_limit_mb*1024**2)
                sk=workspace_split(g,f,c,o,r,s,p,q,torch.cuda.get_device_properties(input.device).multi_processor_count,budget) if split_k is None else split_k
                if sk not in (1,2,4,8,16) or g*sk>65535: raise ValueError('Invalid split_k')
                ne=g*o*c*r*s
                if sk>1 and (ne*sk*4>budget or ne*sk>=2**30): raise ValueError('Split-K exceeds workspace/indexing budget')
                gW=torch.empty_like(weight_r)
                gp=_gw_plan19(g,f,c,o,p,q,sk,budget) if plan.get('gw_candidate19') else {}
                if gp.get('gw_groups19',0):
                    _gw_packed19(ggI_opt,gO_r,gW,gp,dims,sk)
                else:
                    tmp=torch.empty((g,sk,o,c,r,s),device=input.device,dtype=torch.float32) if sk>1 else gW
                    if plan.get('gw_static20'):
                        _gw_static20[lambda m:(triton.cdiv(c,m['BM'])*triton.cdiv(o,m['BN']),g*sk,9)](ggI_opt,gO_r,tmp,*dims,sk,enable_fp_fusion=False)
                    else:
                        _gw[lambda m:(r*s*triton.cdiv(c,m['BM'])*triton.cdiv(o,m['BN']) if m['FIXED_RS'] else triton.cdiv(o,m['BM'])*triton.cdiv(c*r*s,m['BN']),g*sk)](ggI_opt,gO_r,tmp,*dims,sk)
                    if sk>1: _reduce[(triton.cdiv(ne,256),)](tmp,gW,ne,o*c*r*s,sk,256)
    return ggO,gI,gW

_convolution_double_backward=convolution_double_backward


@triton.jit
def _pack_tiles_all(X,V,start,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,H:tl.constexpr,W:tl.constexpr,
                    P:tl.constexpr,Q:tl.constexpr,PH:tl.constexpr,PW:tl.constexpr,NT:tl.constexpr,
                    DOUT:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);g=tl.program_id(1)
    ci=i//NT;n=i%NT;t=start+n
    f=t//(((P+1)//2)*((Q+1)//2));y=t//((Q+1)//2)%((P+1)//2)*2;x=t%((Q+1)//2)*2
    if not DOUT:y-=PH;x-=PW
    base=((g*F+f)*C+ci)*H*W
    valid=(ci<C)&(f<F)

    if DOUT:
        d00=tl.load(X+base+(y+0)*W+x+0,valid&(y+0>=0)&(y+0<H)&(x+0>=0)&(x+0<W),0)
        d01=tl.load(X+base+(y+0)*W+x+1,valid&(y+0>=0)&(y+0<H)&(x+1>=0)&(x+1<W),0)
        d10=tl.load(X+base+(y+1)*W+x+0,valid&(y+1>=0)&(y+1<H)&(x+0>=0)&(x+0<W),0)
        d11=tl.load(X+base+(y+1)*W+x+1,valid&(y+1>=0)&(y+1<H)&(x+1>=0)&(x+1<W),0)
        a00=d00
        a10=d00+d10
        a20=d00-d10
        a30=d10
        a01=d01
        a11=d01+d11
        a21=d01-d11
        a31=d11
        v00=a00
        v01=a00+a01
        v02=a00-a01
        v03=a01
        v10=a10
        v11=a10+a11
        v12=a10-a11
        v13=a11
        v20=a20
        v21=a20+a21
        v22=a20-a21
        v23=a21
        v30=a30
        v31=a30+a31
        v32=a30-a31
        v33=a31
    else:
        d00=tl.load(X+base+(y+0)*W+x+0,valid&(y+0>=0)&(y+0<H)&(x+0>=0)&(x+0<W),0)
        d01=tl.load(X+base+(y+0)*W+x+1,valid&(y+0>=0)&(y+0<H)&(x+1>=0)&(x+1<W),0)
        d02=tl.load(X+base+(y+0)*W+x+2,valid&(y+0>=0)&(y+0<H)&(x+2>=0)&(x+2<W),0)
        d03=tl.load(X+base+(y+0)*W+x+3,valid&(y+0>=0)&(y+0<H)&(x+3>=0)&(x+3<W),0)
        d10=tl.load(X+base+(y+1)*W+x+0,valid&(y+1>=0)&(y+1<H)&(x+0>=0)&(x+0<W),0)
        d11=tl.load(X+base+(y+1)*W+x+1,valid&(y+1>=0)&(y+1<H)&(x+1>=0)&(x+1<W),0)
        d12=tl.load(X+base+(y+1)*W+x+2,valid&(y+1>=0)&(y+1<H)&(x+2>=0)&(x+2<W),0)
        d13=tl.load(X+base+(y+1)*W+x+3,valid&(y+1>=0)&(y+1<H)&(x+3>=0)&(x+3<W),0)
        d20=tl.load(X+base+(y+2)*W+x+0,valid&(y+2>=0)&(y+2<H)&(x+0>=0)&(x+0<W),0)
        d21=tl.load(X+base+(y+2)*W+x+1,valid&(y+2>=0)&(y+2<H)&(x+1>=0)&(x+1<W),0)
        d22=tl.load(X+base+(y+2)*W+x+2,valid&(y+2>=0)&(y+2<H)&(x+2>=0)&(x+2<W),0)
        d23=tl.load(X+base+(y+2)*W+x+3,valid&(y+2>=0)&(y+2<H)&(x+3>=0)&(x+3<W),0)
        d30=tl.load(X+base+(y+3)*W+x+0,valid&(y+3>=0)&(y+3<H)&(x+0>=0)&(x+0<W),0)
        d31=tl.load(X+base+(y+3)*W+x+1,valid&(y+3>=0)&(y+3<H)&(x+1>=0)&(x+1<W),0)
        d32=tl.load(X+base+(y+3)*W+x+2,valid&(y+3>=0)&(y+3<H)&(x+2>=0)&(x+2<W),0)
        d33=tl.load(X+base+(y+3)*W+x+3,valid&(y+3>=0)&(y+3<H)&(x+3>=0)&(x+3<W),0)
        a00=d00-d02
        a01=d01+d02
        a02=d02-d01
        a03=d03-d01
        a10=d10-d12
        a11=d11+d12
        a12=d12-d11
        a13=d13-d11
        a20=d20-d22
        a21=d21+d22
        a22=d22-d21
        a23=d23-d21
        a30=d30-d32
        a31=d31+d32
        a32=d32-d31
        a33=d33-d31
        v00=a00-a20
        v10=a10+a20
        v20=a20-a10
        v30=a30-a10
        v01=a01-a21
        v11=a11+a21
        v21=a21-a11
        v31=a31-a11
        v02=a02-a22
        v12=a12+a22
        v22=a22-a12
        v32=a32-a12
        v03=a03-a23
        v13=a13+a23
        v23=a23-a13
        v33=a33-a13
    tl.store(V+((0*G+g)*C+ci)*NT+n,v00,ci<C)
    tl.store(V+((1*G+g)*C+ci)*NT+n,v01,ci<C)
    tl.store(V+((2*G+g)*C+ci)*NT+n,v02,ci<C)
    tl.store(V+((3*G+g)*C+ci)*NT+n,v03,ci<C)
    tl.store(V+((4*G+g)*C+ci)*NT+n,v10,ci<C)
    tl.store(V+((5*G+g)*C+ci)*NT+n,v11,ci<C)
    tl.store(V+((6*G+g)*C+ci)*NT+n,v12,ci<C)
    tl.store(V+((7*G+g)*C+ci)*NT+n,v13,ci<C)
    tl.store(V+((8*G+g)*C+ci)*NT+n,v20,ci<C)
    tl.store(V+((9*G+g)*C+ci)*NT+n,v21,ci<C)
    tl.store(V+((10*G+g)*C+ci)*NT+n,v22,ci<C)
    tl.store(V+((11*G+g)*C+ci)*NT+n,v23,ci<C)
    tl.store(V+((12*G+g)*C+ci)*NT+n,v30,ci<C)
    tl.store(V+((13*G+g)*C+ci)*NT+n,v31,ci<C)
    tl.store(V+((14*G+g)*C+ci)*NT+n,v32,ci<C)
    tl.store(V+((15*G+g)*C+ci)*NT+n,v33,ci<C)


def _gw6_configs():
    return [triton.Config(dict(BM=m,BN=n,BK=k),num_warps=w,num_stages=2)
            for m,n,k,w in [(16,64,32,4),(32,32,32,4),(32,64,32,4),(64,32,32,4),
                           (64,64,32,4),(32,64,64,4),(64,64,16,4),(32,128,32,8)]]

@triton.autotune(configs=_gw6_configs(),key=['G','C','O','NT','SK','FIRST','PREC'],restore_value=['Sum','Corr'])
@triton.jit
def _packed_gw_acc(V,D,Sum,Corr,G:tl.constexpr,C:tl.constexpr,O:tl.constexpr,NT:tl.constexpr,
                   SK:tl.constexpr,FIRST:tl.constexpr,PREC:tl.constexpr,
                   BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0);g=tl.program_id(1);zs=tl.program_id(2);z=zs%16;part=zs//16
    oc=pid//tl.cdiv(C,BN)*BM+tl.arange(0,BM);ci=pid%tl.cdiv(C,BN)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for kb in range(tl.cdiv(NT,SK*BK)):
        k=(kb*SK+part)*BK+kk
        a=tl.load(D+((z*G+g)*O+oc[:,None])*NT+k[None,:],(oc[:,None]<O)&(k[None,:]<NT),0)
        b=tl.load(V+((z*G+g)*C+ci[None,:])*NT+k[:,None],(ci[None,:]<C)&(k[:,None]<NT),0)
        acc=tl.dot(a,b,acc,input_precision=PREC)
    off=(((g*SK+part)*16+z)*O+oc[:,None])*C+ci[None,:]
    valid=(oc[:,None]<O)&(ci[None,:]<C)
    if FIRST:
        updated=acc;corr=tl.zeros((BM,BN),tl.float32)
    else:
        old=tl.load(Sum+off,valid,0);prev=tl.load(Corr+off,valid,0)
        delta=acc-prev;updated=old+delta;corr=(updated-old)-delta
    tl.store(Sum+off,updated,valid);tl.store(Corr+off,corr,valid)


def _v6_winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget,algorithm='auto',split_k=None):
    if algorithm=='winograd_v5':
        return dict(_v5_winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget,'winograd',split_k),version=5)
    supported=dtype==torch.float32 and (r,s,sh,sw,dh,dw)==(3,3,1,1,1,1) and 0<=ph<=2 and 0<=pw<=2
    if algorithm=='winograd' and not supported:raise ValueError('Winograd requires FP32 3x3 stride=dilation=1 padding<=2')
    use=supported and algorithm=='winograd'
    sk=1 if split_k is None else split_k
    if sk not in (1,2,4,8,16):raise ValueError('Invalid split_k')
    budget=min(budget,(2**30-1)*4)
    def choose(total,fixed,per_tile):
        if not use:return 0,0,0
        # 1024 lost on A5000 in the measured conv2 workload. Keep a bounded
        # target; extra budget increases group capacity, not reduction length.
        nt=min(512 if g<=2 else 256,max(32,512//sk),max(32,triton.next_power_of_2(total)))
        while nt>=32:
            per=64*(fixed+per_tile*nt)
            if per<=budget:
                gb=min(g,4,budget//per)
                return nt,gb,per*gb
            nt//=2
        return 0,0,0
    total=f*triton.cdiv(p,2)*triton.cdiv(q,2)
    nt0,gb0,b0=choose(total,2*c*o,c+o)
    nt1,gb1,b1=choose(f*triton.cdiv(h,2)*triton.cdiv(w,2),c*o,c+o)
    ntw,gbw,bw=choose(total,2*sk*c*o,c+o)
    ntj,gbj,bj=choose(total,(2+2*sk)*c*o,c+2*o)
    return dict(version=6,ggO_nt=nt0,gI_nt=nt1,gW_split=sk if ntw else 0,gW_nt=ntw,
                ggO_groups=gb0,gI_groups=gb1,gW_groups=gbw,ggO_pack=bool(nt0),gI_pack=bool(nt1),
                joint_nt=ntj,joint_groups=gbj,scratch_bytes=max(b0,b1,bw,bj))


def _pack6(x,v,start,g,f,c,h,w,p,q,ph,pw,nt,dout=False):
    _pack_tiles_all[(triton.cdiv(c*nt,128),g)](x,v,start,g,f,c,h,w,p,q,ph,pw,nt,dout,128)


def _gw6(v,d,acc,corr,g,c,o,nt,sk,first,precision):
    _packed_gw_acc[lambda m:(triton.cdiv(o,m['BM'])*triton.cdiv(c,m['BN']),g,16*sk)](
        v,d,acc,corr,g,c,o,nt,sk,first,precision,enable_fp_fusion=False)


def _joint6(x,w,ai,aw,bias,d,out,gw,nt,group_batch,sk,ph,pw,precision):
    """Reuse transformed ggI for ggO and gW. One bounded group/tile chunk.
    ggO = conv(ggI,w) + conv(x,ggW) + ggb; gW = wgrad(ggI,gO).
    The ggI buffer can be overwritten by x after gW consumes it.
    """
    g,f,c,h,wi=x.shape;_,_,o,p,q=d.shape;total=f*triton.cdiv(p,2)*triton.cdiv(q,2)
    for base in range(0,g,group_batch):
        end=min(g,base+group_batch);gb=end-base
        u1=torch.empty((16,gb,o,c),device=x.device,dtype=torch.float32)
        _win_weight_pack[(triton.cdiv(o,16)*triton.cdiv(c,32),gb,16)](w[base:end],u1,gb,c,o,False)
        u2=None
        if aw is not None:
            u2=torch.empty_like(u1)
            _win_weight_pack[(triton.cdiv(o,16)*triton.cdiv(c,32),gb,16)](aw[base:end],u2,gb,c,o,False)
        v=torch.empty((16,gb,c,nt),device=x.device,dtype=torch.float32)
        dv=torch.empty((16,gb,o,nt),device=x.device,dtype=torch.float32)
        m=torch.empty_like(dv)
        acc=torch.empty((gb,sk,16,o,c),device=x.device,dtype=torch.float32);corr=torch.empty_like(acc)
        for start in range(0,total,nt):
            _pack6(ai[base:end],v,start,gb,f,c,h,wi,p,q,ph,pw,nt)
            _launch_packed_conv(u1,v,m,gb,c,o,nt,False,precision)
            _pack6(d[base:end],dv,start,gb,f,o,p,q,p,q,0,0,nt,True)
            _gw6(v,dv,acc,corr,gb,c,o,nt,sk,start==0,precision)
            if aw is not None:
                _pack6(x[base:end],v,start,gb,f,c,h,wi,p,q,ph,pw,nt)
                _launch_packed_conv(u2,v,m,gb,c,o,nt,True,precision)
            _win_out[(triton.cdiv(gb*o*nt,128),)](m,bias[base:end] if bias is not None else x,out[base:end],start,gb,f,o,p,q,nt,bias is not None,128)
        _win_gw_out[(triton.cdiv(gb*o*c,64),)](acc,gw[base:end],gb,o,c,sk,64)
        del u1,u2,v,dv,m,acc,corr


def _weight_grad6(ai,d,out,nt,group_batch,sk,ph,pw,precision):
    g,f,c,h,w=ai.shape;_,_,o,p,q=d.shape;total=f*triton.cdiv(p,2)*triton.cdiv(q,2)
    for base in range(0,g,group_batch):
        end=min(g,base+group_batch);gb=end-base
        v=torch.empty((16,gb,c,nt),device=ai.device,dtype=torch.float32)
        dv=torch.empty((16,gb,o,nt),device=ai.device,dtype=torch.float32)
        acc=torch.empty((gb,sk,16,o,c),device=ai.device,dtype=torch.float32);corr=torch.empty_like(acc)
        for start in range(0,total,nt):
            _pack6(ai[base:end],v,start,gb,f,c,h,w,p,q,ph,pw,nt)
            _pack6(d[base:end],dv,start,gb,f,o,p,q,p,q,0,0,nt,True)
            _gw6(v,dv,acc,corr,gb,c,o,nt,sk,start==0,precision)
        _win_gw_out[(triton.cdiv(gb*o*c,64),)](acc,out[base:end],gb,o,c,sk,64)
        del v,dv,acc,corr

def _forward6(x,wt,ai,aw,bias,out,nt,group_batch,ph,pw,hi,hw,flip,precision):
    g,f,c,h,w=x.shape;_,_,o,p,q=out.shape
    total=f*triton.cdiv(p,2)*triton.cdiv(q,2)
    for base in range(0,g,group_batch):
        end=min(g,base+group_batch);gb=end-base
        weights=[];sources=[]
        if hi:weights.append(wt[base:end]);sources.append(ai[base:end])
        if hw:weights.append(aw[base:end]);sources.append(x[base:end])
        us=[]
        for weight in weights:
            u=torch.empty((16,gb,o,c),device=x.device,dtype=torch.float32)
            _win_weight_pack[(triton.cdiv(o,16)*triton.cdiv(c,32),gb,16)](weight,u,gb,c,o,flip)
            us.append(u)
        v=torch.empty((16,gb,c,nt),device=x.device,dtype=torch.float32)
        m=torch.empty((16,gb,o,nt),device=x.device,dtype=torch.float32)
        for start in range(0,total,nt):
            for term,(source,u) in enumerate(zip(sources,us)):
                _pack6(source,v,start,gb,f,c,h,w,p,q,ph,pw,nt)
                # Separate output buffers are unnecessary: the second term adds to M.
                # Tune overwrite mode only; reuse its config for in-place addition.
                _launch_packed_conv(u,v,m,gb,c,o,nt,term!=0,precision)
            _win_out[(triton.cdiv(gb*o*nt,128),)](m,bias[base:end] if bias is not None else x,out[base:end],start,gb,f,o,p,q,nt,bias is not None,128)
        # Do not retain the previous group's workspaces during next allocation.
        del us,u,v,m



def _mm7_configs():
    return [triton.Config(dict(BM=m,BN=n,BK=k),num_warps=w,num_stages=2)
            for m,n,k,w in [(32,64,32,4),(64,64,16,4),(64,64,32,4),
                           (32,128,32,4),(64,128,16,4),(64,128,32,8),
                           (128,32,32,4),(128,64,16,4),(128,64,32,8)]]
@triton.autotune(configs=_mm7_configs(),key=['G','C','O','NT','ADD','PREC'])
@triton.jit
def _packed_conv7(U,V,M,G:tl.constexpr,C:tl.constexpr,O:tl.constexpr,NT:tl.constexpr,
                 ADD:tl.constexpr,PREC:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0);g=tl.program_id(1);z=tl.program_id(2)
    nn=tl.cdiv(NT,BN);nm=tl.cdiv(O,BM)
    first=pid//(4*nn)*4;count=tl.minimum(nm-first,4)
    oc=(first+pid%(4*nn)%count)*BM+tl.arange(0,BM)
    n=(pid%(4*nn)//count)*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for cb in range(tl.cdiv(C,BK)):
        ci=cb*BK+kk
        a=tl.load(U+((z*G+g)*O+oc[:,None])*C+ci[None,:],(oc[:,None]<O)&(ci[None,:]<C),0)
        b=tl.load(V+((z*G+g)*C+ci[:,None])*NT+n[None,:],(ci[:,None]<C)&(n[None,:]<NT),0)
        acc=tl.dot(a,b,acc,input_precision=PREC)
    off=((z*G+g)*O+oc[:,None])*NT+n[None,:];mask=(oc[:,None]<O)&(n[None,:]<NT)
    if ADD:acc+=tl.load(M+off,mask,0)
    tl.store(M+off,acc,mask)


@triton.jit
def _pack_dout7(X,V,start,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
                H:tl.constexpr,W:tl.constexpr,NT:tl.constexpr):
    pid=tl.program_id(0);g=tl.program_id(1)
    ci=(pid//tl.cdiv(NT,32)*16+tl.arange(0,16))[:,None]
    n=(pid%tl.cdiv(NT,32)*32+tl.arange(0,32))[None,:];t=start+n
    f=t//(tl.cdiv(H,2)*tl.cdiv(W,2));y=t//tl.cdiv(W,2)%tl.cdiv(H,2)*2;x=t%tl.cdiv(W,2)*2
    base=((g*F+f)*C+ci)*H*W
    valid=(ci<C)&(f<F)&(n<NT)
    d00=tl.load(X+base+y*W+x,valid&(y<H)&(x<W),0)
    d01=tl.load(X+base+y*W+x+1,valid&(y<H)&(x+1<W),0)
    d10=tl.load(X+base+(y+1)*W+x,valid&(y+1<H)&(x<W),0)
    d11=tl.load(X+base+(y+1)*W+x+1,valid&(y+1<H)&(x+1<W),0)
    a00=d00
    a10=d00+d10
    a20=d00-d10
    a30=d10
    a01=d01
    a11=d01+d11
    a21=d01-d11
    a31=d11
    v00=a00
    v01=a00+a01
    v02=a00-a01
    v03=a01
    v10=a10
    v11=a10+a11
    v12=a10-a11
    v13=a11
    v20=a20
    v21=a20+a21
    v22=a20-a21
    v23=a21
    v30=a30
    v31=a30+a31
    v32=a30-a31
    v33=a31
    tl.store(V+((0*G+g)*NT+n)*C+ci,v00,(ci<C)&(n<NT))
    tl.store(V+((1*G+g)*NT+n)*C+ci,v01,(ci<C)&(n<NT))
    tl.store(V+((2*G+g)*NT+n)*C+ci,v02,(ci<C)&(n<NT))
    tl.store(V+((3*G+g)*NT+n)*C+ci,v03,(ci<C)&(n<NT))
    tl.store(V+((4*G+g)*NT+n)*C+ci,v10,(ci<C)&(n<NT))
    tl.store(V+((5*G+g)*NT+n)*C+ci,v11,(ci<C)&(n<NT))
    tl.store(V+((6*G+g)*NT+n)*C+ci,v12,(ci<C)&(n<NT))
    tl.store(V+((7*G+g)*NT+n)*C+ci,v13,(ci<C)&(n<NT))
    tl.store(V+((8*G+g)*NT+n)*C+ci,v20,(ci<C)&(n<NT))
    tl.store(V+((9*G+g)*NT+n)*C+ci,v21,(ci<C)&(n<NT))
    tl.store(V+((10*G+g)*NT+n)*C+ci,v22,(ci<C)&(n<NT))
    tl.store(V+((11*G+g)*NT+n)*C+ci,v23,(ci<C)&(n<NT))
    tl.store(V+((12*G+g)*NT+n)*C+ci,v30,(ci<C)&(n<NT))
    tl.store(V+((13*G+g)*NT+n)*C+ci,v31,(ci<C)&(n<NT))
    tl.store(V+((14*G+g)*NT+n)*C+ci,v32,(ci<C)&(n<NT))
    tl.store(V+((15*G+g)*NT+n)*C+ci,v33,(ci<C)&(n<NT))
@triton.autotune(configs=_mm7_configs(),key=['G','C','O','NT','SK','FIRST','PREC'],restore_value=['Sum','Corr'])
@triton.jit
def _packed_gw7(V,D,Sum,Corr,G:tl.constexpr,C:tl.constexpr,O:tl.constexpr,NT:tl.constexpr,
                   SK:tl.constexpr,FIRST:tl.constexpr,PREC:tl.constexpr,
                   BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0);g=tl.program_id(1);zs=tl.program_id(2);z=zs%16;part=zs//16
    ci=pid//tl.cdiv(O,BN)*BM+tl.arange(0,BM);oc=pid%tl.cdiv(O,BN)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for kb in range(tl.cdiv(NT,SK*BK)):
        k=(kb*SK+part)*BK+kk
        a=tl.load(V+((z*G+g)*C+ci[:,None])*NT+k[None,:],(ci[:,None]<C)&(k[None,:]<NT),0)
        b=tl.load(D+((z*G+g)*NT+k[:,None])*O+oc[None,:],(oc[None,:]<O)&(k[:,None]<NT),0)
        acc=tl.dot(a,b,acc,input_precision=PREC)
    off=(((g*SK+part)*16+z)*O+oc[None,:])*C+ci[:,None]
    valid=(oc[None,:]<O)&(ci[:,None]<C)
    if FIRST:
        updated=acc;corr=tl.zeros((BM,BN),tl.float32)
    else:
        old=tl.load(Sum+off,valid,0);prev=tl.load(Corr+off,valid,0)
        delta=acc-prev;updated=old+delta;corr=(updated-old)-delta
    tl.store(Sum+off,updated,valid);tl.store(Corr+off,corr,valid)


def _pack7(x,v,start,g,f,c,h,w,p,q,ph,pw,nt,dout=False):
    if dout:
        _pack_dout7[(triton.cdiv(c,16)*triton.cdiv(nt,32),g)](x,v,start,g,f,c,h,w,nt)
    else:_pack6(x,v,start,g,f,c,h,w,p,q,ph,pw,nt)


def _gw7(v,d,acc,corr,g,c,o,nt,sk,first,precision):
    _packed_gw7[lambda m:(triton.cdiv(c,m['BM'])*triton.cdiv(o,m['BN']),g,16*sk)](
        v,d,acc,corr,g,c,o,nt,sk,first,precision,enable_fp_fusion=False)

def _launch_conv7(u,v,m,g,c,o,nt,add,precision):
    grid=lambda meta:(triton.cdiv(o,meta['BM'])*triton.cdiv(nt,meta['BN']),g,16)
    if not add:
        _packed_conv7[grid](u,v,m,g,c,o,nt,False,precision)
    else:
        # Avoid in-place autotune corruption: use the already measured overwrite
        # configuration. No input/output values are retained between API calls.
        cfg=_packed_conv7.best_config
        _packed_conv7.fn[grid](u,v,m,g,c,o,nt,True,precision,**cfg.kwargs,num_warps=cfg.num_warps,num_stages=cfg.num_stages)


def _joint7(x,w,ai,aw,bias,d,out,gw,nt,group_batch,sk,ph,pw,precision):
    """Reuse transformed ggI for ggO and gW. One bounded group/tile chunk.
    ggO = conv(ggI,w) + conv(x,ggW) + ggb; gW = wgrad(ggI,gO).
    The ggI buffer can be overwritten by x after gW consumes it.
    """
    g,f,c,h,wi=x.shape;_,_,o,p,q=d.shape;total=f*triton.cdiv(p,2)*triton.cdiv(q,2)
    for base in range(0,g,group_batch):
        end=min(g,base+group_batch);gb=end-base
        u1=torch.empty((16,gb,o,c),device=x.device,dtype=torch.float32)
        _win_weight_pack[(triton.cdiv(o,16)*triton.cdiv(c,32),gb,16)](w[base:end],u1,gb,c,o,False)
        u2=None
        if aw is not None:
            u2=torch.empty_like(u1)
            _win_weight_pack[(triton.cdiv(o,16)*triton.cdiv(c,32),gb,16)](aw[base:end],u2,gb,c,o,False)
        v=torch.empty((16,gb,c,nt),device=x.device,dtype=torch.float32)
        dv=torch.empty((16,gb,nt,o),device=x.device,dtype=torch.float32)
        m=torch.empty((16,gb,o,nt),device=x.device,dtype=torch.float32)
        acc=torch.empty((gb,sk,16,o,c),device=x.device,dtype=torch.float32);corr=torch.empty_like(acc)
        for start in range(0,total,nt):
            _pack7(ai[base:end],v,start,gb,f,c,h,wi,p,q,ph,pw,nt)
            _launch_conv7(u1,v,m,gb,c,o,nt,False,precision)
            _pack7(d[base:end],dv,start,gb,f,o,p,q,p,q,0,0,nt,True)
            _gw7(v,dv,acc,corr,gb,c,o,nt,sk,start==0,precision)
            if aw is not None:
                _pack7(x[base:end],v,start,gb,f,c,h,wi,p,q,ph,pw,nt)
                _launch_conv7(u2,v,m,gb,c,o,nt,True,precision)
            _win_out[(triton.cdiv(gb*o*nt,128),)](m,bias[base:end] if bias is not None else x,out[base:end],start,gb,f,o,p,q,nt,bias is not None,128)
        _win_gw_out[(triton.cdiv(gb*o*c,64),)](acc,gw[base:end],gb,o,c,sk,64)
        del u1,u2,v,dv,m,acc,corr


def _weight_grad7(ai,d,out,nt,group_batch,sk,ph,pw,precision):
    g,f,c,h,w=ai.shape;_,_,o,p,q=d.shape;total=f*triton.cdiv(p,2)*triton.cdiv(q,2)
    for base in range(0,g,group_batch):
        end=min(g,base+group_batch);gb=end-base
        v=torch.empty((16,gb,c,nt),device=ai.device,dtype=torch.float32)
        dv=torch.empty((16,gb,nt,o),device=ai.device,dtype=torch.float32)
        acc=torch.empty((gb,sk,16,o,c),device=ai.device,dtype=torch.float32);corr=torch.empty_like(acc)
        for start in range(0,total,nt):
            _pack7(ai[base:end],v,start,gb,f,c,h,w,p,q,ph,pw,nt)
            _pack7(d[base:end],dv,start,gb,f,o,p,q,p,q,0,0,nt,True)
            _gw7(v,dv,acc,corr,gb,c,o,nt,sk,start==0,precision)
        _win_gw_out[(triton.cdiv(gb*o*c,64),)](acc,out[base:end],gb,o,c,sk,64)
        del v,dv,acc,corr

def _forward7(x,wt,ai,aw,bias,out,nt,group_batch,ph,pw,hi,hw,flip,precision):
    g,f,c,h,w=x.shape;_,_,o,p,q=out.shape
    total=f*triton.cdiv(p,2)*triton.cdiv(q,2)
    for base in range(0,g,group_batch):
        end=min(g,base+group_batch);gb=end-base
        weights=[];sources=[]
        if hi:weights.append(wt[base:end]);sources.append(ai[base:end])
        if hw:weights.append(aw[base:end]);sources.append(x[base:end])
        us=[]
        for weight in weights:
            u=torch.empty((16,gb,o,c),device=x.device,dtype=torch.float32)
            _win_weight_pack[(triton.cdiv(o,16)*triton.cdiv(c,32),gb,16)](weight,u,gb,c,o,flip)
            us.append(u)
        v=torch.empty((16,gb,c,nt),device=x.device,dtype=torch.float32)
        m=torch.empty((16,gb,o,nt),device=x.device,dtype=torch.float32)
        for start in range(0,total,nt):
            for term,(source,u) in enumerate(zip(sources,us)):
                _pack7(source,v,start,gb,f,c,h,w,p,q,ph,pw,nt)
                # Separate output buffers are unnecessary: the second term adds to M.
                # Tune overwrite mode only; reuse its config for in-place addition.
                _launch_conv7(u,v,m,gb,c,o,nt,term!=0,precision)
            _win_out[(triton.cdiv(gb*o*nt,128),)](m,bias[base:end] if bias is not None else x,out[base:end],start,gb,f,o,p,q,nt,bias is not None,128)
        # Do not retain the previous group's workspaces during next allocation.
        del us,u,v,m


def _v7_winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget,algorithm='auto',split_k=None):
    args=(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget)
    if algorithm=='winograd_v6':return _v6_winograd_plan(*args,'winograd',split_k)
    if algorithm not in ('winograd','winograd_joint','winograd_separate'):
        return _v6_winograd_plan(*args,algorithm,split_k)
    plan=_v6_winograd_plan(*args,'winograd',split_k)
    plan['version']=7
    # Explicit, provisional shape/workspace policy. Not a measured global optimum.
    # Share ggI only when it does not increase the number of group chunks.
    joint=bool(plan['joint_nt'])
    if algorithm=='winograd_separate':joint=False
    elif algorithm=='winograd':
        joint=joint and triton.cdiv(g,max(1,plan['joint_groups']))<=triton.cdiv(g,max(1,plan['gW_groups']))
    if not joint:plan['joint_nt']=0;plan['joint_groups']=0
    plan['schedule']='joint' if joint else 'separate'
    return plan


def _dual8_configs():
    return [triton.Config(dict(BM=m,BN=n,BK=k),num_warps=w,num_stages=2)
            for m,n,k,w in [(32,32,32,4),(32,64,32,4),(64,32,32,4),
                           (64,64,16,4),(64,64,32,4),(32,128,32,8)]]


@triton.autotune(configs=_dual8_configs(),key=['G','C','O','NT','PREC'])
@triton.jit
def _packed_dual8(U,V,U2,V2,M,G:tl.constexpr,C:tl.constexpr,O:tl.constexpr,NT:tl.constexpr,
                  PREC:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0);g=tl.program_id(1);z=tl.program_id(2)
    nn=tl.cdiv(NT,BN);nm=tl.cdiv(O,BM)
    first=pid//(4*nn)*4;count=tl.minimum(nm-first,4)
    oc=(first+pid%(4*nn)%count)*BM+tl.arange(0,BM)
    n=(pid%(4*nn)//count)*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
    a1=tl.zeros((BM,BN),tl.float32);a2=tl.zeros((BM,BN),tl.float32)
    # Separate accumulators preserve the two contraction sums before addition.
    for cb in range(tl.cdiv(C,BK)):
        ci=cb*BK+kk
        up=((z*G+g)*O+oc[:,None])*C+ci[None,:]
        vp=((z*G+g)*C+ci[:,None])*NT+n[None,:]
        um=(oc[:,None]<O)&(ci[None,:]<C);vm=(ci[:,None]<C)&(n[None,:]<NT)
        a=tl.load(U+up,um,0);b=tl.load(V+vp,vm,0)
        a1=tl.dot(a,b,a1,input_precision=PREC)
        a=tl.load(U2+up,um,0);b=tl.load(V2+vp,vm,0)
        a2=tl.dot(a,b,a2,input_precision=PREC)
    off=((z*G+g)*O+oc[:,None])*NT+n[None,:]
    tl.store(M+off,a1+a2,(oc[:,None]<O)&(n[None,:]<NT))


def _joint8(x,w,ai,aw,bias,d,out,gw,nt,group_batch,sk,ph,pw,precision):
    g,f,c,h,wi=x.shape;_,_,o,p,q=d.shape;total=f*triton.cdiv(p,2)*triton.cdiv(q,2)
    for base in range(0,g,group_batch):
        end=min(g,base+group_batch);gb=end-base
        u1=torch.empty((16,gb,o,c),device=x.device,dtype=torch.float32)
        _win_weight_pack[(triton.cdiv(o,16)*triton.cdiv(c,32),gb,16)](w[base:end],u1,gb,c,o,False)
        u2=None
        if aw is not None:
            u2=torch.empty_like(u1)
            _win_weight_pack[(triton.cdiv(o,16)*triton.cdiv(c,32),gb,16)](aw[base:end],u2,gb,c,o,False)
        v=torch.empty((16,gb,c,nt),device=x.device,dtype=torch.float32)
        # gW consumes D before this same allocation is overwritten by transformed X.
        # Each use packs densely from offset zero; max(C,O) is capacity, not stride.
        scratch=torch.empty(16*gb*max(c,o)*nt,device=x.device,dtype=torch.float32)
        m=torch.empty((16,gb,o,nt),device=x.device,dtype=torch.float32)
        acc=torch.empty((gb,sk,16,o,c),device=x.device,dtype=torch.float32);corr=torch.empty_like(acc)
        for start in range(0,total,nt):
            _pack7(ai[base:end],v,start,gb,f,c,h,wi,p,q,ph,pw,nt)
            _pack7(d[base:end],scratch,start,gb,f,o,p,q,p,q,0,0,nt,True)
            _gw7(v,scratch,acc,corr,gb,c,o,nt,sk,start==0,precision)
            if aw is not None:
                _pack7(x[base:end],scratch,start,gb,f,c,h,wi,p,q,ph,pw,nt)
                _packed_dual8[lambda meta:(triton.cdiv(o,meta['BM'])*triton.cdiv(nt,meta['BN']),gb,16)](
                    u1,v,u2,scratch,m,gb,c,o,nt,precision)
            else:_launch_conv7(u1,v,m,gb,c,o,nt,False,precision)
            _win_out[(triton.cdiv(gb*o*nt,128),)](m,bias[base:end] if bias is not None else x,out[base:end],start,gb,f,o,p,q,nt,bias is not None,128)
        _win_gw_out[(triton.cdiv(gb*o*c,64),)](acc,gw[base:end],gb,o,c,sk,64)
        del u1,u2,v,scratch,m,acc,corr


def _v8_winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget,algorithm='auto',split_k=None):
    args=(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget)
    if algorithm=='winograd_v7':return _v7_winograd_plan(*args,'winograd',split_k)
    if algorithm not in ('winograd','winograd_joint','winograd_separate'):
        return _v7_winograd_plan(*args,algorithm,split_k)
    # Reuse validation only. The new execution planner has no fixed group cap.
    _v6_winograd_plan(*args,'winograd',split_k)
    sk=1 if split_k is None else split_k
    budget=min(budget,(2**30-1)*4)
    def choose(total,fixed,per_tile):
        nt=min(512,max(32,512//sk),max(32,triton.next_power_of_2(total)))
        while nt>=32:
            per=64*(fixed+per_tile*nt)
            if per<=budget:
                cap=min(g,budget//per)
                # Reduce unused capacity when fewer groups suffice for the same chunk count.
                gb=triton.cdiv(g,triton.cdiv(g,cap))
                return nt,gb,per*gb
            nt//=2
        return 0,0,0
    total=f*triton.cdiv(p,2)*triton.cdiv(q,2)
    nt0,gb0,b0=choose(total,2*c*o,c+o)
    nt1,gb1,b1=choose(f*triton.cdiv(h,2)*triton.cdiv(w,2),c*o,c+o)
    ntw,gbw,bw=choose(total,2*sk*c*o,c+o)
    ntj,gbj,bj=choose(total,(2+2*sk)*c*o,c+o+max(c,o))
    if algorithm=='winograd_separate':ntj=gbj=bj=0
    return dict(version=8,ggO_nt=nt0,gI_nt=nt1,gW_split=sk if ntw else 0,gW_nt=ntw,
                ggO_groups=gb0,gI_groups=gb1,gW_groups=gbw,ggO_pack=bool(nt0),gI_pack=bool(nt1),
                joint_nt=ntj,joint_groups=gbj,scratch_bytes=max(b0,b1,bw,bj),
                schedule='joint_dual' if ntj else 'separate',group_cap='workspace',dual_ggO=bool(ntj))


@triton.jit
def _pack_pair9(X,AI,VX,VI,start,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
                H:tl.constexpr,W:tl.constexpr,P:tl.constexpr,Q:tl.constexpr,
                PH:tl.constexpr,PW:tl.constexpr,NT:tl.constexpr):
    # Independent CTAs, one launch. No shared state or cross-group permutation.
    if tl.program_id(2)==0:
        _pack_tiles_all(AI,VI,start,G,F,C,H,W,P,Q,PH,PW,NT,False,128)
    else:
        _pack_tiles_all(X,VX,start,G,F,C,H,W,P,Q,PH,PW,NT,False,128)


@triton.jit
def _weight_pair9(W,AW,U,AU,G:tl.constexpr,C:tl.constexpr,O:tl.constexpr):
    pid=tl.program_id(0);g=tl.program_id(1);plane=tl.program_id(2);z=plane%16
    oc=pid//tl.cdiv(C,32)*16+tl.arange(0,16)
    ci=pid%tl.cdiv(C,32)*32+tl.arange(0,32)
    if plane<16:
        v=_win_weight(W,g,oc[:,None],ci[None,:],z//4,z%4,C,O,False)
        tl.store(U+((z*G+g)*O+oc[:,None])*C+ci[None,:],v,(oc[:,None]<O)&(ci[None,:]<C))
    else:
        v=_win_weight(AW,g,oc[:,None],ci[None,:],z//4,z%4,C,O,False)
        tl.store(AU+((z*G+g)*O+oc[:,None])*C+ci[None,:],v,(oc[:,None]<O)&(ci[None,:]<C))


def _joint9(x,w,ai,aw,bias,d,out,gw,nt,group_batch,sk,ph,pw,precision):
    """Large spatial chunks; each gW partial reduces <=512 tiles by default.

    V(ggI) serves both ggO and gW. Scratch first holds transformed gO in
    [16,G,NT,O], then ggO's GEMM result in [16,G,O,NT]. The two lifetimes do
    not overlap. VX holds transformed input. All buffers die within this call.
    """
    g,f,c,h,wi=x.shape;_,_,o,p,q=d.shape
    total=f*triton.cdiv(p,2)*triton.cdiv(q,2)
    for base in range(0,g,group_batch):
        end=min(g,base+group_batch);gb=end-base
        u=torch.empty((16,gb,o,c),device=x.device,dtype=torch.float32)
        au=torch.empty_like(u) if aw is not None else None
        if aw is not None:
            _weight_pair9[(triton.cdiv(o,16)*triton.cdiv(c,32),gb,32)](w[base:end],aw[base:end],u,au,gb,c,o)
        else:_win_weight_pack[(triton.cdiv(o,16)*triton.cdiv(c,32),gb,16)](w[base:end],u,gb,c,o,False)
        v=torch.empty((16,gb,c,nt),device=x.device,dtype=torch.float32)
        vx=torch.empty_like(v) if aw is not None else None
        scratch=torch.empty(16*gb*o*nt,device=x.device,dtype=torch.float32)
        acc=torch.empty((gb,sk,16,o,c),device=x.device,dtype=torch.float32)
        corr=torch.empty_like(acc)
        # Hold group slices once, rather than reconstructing them each tile chunk.
        xx=x[base:end];aa=ai[base:end];dd=d[base:end];yy=out[base:end]
        bb=bias[base:end] if bias is not None else xx
        for start in range(0,total,nt):
            if aw is not None:
                _pack_pair9[(triton.cdiv(c*nt,128),gb,2)](xx,aa,vx,v,start,gb,f,c,h,wi,p,q,ph,pw,nt)
            else:_pack7(aa,v,start,gb,f,c,h,wi,p,q,ph,pw,nt)
            _pack7(dd,scratch,start,gb,f,o,p,q,p,q,0,0,nt,True)
            _gw7(v,scratch,acc,corr,gb,c,o,nt,sk,start==0,precision)
            if aw is not None:
                _packed_dual8[lambda meta:(triton.cdiv(o,meta['BM'])*triton.cdiv(nt,meta['BN']),gb,16)](
                    u,v,au,vx,scratch,gb,c,o,nt,precision)
            else:_launch_conv7(u,v,scratch,gb,c,o,nt,False,precision)
            _win_out[(triton.cdiv(gb*o*nt,128),)](scratch,bb,yy,start,gb,f,o,p,q,nt,bias is not None,128)
        _win_gw_out[(triton.cdiv(gb*o*c,64),)](acc,gw[base:end],gb,o,c,sk,64)
        del u,au,v,vx,scratch,acc,corr


def _v9_winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget,algorithm='auto',split_k=None):
    args=(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget)
    if algorithm=='winograd_v8':return _v8_winograd_plan(*args,'winograd',split_k)
    if algorithm!='winograd_large':return _v8_winograd_plan(*args,algorithm,split_k)
    # Validate parameters with the existing path. Explicit opt-in only; no claim
    # that this experimental schedule wins on every supported shape.
    _v6_winograd_plan(*args,'winograd',split_k)
    budget=min(int(budget),(2**30-1)*4)
    total=f*triton.cdiv(p,2)*triton.cdiv(q,2)
    total_i=f*triton.cdiv(h,2)*triton.cdiv(w,2)
    def pick(total,kind):
        target=4096 if kind in ('gI','ggO') else 2048
        nt=min(target,max(32,triton.next_power_of_2(total)))
        while nt>=32:
            # Crucially, increasing split-K does NOT shrink the packed tile chunk.
            # The old planner shrank NT as SK increased, multiplying host loops.
            sk=split_k if split_k is not None else max(1,triton.cdiv(nt,512))
            if kind=='joint':fixed=(2+2*sk)*c*o;per_tile=2*c+o
            elif kind=='gW':fixed=2*sk*c*o;per_tile=c+o
            elif kind=='ggO':fixed=2*c*o;per_tile=c+o
            else:fixed=c*o;per_tile=c+o
            per=64*(fixed+per_tile*nt)
            if per<=budget:
                cap=min(g,budget//per)
                gb=triton.cdiv(g,triton.cdiv(g,cap))
                return nt,gb,per*gb,sk
            nt//=2
        return 0,0,0,0
    ntj,gbj,bj,skj=pick(total,'joint')
    ntw,gbw,bw,skw=pick(total,'gW')
    nt0,gb0,b0,_=pick(total,'ggO')
    nt1,gb1,b1,_=pick(total_i,'gI')
    # Joint and isolated gW must use the same split to keep plan/execution aligned.
    sk=skj if ntj else skw
    if ntw and sk!=skw:
        per=64*(2*sk*c*o+(c+o)*ntw)
        if per>budget:ntw=gbw=bw=0
        else:gbw=min(g,budget//per);bw=gbw*per
    return dict(version=9,ggO_nt=nt0,gI_nt=nt1,gW_split=sk if ntw or ntj else 0,
                gW_nt=ntw,ggO_groups=gb0,gI_groups=gb1,gW_groups=gbw,
                ggO_pack=bool(nt0),gI_pack=bool(nt1),joint_nt=ntj,joint_groups=gbj,
                scratch_bytes=max(b0,b1,bw,bj),schedule='large_joint' if ntj else 'large_separate',
                group_cap='workspace',dual_ggO=bool(ntj),
                joint_tile_chunks=triton.cdiv(total,ntj) if ntj else 0,
                joint_group_chunks=triton.cdiv(g,gbj) if gbj else 0,
                gW_tiles_per_partial=triton.cdiv(ntj or ntw,sk) if sk else 0)


def _tile_chunks10(total,capacity):
    """Power-of-two chunks, at most 31 padded tiles at the end.
    Keep the accumulation split/stride fixed even when the packed NT changes.
    """
    if total<1 or capacity<32 or capacity&(capacity-1):raise ValueError('Invalid tile range')
    start=0
    while start<total:
        remaining=total-start
        n=min(capacity,max(32,1<<(remaining.bit_length()-1)))
        yield start,n
        start+=n


def _joint10(x,w,ai,aw,bias,d,out,gw,nt,group_batch,sk,ph,pw,precision,chunker=_tile_chunks10,weight_all=False):
    """Large spatial chunks; each gW partial reduces <=512 tiles by default.

    V(ggI) serves both ggO and gW. Scratch first holds transformed gO in
    [16,G,NT,O], then ggO's GEMM result in [16,G,O,NT]. The two lifetimes do
    not overlap. VX holds transformed input. All buffers die within this call.
    """
    g,f,c,h,wi=x.shape;_,_,o,p,q=d.shape
    total=f*triton.cdiv(p,2)*triton.cdiv(q,2)
    capacity=nt
    for base in range(0,g,group_batch):
        nt=capacity
        end=min(g,base+group_batch);gb=end-base
        u=torch.empty((16,gb,o,c),device=x.device,dtype=torch.float32)
        au=torch.empty_like(u) if aw is not None else None
        if weight_all:
            _win_weight_all14[(triton.cdiv(o*c,128),gb,2 if aw is not None else 1)](
                w[base:end],aw[base:end] if aw is not None else w[base:end],u,au if au is not None else u,gb,c,o,False,128)
        elif aw is not None:
            _weight_pair9[(triton.cdiv(o,16)*triton.cdiv(c,32),gb,32)](w[base:end],aw[base:end],u,au,gb,c,o)
        else:_win_weight_pack[(triton.cdiv(o,16)*triton.cdiv(c,32),gb,16)](w[base:end],u,gb,c,o,False)
        v=torch.empty((16,gb,c,nt),device=x.device,dtype=torch.float32)
        vx=torch.empty_like(v) if aw is not None else None
        scratch=torch.empty(16*gb*o*nt,device=x.device,dtype=torch.float32)
        acc=torch.empty((gb,sk,16,o,c),device=x.device,dtype=torch.float32)
        corr=torch.empty_like(acc)
        # Hold group slices once, rather than reconstructing them each tile chunk.
        xx=x[base:end];aa=ai[base:end];dd=d[base:end];yy=out[base:end]
        bb=bias[base:end] if bias is not None else xx
        for start,nt in chunker(total,capacity):
            if aw is not None:
                _pack_pair9[(triton.cdiv(c*nt,128),gb,2)](xx,aa,vx,v,start,gb,f,c,h,wi,p,q,ph,pw,nt)
            else:_pack7(aa,v,start,gb,f,c,h,wi,p,q,ph,pw,nt)
            _pack7(dd,scratch,start,gb,f,o,p,q,p,q,0,0,nt,True)
            _gw7(v,scratch,acc,corr,gb,c,o,nt,sk,start==0,precision)
            if aw is not None:
                _packed_dual8[lambda meta:(triton.cdiv(o,meta['BM'])*triton.cdiv(nt,meta['BN']),gb,16)](
                    u,v,au,vx,scratch,gb,c,o,nt,precision)
            else:_launch_conv7(u,v,scratch,gb,c,o,nt,False,precision)
            _win_out[(triton.cdiv(gb*o*nt,128),)](scratch,bb,yy,start,gb,f,o,p,q,nt,bias is not None,128)
        _win_gw_out[(triton.cdiv(gb*o*c,64),)](acc,gw[base:end],gb,o,c,sk,64)
        del u,au,v,vx,scratch,acc,corr


def _forward10(x,wt,ai,aw,bias,out,nt,group_batch,ph,pw,hi,hw,flip,precision,chunker=_tile_chunks10,weight_all=False):
    g,f,c,h,w=x.shape;_,_,o,p,q=out.shape
    total=f*triton.cdiv(p,2)*triton.cdiv(q,2)
    capacity=nt
    for base in range(0,g,group_batch):
        nt=capacity
        end=min(g,base+group_batch);gb=end-base
        weights=[];sources=[]
        if hi:weights.append(wt[base:end]);sources.append(ai[base:end])
        if hw:weights.append(aw[base:end]);sources.append(x[base:end])
        us=[]
        for weight in weights:
            u=torch.empty((16,gb,o,c),device=x.device,dtype=torch.float32)
            if weight_all:
                _win_weight_all14[(triton.cdiv(o*c,128),gb,1)](weight,weight,u,u,gb,c,o,flip,128)
            else:_win_weight_pack[(triton.cdiv(o,16)*triton.cdiv(c,32),gb,16)](weight,u,gb,c,o,flip)
            us.append(u)
        v=torch.empty((16,gb,c,nt),device=x.device,dtype=torch.float32)
        m=torch.empty((16,gb,o,nt),device=x.device,dtype=torch.float32)
        for start,nt in chunker(total,capacity):
            for term,(source,u) in enumerate(zip(sources,us)):
                _pack7(source,v,start,gb,f,c,h,w,p,q,ph,pw,nt)
                # Separate output buffers are unnecessary: the second term adds to M.
                # Tune overwrite mode only; reuse its config for in-place addition.
                _launch_conv7(u,v,m,gb,c,o,nt,term!=0,precision)
            _win_out[(triton.cdiv(gb*o*nt,128),)](m,bias[base:end] if bias is not None else x,out[base:end],start,gb,f,o,p,q,nt,bias is not None,128)
        # Do not retain the previous group's workspaces during next allocation.
        del us,u,v,m


def _weight_grad10(ai,d,out,nt,group_batch,sk,ph,pw,precision,chunker=_tile_chunks10):
    g,f,c,h,w=ai.shape;_,_,o,p,q=d.shape;total=f*triton.cdiv(p,2)*triton.cdiv(q,2)
    capacity=nt
    for base in range(0,g,group_batch):
        nt=capacity
        end=min(g,base+group_batch);gb=end-base
        v=torch.empty((16,gb,c,nt),device=ai.device,dtype=torch.float32)
        dv=torch.empty((16,gb,nt,o),device=ai.device,dtype=torch.float32)
        acc=torch.empty((gb,sk,16,o,c),device=ai.device,dtype=torch.float32);corr=torch.empty_like(acc)
        for start,nt in chunker(total,capacity):
            _pack7(ai[base:end],v,start,gb,f,c,h,w,p,q,ph,pw,nt)
            _pack7(d[base:end],dv,start,gb,f,o,p,q,p,q,0,0,nt,True)
            _gw7(v,dv,acc,corr,gb,c,o,nt,sk,start==0,precision)
        _win_gw_out[(triton.cdiv(gb*o*c,64),)](acc,out[base:end],gb,o,c,sk,64)
        del v,dv,acc,corr


def _tile_chunks13(total,capacity):
    """One aligned remainder instead of several power-of-two remainders."""
    if total<1 or capacity<32 or capacity%32:raise ValueError('Invalid aligned tile range')
    for start in range(0,total,capacity):
        yield start,min(capacity,triton.cdiv(total-start,32)*32)


def _exact_plan13(plan,g,f,c,o,h,w,p,q,budget):
    plan=dict(plan)
    sk=plan['gW_split']
    sizes=[]
    for name,total in [('joint',f*triton.cdiv(p,2)*triton.cdiv(q,2)),
                       ('ggO',f*triton.cdiv(p,2)*triton.cdiv(q,2)),
                       ('gW',f*triton.cdiv(p,2)*triton.cdiv(q,2)),
                       ('gI',f*triton.cdiv(h,2)*triton.cdiv(w,2))]:
        capacity=plan[name+'_nt']
        if not capacity:continue
        nt=min(capacity,triton.cdiv(total,32)*32)
        fixed={'joint':(2+2*sk)*c*o,'gW':2*sk*c*o,'ggO':2*c*o,'gI':c*o}[name]
        per=64*(fixed+(2*c+o if name=='joint' else c+o)*nt)
        cap=min(g,budget//per)
        if cap<1:raise AssertionError('Exact chunks must not increase workspace')
        gb=triton.cdiv(g,triton.cdiv(g,cap))
        plan[name+'_nt']=nt;plan[name+'_groups']=gb
        chunks=list(_tile_chunks13(total,nt))
        plan[name+'_tile_chunks']=len(chunks)
        plan[name+'_group_chunks']=triton.cdiv(g,gb)
        plan[name+'_tail_sizes']=[n for _,n in chunks if n!=nt]
        plan[name+'_padded_tiles']=sum(n for _,n in chunks)-total
        sizes.append(per*gb)
    plan.update(exact_chunks=True,scratch_bytes=max(sizes,default=0),schedule='aligned_remainder',
                tail_policy='single_remainder_multiple32',
                gW_tiles_per_partial=triton.cdiv(plan['joint_nt'] or plan['gW_nt'],sk) if sk else 0)
    return plan


def winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget,algorithm='auto',split_k=None):
    if algorithm in ('stem7_gi','stem7_gi_ggo','stem7_gi_gw','stem7_full'):
        supported=(dtype==torch.float32 and c==3 and
                   (r,s,sh,sw,ph,pw,dh,dw)==(7,7,2,2,3,3,1,1))
        if not supported:
            raise ValueError('v29 stem7 candidates require FP32 RGB 7x7 stride2 padding3 dilation1')
        plan=winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,
                           dtype,budget,'implicit',split_k)
        cap=min(max(0,int(budget)),128*1024**2)

        def choose(total,per_column,fixed=0,max_nt=65536):
            total=((int(total)+31)//32)*32
            top=(min(max_nt,total)//32)*32
            candidates={top}
            nt=32
            while nt<=top:
                candidates.add(nt);nt*=2
            best=None
            for nt in sorted(candidates):
                per=int(fixed)+int(per_column)*nt
                if per<=cap:
                    gb=min(g,cap//per)
                    if gb:
                        chunks=(total+nt-1)//nt;waves=(g+gb-1)//gb
                        # Minimize host launch rounds first, then padded GEMM
                        # columns. Prefer wider group waves on exact ties.
                        score=(chunks*waves,chunks*nt,-gb,-nt)
                        if best is None or score<best[0]:
                            best=(score,int(nt),int(gb),int(per*gb))
            return (0,0,0) if best is None else best[1:]

        # ggO holds one packed im2col matrix per active tensor term. The plan
        # uses the worst-case two terms so optional-input calls never exceed it.
        ggo_nt,ggo_g,ggo_scratch=choose(f*p*q,2*4*c*49)

        # Every dgrad parity has K=O*NR*NS; the largest 7x7 phase is 4x4.
        # The phase weight matrix and packed gO matrix coexist.
        kmax=o*16
        gi_total=f*((h+1)//2)*((w+1)//2)
        gi_nt,gi_g,gi_scratch=choose(gi_total,4*kmax,4*c*kmax)

        want_gi=algorithm in ('stem7_gi','stem7_gi_ggo','stem7_gi_gw','stem7_full')
        want_gw=algorithm in ('stem7_gi_gw','stem7_full')
        want_ggo=algorithm in ('stem7_gi_ggo','stem7_full')
        use_gi=bool(want_gi and gi_nt and gi_g)
        use_gw=bool(want_gw)  # direct cropped kernel needs no packing workspace
        use_ggo=bool(want_ggo and ggo_nt and ggo_g)
        scratch=max(plan.get('scratch_bytes',0),
                    gi_scratch if use_gi else 0,
                    ggo_scratch if use_ggo else 0)
        return dict(plan,stem7_gi28=use_gi,stem7_gw28=use_gw,
                    stem7_ggo28=use_ggo,
                    stem7_gi_nt28=gi_nt,stem7_gi_groups28=gi_g,
                    stem7_ggo_nt28=ggo_nt,stem7_ggo_groups28=ggo_g,
                    stem7_gi_scratch28=gi_scratch if use_gi else 0,
                    stem7_ggo_scratch28=ggo_scratch if use_ggo else 0,
                    stem7_workspace_cap28=cap,scratch_bytes=scratch,
                    dispatch_route=algorithm,
                    dispatch_reason=('rgb7x7_parity_gi_packed_ggo_implicit_gw'
                        if algorithm=='stem7_gi_ggo' else
                        'rgb7x7_parity_cropped_and_packed_stages'),
                    dispatch_policy='v29_opt_in_target_gpu_validation')
    if algorithm=='pointwise_stride2':
        supported=dtype==torch.float32 and (r,s,sh,sw,ph,pw,dh,dw)==(1,1,2,2,0,0,1,1)
        if not supported:
            raise ValueError('pointwise_stride2 requires FP32 1x1 stride2 padding0 dilation1')
        plan=winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget,'implicit',split_k)
        return dict(plan,pointwise26=True,dispatch_route='pointwise_stride2',
                    dispatch_reason='sampled_lattice_direct_gemm',
                    dispatch_policy='v27_auto_for_measured_complete_fp32_calls')
    if algorithm in ('stride2_joint_split_gw4','stride2_joint_split_gw8'):
        supported=(dtype==torch.float32 and
                   (r,s,sh,sw,dh,dw)==(3,3,2,2,1,1) and min(h,w)>=2)
        if not supported:
            raise ValueError(algorithm+' requires FP32 3x3 stride2 dilation1')
        gw_split=4 if algorithm.endswith('4') else 8
        # Start from v31 so isolated calls retain the measured wide ggO/gI
        # schedules. Only complete ggO+gW calls with a genuinely long N
        # reduction enter the v32 packed split-K experiment.
        plan=winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,
                           dtype,budget,'stride2_joint_onebuf_wide_gi',
                           split_k)
        cap=min(max(0,int(budget)),128*1024**2)
        total=f*p*q
        large_n=total>=16384
        if not large_n or split_k is not None:
            return dict(plan,joint_split_gw32=False,gw_split32=gw_split,
                        split_gw_fallback32=('explicit_split_k_uses_v31'
                            if split_k is not None else
                            'short_reduction_uses_v31'),
                        dispatch_route=algorithm,
                        dispatch_reason='v31_fallback_outside_large_n',
                        dispatch_policy='v32_opt_in_target_gpu_validation')
        aligned=triton.cdiv(total,32)*32
        best=None
        # The packed activation buffer, split-K partial gW tensors and one
        # compensated running gW value coexist. Search the complete live set,
        # rather than planning each allocation independently.
        per_gw=o*c*9*4
        for gb in range(1,g+1):
            waves=triton.cdiv(g,gb)
            for requested in range(1,min(64,triton.cdiv(aligned,32))+1):
                nt=triton.cdiv(triton.cdiv(total,requested),32)*32
                chunks=triton.cdiv(total,nt)
                pack=gb*c*9*nt*4
                partial=gb*gw_split*per_gw
                correction=gb*per_gw
                scratch=pack+partial+correction
                if (scratch>cap or gb*gw_split*o*c*9>=2**30):
                    continue
                # Minimize the number of group-wave/chunk launches. At equal
                # launch count keep more groups resident before preferring a
                # smaller number of chunks.
                score=(waves*chunks,waves,chunks,-gb,-nt)
                candidate=(score,gb,nt,chunks,scratch,pack,
                           partial,correction)
                if best is None or candidate[0]<best[0]:best=candidate
        if best is None:
            return dict(plan,joint_split_gw32=False,gw_split32=gw_split,
                        split_gw_fallback32='workspace_or_index_guard_uses_v31',
                        dispatch_route=algorithm,
                        dispatch_reason='v31_fallback_workspace_guard',
                        dispatch_policy='v32_opt_in_target_gpu_validation')
        _,gb,nt,chunks,scratch,pack,partial,correction=best
        return dict(plan,joint_onebuf30=False,joint_split_gw32=True,
                    joint_wide_gi31=True,gw_split32=gw_split,
                    split_gw_nt32=nt,split_gw_groups32=gb,
                    split_gw_chunks32=chunks,
                    split_gw_group_waves32=triton.cdiv(g,gb),
                    split_gw_pack_bytes32=pack,
                    split_gw_partial_bytes32=partial,
                    split_gw_correction_bytes32=correction,
                    split_gw_scratch32=scratch,split_gw_cap32=cap,
                    scratch_bytes=max(plan.get('gi_scratch17',0),scratch),
                    dispatch_route=algorithm,
                    dispatch_reason='large_n_onebuf_packed_splitk_gw',
                    dispatch_policy='v32_opt_in_target_gpu_validation')
    if algorithm in ('stride2_joint_onebuf','stride2_joint_onebuf_wide_gi'):
        wide_gi=algorithm=='stride2_joint_onebuf_wide_gi'
        supported=(dtype==torch.float32 and
                   (r,s,sh,sw,dh,dw)==(3,3,2,2,1,1) and min(h,w)>=2)
        if not supported:
            raise ValueError(algorithm+' requires FP32 3x3 stride2 dilation1')
        # The v30 candidate retains the validated v18 parity-packed gI plan;
        # v31 changes only its chunk/group schedule to v27's wider plan. The
        # complete ggO+gW path has independent workspace keys, so partial calls
        # safely fall back to the corresponding inherited schedule.
        plan=winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,
                           dtype,budget,
                           'stride2_wide_chunks' if wide_gi else
                           'stride2_ggo_tail',split_k)
        cap=min(max(0,int(budget)),128*1024**2)
        total=f*p*q
        aligned=(total+31)//32*32
        best=None
        # Search by requested chunk count instead of every possible NT.  For a
        # fixed chunk count the balanced NT minimizes padding and workspace.
        # Prefer the fewest wave*chunk launches; on ties prefer fewer chunks
        # per group (large N already supplies enough CTA parallelism).
        for gb in range(1,g+1):
            waves=(g+gb-1)//gb
            for requested in range(1,min(64,(aligned+31)//32)+1):
                nt=((total+requested-1)//requested+31)//32*32
                chunks=(total+nt-1)//nt
                # One output-sized correction buffer supports compensated
                # accumulation across chunks; unlike v25 it does not scale
                # with the number of chunks.
                correction=o*c*9*4
                scratch=gb*(c*9*nt*4+correction)
                if (scratch>cap or
                        gb*chunks*o*c*9>=2**30):
                    continue
                score=(waves*chunks,chunks,waves,-gb)
                candidate=(score,gb,nt,chunks,scratch,correction)
                if best is None or candidate[0]<best[0]:
                    best=candidate
        enabled=best is not None and split_k is None
        if not enabled:
            return dict(plan,joint_onebuf30=False,
                        onebuf_fallback30=('explicit_split_k_uses_v23_fallback'
                            if split_k is not None else 'workspace_or_index_guard'),
                        onebuf_cap30=cap,
                        dispatch_route=algorithm,
                        dispatch_policy=('v31_opt_in_target_gpu_validation'
                            if wide_gi else
                            'v30_opt_in_target_gpu_validation'))
        _,gb,nt,chunks,scratch,correction=best
        return dict(plan,joint_onebuf30=True,
                    onebuf_nt30=nt,onebuf_groups30=gb,
                    onebuf_chunks30=chunks,
                    onebuf_group_waves30=(g+gb-1)//gb,
                    onebuf_total_columns30=total,
                    onebuf_pack_bytes30=gb*c*9*nt*4,
                    onebuf_correction_bytes30=gb*correction,
                    onebuf_scratch30=scratch,onebuf_cap30=cap,
                    # Full v31 bypasses v27's wide ggO workspace; only its
                    # wide gI scratch can overlap in the plan (execution is
                    # sequential), so do not report unused ggO packing.
                    scratch_bytes=max(
                        plan.get('gi_scratch17',0) if wide_gi else
                        plan.get('scratch_bytes',0),scratch),
                    joint_wide_gi31=wide_gi,
                    dispatch_route=algorithm,
                    dispatch_reason=('single_pack_buffer_ggo_gw_plus_wide_gi'
                        if wide_gi else
                        'single_pack_buffer_reused_by_ggo_and_gw'),
                    dispatch_policy=('v31_opt_in_target_gpu_validation'
                        if wide_gi else
                        'v30_opt_in_target_gpu_validation'))
    if algorithm=='stride2_wide_chunks':
        supported=dtype==torch.float32 and (r,s,sh,sw,dh,dw)==(3,3,2,2,1,1) and min(h,w)>=2
        if not supported:
            raise ValueError('stride2_wide_chunks requires FP32 3x3 stride2 dilation1')
        # Preserve v23's exact two-accumulator ggO, v18 parity-packed gI and
        # the measured generic gW. Only enlarge the two packing schedules.
        # v14-v23 deliberately capped them at 32 MiB; the target experiment
        # supplies 128 MiB and s3b1_conv1 otherwise executes eight ggO chunks
        # and roughly four chunks for every gI parity.
        plan=winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,
                           dtype,budget,'stride2_ggo_tail',split_k)
        cap=min(max(0,int(budget)),128*1024**2)
        old_ggo_nt=plan.get('ggo_nt14',0);old_gi_nt=plan.get('gi_nt17',0)

        # Keep at least about 1024 columns per group wave when G is large,
        # then use the remaining budget to maximize NT. This prevents a wide
        # schedule from becoming many tiny all-G chunks.
        ggo_total=((f*p*q+31)//32)*32
        ggo_goal=min(ggo_total,1024)
        ggo_per_goal=2*4*c*9*ggo_goal
        ggo_groups=min(g,cap//ggo_per_goal) if ggo_per_goal else 0
        if not ggo_groups and cap>=2*4*c*9*32:ggo_groups=1
        ggo_cap=cap//(ggo_groups*2*4*c*9) if ggo_groups else 0
        ggo_nt=min(ggo_total,(ggo_cap//32)*32)

        # One phase-local weight matrix and one activation matrix are live.
        # kmax=4*O is the largest of the four stride-2 parity contractions.
        gi_total=((f*((h+1)//2)*((w+1)//2)+31)//32)*32
        kmax=4*o;gi_goal=min(gi_total,1024)
        gi_per_goal=4*kmax*(c+gi_goal)
        gi_groups=min(g,cap//gi_per_goal) if gi_per_goal else 0
        if not gi_groups and cap>=4*kmax*(c+32):gi_groups=1
        gi_cap=cap//(gi_groups*4*kmax)-c if gi_groups else 0
        gi_nt=min(gi_total,(max(0,gi_cap)//32)*32)

        enabled=bool(plan.get('ggo_tail23') and plan.get('gi_tail18') and
                     ggo_groups and gi_groups and ggo_nt and gi_nt)
        if not enabled:
            return dict(plan,wide_chunks27=False,wide_chunks_fallback27='workspace_or_base_guard',
                        wide_chunks_cap27=cap,dispatch_policy='v27_opt_in_target_gpu_validation')
        ggo_scratch=ggo_groups*2*4*c*9*ggo_nt
        gi_scratch=gi_groups*4*kmax*(c+gi_nt)
        phase_stats=[]
        for rh in range(2):
            for rw in range(2):
                total=f*((h+1-rh)//2)*((w+1-rw)//2)
                rem=total%gi_nt
                tail=((rem+31)//32)*32 if rem else 0
                phase_stats.append(dict(phase=rh*2+rw,total_columns=total,
                    full_nt=gi_nt,tail_columns=rem,tail_nt=tail,
                    padded_columns=tail-rem,previous_padded_columns=(-total)%gi_nt))
        ggo_rem=f*p*q%ggo_nt
        return dict(plan,wide_chunks27=True,
                    ggo_nt14=ggo_nt,ggo_groups14=ggo_groups,
                    gi_nt17=gi_nt,gi_groups17=gi_groups,
                    ggo_full_nt23=ggo_nt,
                    ggo_tail_nt23=((ggo_rem+31)//32)*32 if ggo_rem else 0,
                    ggo_previous_padding23=(-f*p*q)%ggo_nt,
                    gi_phase_chunks18=phase_stats,
                    old_ggo_nt27=old_ggo_nt,old_gi_nt27=old_gi_nt,
                    ggo_chunks27=(f*p*q+ggo_nt-1)//ggo_nt,
                    gi_max_phase_chunks27=(f*((h+1)//2)*((w+1)//2)+gi_nt-1)//gi_nt,
                    ggo_scratch27=ggo_scratch,gi_scratch27=gi_scratch,
                    wide_chunks_cap27=cap,packing_limit_bytes=cap,
                    scratch_bytes=max(plan.get('scratch_bytes',0),ggo_scratch,gi_scratch),
                    dispatch_route='stride2_wide_chunks',
                    dispatch_reason='use_available_workspace_for_fewer_ggo_gi_chunks',
                    dispatch_policy='v27_opt_in_target_gpu_validation')
    if algorithm=='stride2_small_spatial':
        supported=dtype==torch.float32 and (r,s,sh,sw,dh,dw)==(3,3,2,2,1,1) and min(h,w)>=2
        if not supported:
            raise ValueError('stride2_small_spatial requires FP32 3x3 stride2 dilation1')
        # Retain v23's best ggO tail/single-accumulator and v18 packed dgrad,
        # but replace the remaining generic wgrad with a late-stage tiling family.
        plan=winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget,'stride2_ggo_single',split_k)
        return dict(plan,gw_small26=True,dispatch_route='stride2_small_spatial',
                    dispatch_reason='v23_ggo_v18_gi_large_tile_cropped_gw',
                    dispatch_policy='v26_opt_in_no_default_change')
    if algorithm in ('stride2_ggo_tail','stride2_ggo_single'):
        plan=winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget,'stride2_packed_gi_tail',split_k)
        enabled=bool(plan.get('ggo_pack14'))
        nt=plan.get('ggo_nt14',0);total=f*p*q
        rem=total%nt if nt else 0
        return dict(plan,ggo_tail23=enabled,ggo_single23=enabled and algorithm=='stride2_ggo_single',
                    ggo_full_nt23=nt,ggo_tail_nt23=triton.cdiv(rem,32)*32 if rem else 0,
                    ggo_previous_padding23=(-total)%nt if nt else 0,
                    dispatch_route=algorithm if enabled else plan['dispatch_route'],
                    dispatch_policy='v23_opt_in_no_default_change')
    if algorithm=='stride2_implicit_gw':
        plan=winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget,'stride2_packed_gi_tail',split_k)
        return dict(plan,gw_static20=True,dispatch_route='stride2_implicit_gw',
                    dispatch_reason='static_filter_cropped_incremental_wgrad',
                    dispatch_policy='v20_opt_in_no_default_change')
    if algorithm=='stride2_packed_gw':
        plan=winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget,'stride2_packed_gi',split_k)
        return dict(plan,gw_candidate19=True,dispatch_policy='v19_opt_in_no_default_change')
    if algorithm=='stride2_packed_gi_tail':
        plan=winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget,'stride2_packed_gi',split_k)
        enabled=bool(plan.get('gi_packed17'))
        stats=[]
        if enabled:
            nt=plan['gi_nt17']
            for rh in range(2):
                for rw in range(2):
                    total=f*((h+1-rh)//2)*((w+1-rw)//2)
                    rem=total%nt
                    tail=triton.cdiv(rem,32)*32 if rem else 0
                    stats.append(dict(phase=rh*2+rw,total_columns=total,full_nt=nt,
                                      tail_columns=rem,tail_nt=tail,
                                      padded_columns=tail-rem,
                                      previous_padded_columns=(-total)%nt))
        return dict(plan,gi_tail18=enabled,gi_phase_chunks18=stats,
                    dispatch_route='stride2_packed_gi_tail' if enabled else plan['dispatch_route'],
                    dispatch_reason='parity_gO_pack_aligned_tail' if enabled else plan['dispatch_reason'],
                    dispatch_policy='v18_opt_in_no_default_change')
    if algorithm=='stride2_packed_gi':
        plan=winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget,'stride2_static_weight',split_k)
        # Bound combined live phase weights + activation packing, not each alone.
        cap=min(int(budget),32*1024**2)
        kmax=4*o
        gb=min(g,cap//(4*kmax*(c+32)))
        nt=min(2048,triton.cdiv(f*triton.cdiv(h,2)*triton.cdiv(w,2),32)*32,
               (cap//(4*gb*kmax)-c)//32*32) if gb else 0
        enabled=bool(gb and nt)
        scratch=4*gb*kmax*(c+nt) if enabled else 0
        return dict(plan,gi_packed17=enabled,gi_groups17=gb,gi_nt17=nt,
                    gi_scratch17=scratch,scratch_bytes=max(plan['scratch_bytes'],scratch),
                    dispatch_route='stride2_packed_gi' if enabled else plan['dispatch_route'],
                    dispatch_reason='bounded_parity_gO_pack' if enabled else 'packing_budget_fallback_v16',
                    dispatch_policy='v17_opt_in_no_default_change')
    if algorithm=='stride2_static_weight':
        plan=winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget,'stride2_packed',split_k)
        # Identical v14 packing and group chunk sizes. Only dgrad's phase
        # indexing/filter loops change; low-budget fallback is unchanged v14.
        enabled=bool(plan.get('gi_weight14'))
        return dict(plan,gi_static_weight16=enabled,
                    dispatch_route='stride2_static_weight' if enabled else plan['dispatch_route'],
                    dispatch_reason='static_parity_reordered_weights' if enabled else 'weight_budget_fallback_v14',
                    dispatch_policy='v16_opt_in_no_default_change')
    if algorithm in ('stride2_static','stride2_shared'):
        plan=winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget,'stride2_packed',split_k)
        if algorithm=='stride2_static':
            plan.update(gi_static15=True,gi_weight14=False,dispatch_route='stride2_static',
                        dispatch_reason='compile_time_parity_original_weights')
        else:
            cap=min(int(budget),32*1024**2)
            # Output gW itself stores the running sum. Only compensation is scratch.
            per_corr=4*o*c*9
            gb=min(g,cap//(per_corr+2*4*c*9*32))
            nt=min(2048,triton.cdiv(f*p*q,32)*32,(cap//gb-per_corr)//(2*4*c*9)//32*32) if gb else 0
            enabled=bool(nt and split_k is None)
            plan.update(shared15=enabled,shared_groups15=gb,shared_nt15=nt,
                        shared_scratch15=gb*(per_corr+2*4*c*9*nt) if enabled else 0,
                        dispatch_route='stride2_shared' if enabled else plan['dispatch_route'],
                        dispatch_reason='reuse_packed_ggI' if enabled else 'shared_budget_or_explicit_split_fallback')
            plan['scratch_bytes']=max(plan['scratch_bytes'],plan['shared_scratch15'])
        plan['dispatch_policy']='v15_opt_in_no_default_change'
        return plan
    if algorithm in ('adaptive14','stride2_weight','stride2_packed','winograd_weight'):
        supported=dtype==torch.float32 and (r,s,sh,sw,dh,dw)==(3,3,2,2,1,1) and min(h,w)>=2
        if algorithm.startswith('stride2_') and not supported:
            raise ValueError('v14 stride2 candidates require FP32 3x3 stride2 dilation1 H,W>=2')
        base='winograd_exact' if algorithm=='winograd_weight' else 'adaptive13'
        plan=winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget,base,split_k)
        if supported:
            # Keep new packing within 32 MiB even when legacy Winograd is
            # allowed 128 MiB: preserve the measured stride2 memory advantage.
            budget=min(int(budget),32*1024**2)
            weight_per=4*c*o*9
            wg=min(g,int(budget)//weight_per)
            # Same allocation reused for every tile chunk; no full-batch im2col.
            pg=min(g,budget//(2*4*c*9*32))
            nt=min(2048,triton.cdiv(f*p*q,32)*32,budget//(pg*2*4*c*9)//32*32) if pg else 0
            packed=algorithm!='stride2_weight' and pg>0
            scratch=max(wg*weight_per,pg*2*4*c*9*nt if packed else 0)
            plan.update(gi_weight14=bool(wg),gi_groups14=wg,ggo_pack14=packed,
                        ggo_nt14=nt if packed else 0,ggo_groups14=pg if packed else 0,
                        packing_limit_bytes=budget,scratch_bytes=scratch,dispatch_route='stride2_packed' if packed else 'stride2_weight' if wg else 'stride2_phase',
                        dispatch_reason='bounded_pack_and_weight_reorder',dispatch_policy='v14_experimental_not_measured')
        elif plan.get('version')==10 and (algorithm=='winograd_weight' or plan.get('exact_chunks')):
            plan.update(weight_all14=True,dispatch_route='winograd_weight',dispatch_reason='reuse_weight_loads',dispatch_policy='v14_experimental_not_measured')
        return plan
    if algorithm in ('adaptive13','winograd_exact'):
        old='adaptive12' if algorithm=='adaptive13' else 'winograd_tail'
        plan=winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget,old,split_k)
        tiles=f*triton.cdiv(p,2)*triton.cdiv(q,2)
        # Opt-in candidate: preserve large-batch and existing direct/phase routes.
        if plan.get('version')==10 and (algorithm=='winograd_exact' or tiles<=1024):
            plan=_exact_plan13(plan,g,f,c,o,h,w,p,q,int(budget))
            plan.update(dispatch_route='winograd_exact',dispatch_reason='coalesce_tail_launches',
                        dispatch_policy='v13_experimental_small_tile_schedule')
        return plan
    if algorithm in ('adaptive12','stride2_phase'):
        supported=(r,s,sh,sw,dh,dw)==(3,3,2,2,1,1) and min(h,w)>=2
        if algorithm=='stride2_phase' and not supported:
            raise ValueError('stride2_phase requires 3x3 stride=2 dilation=1 H,W>=2')
        tiles=f*triton.cdiv(p,2)*triton.cdiv(q,2)
        dense=dtype==torch.float32 and (r,s,sh,sw,dh,dw)==(3,3,1,1,1,1) and c>3 and 0<=ph<=2 and 0<=pw<=2
        # Provisional broad workload rule, deliberately independent of model names.
        # 256 tiles bounds the per-group contraction dimension: amortization of
        # transforms/weight packing is poor in measured small-spatial cases.
        route='winograd_tail' if dense and tiles>256 else 'implicit'
        plan=winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget,route,split_k)
        return dict(plan,phase_gI=supported,dispatch_route='stride2_phase' if supported else route,
                    dispatch_reason='parity_dgrad' if supported else 'small_tile_direct' if dense and tiles<=256 else 'dense_tail' if dense else 'preserve_generic',
                    per_group_tiles=tiles,dispatch_policy='v12_candidate_tile_cutoff256_not_proven_optimal')
    # Structural candidate dispatch, NOT a claim of the fastest route per shape.
    # Keep RGB stems, pointwise and strided convolutions on the original path.
    # No batch/group thresholds or benchmark-result lookup tables.
    if algorithm=='shape_safe':
        dense=(c>3 and dtype==torch.float32 and
               (r,s,sh,sw,dh,dw)==(3,3,1,1,1,1) and 0<=ph<=2 and 0<=pw<=2)
        route='winograd_tail' if dense else 'implicit'
        plan=winograd_plan(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget,route,split_k)
        return dict(plan,dispatch_route=route,dispatch_reason='dense_3x3_candidate' if dense else 'preserve_generic',
                    dispatch_policy='structural_v11_not_autotuned')
    args=(g,f,c,o,h,w,p,q,r,s,sh,sw,ph,pw,dh,dw,dtype,budget)
    if algorithm=='winograd_v9':return _v9_winograd_plan(*args,'winograd_large',split_k)
    if algorithm not in ('winograd_tail','winograd_group'):
        return _v9_winograd_plan(*args,algorithm,split_k)
    plan=_v9_winograd_plan(*args,'winograd_large',split_k)
    total=f*triton.cdiv(p,2)*triton.cdiv(q,2)
    total_i=f*triton.cdiv(h,2)*triton.cdiv(w,2)
    if algorithm=='winograd_group':
        # Explore more concurrent outer groups. Prefer a chunk fitting all G;
        # if even NT=32 cannot fit all G, use v9's bounded group plan instead.
        # The independent gI/ggO capacities need not match the joint gW capacity.
        def pick(target,total,kind,forced_sk=None):
            nt=min(target,max(32,triton.next_power_of_2(total)))
            while nt>=32:
                sk=forced_sk if forced_sk is not None else (split_k if split_k is not None else max(1,triton.cdiv(nt,512)))
                fixed={'joint':(2+2*sk)*c*o,'gW':2*sk*c*o,'gI':c*o,'ggO':2*c*o}[kind]
                tile=2*c+o if kind=='joint' else c+o
                size=64*g*(fixed+tile*nt)
                if size<=budget:return nt,g,size,sk
                nt//=2
            return None
        j=pick(2048,total,'joint')
        if j is not None:
            ntj,gbj,bj,sk=j
            wplan=pick(ntj,total,'gW',sk)
            assert wplan is not None  # joint needs strictly more memory
            ntw,gbw,bw,_=wplan
            plan.update(joint_nt=ntj,joint_groups=gbj,gW_nt=ntw,gW_groups=gbw,gW_split=sk)
            sizes=[bj,bw]
            for name,t in [('ggO',total),('gI',total_i)]:
                v=pick(4096,t,name)
                if v is None:raise AssertionError('forward workspace must fit when joint fits')
                nt,gb,size,_=v;plan[name+'_nt']=nt;plan[name+'_groups']=gb
                plan[name+'_pack']=True;sizes.append(size)
            plan['scratch_bytes']=max(sizes)
    plan['version']=10
    plan['schedule']='tail_more_groups' if algorithm=='winograd_group' else 'tail_only'
    plan['tail_policy']='power_of_two_decomposition_min32'
    plan['joint_tile_chunks']=len(list(_tile_chunks10(total,plan['joint_nt']))) if plan['joint_nt'] else 0
    plan['joint_group_chunks']=triton.cdiv(g,plan['joint_groups']) if plan['joint_groups'] else 0
    plan['gW_tiles_per_partial']=triton.cdiv(plan['joint_nt'] or plan['gW_nt'],plan['gW_split']) if plan['gW_split'] else 0
    for name,t in [('joint',total),('ggO',total),('gI',total_i),('gW',total)]:
        nt=plan[name+'_nt']
        if nt:
            chunks=list(_tile_chunks10(t,nt))
            plan[name+'_tail_sizes']=[n for _,n in chunks if n!=nt]
            plan[name+'_padded_tiles']=sum(n for _,n in chunks)-t
    return plan

# Stride-2 dgrad: partition input positions by parity. A CTA only multiplies
# filter offsets compatible with its parity; no expanded zero-filled dY.
# For 3x3 filters, the four phases use 1, 2, 2 and 4 offsets instead of 9 each.
@triton.autotune(configs=_configs(),key=_SHAPE_KEY)
@triton.jit
def _gi_phase12(D,AW,Y,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
        H:tl.constexpr,W:tl.constexpr,O:tl.constexpr,R:tl.constexpr,S:tl.constexpr,
        P:tl.constexpr,Q:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
        PH:tl.constexpr,PW:tl.constexpr,DH:tl.constexpr,DW:tl.constexpr,
        PREC:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    tile=tl.program_id(0);g=tl.program_id(1);phase=tl.program_id(2)
    rh=phase//2;rw=phase%2
    hh=(H+1-rh)//2;ww=(W+1-rw)//2
    # Uniform grid covers the largest phase; inactive columns are masked.
    nn=tl.cdiv(F*tl.cdiv(H,2)*tl.cdiv(W,2),BN)
    ci=tile//nn*BM+tl.arange(0,BM)
    n=tile%nn*BN+tl.arange(0,BN)
    batch=n//(hh*ww);iy=(n//ww%hh)*2+rh;ix=(n%ww)*2+rw
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    r0=(rh+PH)%2;s0=(rw+PW)%2
    for rr in range(r0,R,2):
        for ss in range(s0,S,2):
            py=(iy+PH-rr)//2;qx=(ix+PW-ss)//2
            valid=(n<F*hh*ww)&(py>=0)&(py<P)&(qx>=0)&(qx<Q)
            for cb in range(tl.cdiv(O,BK)):
                oc=cb*BK+kk
                a=tl.load(AW+g*O*C*R*S+oc[None,:]*C*R*S+ci[:,None]*R*S+rr*S+ss,
                          (ci[:,None]<C)&(oc[None,:]<O),0)
                b=tl.load(D+(g*F+batch[None,:])*O*P*Q+oc[:,None]*P*Q+py[None,:]*Q+qx[None,:],
                          (oc[:,None]<O)&valid[None,:],0)
                acc=tl.dot(a,b,acc,input_precision=PREC)
    tl.store(Y+((g*F+batch[None,:])*C+ci[:,None])*H*W+iy[None,:]*W+ix[None,:],acc,
             (ci[:,None]<C)&(n[None,:]<F*hh*ww))


@triton.jit
def _gi_weight_pack14(W,U,G:tl.constexpr,C:tl.constexpr,O:tl.constexpr):
    # [G,O,C,9] -> [G,9,C,O]; O is the dgrad reduction dimension.
    pid=tl.program_id(0);g=tl.program_id(1);rs=tl.program_id(2)
    ci=(pid//tl.cdiv(O,32)*16+tl.arange(0,16))[:,None]
    oc=(pid%tl.cdiv(O,32)*32+tl.arange(0,32))[None,:]
    val=tl.load(W+((g*O+oc)*C+ci)*9+rs,(ci<C)&(oc<O),0)
    tl.store(U+((g*9+rs)*C+ci)*O+oc,val,(ci<C)&(oc<O))


def _gi_weight14(d,aw,out,group_batch,dims):
    g,f,c,h,w,o,r,s,p,q,sh,sw,ph,pw,dh,dw,prec=dims
    for start in range(0,g,group_batch):
        end=min(g,start+group_batch);gb=end-start
        packed=torch.empty((gb,9,c,o),device=d.device,dtype=torch.float32)
        _gi_weight_pack14[(triton.cdiv(c,16)*triton.cdiv(o,32),gb,9)](aw[start:end],packed,gb,c,o)
        _gi_reordered14[lambda m:(triton.cdiv(c,m['BM'])*triton.cdiv(f*triton.cdiv(h,2)*triton.cdiv(w,2),m['BN']),gb,4)](
            d[start:end],packed,out[start:end],gb,*dims[1:])
        del packed


@triton.jit
def _ggo_pack14(X,AI,V0,V1,start,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
                H:tl.constexpr,W:tl.constexpr,P:tl.constexpr,Q:tl.constexpr,
                PH:tl.constexpr,PW:tl.constexpr,NT:tl.constexpr,
                HI:tl.constexpr,HW:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);g=tl.program_id(1)
    k=i//NT;n=i%NT;t=start+n
    batch=t//(P*Q);pos=t%(P*Q)
    iy=pos//Q*2-PH+k//3%3;ix=pos%Q*2-PW+k%3
    off=((g*F+batch)*C+k//9)*H*W+iy*W+ix
    valid=(k<C*9)&(t<F*P*Q)&(iy>=0)&(iy<H)&(ix>=0)&(ix<W)
    dst=(g*C*9+k)*NT+n
    if HI and HW:
        if tl.program_id(2)==0:
            v=tl.load(AI+off,valid,0);tl.store(V0+dst,v,k<C*9)
        else:
            v=tl.load(X+off,valid,0);tl.store(V1+dst,v,k<C*9)
    elif HI:
        v=tl.load(AI+off,valid,0);tl.store(V0+dst,v,k<C*9)
    else:
        v=tl.load(X+off,valid,0);tl.store(V1+dst,v,k<C*9)


@triton.autotune(configs=_mm7_configs(),key=['G','F','C','O','P','Q','NT','HI','HW','HB','PREC'])
@triton.jit
def _ggo_mm14(W,AW,V0,V1,Bias,Y,start,G:tl.constexpr,F:tl.constexpr,
              C:tl.constexpr,O:tl.constexpr,P:tl.constexpr,Q:tl.constexpr,NT:tl.constexpr,
              HI:tl.constexpr,HW:tl.constexpr,HB:tl.constexpr,PREC:tl.constexpr,
              BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0);g=tl.program_id(1)
    nn=tl.cdiv(NT,BN);nm=tl.cdiv(O,BM)
    first=pid//(4*nn)*4;count=tl.minimum(nm-first,4)
    mi=first+pid%(4*nn)%count;ni=pid%(4*nn)//count
    oc=mi*BM+tl.arange(0,BM);n=ni*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    a0=tl.zeros((BM,BN),tl.float32);a1=tl.zeros((BM,BN),tl.float32)
    for kb in range(tl.cdiv(C*9,BK)):
        k=kb*BK+kk
        wo=(g*O+oc[:,None])*C*9+k[None,:]
        vo=(g*C*9+k[:,None])*NT+n[None,:]
        wm=(oc[:,None]<O)&(k[None,:]<C*9)
        vm=(k[:,None]<C*9)&(n[None,:]<NT)
        if HI:a0=tl.dot(tl.load(W+wo,wm,0),tl.load(V0+vo,vm,0),a0,input_precision=PREC)
        if HW:a1=tl.dot(tl.load(AW+wo,wm,0),tl.load(V1+vo,vm,0),a1,input_precision=PREC)
    val=a0+a1
    if HB:val+=tl.load(Bias+g*O+oc,oc<O,0)[:,None]
    t=start+n;batch=t//(P*Q);pos=t%(P*Q)
    tl.store(Y+((g*F+batch[None,:])*O+oc[:,None])*P*Q+pos[None,:],val,
             (oc[:,None]<O)&(n[None,:]<NT)&(t[None,:]<F*P*Q))


def _ggo_packed14(x,w,ai,aw,bias,out,plan,ph,pw,precision):
    g,f,c,h,wi=x.shape;_,_,o,p,q=out.shape
    hi=ai is not None;hw=aw is not None;hb=bias is not None
    nt=plan['ggo_nt14'];group_batch=plan['ggo_groups14']
    for base in range(0,g,group_batch):
        end=min(g,base+group_batch);gb=end-base
        v=torch.empty((gb,c*9,nt),device=x.device,dtype=torch.float32)
        v2=torch.empty_like(v) if hi and hw else v
        xx=x[base:end];aa=ai[base:end] if hi else xx
        ww=w[base:end];aw1=aw[base:end] if hw else ww
        bb=bias[base:end] if hb else xx;yy=out[base:end]
        for start in range(0,f*p*q,nt):
            _ggo_pack14[(triton.cdiv(c*9*nt,256),gb,2 if hi and hw else 1)](
                xx,aa,v,v2,start,gb,f,c,h,wi,p,q,ph,pw,nt,hi,hw,256)
            _ggo_mm14[lambda m:(triton.cdiv(o,m['BM'])*triton.cdiv(nt,m['BN']),gb)](
                ww,aw1,v,v2,bb,yy,start,gb,f,c,o,p,q,nt,hi,hw,hb,precision)
        del v,v2


@triton.autotune(configs=_configs(),key=_SHAPE_KEY)
@triton.jit
def _gi_reordered14(D,AW,Y,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
        H:tl.constexpr,W:tl.constexpr,O:tl.constexpr,R:tl.constexpr,S:tl.constexpr,
        P:tl.constexpr,Q:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
        PH:tl.constexpr,PW:tl.constexpr,DH:tl.constexpr,DW:tl.constexpr,
        PREC:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    tile=tl.program_id(0);g=tl.program_id(1);phase=tl.program_id(2)
    rh=phase//2;rw=phase%2
    hh=(H+1-rh)//2;ww=(W+1-rw)//2
    # Uniform grid covers the largest phase; inactive columns are masked.
    nn=tl.cdiv(F*tl.cdiv(H,2)*tl.cdiv(W,2),BN)
    ci=tile//nn*BM+tl.arange(0,BM)
    n=tile%nn*BN+tl.arange(0,BN)
    batch=n//(hh*ww);iy=(n//ww%hh)*2+rh;ix=(n%ww)*2+rw
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    r0=(rh+PH)%2;s0=(rw+PW)%2
    for rr in range(r0,R,2):
        for ss in range(s0,S,2):
            py=(iy+PH-rr)//2;qx=(ix+PW-ss)//2
            valid=(n<F*hh*ww)&(py>=0)&(py<P)&(qx>=0)&(qx<Q)
            for cb in range(tl.cdiv(O,BK)):
                oc=cb*BK+kk
                a=tl.load(AW+((g*9+rr*3+ss)*C+ci[:,None])*O+oc[None,:],
                          (ci[:,None]<C)&(oc[None,:]<O),0)
                b=tl.load(D+(g*F+batch[None,:])*O*P*Q+oc[:,None]*P*Q+py[None,:]*Q+qx[None,:],
                          (oc[:,None]<O)&valid[None,:],0)
                acc=tl.dot(a,b,acc,input_precision=PREC)
    tl.store(Y+((g*F+batch[None,:])*C+ci[:,None])*H*W+iy[None,:]*W+ix[None,:],acc,
             (ci[:,None]<C)&(n[None,:]<F*hh*ww))



@triton.jit
def _win_weight_all14(W,AW,U,AU,G:tl.constexpr,C:tl.constexpr,O:tl.constexpr,
                      FLIP:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);g=tl.program_id(1)
    oc=i//C;ci=i%C;valid=i<O*C
    source=W;target=U
    if tl.program_id(2)!=0:source=AW;target=AU
    if FLIP:base=((g*C+ci)*O+oc)*9
    else:base=(g*O*C+i)*9
    if FLIP:w00=tl.load(source+base+8,valid,0)
    else:w00=tl.load(source+base+0,valid,0)
    if FLIP:w01=tl.load(source+base+7,valid,0)
    else:w01=tl.load(source+base+1,valid,0)
    if FLIP:w02=tl.load(source+base+6,valid,0)
    else:w02=tl.load(source+base+2,valid,0)
    if FLIP:w10=tl.load(source+base+5,valid,0)
    else:w10=tl.load(source+base+3,valid,0)
    if FLIP:w11=tl.load(source+base+4,valid,0)
    else:w11=tl.load(source+base+4,valid,0)
    if FLIP:w12=tl.load(source+base+3,valid,0)
    else:w12=tl.load(source+base+5,valid,0)
    if FLIP:w20=tl.load(source+base+2,valid,0)
    else:w20=tl.load(source+base+6,valid,0)
    if FLIP:w21=tl.load(source+base+1,valid,0)
    else:w21=tl.load(source+base+7,valid,0)
    if FLIP:w22=tl.load(source+base+0,valid,0)
    else:w22=tl.load(source+base+8,valid,0)
    t00=w00
    t10=0.5*(w00+w10+w20)
    t20=0.5*(w00-w10+w20)
    t30=w20
    t01=w01
    t11=0.5*(w01+w11+w21)
    t21=0.5*(w01-w11+w21)
    t31=w21
    t02=w02
    t12=0.5*(w02+w12+w22)
    t22=0.5*(w02-w12+w22)
    t32=w22
    tl.store(target+(0*G+g)*O*C+i,t00,valid)
    tl.store(target+(1*G+g)*O*C+i,0.5*(t00+t01+t02),valid)
    tl.store(target+(2*G+g)*O*C+i,0.5*(t00-t01+t02),valid)
    tl.store(target+(3*G+g)*O*C+i,t02,valid)
    tl.store(target+(4*G+g)*O*C+i,t10,valid)
    tl.store(target+(5*G+g)*O*C+i,0.5*(t10+t11+t12),valid)
    tl.store(target+(6*G+g)*O*C+i,0.5*(t10-t11+t12),valid)
    tl.store(target+(7*G+g)*O*C+i,t12,valid)
    tl.store(target+(8*G+g)*O*C+i,t20,valid)
    tl.store(target+(9*G+g)*O*C+i,0.5*(t20+t21+t22),valid)
    tl.store(target+(10*G+g)*O*C+i,0.5*(t20-t21+t22),valid)
    tl.store(target+(11*G+g)*O*C+i,t22,valid)
    tl.store(target+(12*G+g)*O*C+i,t30,valid)
    tl.store(target+(13*G+g)*O*C+i,0.5*(t30+t31+t32),valid)
    tl.store(target+(14*G+g)*O*C+i,0.5*(t30-t31+t32),valid)
    tl.store(target+(15*G+g)*O*C+i,t32,valid)


@triton.jit
def _gi_parity15(D,AW,Y,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
        H:tl.constexpr,W:tl.constexpr,O:tl.constexpr,R:tl.constexpr,S:tl.constexpr,
        P:tl.constexpr,Q:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
        PH:tl.constexpr,PW:tl.constexpr,DH:tl.constexpr,DW:tl.constexpr,
        PREC:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,PHASE:tl.constexpr):
    tile=tl.program_id(0);g=tl.program_id(1);phase:tl.constexpr=PHASE
    rh:tl.constexpr=phase//2;rw:tl.constexpr=phase%2
    hh:tl.constexpr=(H+1-rh)//2;ww:tl.constexpr=(W+1-rw)//2
    # Uniform grid covers the largest phase; inactive columns are masked.
    nn=tl.cdiv(F*tl.cdiv(H,2)*tl.cdiv(W,2),BN)
    ci=tile//nn*BM+tl.arange(0,BM)
    n=tile%nn*BN+tl.arange(0,BN)
    batch=n//(hh*ww);iy=(n//ww%hh)*2+rh;ix=(n%ww)*2+rw
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    r0:tl.constexpr=(rh+PH)%2;s0:tl.constexpr=(rw+PW)%2
    for rr in tl.static_range(r0,R,2):
        for ss in tl.static_range(s0,S,2):
            py=(iy+PH-rr)//2;qx=(ix+PW-ss)//2
            valid=(n<F*hh*ww)&(py>=0)&(py<P)&(qx>=0)&(qx<Q)
            for cb in range(tl.cdiv(O,BK)):
                oc=cb*BK+kk
                a=tl.load(AW+(g*O+oc[None,:])*C*9+ci[:,None]*9+rr*3+ss,
                          (ci[:,None]<C)&(oc[None,:]<O),0)
                b=tl.load(D+(g*F+batch[None,:])*O*P*Q+oc[:,None]*P*Q+py[None,:]*Q+qx[None,:],
                          (oc[:,None]<O)&valid[None,:],0)
                acc=tl.dot(a,b,acc,input_precision=PREC)
    tl.store(Y+((g*F+batch[None,:])*C+ci[:,None])*H*W+iy[None,:]*W+ix[None,:],acc,
             (ci[:,None]<C)&(n[None,:]<F*hh*ww))


@triton.autotune(configs=_configs(),key=_SHAPE_KEY)
@triton.jit
def _gi_static15(D,AW,Y,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
        H:tl.constexpr,W:tl.constexpr,O:tl.constexpr,R:tl.constexpr,S:tl.constexpr,
        P:tl.constexpr,Q:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
        PH:tl.constexpr,PW:tl.constexpr,DH:tl.constexpr,DW:tl.constexpr,
        PREC:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    for phase_id in tl.static_range(4):
        if tl.program_id(2)==phase_id:
            _gi_parity15(D,AW,Y,G,F,C,H,W,O,R,S,P,Q,SH,SW,PH,PW,DH,DW,PREC,BM,BN,BK,phase_id)


@triton.autotune(configs=_mm7_configs(),key=['G','F','C','O','P','Q','NT','FIRST','PREC'],restore_value=['Y','Corr'])
@triton.jit
def _shared_gw15(D,V,Y,Corr,start,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
                 O:tl.constexpr,P:tl.constexpr,Q:tl.constexpr,NT:tl.constexpr,
                 FIRST:tl.constexpr,PREC:tl.constexpr,
                 BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0);g=tl.program_id(1)
    nn=tl.cdiv(C*9,BN)
    oc=pid//nn*BM+tl.arange(0,BM);k=pid%nn*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for nb in range(tl.cdiv(NT,BK)):
        n=nb*BK+kk;t=start+n;batch=t//(P*Q);pos=t%(P*Q)
        d=tl.load(D+((g*F+batch[None,:])*O+oc[:,None])*P*Q+pos[None,:],
                  (oc[:,None]<O)&(n[None,:]<NT)&(t[None,:]<F*P*Q),0)
        v=tl.load(V+(g*C*9+k[None,:])*NT+n[:,None],(k[None,:]<C*9)&(n[:,None]<NT),0)
        acc=tl.dot(d,v,acc,input_precision=PREC)
    off=(g*O+oc[:,None])*C*9+k[None,:];mask=(oc[:,None]<O)&(k[None,:]<C*9)
    if FIRST:
        updated=acc;correction=tl.full((BM,BN),0,tl.float32)
    else:
        old=tl.load(Y+off,mask,0);prev=tl.load(Corr+off,mask,0)
        delta=acc-prev;updated=old+delta;correction=(updated-old)-delta
    tl.store(Y+off,updated,mask);tl.store(Corr+off,correction,mask)


def _shared15(x,w,ai,aw,bias,d,y,gw,plan,ph,pw,precision):
    g,f,c,h,wi=x.shape;_,_,o,p,q=d.shape
    nt=plan['shared_nt15'];gbatch=plan['shared_groups15'];hw=aw is not None;hb=bias is not None
    for base in range(0,g,gbatch):
        end=min(g,base+gbatch);gb=end-base
        v=torch.empty((gb,c*9,nt),device=x.device,dtype=torch.float32)
        v2=torch.empty_like(v) if hw else v
        corr=torch.empty((gb,o,c,3,3),device=x.device,dtype=torch.float32)
        xx=x[base:end];aa=ai[base:end];ww=w[base:end]
        aw1=aw[base:end] if hw else ww;bb=bias[base:end] if hb else xx
        yy=y[base:end];dd=d[base:end];out=gw[base:end]
        for start in range(0,f*p*q,nt):
            _ggo_pack14[(triton.cdiv(c*9*nt,256),gb,2 if hw else 1)](
                xx,aa,v,v2,start,gb,f,c,h,wi,p,q,ph,pw,nt,True,hw,256)
            _ggo_mm14[lambda m:(triton.cdiv(o,m['BM'])*triton.cdiv(nt,m['BN']),gb)](
                ww,aw1,v,v2,bb,yy,start,gb,f,c,o,p,q,nt,True,hw,hb,precision)
            _shared_gw15[lambda m:(triton.cdiv(o,m['BM'])*triton.cdiv(c*9,m['BN']),gb)](
                dd,v,out,corr,start,gb,f,c,o,p,q,nt,start==0,precision,enable_fp_fusion=False)
        del v,v2,corr


@triton.jit
def _gi_parity_weight16(D,AW,Y,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
        H:tl.constexpr,W:tl.constexpr,O:tl.constexpr,R:tl.constexpr,S:tl.constexpr,
        P:tl.constexpr,Q:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
        PH:tl.constexpr,PW:tl.constexpr,DH:tl.constexpr,DW:tl.constexpr,
        PREC:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,PHASE:tl.constexpr):
    tile=tl.program_id(0);g=tl.program_id(1);phase:tl.constexpr=PHASE
    rh:tl.constexpr=phase//2;rw:tl.constexpr=phase%2
    hh:tl.constexpr=(H+1-rh)//2;ww:tl.constexpr=(W+1-rw)//2
    # Uniform grid covers the largest phase; inactive columns are masked.
    nn=tl.cdiv(F*tl.cdiv(H,2)*tl.cdiv(W,2),BN)
    ci=tile//nn*BM+tl.arange(0,BM)
    n=tile%nn*BN+tl.arange(0,BN)
    batch=n//(hh*ww);iy=(n//ww%hh)*2+rh;ix=(n%ww)*2+rw
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    r0:tl.constexpr=(rh+PH)%2;s0:tl.constexpr=(rw+PW)%2
    for rr in tl.static_range(r0,R,2):
        for ss in tl.static_range(s0,S,2):
            py=(iy+PH-rr)//2;qx=(ix+PW-ss)//2
            valid=(n<F*hh*ww)&(py>=0)&(py<P)&(qx>=0)&(qx<Q)
            for cb in range(tl.cdiv(O,BK)):
                oc=cb*BK+kk
                a=tl.load(AW+((g*9+rr*3+ss)*C+ci[:,None])*O+oc[None,:],
                          (ci[:,None]<C)&(oc[None,:]<O),0)
                b=tl.load(D+(g*F+batch[None,:])*O*P*Q+oc[:,None]*P*Q+py[None,:]*Q+qx[None,:],
                          (oc[:,None]<O)&valid[None,:],0)
                acc=tl.dot(a,b,acc,input_precision=PREC)
    tl.store(Y+((g*F+batch[None,:])*C+ci[:,None])*H*W+iy[None,:]*W+ix[None,:],acc,
             (ci[:,None]<C)&(n[None,:]<F*hh*ww))


@triton.autotune(configs=_configs(),key=_SHAPE_KEY)
@triton.jit
def _gi_static_weight16(D,AW,Y,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
        H:tl.constexpr,W:tl.constexpr,O:tl.constexpr,R:tl.constexpr,S:tl.constexpr,
        P:tl.constexpr,Q:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
        PH:tl.constexpr,PW:tl.constexpr,DH:tl.constexpr,DW:tl.constexpr,
        PREC:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    for phase_id in tl.static_range(4):
        if tl.program_id(2)==phase_id:
            _gi_parity_weight16(D,AW,Y,G,F,C,H,W,O,R,S,P,Q,SH,SW,PH,PW,DH,DW,PREC,BM,BN,BK,phase_id)


def _gi_weight16(d,aw,out,group_batch,dims):
    g,f,c,h,w,o,r,s,p,q,sh,sw,ph,pw,dh,dw,prec=dims
    for start in range(0,g,group_batch):
        end=min(g,start+group_batch);gb=end-start
        packed=torch.empty((gb,9,c,o),device=d.device,dtype=torch.float32)
        _gi_weight_pack14[(triton.cdiv(c,16)*triton.cdiv(o,32),gb,9)](aw[start:end],packed,gb,c,o)
        _gi_static_weight16[lambda m:(triton.cdiv(c,m['BM'])*triton.cdiv(f*triton.cdiv(h,2)*triton.cdiv(w,2),m['BN']),gb,4)](
            d[start:end],packed,out[start:end],gb,*dims[1:])
        del packed



# v17 opt-in dgrad: phase-local contiguous KxN activation packing.
@triton.jit
def _gi_pack_d17(D,V,F:tl.constexpr,O:tl.constexpr,H:tl.constexpr,W:tl.constexpr,
        P:tl.constexpr,Q:tl.constexpr,PH:tl.constexpr,PW:tl.constexpr,
        RH:tl.constexpr,RW:tl.constexpr,NT:tl.constexpr,START,BLOCK:tl.constexpr):
    g=tl.program_id(1);i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    HH:tl.constexpr=(H+1-RH)//2;WW:tl.constexpr=(W+1-RW)//2
    R0:tl.constexpr=(RH+PH)%2;S0:tl.constexpr=(RW+PW)%2
    NR:tl.constexpr=(3-R0+1)//2;NS:tl.constexpr=(3-S0+1)//2
    K:tl.constexpr=O*NR*NS
    k=i//NT;n=START+i%NT;oc=k%O;rs=k//O
    rr=R0+2*(rs//NS);ss=S0+2*(rs%NS)
    b=n//(HH*WW);iy=(n//WW%HH)*2+RH;ix=(n%WW)*2+RW
    py=(iy+PH-rr)//2;qx=(ix+PW-ss)//2
    x=tl.load(D+((g*F+b)*O+oc)*P*Q+py*Q+qx,
              (i<K*NT)&(b<F)&(py>=0)&(py<P)&(qx>=0)&(qx<Q),0)
    tl.store(V+g*K*NT+i,x,i<K*NT)


@triton.jit
def _gi_pack_w17(AW,A,C:tl.constexpr,O:tl.constexpr,PH:tl.constexpr,PW:tl.constexpr,
        RH:tl.constexpr,RW:tl.constexpr,BC:tl.constexpr,BO:tl.constexpr):
    R0:tl.constexpr=(RH+PH)%2;S0:tl.constexpr=(RW+PW)%2
    NR:tl.constexpr=(3-R0+1)//2;NS:tl.constexpr=(3-S0+1)//2
    K:tl.constexpr=O*NR*NS
    g=tl.program_id(1);rs=tl.program_id(2)
    ci=tl.program_id(0)//tl.cdiv(O,BO)*BC+tl.arange(0,BC)
    oc=tl.program_id(0)%tl.cdiv(O,BO)*BO+tl.arange(0,BO)
    rr=R0+2*(rs//NS);ss=S0+2*(rs%NS)
    x=tl.load(AW+((g*O+oc[None,:])*C+ci[:,None])*9+rr*3+ss,
              (ci[:,None]<C)&(oc[None,:]<O),0)
    tl.store(A+(g*C+ci[:,None])*K+rs*O+oc[None,:],x,
             (ci[:,None]<C)&(oc[None,:]<O))


@triton.autotune(configs=_configs(),key=['G','F','C','O','H','W','PH','PW','RH','RW','NT','PREC'])
@triton.jit
def _gi_mm17(A,V,Y,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,O:tl.constexpr,
        H:tl.constexpr,W:tl.constexpr,PH:tl.constexpr,PW:tl.constexpr,
        RH:tl.constexpr,RW:tl.constexpr,NT:tl.constexpr,START,PREC:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    HH:tl.constexpr=(H+1-RH)//2;WW:tl.constexpr=(W+1-RW)//2
    R0:tl.constexpr=(RH+PH)%2;S0:tl.constexpr=(RW+PW)%2
    K:tl.constexpr=O*((3-R0+1)//2)*((3-S0+1)//2)
    g=tl.program_id(1);pid=tl.program_id(0)
    ci=pid//tl.cdiv(NT,BN)*BM+tl.arange(0,BM)
    ni=pid%tl.cdiv(NT,BN)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for base in range(tl.cdiv(K,BK)):
        k=base*BK+kk
        a=tl.load(A+(g*C+ci[:,None])*K+k[None,:],(ci[:,None]<C)&(k[None,:]<K),0)
        v=tl.load(V+(g*K+k[:,None])*NT+ni[None,:],(k[:,None]<K)&(ni[None,:]<NT),0)
        acc=tl.dot(a,v,acc,input_precision=PREC)
    n=START+ni;b=n//(HH*WW);iy=(n//WW%HH)*2+RH;ix=(n%WW)*2+RW
    tl.store(Y+((g*F+b[None,:])*C+ci[:,None])*H*W+iy[None,:]*W+ix[None,:],acc,
             (ci[:,None]<C)&(ni[None,:]<NT)&(b[None,:]<F))


def _gi_packed17(d,aw,out,plan,dims):
    g,f,c,h,w,o,r,s,p,q,sh,sw,ph,pw,dh,dw,prec=dims
    group_batch=plan['gi_groups17'];nt=plan['gi_nt17']
    for start in range(0,g,group_batch):
        end=min(g,start+group_batch);gb=end-start
        for rh in range(2):
            for rw in range(2):
                nr=(4-(rh+ph)%2)//2;ns=(4-(rw+pw)%2)//2;k=o*nr*ns
                total=f*((h+1-rh)//2)*((w+1-rw)//2)
                # Allocate once per phase and reuse for every tile chunk.
                a=torch.empty((gb,c,k),device=d.device,dtype=torch.float32)
                v=torch.empty((gb,k,nt),device=d.device,dtype=torch.float32)
                _gi_pack_w17[(triton.cdiv(c,16)*triton.cdiv(o,32),gb,nr*ns)](
                    aw[start:end],a,c,o,ph,pw,rh,rw,16,32)
                for pos in range(0,total,nt):
                    _gi_pack_d17[(triton.cdiv(k*nt,256),gb)](
                        d[start:end],v,f,o,h,w,p,q,ph,pw,rh,rw,nt,pos,256)
                    _gi_mm17[lambda m:(triton.cdiv(c,m['BM'])*triton.cdiv(nt,m['BN']),gb)](
                        a,v,out[start:end],gb,f,c,o,h,w,ph,pw,rh,rw,nt,pos,prec)
                del a,v


def _gi_packed18(d,aw,out,plan,dims):
    g,f,c,h,w,o,r,s,p,q,sh,sw,ph,pw,dh,dw,prec=dims
    group_batch=plan['gi_groups17'];nt=plan['gi_nt17']
    for start in range(0,g,group_batch):
        end=min(g,start+group_batch);gb=end-start
        for rh in range(2):
            for rw in range(2):
                nr=(4-(rh+ph)%2)//2;ns=(4-(rw+pw)%2)//2;k=o*nr*ns
                total=f*((h+1-rh)//2)*((w+1-rw)//2)
                # Allocate once per phase and reuse for every tile chunk.
                a=torch.empty((gb,c,k),device=d.device,dtype=torch.float32)
                v=torch.empty((gb,k,nt),device=d.device,dtype=torch.float32)
                _gi_pack_w17[(triton.cdiv(c,16)*triton.cdiv(o,32),gb,nr*ns)](
                    aw[start:end],a,c,o,ph,pw,rh,rw,16,32)
                for pos in range(0,total,nt):
                    # Reinterpret the same buffer prefix as [gb,k,active_nt].
                    # Packing and GEMM must use the SAME compact group/row stride.
                    active_nt=min(nt,triton.cdiv(total-pos,32)*32)
                    _gi_pack_d17[(triton.cdiv(k*active_nt,256),gb)](
                        d[start:end],v,f,o,h,w,p,q,ph,pw,rh,rw,active_nt,pos,256)
                    _gi_mm17[lambda m:(triton.cdiv(c,m['BM'])*triton.cdiv(active_nt,m['BN']),gb)](
                        a,v,out[start:end],gb,f,c,o,h,w,ph,pw,rh,rw,active_nt,pos,prec)
                del a,v


def _gw_plan19(g,f,c,o,p,q,split,budget):
    # All split-K partitions execute independently. If one group's complete
    # operands cannot fit, keep the previous implicit kernel (no serial waves).
    nt=triton.cdiv(f*p*q,32)*32
    operands=4*(c*9+o)*nt
    partial=4*split*o*c*9 if split>1 else 0
    gb=min(g,int(budget)//(operands+partial))
    return dict(gw_groups19=gb,gw_nt19=nt,gw_split19=split,
                gw_scratch19=gb*(operands+partial),
                gw_reason19='packed_parallel_splitk' if gb else 'workspace_fallback_implicit')


@triton.jit
def _gw_pack_d19(D,A,F:tl.constexpr,O:tl.constexpr,P:tl.constexpr,Q:tl.constexpr,
                 NT:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);g=tl.program_id(1)
    oc=i//NT;n=i%NT;b=n//(P*Q);pos=n%(P*Q)
    x=tl.load(D+((g*F+b)*O+oc)*P*Q+pos,(oc<O)&(b<F),0)
    tl.store(A+g*O*NT+i,x,oc<O)


@triton.autotune(configs=_configs(),key=['G','C','O','NT','TOTAL','SPLIT','PREC'])
@triton.jit
def _gw_mm19(A,V,T,G:tl.constexpr,C:tl.constexpr,O:tl.constexpr,
        NT:tl.constexpr,TOTAL:tl.constexpr,SPLIT:tl.constexpr,PREC:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0);gs=tl.program_id(1);g=gs//SPLIT;sp=gs%SPLIT
    oc=pid//tl.cdiv(C*9,BN)*BM+tl.arange(0,BM)
    ci=pid%tl.cdiv(C*9,BN)*BN+tl.arange(0,BN)
    # Same split across all output tiles. Only the final reduction writes gW.
    chunk:tl.constexpr=((TOTAL+SPLIT*BK-1)//(SPLIT*BK))*BK
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for base in range(chunk//BK):
        n=sp*chunk+base*BK+kk
        a=tl.load(A+(g*O+oc[:,None])*NT+n[None,:],(oc[:,None]<O)&(n[None,:]<TOTAL),0)
        v=tl.load(V+(g*C*9+ci[None,:])*NT+n[:,None],(ci[None,:]<C*9)&(n[:,None]<TOTAL),0)
        acc=tl.dot(a,v,acc,input_precision=PREC)
    tl.store(T+(gs*O+oc[:,None])*C*9+ci[None,:],acc,
             (oc[:,None]<O)&(ci[None,:]<C*9))


def _gw_packed19(ai,d,out,plan,dims,split):
    g,f,c,h,w,o,r,s,p,q,sh,sw,ph,pw,dh,dw,prec=dims
    group_batch=plan['gw_groups19'];nt=plan['gw_nt19']
    for start in range(0,g,group_batch):
        end=min(g,start+group_batch);gb=end-start
        v=torch.empty((gb,c*9,nt),device=d.device,dtype=torch.float32)
        a=torch.empty((gb,o,nt),device=d.device,dtype=torch.float32)
        tmp=torch.empty((gb,split,o,c,3,3),device=d.device,dtype=torch.float32) if split>1 else out[start:end]
        _ggo_pack14[(triton.cdiv(c*9*nt,256),gb)](ai[start:end],ai[start:end],v,v,0,
                    gb,f,c,h,w,p,q,ph,pw,nt,True,False,256)
        _gw_pack_d19[(triton.cdiv(o*nt,256),gb)](d[start:end],a,f,o,p,q,nt,256)
        _gw_mm19[lambda m:(triton.cdiv(o,m['BM'])*triton.cdiv(c*9,m['BN']),gb*split)](
                    a,v,tmp,gb,c,o,nt,f*p*q,split,prec)
        if split>1:_reduce[(triton.cdiv(gb*o*c*9,256),)](tmp,out[start:end],gb*o*c*9,o*c*9,split,256)
        del a,v,tmp


@triton.jit
def _gw_filter20(AI,D,Y,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
        H:tl.constexpr,W:tl.constexpr,O:tl.constexpr,R:tl.constexpr,S:tl.constexpr,
        P:tl.constexpr,Q:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
        PH:tl.constexpr,PW:tl.constexpr,DH:tl.constexpr,DW:tl.constexpr,
        PREC:tl.constexpr,SPLIT:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
        RS:tl.constexpr):
    rr:tl.constexpr=RS//3;ss:tl.constexpr=RS%3
    # Enumerate only output positions that map inside this filter's input.
    ylo:tl.constexpr=(PH-rr+1)//2 if PH>rr else 0
    xlo:tl.constexpr=(PW-ss+1)//2 if PW>ss else 0
    ye:tl.constexpr=(H-1+PH-rr)//2+1
    yhi:tl.constexpr=ye if ye<P else P
    xe:tl.constexpr=(W-1+PW-ss)//2+1
    xhi:tl.constexpr=xe if xe<Q else Q
    vh:tl.constexpr=yhi-ylo if yhi>ylo else 0
    vw:tl.constexpr=xhi-xlo if xhi>xlo else 0
    pid=tl.program_id(0);gs=tl.program_id(1);g=gs//SPLIT;part=gs%SPLIT
    ci=pid//tl.cdiv(O,BN)*BM+tl.arange(0,BM)
    oc=pid%tl.cdiv(O,BN)*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),tl.float32)
    if vh>0 and vw>0:
        k=part*BK+tl.arange(0,BK)
        f=k//(vh*vw);yy=k//vw%vh;xx=k%vw
        aoff=(g*F+f)*C*H*W+((yy+ylo)*2-PH+rr)*W+(xx+xlo)*2-PW+ss
        doff=(g*F+f)*O*P*Q+(yy+ylo)*Q+xx+xlo
        step:tl.constexpr=SPLIT*BK
        dx:tl.constexpr=step%vw;dy:tl.constexpr=(step//vw)%vh;df:tl.constexpr=step//(vh*vw)
        CHUNKED:tl.constexpr=F*vh*vw>=SPLIT*8192
        INNER:tl.constexpr=512//BK if CHUNKED else 1
        correction=tl.zeros((BM,BN),tl.float32)
        for chunk in range((F*vh*vw+SPLIT*BK*INNER-1)//(SPLIT*BK*INNER)):
            partial=tl.zeros((BM,BN),tl.float32)
            for block in range(INNER):
                a=tl.load(AI+ci[:,None]*H*W+aoff[None,:],(ci[:,None]<C)&(f[None,:]<F),0)
                d=tl.load(D+oc[None,:]*P*Q+doff[:,None],(oc[None,:]<O)&(f[:,None]<F),0)
                if CHUNKED:partial=tl.dot(a,d,partial,input_precision=PREC)
                else:acc=tl.dot(a,d,acc,input_precision=PREC)
                # Update coordinates and pointer offsets with adds/carries only.
                # No batch/spatial division or convolution bounds tests in loop.
                tx=xx+dx;cx=tx>=vw;nx=tx-tl.where(cx,vw,0)
                ty=yy+dy+cx.to(tl.int32);cy=ty>=vh;ny=ty-tl.where(cy,vh,0)
                fd=df+cy.to(tl.int32);xd=nx-xx;yd=ny-yy
                aoff+=fd*C*H*W+yd*2*W+xd*2
                doff+=fd*O*P*Q+yd*Q+xd
                f+=fd;xx=nx;yy=ny
            if CHUNKED:
                delta=partial-correction;updated=acc+delta
                correction=(updated-acc)-delta;acc=updated
    tl.store(Y+(gs*O+oc[None,:])*C*9+ci[:,None]*9+RS,acc,
             (ci[:,None]<C)&(oc[None,:]<O))


@triton.autotune(configs=_configs(),key=_SHAPE_KEY+['SPLIT'])
@triton.jit
def _gw_static20(AI,D,Y,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
        H:tl.constexpr,W:tl.constexpr,O:tl.constexpr,R:tl.constexpr,S:tl.constexpr,
        P:tl.constexpr,Q:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
        PH:tl.constexpr,PW:tl.constexpr,DH:tl.constexpr,DW:tl.constexpr,
        PREC:tl.constexpr,SPLIT:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    for rs in tl.static_range(9):
        if tl.program_id(2)==rs:
            _gw_filter20(AI,D,Y,G,F,C,H,W,O,R,S,P,Q,SH,SW,PH,PW,DH,DW,PREC,SPLIT,BM,BN,BK,rs)


def _small_wgrad_configs26():
    # Late ResNet stages have large C/O but only 8x8 or 4x4 output lattices.
    # Wider channel tiles expose more reuse than the generic 32x64-biased set.
    return [triton.Config(dict(BM=m,BN=n,BK=k),num_warps=w,num_stages=st)
            for m,n,k,w,st in [(32,64,32,4,3),(64,64,16,4,3),
                (64,64,32,4,3),(64,128,16,8,3),(64,128,32,8,3),
                (128,32,32,4,3),(128,64,16,8,3),(128,64,32,8,3)]]


@triton.autotune(configs=_small_wgrad_configs26(),key=_SHAPE_KEY+['SPLIT'])
@triton.jit
def _gw_small26(AI,D,Y,G:tl.constexpr,F:tl.constexpr,C:tl.constexpr,
        H:tl.constexpr,W:tl.constexpr,O:tl.constexpr,R:tl.constexpr,S:tl.constexpr,
        P:tl.constexpr,Q:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
        PH:tl.constexpr,PW:tl.constexpr,DH:tl.constexpr,DW:tl.constexpr,
        PREC:tl.constexpr,SPLIT:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    for rs in tl.static_range(9):
        if tl.program_id(2)==rs:
            _gw_filter20(AI,D,Y,G,F,C,H,W,O,R,S,P,Q,SH,SW,PH,PW,DH,DW,
                         PREC,SPLIT,BM,BN,BK,rs)


@triton.autotune(configs=_mm7_configs(),key=['G','F','C','O','P','Q','NT','HI','HW','HB','PREC'])
@triton.jit
def _ggo_single23(W,AW,V0,V1,Bias,Y,start,G:tl.constexpr,F:tl.constexpr,
                  C:tl.constexpr,O:tl.constexpr,P:tl.constexpr,Q:tl.constexpr,NT:tl.constexpr,
                  HI:tl.constexpr,HW:tl.constexpr,HB:tl.constexpr,PREC:tl.constexpr,
                  BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    # Virtual concatenation [W,ggW] @ [V(ggI);V(input)]. No concat allocation.
    # One accumulator, unlike the two independent sums in _ggo_mm14.
    pid=tl.program_id(0);g=tl.program_id(1)
    nn=tl.cdiv(NT,BN);nm=tl.cdiv(O,BM)
    first=pid//(4*nn)*4;count=tl.minimum(nm-first,4)
    mi=first+pid%(4*nn)%count;ni=pid%(4*nn)//count
    oc=mi*BM+tl.arange(0,BM);n=ni*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    KBLOCKS:tl.constexpr=(C*9+BK-1)//BK
    TERMS:tl.constexpr=2 if HI and HW else 1
    for kb in range(KBLOCKS*TERMS):
        if HI and HW:
            second=kb>=KBLOCKS
            k=(kb-tl.where(second,KBLOCKS,0))*BK+kk
            wp=tl.where(second,AW,W);vp=tl.where(second,V1,V0)
        else:
            k=kb*BK+kk
            wp=W if HI else AW;vp=V0 if HI else V1
        wo=(g*O+oc[:,None])*C*9+k[None,:]
        vo=(g*C*9+k[:,None])*NT+n[None,:]
        a=tl.load(wp+wo,(oc[:,None]<O)&(k[None,:]<C*9),0)
        b=tl.load(vp+vo,(k[:,None]<C*9)&(n[None,:]<NT),0)
        acc=tl.dot(a,b,acc,input_precision=PREC)
    if HB:acc+=tl.load(Bias+g*O+oc,oc<O,0)[:,None]
    t=start+n;batch=t//(P*Q);pos=t%(P*Q)
    tl.store(Y+((g*F+batch[None,:])*O+oc[:,None])*P*Q+pos[None,:],acc,
             (oc[:,None]<O)&(n[None,:]<NT)&(t[None,:]<F*P*Q))


def _ggo_packed23(x,w,ai,aw,bias,out,plan,ph,pw,precision):
    g,f,c,h,wi=x.shape;_,_,o,p,q=out.shape
    hi=ai is not None;hw=aw is not None;hb=bias is not None
    nt=plan['ggo_nt14'];group_batch=plan['ggo_groups14']
    mm=_ggo_single23 if plan.get('ggo_single23') else _ggo_mm14
    for base in range(0,g,group_batch):
        end=min(g,base+group_batch);gb=end-base
        v=torch.empty((gb,c*9,nt),device=x.device,dtype=torch.float32)
        v2=torch.empty_like(v) if hi and hw else v
        xx=x[base:end];aa=ai[base:end] if hi else xx
        ww=w[base:end];aw1=aw[base:end] if hw else ww
        bb=bias[base:end] if hb else xx;yy=out[base:end]
        for start in range(0,f*p*q,nt):
            active_nt=min(nt,triton.cdiv(f*p*q-start,32)*32)
            # Reinterpret buffer prefix compactly; both pack and mm receive
            # active_nt as group/row stride. Original allocation is reused.
            _ggo_pack14[(triton.cdiv(c*9*active_nt,256),gb,2 if hi and hw else 1)](
                xx,aa,v,v2,start,gb,f,c,h,wi,p,q,ph,pw,active_nt,hi,hw,256)
            mm[lambda m:(triton.cdiv(o,m['BM'])*triton.cdiv(active_nt,m['BN']),gb)](
                ww,aw1,v,v2,bb,yy,start,gb,f,c,o,p,q,active_nt,hi,hw,hb,precision)
        del v,v2


# v30 complete-call path.  One packed buffer is deliberately shared by all
# three contractions: pack ggI -> ggO(W, ggI) + gW(gO, ggI), then overwrite it
# with packed input -> ggO(ggW, input).  The two ggO launches preserve the
# eager two-convolution rounding boundary while eliminating v27's second
# im2col buffer and v25's many large gW partials on the important B=100/G=4
# stage-3 downsample.
@triton.autotune(configs=_mm7_configs(),
                 key=['G','F','C','O','P','Q','NT','ADD','HB','PREC'],
                 restore_value=['Y'])
@triton.jit
def _ggo_onebuf_mm30(W,V,Bias,Y,start,G:tl.constexpr,F:tl.constexpr,
        C:tl.constexpr,O:tl.constexpr,P:tl.constexpr,Q:tl.constexpr,
        NT:tl.constexpr,ADD:tl.constexpr,HB:tl.constexpr,PREC:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0);g=tl.program_id(1)
    nn=tl.cdiv(NT,BN);nm=tl.cdiv(O,BM)
    first=pid//(4*nn)*4;count=tl.minimum(nm-first,4)
    mi=first+pid%(4*nn)%count;ni=pid%(4*nn)//count
    oc=mi*BM+tl.arange(0,BM);n=ni*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for kb in range(tl.cdiv(C*9,BK)):
        k=kb*BK+kk
        wo=(g*O+oc[:,None])*C*9+k[None,:]
        vo=(g*C*9+k[:,None])*NT+n[None,:]
        acc=tl.dot(tl.load(W+wo,(oc[:,None]<O)&(k[None,:]<C*9),0),
                   tl.load(V+vo,(k[:,None]<C*9)&(n[None,:]<NT),0),
                   acc,input_precision=PREC)
    t=start+n;batch=t//(P*Q);pos=t%(P*Q)
    yo=((g*F+batch[None,:])*O+oc[:,None])*P*Q+pos[None,:]
    mask=(oc[:,None]<O)&(n[None,:]<NT)&(t[None,:]<F*P*Q)
    value=acc.to(Y.dtype.element_ty)
    if ADD:
        value=(tl.load(Y+yo,mask,0)+value).to(Y.dtype.element_ty)
    if HB:
        value+=tl.load(Bias+g*O+oc,oc<O,0)[:,None]
    tl.store(Y+yo,value,mask)


def _joint_onebuf30(x,w,ai,aw,bias,d,out,gw,plan,ph,pw,precision):
    g,f,c,h,wi=x.shape;_,_,o,p,q=out.shape
    hw=aw is not None;hb=bias is not None
    nt=plan['onebuf_nt30'];group_batch=plan['onebuf_groups30']
    chunks=plan['onebuf_chunks30'];total=f*p*q
    for base in range(0,g,group_batch):
        end=min(g,base+group_batch);gb=end-base
        v=torch.empty((gb,c*9,nt),device=x.device,dtype=torch.float32)
        xx=x[base:end];aa=ai[base:end];ww=w[base:end]
        aw1=aw[base:end] if hw else ww
        bb=bias[base:end] if hb else xx
        yy=out[base:end];dd=d[base:end];target=gw[base:end]
        correction=torch.empty_like(target)
        for part,start in enumerate(range(0,total,nt)):
            active_nt=min(nt,triton.cdiv(total-start,32)*32)
            _ggo_pack14[(triton.cdiv(c*9*active_nt,256),gb,1)](
                xx,aa,v,v,start,gb,f,c,h,wi,p,q,ph,pw,active_nt,
                True,False,256)
            _ggo_onebuf_mm30[
                lambda m:(triton.cdiv(o,m['BM'])*
                          triton.cdiv(active_nt,m['BN']),gb)](
                ww,v,bb,yy,start,gb,f,c,o,p,q,active_nt,
                False,hb and not hw,precision)
            _shared_gw15[
                lambda m:(triton.cdiv(o,m['BM'])*
                          triton.cdiv(c*9,m['BN']),gb)](
                dd,v,target,correction,start,gb,f,c,o,p,q,active_nt,
                part==0,precision)
            if hw:
                _ggo_pack14[(triton.cdiv(c*9*active_nt,256),gb,1)](
                    xx,xx,v,v,start,gb,f,c,h,wi,p,q,ph,pw,active_nt,
                    False,True,256)
                _ggo_onebuf_mm30[
                    lambda m:(triton.cdiv(o,m['BM'])*
                              triton.cdiv(active_nt,m['BN']),gb)](
                    aw1,v,bb,yy,start,gb,f,c,o,p,q,active_nt,
                    True,hb,precision)
        del v,correction


# v32 large-N experiment. The mathematical decomposition is unchanged from
# v30/v31, but gW no longer assigns the complete long N reduction to one CTA.
# Split-K partials are reduced once per packed chunk, while the same packed ggI
# matrix remains available to ggO. Larger output tiles are tested only in this
# opt-in path so earlier candidates and the default selector stay untouched.
def _large_mm_configs32():
    return [triton.Config(dict(BM=m,BN=n,BK=k),num_warps=w,num_stages=st)
            for m,n,k,w,st in [(32,64,32,4,3),(32,128,32,4,3),
                (64,64,32,4,3),(64,64,64,4,3),(64,128,32,8,3),
                (128,64,32,8,3),(128,128,32,8,2)]]


@triton.autotune(configs=_large_mm_configs32(),
                 key=['G','F','C','O','P','Q','NT','ADD','HB','PREC'],
                 restore_value=['Y'])
@triton.jit
def _ggo_onebuf_mm32(W,V,Bias,Y,start,G:tl.constexpr,F:tl.constexpr,
        C:tl.constexpr,O:tl.constexpr,P:tl.constexpr,Q:tl.constexpr,
        NT:tl.constexpr,ADD:tl.constexpr,HB:tl.constexpr,PREC:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0);g=tl.program_id(1)
    nn=tl.cdiv(NT,BN);nm=tl.cdiv(O,BM)
    first=pid//(4*nn)*4;count=tl.minimum(nm-first,4)
    mi=first+pid%(4*nn)%count;ni=pid%(4*nn)//count
    oc=mi*BM+tl.arange(0,BM);n=ni*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for kb in range(tl.cdiv(C*9,BK)):
        k=kb*BK+kk
        wo=(g*O+oc[:,None])*C*9+k[None,:]
        vo=(g*C*9+k[:,None])*NT+n[None,:]
        acc=tl.dot(tl.load(W+wo,(oc[:,None]<O)&(k[None,:]<C*9),0),
                   tl.load(V+vo,(k[:,None]<C*9)&(n[None,:]<NT),0),
                   acc,input_precision=PREC)
    t=start+n;batch=t//(P*Q);pos=t%(P*Q)
    yo=((g*F+batch[None,:])*O+oc[:,None])*P*Q+pos[None,:]
    mask=(oc[:,None]<O)&(n[None,:]<NT)&(t[None,:]<F*P*Q)
    value=acc.to(Y.dtype.element_ty)
    if ADD:value=(tl.load(Y+yo,mask,0)+value).to(Y.dtype.element_ty)
    if HB:value+=tl.load(Bias+g*O+oc,oc<O,0)[:,None]
    tl.store(Y+yo,value,mask)


@triton.autotune(configs=_large_mm_configs32(),
                 key=['G','F','C','O','P','Q','NT','SPLIT','PREC'])
@triton.jit
def _shared_gw_split32(D,V,T,start,G:tl.constexpr,F:tl.constexpr,
        C:tl.constexpr,O:tl.constexpr,P:tl.constexpr,Q:tl.constexpr,
        NT:tl.constexpr,SPLIT:tl.constexpr,PREC:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0);gs=tl.program_id(1);g=gs//SPLIT;part=gs%SPLIT
    nn=tl.cdiv(C*9,BN)
    oc=pid//nn*BM+tl.arange(0,BM);k=pid%nn*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    # Contiguous split ranges retain coalescing in both packed V and D. The
    # final split is masked instead of padding or materializing another input.
    # Triton 3.1 cannot lower tl.cdiv() inside a locally annotated constexpr
    # assignment (it reports the ast.Call itself). Use the same pure integer
    # compile-time form already exercised by _gw_mm19 and _pointwise_gw26.
    chunk:tl.constexpr=((NT+SPLIT*BK-1)//(SPLIT*BK))*BK
    for nb in range(chunk//BK):
        n=part*chunk+nb*BK+kk;t=start+n
        batch=t//(P*Q);pos=t%(P*Q)
        d=tl.load(D+((g*F+batch[None,:])*O+oc[:,None])*P*Q+pos[None,:],
                  (oc[:,None]<O)&(n[None,:]<NT)&(t[None,:]<F*P*Q),0)
        v=tl.load(V+(g*C*9+k[None,:])*NT+n[:,None],
                  (k[None,:]<C*9)&(n[:,None]<NT),0)
        acc=tl.dot(d,v,acc,input_precision=PREC)
    off=(gs*O+oc[:,None])*C*9+k[None,:]
    tl.store(T+off,acc,(oc[:,None]<O)&(k[None,:]<C*9))


@triton.jit
def _gw_reduce_acc32(T,Y,Corr,N:tl.constexpr,PER:tl.constexpr,
        SPLIT:tl.constexpr,FIRST:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);g=i//PER;j=i%PER
    valid=i<N;acc=tl.full((BLOCK,),0,tl.float32)
    for s in tl.static_range(SPLIT):
        acc+=tl.load(T+(g*SPLIT+s)*PER+j,valid,0)
    if FIRST:
        updated=acc;correction=tl.full((BLOCK,),0,tl.float32)
    else:
        old=tl.load(Y+i,valid,0);prev=tl.load(Corr+i,valid,0)
        delta=acc-prev;updated=old+delta;correction=(updated-old)-delta
    tl.store(Y+i,updated,valid);tl.store(Corr+i,correction,valid)


def _joint_split_gw32(x,w,ai,aw,bias,d,out,gw,plan,ph,pw,precision):
    g,f,c,h,wi=x.shape;_,_,o,p,q=out.shape
    hw=aw is not None;hb=bias is not None
    nt=plan['split_gw_nt32'];group_batch=plan['split_gw_groups32']
    split=plan['gw_split32'];total=f*p*q
    for base in range(0,g,group_batch):
        end=min(g,base+group_batch);gb=end-base
        v=torch.empty((gb,c*9,nt),device=x.device,dtype=torch.float32)
        partial=torch.empty((gb,split,o,c,3,3),device=x.device,
                            dtype=torch.float32)
        xx=x[base:end];aa=ai[base:end];ww=w[base:end]
        aw1=aw[base:end] if hw else ww
        bb=bias[base:end] if hb else xx
        yy=out[base:end];dd=d[base:end];target=gw[base:end]
        correction=torch.empty_like(target)
        for part,start in enumerate(range(0,total,nt)):
            active_nt=min(nt,triton.cdiv(total-start,32)*32)
            _ggo_pack14[(triton.cdiv(c*9*active_nt,256),gb,1)](
                xx,aa,v,v,start,gb,f,c,h,wi,p,q,ph,pw,active_nt,
                True,False,256)
            _ggo_onebuf_mm32[
                lambda m:(triton.cdiv(o,m['BM'])*
                          triton.cdiv(active_nt,m['BN']),gb)](
                ww,v,bb,yy,start,gb,f,c,o,p,q,active_nt,
                False,hb and not hw,precision)
            _shared_gw_split32[
                lambda m:(triton.cdiv(o,m['BM'])*
                          triton.cdiv(c*9,m['BN']),gb*split)](
                dd,v,partial,start,gb,f,c,o,p,q,active_nt,split,precision)
            ne=gb*o*c*9
            _gw_reduce_acc32[(triton.cdiv(ne,256),)](
                partial,target,correction,ne,o*c*9,split,part==0,256,
                enable_fp_fusion=False)
            if hw:
                _ggo_pack14[(triton.cdiv(c*9*active_nt,256),gb,1)](
                    xx,xx,v,v,start,gb,f,c,h,wi,p,q,ph,pw,active_nt,
                    False,True,256)
                _ggo_onebuf_mm32[
                    lambda m:(triton.cdiv(o,m['BM'])*
                              triton.cdiv(active_nt,m['BN']),gb)](
                    aw1,v,bb,yy,start,gb,f,c,o,p,q,active_nt,
                    True,hb,precision)
        del v,partial,correction









# # ############################
# # ############################
# # ############################
# # ############################
# # ############################
#  # 1 st bwd
 
# # ############################
# # ############################
# # ############################
# # ############################
# # ############################
# # ############################


# """Group-first ConvNet 3x3 stride=padding=dilation=1 convolution backward.

# conv_bwd(x, weight, grad_output, stride=1, padding=1, dilation=1, groups=1)
# returns grad_x [G,B,C,H,W], grad_weight [G,O,C,3,3], grad_bias [G,O].

# v13 adds compact RGB-input kernels for ConvNet's first convolution.  Its dX
# kernel reuses each dY load across exactly three scalar input-channel
# accumulators, while its dW/dB kernel contracts the compact C*3*3=27 axis and
# uses split reduction only when needed for GPU occupancy.  v12 adds a
# two-launch, zero-workspace direct path for small dense workloads: one
# implicit-GEMM kernel writes dX directly and the existing direct row kernel
# computes dW/dB together.  A bmm+fold dX ablation is also exposed.
# v11: Winograd F(4x4,3x3) for both dX and dW.  The joint candidate
# loads each 6x6 dY tile once and emits both transforms needed by dX and dW.
# dX uses the Triton contraction; dW uses batched GEMM directly on native
# Winograd buffers.  v10 retains the all-Triton joint contraction as a control.
# v9 retains v7 dX, Winograd F(4x4,3x3) dW, and separate dB.  F(4,3)
# uses 36 transform components for sixteen output positions rather than v7's
# 16 components for four positions.  The dW GEMM deliberately retains the
# proven manual-pointer 64x64x16/4-warp schedule after v8's block-pointer
# 64x64x32/8-warp experiment regressed on RTX A5000.
# Triton contractions use IEEE FP32.  Batched GEMM follows PyTorch's matmul
# precision setting; the benchmark explicitly disables TF32 for both sides.
# Temporary buffers are reused within this call and limited to 128 MiB per
# phase. No transformed values or tensors are cached across calls.
# conv_bwd_v7 and conv_bwd_v5 retain the previous explicit baselines.  The v8
# GEMM remains available only as a diagnostic launch helper.
# """
# import operator
# import torch
# import triton
# import triton.language as tl

# WORKSPACE_LIMIT_BYTES = 128 * 1024**2
# WG_TILE_V7 = (64, 64, 16)
# WG_TILE = (64, 64, 32)
# WG_WARPS = 8
# F4_COMPONENTS = 36

# def _pair(value):
#     try:
#         n = operator.index(value)
#     except TypeError:
#         values = tuple(map(operator.index, value))
#         if len(values) != 2:
#             raise ValueError("expected an integer or a pair of integers")
#         return values
#     return n, n

# @triton.jit
# def _dw_db_row_tile(X,D,DWOUT,DBOUT,C:tl.constexpr,O:tl.constexpr,
#                     H:tl.constexpr,W:tl.constexpr,B:tl.constexpr,ROW:tl.constexpr,
#                     BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
#     tile=tl.program_id(0)
#     g=tl.program_id(1)
#     no:tl.constexpr=(O+BN-1)//BN
#     ci=tile//no*BM+tl.arange(0,BM)
#     oc=tile%no*BN+tl.arange(0,BN)
#     vh:tl.constexpr=H-(1 if ROW!=1 else 0)
#     iy0:tl.constexpr=1 if ROW==2 else 0
#     dy0:tl.constexpr=1 if ROW==0 else 0
#     kk=tl.arange(0,BK)
#     a0=tl.zeros((BM,BN),tl.float32)
#     a1=tl.zeros((BM,BN),tl.float32)
#     a2=tl.zeros((BM,BN),tl.float32)
#     bias=tl.full((BN,),0,tl.float32)
#     for block in range((B*vh*W+BK-1)//BK):
#         k=block*BK+kk
#         batch=k//(vh*W)
#         pos=k%(vh*W)
#         y=pos//W
#         x=pos%W
#         valid=k<B*vh*W
#         dv=tl.load(D+(g*B+batch[:,None])*O*H*W+oc[None,:]*H*W+(y[:,None]+dy0)*W+x[:,None],
#                    (oc[None,:]<O)&valid[:,None],0)
#         base=(g*B+batch[None,:])*C*H*W+ci[:,None]*H*W+(y[None,:]+iy0)*W+x[None,:]
#         mask=(ci[:,None]<C)&valid[None,:]
#         x0=tl.load(X+base-1,mask&(x[None,:]>0),0)
#         a0=tl.dot(x0,dv,a0,input_precision="ieee")
#         x1=tl.load(X+base,mask,0)
#         a1=tl.dot(x1,dv,a1,input_precision="ieee")
#         x2=tl.load(X+base+1,mask&(x[None,:]+1<W),0)
#         a2=tl.dot(x2,dv,a2,input_precision="ieee")
#         if ROW==1:
#             if tile//no==0:
#                 bias+=tl.sum(dv,0)
#     target=DWOUT+g*O*C*9+oc[None,:]*C*9+ci[:,None]*9+ROW*3
#     mask=(ci[:,None]<C)&(oc[None,:]<O)
#     tl.store(target,a0,mask)
#     tl.store(target+1,a1,mask)
#     tl.store(target+2,a2,mask)
#     if ROW==1:
#         if tile//no==0:
#             tl.store(DBOUT+g*O+oc,bias,oc<O)

# @triton.jit
# def _dw_db_row(X,D,DWOUT,DBOUT,C:tl.constexpr,O:tl.constexpr,
#                H:tl.constexpr,W:tl.constexpr,B:tl.constexpr,
#                BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
#     for row in tl.static_range(3):
#         if tl.program_id(2)==row:
#             _dw_db_row_tile(X,D,DWOUT,DBOUT,C,O,H,W,B,row,BM,BN,BK)

# @triton.jit
# def _db_reduce(D, DBOUT, B: tl.constexpr, O: tl.constexpr,
#                HW: tl.constexpr, BLOCK: tl.constexpr):
#     oc = tl.program_id(0)
#     g = tl.program_id(1)
#     offsets = tl.arange(0, BLOCK)
#     acc = tl.zeros((BLOCK,), tl.float32)
#     # Bounded per-CTA state even as B*HW grows; every dY is read once.
#     for block in range((B * HW + BLOCK - 1) // BLOCK):
#         k = block * BLOCK + offsets
#         batch = k // HW
#         pos = k % HW
#         value = tl.load(D + ((g * B + batch) * O + oc) * HW + pos,
#                         k < B * HW, 0)
#         acc += value
#     tl.store(DBOUT + g * O + oc, tl.sum(acc, 0))

# @triton.jit
# def _gx_gemm(D,W,Z,B:tl.constexpr,BATCH,
#              C:tl.constexpr,O:tl.constexpr,HW:tl.constexpr,
#              BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
#     tile=tl.program_id(0)
#     g=tl.program_id(1)
#     nn:tl.constexpr=(HW+BN-1)//BN
#     crs=tile//nn*BM+tl.arange(0,BM)
#     n=tile%nn*BN+tl.arange(0,BN)
#     kk=tl.arange(0,BK)
#     acc=tl.zeros((BM,BN),tl.float32)
#     for block in range((O+BK-1)//BK):
#         oc=block*BK+kk
#         weights=tl.load(W+(g*O+oc[None,:])*C*9+crs[:,None],
#                         (crs[:,None]<C*9)&(oc[None,:]<O),0)
#         dy=tl.load(D+((g*B+BATCH)*O+oc[:,None])*HW+n[None,:],
#                    (oc[:,None]<O)&(n[None,:]<HW),0)
#         acc=tl.dot(weights,dy,acc,input_precision="ieee")
#     tl.store(Z+(g*C*9+crs[:,None])*HW+n[None,:],acc,
#              (crs[:,None]<C*9)&(n[None,:]<HW))

# @triton.jit
# def _gx_fold(Z,DX,G:tl.constexpr,B:tl.constexpr,BATCH,
#              C:tl.constexpr,H:tl.constexpr,W:tl.constexpr,BLOCK:tl.constexpr):
#     i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
#     pos=i%(H*W)
#     ci=i//(H*W)%C
#     g=i//(C*H*W)
#     iy=pos//W
#     ix=pos%W
#     acc=tl.full((BLOCK,),0,tl.float32)
#     for rr in tl.static_range(3):
#         for ss in tl.static_range(3):
#             py=iy+1-rr
#             px=ix+1-ss
#             offset=((g*C+ci)*9+rr*3+ss)*H*W+py*W+px
#             value=tl.load(Z+offset,(i<G*C*H*W)&(py>=0)&(py<H)&(px>=0)&(px<W),0)
#             acc+=value
#     tl.store(DX+((g*B+BATCH)*C+ci)*H*W+pos,acc,i<G*C*H*W)


# @triton.jit
# def _gx_implicit_direct(D,WEIGHT,DX,B:tl.constexpr,C:tl.constexpr,
#                         O:tl.constexpr,H:tl.constexpr,WI:tl.constexpr,
#                         BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
#     """Write dX directly: reduce [O,3,3] without materialized columns/fold."""
#     tile=tl.program_id(0);g=tl.program_id(1)
#     hw:tl.constexpr=H*WI
#     nall:tl.constexpr=B*hw
#     nn:tl.constexpr=(nall+BN-1)//BN
#     ci=tile//nn*BM+tl.arange(0,BM)
#     n=tile%nn*BN+tl.arange(0,BN)
#     batch=n//hw;pos=n%hw;iy=pos//WI;ix=pos%WI
#     kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
#     # k = (filter position)*O + output channel.  With O=128 and BK=16/32,
#     # each dot block normally keeps one spatial offset and contiguous dY
#     # channels while still reducing the complete O*3*3 dimension.
#     for block in range((O*9+BK-1)//BK):
#         k=block*BK+kk;rs=k//O;oc=k%O
#         rr=rs//3;ss=rs%3
#         weight=tl.load(
#             WEIGHT+((g*O+oc[None,:])*C+ci[:,None])*9+rs[None,:],
#             (ci[:,None]<C)&(k[None,:]<O*9),0)
#         py=iy[None,:]+1-rr[:,None]
#         px=ix[None,:]+1-ss[:,None]
#         dy=tl.load(
#             D+((g*B+batch[None,:])*O+oc[:,None])*hw+py*WI+px,
#             (k[:,None]<O*9)&(n[None,:]<nall)&
#             (py>=0)&(py<H)&(px>=0)&(px<WI),0)
#         acc=tl.dot(weight,dy,acc,input_precision="ieee")
#     tl.store(DX+((g*B+batch[None,:])*C+ci[:,None])*hw+pos[None,:],acc,
#              (ci[:,None]<C)&(n[None,:]<nall))


# @triton.jit
# def _rgb_dx_direct3(D, WEIGHT, DX, B:tl.constexpr, O:tl.constexpr,
#                     H:tl.constexpr, WI:tl.constexpr,
#                     BLOCK_N:tl.constexpr, BLOCK_K:tl.constexpr):
#     """C=3 dX without padded channel tiles or a materialized im2col buffer."""
#     n=tl.program_id(0)*BLOCK_N+tl.arange(0,BLOCK_N)
#     g=tl.program_id(1)
#     hw:tl.constexpr=H*WI
#     nall:tl.constexpr=B*hw
#     batch=n//hw;pos=n%hw;iy=pos//WI;ix=pos%WI
#     kk=tl.arange(0,BLOCK_K)
#     a0=tl.zeros((BLOCK_N,),tl.float32)
#     a1=tl.zeros((BLOCK_N,),tl.float32)
#     a2=tl.zeros((BLOCK_N,),tl.float32)
#     # A dY tile is shared by the three real input channels.  This avoids the
#     # 16/32-channel padding paid by a conventional small-M dot product.
#     for rs in tl.static_range(9):
#         # Keep the static expressions inline. Triton 3.1 rejects a constexpr
#         # local that is rebound by successive unrolled iterations.
#         py=iy+1-(rs//3);px=ix+1-(rs%3)
#         valid_pos=(n<nall)&(py>=0)&(py<H)&(px>=0)&(px<WI)
#         for ob in range((O+BLOCK_K-1)//BLOCK_K):
#             oc=ob*BLOCK_K+kk
#             dy=tl.load(D+((g*B+batch[None,:])*O+oc[:,None])*hw+
#                        py[None,:]*WI+px[None,:],
#                        (oc[:,None]<O)&valid_pos[None,:],0.0)
#             wbase=WEIGHT+(g*O+oc)*27+rs
#             w0=tl.load(wbase+0*9,oc<O,0.0)
#             w1=tl.load(wbase+1*9,oc<O,0.0)
#             w2=tl.load(wbase+2*9,oc<O,0.0)
#             a0+=tl.sum(dy*w0[:,None],axis=0)
#             a1+=tl.sum(dy*w1[:,None],axis=0)
#             a2+=tl.sum(dy*w2[:,None],axis=0)
#     out=DX+(g*B+batch)*3*hw+pos
#     tl.store(out+0*hw,a0,n<nall)
#     tl.store(out+1*hw,a1,n<nall)
#     tl.store(out+2*hw,a2,n<nall)


# @triton.jit
# def _rgb_dw_db_flat(X,D,DW,DB,PDW,PDB,B:tl.constexpr,O:tl.constexpr,
#                     H:tl.constexpr,WI:tl.constexpr,SPLIT:tl.constexpr,
#                     PARTIAL:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,
#                     BK:tl.constexpr):
#     """[O,K]@[K,27] dW and dB; optional contiguous split-K partials."""
#     pid=tl.program_id(0);gs=tl.program_id(1)
#     g=gs//SPLIT;part=gs%SPLIT
#     oc=pid*BM+tl.arange(0,BM)
#     crs=tl.arange(0,BN);ci=crs//9;rs=crs%9;rr=rs//3;ss=rs%3
#     hw:tl.constexpr=H*WI
#     total:tl.constexpr=B*hw
#     chunk:tl.constexpr=((total+SPLIT-1)//SPLIT+BK-1)//BK*BK
#     kk=tl.arange(0,BK)
#     acc=tl.zeros((BM,BN),tl.float32)
#     bias=tl.zeros((BM,),tl.float32)
#     for block in range((chunk+BK-1)//BK):
#         k=part*chunk+block*BK+kk
#         batch=k//hw;pos=k%hw;y=pos//WI;xpos=pos%WI
#         valid=k<total
#         dv=tl.load(D+((g*B+batch[None,:])*O+oc[:,None])*hw+pos[None,:],
#                    (oc[:,None]<O)&valid[None,:],0.0)
#         iy=y[:,None]+rr[None,:]-1
#         ix=xpos[:,None]+ss[None,:]-1
#         xv=tl.load(X+((g*B+batch[:,None])*3+ci[None,:])*hw+iy*WI+ix,
#                    valid[:,None]&(crs[None,:]<27)&
#                    (iy>=0)&(iy<H)&(ix>=0)&(ix<WI),0.0)
#         acc=tl.dot(dv,xv,acc,input_precision="ieee")
#         bias+=tl.sum(dv,axis=1)
#     mask=(oc[:,None]<O)&(crs[None,:]<27)
#     if PARTIAL:
#         tl.store(PDW+((g*SPLIT+part)*O+oc[:,None])*27+crs[None,:],acc,mask)
#         tl.store(PDB+(g*SPLIT+part)*O+oc,bias,oc<O)
#     else:
#         tl.store(DW+(g*O+oc[:,None])*27+crs[None,:],acc,mask)
#         tl.store(DB+g*O+oc,bias,oc<O)


# @triton.jit
# def _rgb_dw_db_finish(PDW,PDB,DW,DB,O:tl.constexpr,SPLIT:tl.constexpr,
#                       BLOCK:tl.constexpr):
#     i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);g=tl.program_id(1)
#     oc=i//28;col=i%28
#     acc=tl.zeros((BLOCK,),tl.float32)
#     for part in range(SPLIT):
#         w=tl.load(PDW+((g*SPLIT+part)*O+oc)*27+col,
#                   (oc<O)&(col<27),0.0)
#         b=tl.load(PDB+(g*SPLIT+part)*O+oc,(oc<O)&(col==27),0.0)
#         acc+=w+b
#     tl.store(DW+(g*O+oc)*27+col,acc,(oc<O)&(col<27))
#     tl.store(DB+g*O+oc,acc,(oc<O)&(col==27))

# def _validate(x, weight, d, stride, padding, dilation, groups):
#     if x.ndim != 5 or weight.ndim != 5 or d.ndim != 5:
#         raise ValueError("expected 5D group-first tensors")
#     g, b, c, h, w = x.shape
#     o = weight.shape[1]
#     if (min(g, b, c, o) < 1 or min(h, w) < 2 or
#         tuple(weight.shape) != (g, o, c, 3, 3) or
#         tuple(d.shape) != (g, b, o, h, w)):
#         raise ValueError("expected x=[G,B,C,H,W], weight=[G,O,C,3,3], dY=[G,B,O,H,W]; H,W>=2")
#     if (_pair(stride) != (1, 1) or _pair(padding) != (1, 1) or
#         _pair(dilation) != (1, 1) or operator.index(groups) not in (1, g)):
#         raise ValueError("this family requires stride=padding=dilation=1; groups=1 or G")
#     if (g > 65535 or max(g*b*c*h*w, g*b*o*h*w, g*o*c*9) >= 2**31):
#         raise ValueError("tensor indexing or CUDA grid limit exceeded")
#     if c * 9 * h * w * 4 > WORKSPACE_LIMIT_BYTES:
#         raise ValueError("one group's dX workspace exceeds the 128 MiB budget")
#     if any(not t.is_cuda or t.dtype != torch.float32 or t.device != x.device
#            for t in (x, weight, d)):
#         raise ValueError("expected FP32 tensors on the same CUDA device")
#     return g, b, c, o, h, w

# def grad_x_workspace(d, dx):
#     """Reuse one batch's buffer across B; chunk G to keep it <=128 MiB."""
#     g, _, c, h, w = dx.shape
#     per_group = c * 9 * h * w * dx.element_size()
#     if per_group > WORKSPACE_LIMIT_BYTES:
#         raise ValueError("one group's dX workspace exceeds the 128 MiB budget")
#     chunk = min(g, max(1, WORKSPACE_LIMIT_BYTES // per_group))
#     return d.new_empty((chunk, c * 9, h * w))

# def launch_gemm_one_batch(d, weight, z, batch):
#     g, b, o, h, w = d.shape
#     c = weight.shape[2]
#     grid = (triton.cdiv(c * 9, 32) * triton.cdiv(h * w, 64), g)
#     _gx_gemm[grid](d, weight, z, b, batch, c, o, h*w, 32, 64, 16,
#                    num_warps=4, num_stages=2)

# def launch_fold_one_batch(z, dx, batch):
#     g, b, c, h, w = dx.shape
#     _gx_fold[(triton.cdiv(g*c*h*w, 128),)](
#         z, dx, g, b, batch, c, h, w, 128, num_warps=4)

# def launch_grad_x(d, weight, dx, workspace):
#     g, b = d.shape[:2]
#     chunk = workspace.shape[0]
#     if chunk < 1:
#         raise ValueError("empty workspace")
#     for base in range(0, g, chunk):
#         end = min(g, base + chunk)
#         dd, ww, out = d[base:end], weight[base:end], dx[base:end]
#         for batch in range(b):
#             launch_gemm_one_batch(dd, ww, workspace, batch)
#             launch_fold_one_batch(workspace, out, batch)


# def launch_grad_x_implicit(d,weight,dx,bm=32,bn=32,bk=16,warps=4):
#     """One-kernel direct dX; intended for small B*H*W workloads."""
#     g,b,o,h,w=d.shape;c=weight.shape[2]
#     grid=(triton.cdiv(c,bm)*triton.cdiv(b*h*w,bn),g)
#     _gx_implicit_direct[grid](d,weight,dx,b,c,o,h,w,bm,bn,bk,
#                               num_warps=warps,num_stages=2)


# def launch_rgb_grad_x_direct3(d,weight,dx,block_n=16,block_k=16,warps=4):
#     """Direct C=3 dX, sharing dY across all three input channels."""
#     g,b,o,h,w=d.shape
#     if weight.shape[2:] != (3,3,3):
#         raise ValueError("RGB dX requires weight=[G,O,3,3,3]")
#     _rgb_dx_direct3[(triton.cdiv(b*h*w,block_n),g)](
#         d,weight,dx,b,o,h,w,block_n,block_k,
#         num_warps=warps,num_stages=2)


# def rgb_dw_split_count(x,d,force_split=None):
#     """Occupancy/work rule used only inside the RGB candidate family."""
#     g,b,_,h,w=x.shape;o=d.shape[2]
#     if force_split is not None:
#         split=operator.index(force_split)
#         if split<1 or split>64:
#             raise ValueError("RGB split must be in [1,64]")
#         return split
#     base_ctas=g*triton.cdiv(o,32)
#     occupancy=triton.cdiv(64,max(1,base_ctas))
#     reduction=triton.cdiv(b*h*w,8192)
#     return min(64,max(1,occupancy,reduction))


# def launch_rgb_grad_weight_bias(x,d,dw,db,force_split=None):
#     """Compact [O,K]@[K,27] RGB dW; dB shares the same dY reads."""
#     g,b,c,h,w=x.shape;o=d.shape[2]
#     if c!=3 or tuple(dw.shape)!=(g,o,3,3,3):
#         raise ValueError("RGB dW requires x channels=3 and dw=[G,O,3,3,3]")
#     split=rgb_dw_split_count(x,d,force_split)
#     grid=(triton.cdiv(o,32),g*split)
#     if split==1:
#         # Dummy partial pointers are never accessed under PARTIAL=False.
#         _rgb_dw_db_flat[grid](x,d,dw,db,dw,db,b,o,h,w,split,False,32,32,16,
#                               num_warps=4,num_stages=2)
#         return split
#     pdw=x.new_empty((g,split,o,27))
#     pdb=x.new_empty((g,split,o))
#     _rgb_dw_db_flat[grid](x,d,dw,db,pdw,pdb,b,o,h,w,split,True,32,32,16,
#                           num_warps=4,num_stages=2)
#     _rgb_dw_db_finish[(triton.cdiv(o*28,256),g)](
#         pdw,pdb,dw,db,o,split,256,num_warps=4)
#     return split


# def launch_grad_x_bmm(d,weight,dx,workspace):
#     """Direct dX GEMM through bmm(out=...), followed by the existing fold."""
#     g,b,o,h,w=d.shape;c=weight.shape[2];chunk=workspace.shape[0]
#     for base in range(0,g,chunk):
#         end=min(g,base+chunk);active=end-base
#         ww=weight[base:end].reshape(active,o,c*9).transpose(1,2)
#         for batch in range(b):
#             dd=d[base:end,batch].reshape(active,o,h*w)
#             torch.bmm(ww,dd,out=workspace[:active])
#             launch_fold_one_batch(workspace,dx[base:end],batch)

# def launch_grad_weight_bias_v5(x, d, dw, db):
#     g, b, c, h, w = x.shape
#     o = d.shape[2]
#     _dw_db_row[(triton.cdiv(c, 32) * triton.cdiv(o, 32), g, 3)](
#         x, d, dw, db, c, o, h, w, b, 32, 32, 16,
#         num_warps=4, num_stages=2)

# def launch_grad_bias(d, db):
#     g, b, o, h, w = d.shape
#     _db_reduce[(o, g)](d, db, b, o, h*w, 1024, num_warps=4)


# @triton.jit
# def _wg_pack_x(X, V, START, B: tl.constexpr, C: tl.constexpr,
#                H: tl.constexpr, W: tl.constexpr, NT: tl.constexpr,
#                BLOCK: tl.constexpr):
#     i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
#     g = tl.program_id(1)
#     ci, n = i // NT, i % NT
#     t = START + n
#     th: tl.constexpr = (H + 1) // 2
#     tw: tl.constexpr = (W + 1) // 2
#     batch = t // (th * tw)
#     y = t // tw % th * 2 - 1
#     x = t % tw * 2 - 1
#     base = ((g * B + batch) * C + ci) * H * W
#     valid = (ci < C) & (batch < B)
#     x00 = tl.load(X+base+y*W+x, valid&(y>=0)&(y<H)&(x>=0)&(x<W), 0)
#     x01 = tl.load(X+base+y*W+x+1, valid&(y>=0)&(y<H)&(x+1>=0)&(x+1<W), 0)
#     x02 = tl.load(X+base+y*W+x+2, valid&(y>=0)&(y<H)&(x+2>=0)&(x+2<W), 0)
#     x03 = tl.load(X+base+y*W+x+3, valid&(y>=0)&(y<H)&(x+3>=0)&(x+3<W), 0)
#     x10 = tl.load(X+base+(y+1)*W+x, valid&(y+1>=0)&(y+1<H)&(x>=0)&(x<W), 0)
#     x11 = tl.load(X+base+(y+1)*W+x+1, valid&(y+1>=0)&(y+1<H)&(x+1>=0)&(x+1<W), 0)
#     x12 = tl.load(X+base+(y+1)*W+x+2, valid&(y+1>=0)&(y+1<H)&(x+2>=0)&(x+2<W), 0)
#     x13 = tl.load(X+base+(y+1)*W+x+3, valid&(y+1>=0)&(y+1<H)&(x+3>=0)&(x+3<W), 0)
#     x20 = tl.load(X+base+(y+2)*W+x, valid&(y+2>=0)&(y+2<H)&(x>=0)&(x<W), 0)
#     x21 = tl.load(X+base+(y+2)*W+x+1, valid&(y+2>=0)&(y+2<H)&(x+1>=0)&(x+1<W), 0)
#     x22 = tl.load(X+base+(y+2)*W+x+2, valid&(y+2>=0)&(y+2<H)&(x+2>=0)&(x+2<W), 0)
#     x23 = tl.load(X+base+(y+2)*W+x+3, valid&(y+2>=0)&(y+2<H)&(x+3>=0)&(x+3<W), 0)
#     x30 = tl.load(X+base+(y+3)*W+x, valid&(y+3>=0)&(y+3<H)&(x>=0)&(x<W), 0)
#     x31 = tl.load(X+base+(y+3)*W+x+1, valid&(y+3>=0)&(y+3<H)&(x+1>=0)&(x+1<W), 0)
#     x32 = tl.load(X+base+(y+3)*W+x+2, valid&(y+3>=0)&(y+3<H)&(x+2>=0)&(x+2<W), 0)
#     x33 = tl.load(X+base+(y+3)*W+x+3, valid&(y+3>=0)&(y+3<H)&(x+3>=0)&(x+3<W), 0)
#     a00, a01, a02, a03 = x00-x20, x01-x21, x02-x22, x03-x23
#     a10, a11, a12, a13 = x10+x20, x11+x21, x12+x22, x13+x23
#     a20, a21, a22, a23 = x20-x10, x21-x11, x22-x12, x23-x13
#     a30, a31, a32, a33 = x30-x10, x31-x11, x32-x12, x33-x13
#     dst = V + g*16*C*NT + ci*NT + n
#     step: tl.constexpr = C*NT
#     # Fill padding with zero, including a partial final tile chunk.
#     mask = ci < C
#     tl.store(dst+0*step, a00-a02, mask)
#     tl.store(dst+1*step, a01+a02, mask)
#     tl.store(dst+2*step, a02-a01, mask)
#     tl.store(dst+3*step, a03-a01, mask)
#     tl.store(dst+4*step, a10-a12, mask)
#     tl.store(dst+5*step, a11+a12, mask)
#     tl.store(dst+6*step, a12-a11, mask)
#     tl.store(dst+7*step, a13-a11, mask)
#     tl.store(dst+8*step, a20-a22, mask)
#     tl.store(dst+9*step, a21+a22, mask)
#     tl.store(dst+10*step, a22-a21, mask)
#     tl.store(dst+11*step, a23-a21, mask)
#     tl.store(dst+12*step, a30-a32, mask)
#     tl.store(dst+13*step, a31+a32, mask)
#     tl.store(dst+14*step, a32-a31, mask)
#     tl.store(dst+15*step, a33-a31, mask)


# @triton.jit
# def _wg_pack_d(D, E, START, B: tl.constexpr, O: tl.constexpr,
#                H: tl.constexpr, W: tl.constexpr, NT: tl.constexpr,
#                BLOCK: tl.constexpr):
#     i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
#     g = tl.program_id(1)
#     oc, n = i // NT, i % NT
#     t = START + n
#     th: tl.constexpr = (H + 1) // 2
#     tw: tl.constexpr = (W + 1) // 2
#     batch = t // (th * tw)
#     y = t // tw % th * 2
#     x = t % tw * 2
#     base = ((g * B + batch) * O + oc) * H * W + y*W+x
#     valid = (oc < O) & (batch < B)
#     a = tl.load(D+base, valid&(y<H)&(x<W), 0)
#     b = tl.load(D+base+1, valid&(y<H)&(x+1<W), 0)
#     c = tl.load(D+base+W, valid&(y+1<H)&(x<W), 0)
#     d = tl.load(D+base+W+1, valid&(y+1<H)&(x+1<W), 0)
#     dst = E + g*16*O*NT + oc*NT + n
#     step: tl.constexpr = O*NT
#     mask = oc < O
#     tl.store(dst+0*step, a, mask)
#     tl.store(dst+1*step, a+b, mask)
#     tl.store(dst+2*step, a-b, mask)
#     tl.store(dst+3*step, b, mask)
#     tl.store(dst+4*step, a+c, mask)
#     tl.store(dst+5*step, (a+c)+(b+d), mask)
#     tl.store(dst+6*step, (a+c)-(b+d), mask)
#     tl.store(dst+7*step, b+d, mask)
#     tl.store(dst+8*step, a-c, mask)
#     tl.store(dst+9*step, (a-c)+(b-d), mask)
#     tl.store(dst+10*step, (a-c)-(b-d), mask)
#     tl.store(dst+11*step, b-d, mask)
#     tl.store(dst+12*step, c, mask)
#     tl.store(dst+13*step, c+d, mask)
#     tl.store(dst+14*step, c-d, mask)
#     tl.store(dst+15*step, d, mask)


# @triton.jit
# def _wg_gemm_v7(V, E, U, C: tl.constexpr, O: tl.constexpr, NT: tl.constexpr,
#              ADD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
#     tile = tl.program_id(0)
#     g = tl.program_id(1)
#     z = tl.program_id(2)
#     nc: tl.constexpr = (C + BN - 1) // BN
#     oc = tile // nc * BM + tl.arange(0, BM)
#     ci = tile % nc * BN + tl.arange(0, BN)
#     kk = tl.arange(0, BK)
#     acc = tl.zeros((BM, BN), tl.float32)
#     for block in range((NT + BK - 1) // BK):
#         k = block * BK + kk
#         a = tl.load(E+((g*16+z)*O+oc[:,None])*NT+k[None,:],
#                     (oc[:,None]<O)&(k[None,:]<NT), 0)
#         b = tl.load(V+((g*16+z)*C+ci[None,:])*NT+k[:,None],
#                     (ci[None,:]<C)&(k[:,None]<NT), 0)
#         acc = tl.dot(a, b, acc, input_precision="ieee")
#     dst = U + ((g*16+z)*O+oc[:,None])*C+ci[None,:]
#     valid = (oc[:,None]<O)&(ci[None,:]<C)
#     if ADD:
#         acc += tl.load(dst, valid, 0)
#     tl.store(dst, acc, valid)


# @triton.jit
# def _wg_gemm_v8(V, E, U, C: tl.constexpr, O: tl.constexpr, NT: tl.constexpr,
#                 ADD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
#     # The same packed operands as v7. Block pointers describe their exact
#     # contiguous axes, and advance by a K tile rather than rebuilding a
#     # full pointer tensor on each iteration. FP32 SIMT, no TF32 conversion.
#     tile = tl.program_id(0)
#     g = tl.program_id(1)
#     z = tl.program_id(2)
#     nc: tl.constexpr = (C + BN - 1) // BN
#     om, cn = tile // nc * BM, tile % nc * BN
#     a_ptr = tl.make_block_ptr(
#         base=E+(g*16+z)*O*NT, shape=(O,NT), strides=(NT,1),
#         offsets=(om,0), block_shape=(BM,BK), order=(1,0))
#     b_ptr = tl.make_block_ptr(
#         base=V+(g*16+z)*C*NT, shape=(NT,C), strides=(1,NT),
#         offsets=(0,cn), block_shape=(BK,BN), order=(0,1))
#     acc = tl.zeros((BM,BN),tl.float32)
#     for block in range((NT+BK-1)//BK):
#         a = tl.load(a_ptr,boundary_check=(0,1),padding_option="zero")
#         b = tl.load(b_ptr,boundary_check=(0,1),padding_option="zero")
#         acc = tl.dot(a,b,acc,input_precision="ieee")
#         a_ptr = tl.advance(a_ptr,(0,BK))
#         b_ptr = tl.advance(b_ptr,(BK,0))
#     oc = om+tl.arange(0,BM)
#     ci = cn+tl.arange(0,BN)
#     dst = U+((g*16+z)*O+oc[:,None])*C+ci[None,:]
#     valid = (oc[:,None]<O)&(ci[None,:]<C)
#     if ADD:
#         acc += tl.load(dst,valid,0)
#     tl.store(dst,acc,valid)


# @triton.jit
# def _wg_inverse(U, DWOUT, G: tl.constexpr, C: tl.constexpr,
#                 O: tl.constexpr, BLOCK: tl.constexpr):
#     i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
#     g, j = i//(O*C), i%(O*C)
#     valid = i < G*O*C
#     base = U + g*16*O*C+j
#     step: tl.constexpr = O*C
#     u0=tl.load(base+0*step,valid,0)
#     u1=tl.load(base+1*step,valid,0)
#     u2=tl.load(base+2*step,valid,0)
#     u3=tl.load(base+3*step,valid,0)
#     u4=tl.load(base+4*step,valid,0)
#     u5=tl.load(base+5*step,valid,0)
#     u6=tl.load(base+6*step,valid,0)
#     u7=tl.load(base+7*step,valid,0)
#     u8=tl.load(base+8*step,valid,0)
#     u9=tl.load(base+9*step,valid,0)
#     u10=tl.load(base+10*step,valid,0)
#     u11=tl.load(base+11*step,valid,0)
#     u12=tl.load(base+12*step,valid,0)
#     u13=tl.load(base+13*step,valid,0)
#     u14=tl.load(base+14*step,valid,0)
#     u15=tl.load(base+15*step,valid,0)
#     a0=u0+0.5*(u4+u8)
#     a1=u1+0.5*(u5+u9)
#     a2=u2+0.5*(u6+u10)
#     a3=u3+0.5*(u7+u11)
#     b0=0.5*(u4-u8)
#     b1=0.5*(u5-u9)
#     b2=0.5*(u6-u10)
#     b3=0.5*(u7-u11)
#     c0=0.5*(u4+u8)+u12
#     c1=0.5*(u5+u9)+u13
#     c2=0.5*(u6+u10)+u14
#     c3=0.5*(u7+u11)+u15
#     tl.store(DWOUT+i*9+0,a0+0.5*(a1+a2),valid)
#     tl.store(DWOUT+i*9+1,0.5*(a1-a2),valid)
#     tl.store(DWOUT+i*9+2,0.5*(a1+a2)+a3,valid)
#     tl.store(DWOUT+i*9+3,b0+0.5*(b1+b2),valid)
#     tl.store(DWOUT+i*9+4,0.5*(b1-b2),valid)
#     tl.store(DWOUT+i*9+5,0.5*(b1+b2)+b3,valid)
#     tl.store(DWOUT+i*9+6,c0+0.5*(c1+c2),valid)
#     tl.store(DWOUT+i*9+7,0.5*(c1-c2),valid)
#     tl.store(DWOUT+i*9+8,0.5*(c1+c2)+c3,valid)


# # Winograd F(4x4,3x3) weight-gradient path.  For one spatial tile,
# # V = B^T x B, E = A dY A^T, dU = E @ V^T, and dW = G^T dU G.
# # The matrices are the standard real-valued minimal-filtering transforms:
# #
# # B^T = [[4,0,-5,0,1,0], [0,-4,-4,1,1,0],
# #        [0,4,-4,-1,1,0], [0,-2,-1,2,1,0],
# #        [0,2,-1,-2,1,0], [0,4,0,-5,0,1]]
# # A^T = [[1,1,1,1,1,0], [0,1,-1,2,-2,0],
# #        [0,1,1,4,4,0], [0,1,-1,8,-8,1]]
# # G   = [[1/4,0,0], [-1/6,-1/6,-1/6], [-1/6,1/6,-1/6],
# #        [1/24,1/12,1/6], [1/24,-1/12,1/6], [0,0,1]]

# @triton.jit
# def _wg_f4_pack_x(X,V,E,START,B:tl.constexpr,C:tl.constexpr,
#                   H:tl.constexpr,W:tl.constexpr,NT:tl.constexpr,
#                   WRITE_E:tl.constexpr,BLOCK:tl.constexpr):
#     i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
#     g = tl.program_id(1)
#     ci, n = i // NT, i % NT
#     th: tl.constexpr = (H + 3) // 4
#     tw: tl.constexpr = (W + 3) // 4
#     t = START + n
#     batch = t // (th * tw)
#     y = t // tw % th * 4 - 1
#     x = t % tw * 4 - 1
#     base = ((g * B + batch) * C + ci) * H * W + y * W + x
#     valid = (ci < C) & (batch < B)

#     d00=tl.load(X+base+0*W+0,valid&(y+0>=0)&(y+0<H)&(x+0>=0)&(x+0<W),0)
#     d01=tl.load(X+base+0*W+1,valid&(y+0>=0)&(y+0<H)&(x+1>=0)&(x+1<W),0)
#     d02=tl.load(X+base+0*W+2,valid&(y+0>=0)&(y+0<H)&(x+2>=0)&(x+2<W),0)
#     d03=tl.load(X+base+0*W+3,valid&(y+0>=0)&(y+0<H)&(x+3>=0)&(x+3<W),0)
#     d04=tl.load(X+base+0*W+4,valid&(y+0>=0)&(y+0<H)&(x+4>=0)&(x+4<W),0)
#     d05=tl.load(X+base+0*W+5,valid&(y+0>=0)&(y+0<H)&(x+5>=0)&(x+5<W),0)
#     d10=tl.load(X+base+1*W+0,valid&(y+1>=0)&(y+1<H)&(x+0>=0)&(x+0<W),0)
#     d11=tl.load(X+base+1*W+1,valid&(y+1>=0)&(y+1<H)&(x+1>=0)&(x+1<W),0)
#     d12=tl.load(X+base+1*W+2,valid&(y+1>=0)&(y+1<H)&(x+2>=0)&(x+2<W),0)
#     d13=tl.load(X+base+1*W+3,valid&(y+1>=0)&(y+1<H)&(x+3>=0)&(x+3<W),0)
#     d14=tl.load(X+base+1*W+4,valid&(y+1>=0)&(y+1<H)&(x+4>=0)&(x+4<W),0)
#     d15=tl.load(X+base+1*W+5,valid&(y+1>=0)&(y+1<H)&(x+5>=0)&(x+5<W),0)
#     d20=tl.load(X+base+2*W+0,valid&(y+2>=0)&(y+2<H)&(x+0>=0)&(x+0<W),0)
#     d21=tl.load(X+base+2*W+1,valid&(y+2>=0)&(y+2<H)&(x+1>=0)&(x+1<W),0)
#     d22=tl.load(X+base+2*W+2,valid&(y+2>=0)&(y+2<H)&(x+2>=0)&(x+2<W),0)
#     d23=tl.load(X+base+2*W+3,valid&(y+2>=0)&(y+2<H)&(x+3>=0)&(x+3<W),0)
#     d24=tl.load(X+base+2*W+4,valid&(y+2>=0)&(y+2<H)&(x+4>=0)&(x+4<W),0)
#     d25=tl.load(X+base+2*W+5,valid&(y+2>=0)&(y+2<H)&(x+5>=0)&(x+5<W),0)
#     d30=tl.load(X+base+3*W+0,valid&(y+3>=0)&(y+3<H)&(x+0>=0)&(x+0<W),0)
#     d31=tl.load(X+base+3*W+1,valid&(y+3>=0)&(y+3<H)&(x+1>=0)&(x+1<W),0)
#     d32=tl.load(X+base+3*W+2,valid&(y+3>=0)&(y+3<H)&(x+2>=0)&(x+2<W),0)
#     d33=tl.load(X+base+3*W+3,valid&(y+3>=0)&(y+3<H)&(x+3>=0)&(x+3<W),0)
#     d34=tl.load(X+base+3*W+4,valid&(y+3>=0)&(y+3<H)&(x+4>=0)&(x+4<W),0)
#     d35=tl.load(X+base+3*W+5,valid&(y+3>=0)&(y+3<H)&(x+5>=0)&(x+5<W),0)
#     d40=tl.load(X+base+4*W+0,valid&(y+4>=0)&(y+4<H)&(x+0>=0)&(x+0<W),0)
#     d41=tl.load(X+base+4*W+1,valid&(y+4>=0)&(y+4<H)&(x+1>=0)&(x+1<W),0)
#     d42=tl.load(X+base+4*W+2,valid&(y+4>=0)&(y+4<H)&(x+2>=0)&(x+2<W),0)
#     d43=tl.load(X+base+4*W+3,valid&(y+4>=0)&(y+4<H)&(x+3>=0)&(x+3<W),0)
#     d44=tl.load(X+base+4*W+4,valid&(y+4>=0)&(y+4<H)&(x+4>=0)&(x+4<W),0)
#     d45=tl.load(X+base+4*W+5,valid&(y+4>=0)&(y+4<H)&(x+5>=0)&(x+5<W),0)
#     d50=tl.load(X+base+5*W+0,valid&(y+5>=0)&(y+5<H)&(x+0>=0)&(x+0<W),0)
#     d51=tl.load(X+base+5*W+1,valid&(y+5>=0)&(y+5<H)&(x+1>=0)&(x+1<W),0)
#     d52=tl.load(X+base+5*W+2,valid&(y+5>=0)&(y+5<H)&(x+2>=0)&(x+2<W),0)
#     d53=tl.load(X+base+5*W+3,valid&(y+5>=0)&(y+5<H)&(x+3>=0)&(x+3<W),0)
#     d54=tl.load(X+base+5*W+4,valid&(y+5>=0)&(y+5<H)&(x+4>=0)&(x+4<W),0)
#     d55=tl.load(X+base+5*W+5,valid&(y+5>=0)&(y+5<H)&(x+5>=0)&(x+5<W),0)

#     # Left multiply by B^T, one column at a time.
#     t00=4*d00-5*d20+d40; t01=4*d01-5*d21+d41; t02=4*d02-5*d22+d42
#     t03=4*d03-5*d23+d43; t04=4*d04-5*d24+d44; t05=4*d05-5*d25+d45
#     t10=-4*d10-4*d20+d30+d40; t11=-4*d11-4*d21+d31+d41
#     t12=-4*d12-4*d22+d32+d42; t13=-4*d13-4*d23+d33+d43
#     t14=-4*d14-4*d24+d34+d44; t15=-4*d15-4*d25+d35+d45
#     t20=4*d10-4*d20-d30+d40; t21=4*d11-4*d21-d31+d41
#     t22=4*d12-4*d22-d32+d42; t23=4*d13-4*d23-d33+d43
#     t24=4*d14-4*d24-d34+d44; t25=4*d15-4*d25-d35+d45
#     t30=-2*d10-d20+2*d30+d40; t31=-2*d11-d21+2*d31+d41
#     t32=-2*d12-d22+2*d32+d42; t33=-2*d13-d23+2*d33+d43
#     t34=-2*d14-d24+2*d34+d44; t35=-2*d15-d25+2*d35+d45
#     t40=2*d10-d20-2*d30+d40; t41=2*d11-d21-2*d31+d41
#     t42=2*d12-d22-2*d32+d42; t43=2*d13-d23-2*d33+d43
#     t44=2*d14-d24-2*d34+d44; t45=2*d15-d25-2*d35+d45
#     t50=4*d10-5*d30+d50; t51=4*d11-5*d31+d51; t52=4*d12-5*d32+d52
#     t53=4*d13-5*d33+d53; t54=4*d14-5*d34+d54; t55=4*d15-5*d35+d55

#     dst=V+g*36*C*NT+ci*NT+n
#     step: tl.constexpr=C*NT
#     mask=ci<C
#     # Right multiply by B, equivalently apply B^T to every transformed row.
#     tl.store(dst+ 0*step,4*t00-5*t02+t04,mask); tl.store(dst+ 1*step,-4*t01-4*t02+t03+t04,mask)
#     tl.store(dst+ 2*step,4*t01-4*t02-t03+t04,mask); tl.store(dst+ 3*step,-2*t01-t02+2*t03+t04,mask)
#     tl.store(dst+ 4*step,2*t01-t02-2*t03+t04,mask); tl.store(dst+ 5*step,4*t01-5*t03+t05,mask)
#     tl.store(dst+ 6*step,4*t10-5*t12+t14,mask); tl.store(dst+ 7*step,-4*t11-4*t12+t13+t14,mask)
#     tl.store(dst+ 8*step,4*t11-4*t12-t13+t14,mask); tl.store(dst+ 9*step,-2*t11-t12+2*t13+t14,mask)
#     tl.store(dst+10*step,2*t11-t12-2*t13+t14,mask); tl.store(dst+11*step,4*t11-5*t13+t15,mask)
#     tl.store(dst+12*step,4*t20-5*t22+t24,mask); tl.store(dst+13*step,-4*t21-4*t22+t23+t24,mask)
#     tl.store(dst+14*step,4*t21-4*t22-t23+t24,mask); tl.store(dst+15*step,-2*t21-t22+2*t23+t24,mask)
#     tl.store(dst+16*step,2*t21-t22-2*t23+t24,mask); tl.store(dst+17*step,4*t21-5*t23+t25,mask)
#     tl.store(dst+18*step,4*t30-5*t32+t34,mask); tl.store(dst+19*step,-4*t31-4*t32+t33+t34,mask)
#     tl.store(dst+20*step,4*t31-4*t32-t33+t34,mask); tl.store(dst+21*step,-2*t31-t32+2*t33+t34,mask)
#     tl.store(dst+22*step,2*t31-t32-2*t33+t34,mask); tl.store(dst+23*step,4*t31-5*t33+t35,mask)
#     tl.store(dst+24*step,4*t40-5*t42+t44,mask); tl.store(dst+25*step,-4*t41-4*t42+t43+t44,mask)
#     tl.store(dst+26*step,4*t41-4*t42-t43+t44,mask); tl.store(dst+27*step,-2*t41-t42+2*t43+t44,mask)
#     tl.store(dst+28*step,2*t41-t42-2*t43+t44,mask); tl.store(dst+29*step,4*t41-5*t43+t45,mask)
#     tl.store(dst+30*step,4*t50-5*t52+t54,mask); tl.store(dst+31*step,-4*t51-4*t52+t53+t54,mask)
#     tl.store(dst+32*step,4*t51-4*t52-t53+t54,mask); tl.store(dst+33*step,-2*t51-t52+2*t53+t54,mask)
#     tl.store(dst+34*step,2*t51-t52-2*t53+t54,mask); tl.store(dst+35*step,4*t51-5*t53+t55,mask)
#     if WRITE_E:
#         # The central 4x4 dY region is exactly the tile used by A dY A^T.
#         # Produce that transform from the already loaded 6x6 tile.
#         p00=d11;p01=d12;p02=d13;p03=d14
#         p10=d11+d21+d31+d41;p11=d12+d22+d32+d42;p12=d13+d23+d33+d43;p13=d14+d24+d34+d44
#         p20=d11-d21+d31-d41;p21=d12-d22+d32-d42;p22=d13-d23+d33-d43;p23=d14-d24+d34-d44
#         p30=d11+2*d21+4*d31+8*d41;p31=d12+2*d22+4*d32+8*d42;p32=d13+2*d23+4*d33+8*d43;p33=d14+2*d24+4*d34+8*d44
#         p40=d11-2*d21+4*d31-8*d41;p41=d12-2*d22+4*d32-8*d42;p42=d13-2*d23+4*d33-8*d43;p43=d14-2*d24+4*d34-8*d44
#         p50=d41;p51=d42;p52=d43;p53=d44
#         edst=E+g*36*C*NT+ci*NT+n
#         tl.store(edst+ 0*step,p00,mask);tl.store(edst+ 1*step,p00+p01+p02+p03,mask);tl.store(edst+ 2*step,p00-p01+p02-p03,mask)
#         tl.store(edst+ 3*step,p00+2*p01+4*p02+8*p03,mask);tl.store(edst+ 4*step,p00-2*p01+4*p02-8*p03,mask);tl.store(edst+ 5*step,p03,mask)
#         tl.store(edst+ 6*step,p10,mask);tl.store(edst+ 7*step,p10+p11+p12+p13,mask);tl.store(edst+ 8*step,p10-p11+p12-p13,mask)
#         tl.store(edst+ 9*step,p10+2*p11+4*p12+8*p13,mask);tl.store(edst+10*step,p10-2*p11+4*p12-8*p13,mask);tl.store(edst+11*step,p13,mask)
#         tl.store(edst+12*step,p20,mask);tl.store(edst+13*step,p20+p21+p22+p23,mask);tl.store(edst+14*step,p20-p21+p22-p23,mask)
#         tl.store(edst+15*step,p20+2*p21+4*p22+8*p23,mask);tl.store(edst+16*step,p20-2*p21+4*p22-8*p23,mask);tl.store(edst+17*step,p23,mask)
#         tl.store(edst+18*step,p30,mask);tl.store(edst+19*step,p30+p31+p32+p33,mask);tl.store(edst+20*step,p30-p31+p32-p33,mask)
#         tl.store(edst+21*step,p30+2*p31+4*p32+8*p33,mask);tl.store(edst+22*step,p30-2*p31+4*p32-8*p33,mask);tl.store(edst+23*step,p33,mask)
#         tl.store(edst+24*step,p40,mask);tl.store(edst+25*step,p40+p41+p42+p43,mask);tl.store(edst+26*step,p40-p41+p42-p43,mask)
#         tl.store(edst+27*step,p40+2*p41+4*p42+8*p43,mask);tl.store(edst+28*step,p40-2*p41+4*p42-8*p43,mask);tl.store(edst+29*step,p43,mask)
#         tl.store(edst+30*step,p50,mask);tl.store(edst+31*step,p50+p51+p52+p53,mask);tl.store(edst+32*step,p50-p51+p52-p53,mask)
#         tl.store(edst+33*step,p50+2*p51+4*p52+8*p53,mask);tl.store(edst+34*step,p50-2*p51+4*p52-8*p53,mask);tl.store(edst+35*step,p53,mask)


# @triton.jit
# def _wg_f4_pack_d(D,E,START,B:tl.constexpr,O:tl.constexpr,
#                   H:tl.constexpr,W:tl.constexpr,NT:tl.constexpr,
#                   BLOCK:tl.constexpr):
#     i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);g=tl.program_id(1)
#     oc,n=i//NT,i%NT
#     th:tl.constexpr=(H+3)//4;tw:tl.constexpr=(W+3)//4
#     t=START+n;batch=t//(th*tw);y=t//tw%th*4;x=t%tw*4
#     base=((g*B+batch)*O+oc)*H*W+y*W+x
#     valid=(oc<O)&(batch<B)
#     d00=tl.load(D+base+0*W+0,valid&(y+0<H)&(x+0<W),0);d01=tl.load(D+base+0*W+1,valid&(y+0<H)&(x+1<W),0)
#     d02=tl.load(D+base+0*W+2,valid&(y+0<H)&(x+2<W),0);d03=tl.load(D+base+0*W+3,valid&(y+0<H)&(x+3<W),0)
#     d10=tl.load(D+base+1*W+0,valid&(y+1<H)&(x+0<W),0);d11=tl.load(D+base+1*W+1,valid&(y+1<H)&(x+1<W),0)
#     d12=tl.load(D+base+1*W+2,valid&(y+1<H)&(x+2<W),0);d13=tl.load(D+base+1*W+3,valid&(y+1<H)&(x+3<W),0)
#     d20=tl.load(D+base+2*W+0,valid&(y+2<H)&(x+0<W),0);d21=tl.load(D+base+2*W+1,valid&(y+2<H)&(x+1<W),0)
#     d22=tl.load(D+base+2*W+2,valid&(y+2<H)&(x+2<W),0);d23=tl.load(D+base+2*W+3,valid&(y+2<H)&(x+3<W),0)
#     d30=tl.load(D+base+3*W+0,valid&(y+3<H)&(x+0<W),0);d31=tl.load(D+base+3*W+1,valid&(y+3<H)&(x+1<W),0)
#     d32=tl.load(D+base+3*W+2,valid&(y+3<H)&(x+2<W),0);d33=tl.load(D+base+3*W+3,valid&(y+3<H)&(x+3<W),0)
#     t00=d00;t01=d01;t02=d02;t03=d03
#     t10=d00+d10+d20+d30;t11=d01+d11+d21+d31;t12=d02+d12+d22+d32;t13=d03+d13+d23+d33
#     t20=d00-d10+d20-d30;t21=d01-d11+d21-d31;t22=d02-d12+d22-d32;t23=d03-d13+d23-d33
#     t30=d00+2*d10+4*d20+8*d30;t31=d01+2*d11+4*d21+8*d31;t32=d02+2*d12+4*d22+8*d32;t33=d03+2*d13+4*d23+8*d33
#     t40=d00-2*d10+4*d20-8*d30;t41=d01-2*d11+4*d21-8*d31;t42=d02-2*d12+4*d22-8*d32;t43=d03-2*d13+4*d23-8*d33
#     t50=d30;t51=d31;t52=d32;t53=d33
#     dst=E+g*36*O*NT+oc*NT+n;step:tl.constexpr=O*NT;mask=oc<O
#     tl.store(dst+ 0*step,t00,mask);tl.store(dst+ 1*step,t00+t01+t02+t03,mask);tl.store(dst+ 2*step,t00-t01+t02-t03,mask)
#     tl.store(dst+ 3*step,t00+2*t01+4*t02+8*t03,mask);tl.store(dst+ 4*step,t00-2*t01+4*t02-8*t03,mask);tl.store(dst+ 5*step,t03,mask)
#     tl.store(dst+ 6*step,t10,mask);tl.store(dst+ 7*step,t10+t11+t12+t13,mask);tl.store(dst+ 8*step,t10-t11+t12-t13,mask)
#     tl.store(dst+ 9*step,t10+2*t11+4*t12+8*t13,mask);tl.store(dst+10*step,t10-2*t11+4*t12-8*t13,mask);tl.store(dst+11*step,t13,mask)
#     tl.store(dst+12*step,t20,mask);tl.store(dst+13*step,t20+t21+t22+t23,mask);tl.store(dst+14*step,t20-t21+t22-t23,mask)
#     tl.store(dst+15*step,t20+2*t21+4*t22+8*t23,mask);tl.store(dst+16*step,t20-2*t21+4*t22-8*t23,mask);tl.store(dst+17*step,t23,mask)
#     tl.store(dst+18*step,t30,mask);tl.store(dst+19*step,t30+t31+t32+t33,mask);tl.store(dst+20*step,t30-t31+t32-t33,mask)
#     tl.store(dst+21*step,t30+2*t31+4*t32+8*t33,mask);tl.store(dst+22*step,t30-2*t31+4*t32-8*t33,mask);tl.store(dst+23*step,t33,mask)
#     tl.store(dst+24*step,t40,mask);tl.store(dst+25*step,t40+t41+t42+t43,mask);tl.store(dst+26*step,t40-t41+t42-t43,mask)
#     tl.store(dst+27*step,t40+2*t41+4*t42+8*t43,mask);tl.store(dst+28*step,t40-2*t41+4*t42-8*t43,mask);tl.store(dst+29*step,t43,mask)
#     tl.store(dst+30*step,t50,mask);tl.store(dst+31*step,t50+t51+t52+t53,mask);tl.store(dst+32*step,t50-t51+t52-t53,mask)
#     tl.store(dst+33*step,t50+2*t51+4*t52+8*t53,mask);tl.store(dst+34*step,t50-2*t51+4*t52-8*t53,mask);tl.store(dst+35*step,t53,mask)


# @triton.jit
# def _wg_f4_joint_pack_d(D,V,E,START,B:tl.constexpr,O:tl.constexpr,
#                         H:tl.constexpr,W:tl.constexpr,NT:tl.constexpr,
#                         BLOCK:tl.constexpr):
#     # One set of 6x6 loads emits both B^T dY B and A dY A^T.
#     _wg_f4_pack_x(D,V,E,START,B,O,H,W,NT,True,BLOCK)


# @triton.jit
# def _wg_f4_pack_weight_dx(W,U,G:tl.constexpr,C:tl.constexpr,O:tl.constexpr,
#                           BLOCK:tl.constexpr):
#     """U[z,c,o] = G * rot180(W[o,c]) * G^T for forward-form dX."""
#     i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);g=tl.program_id(1)
#     ci=i//O;oc=i%O;valid=ci<C
#     base=W+(g*O+oc)*C*9+ci*9
#     # conv_transpose(dY,W) == conv(dY, rot180(W)^T), padding one.
#     w00=tl.load(base+8,valid,0);w01=tl.load(base+7,valid,0);w02=tl.load(base+6,valid,0)
#     w10=tl.load(base+5,valid,0);w11=tl.load(base+4,valid,0);w12=tl.load(base+3,valid,0)
#     w20=tl.load(base+2,valid,0);w21=tl.load(base+1,valid,0);w22=tl.load(base+0,valid,0)
#     # Left multiply by G.
#     t00=.25*w00;t01=.25*w01;t02=.25*w02
#     t10=-(w00+w10+w20)/6;t11=-(w01+w11+w21)/6;t12=-(w02+w12+w22)/6
#     t20=(-w00+w10-w20)/6;t21=(-w01+w11-w21)/6;t22=(-w02+w12-w22)/6
#     t30=w00/24+w10/12+w20/6;t31=w01/24+w11/12+w21/6;t32=w02/24+w12/12+w22/6
#     t40=w00/24-w10/12+w20/6;t41=w01/24-w11/12+w21/6;t42=w02/24-w12/12+w22/6
#     t50=w20;t51=w21;t52=w22
#     dst=U+g*36*C*O+ci*O+oc;step:tl.constexpr=C*O
#     # Right multiply by G^T.
#     tl.store(dst+ 0*step,.25*t00,valid);tl.store(dst+ 1*step,-(t00+t01+t02)/6,valid)
#     tl.store(dst+ 2*step,(-t00+t01-t02)/6,valid);tl.store(dst+ 3*step,t00/24+t01/12+t02/6,valid)
#     tl.store(dst+ 4*step,t00/24-t01/12+t02/6,valid);tl.store(dst+ 5*step,t02,valid)
#     tl.store(dst+ 6*step,.25*t10,valid);tl.store(dst+ 7*step,-(t10+t11+t12)/6,valid)
#     tl.store(dst+ 8*step,(-t10+t11-t12)/6,valid);tl.store(dst+ 9*step,t10/24+t11/12+t12/6,valid)
#     tl.store(dst+10*step,t10/24-t11/12+t12/6,valid);tl.store(dst+11*step,t12,valid)
#     tl.store(dst+12*step,.25*t20,valid);tl.store(dst+13*step,-(t20+t21+t22)/6,valid)
#     tl.store(dst+14*step,(-t20+t21-t22)/6,valid);tl.store(dst+15*step,t20/24+t21/12+t22/6,valid)
#     tl.store(dst+16*step,t20/24-t21/12+t22/6,valid);tl.store(dst+17*step,t22,valid)
#     tl.store(dst+18*step,.25*t30,valid);tl.store(dst+19*step,-(t30+t31+t32)/6,valid)
#     tl.store(dst+20*step,(-t30+t31-t32)/6,valid);tl.store(dst+21*step,t30/24+t31/12+t32/6,valid)
#     tl.store(dst+22*step,t30/24-t31/12+t32/6,valid);tl.store(dst+23*step,t32,valid)
#     tl.store(dst+24*step,.25*t40,valid);tl.store(dst+25*step,-(t40+t41+t42)/6,valid)
#     tl.store(dst+26*step,(-t40+t41-t42)/6,valid);tl.store(dst+27*step,t40/24+t41/12+t42/6,valid)
#     tl.store(dst+28*step,t40/24-t41/12+t42/6,valid);tl.store(dst+29*step,t42,valid)
#     tl.store(dst+30*step,.25*t50,valid);tl.store(dst+31*step,-(t50+t51+t52)/6,valid)
#     tl.store(dst+32*step,(-t50+t51-t52)/6,valid);tl.store(dst+33*step,t50/24+t51/12+t52/6,valid)
#     tl.store(dst+34*step,t50/24-t51/12+t52/6,valid);tl.store(dst+35*step,t52,valid)


# @triton.jit
# def _wg_f4_dx_gemm(U,V,M,C:tl.constexpr,O:tl.constexpr,NT:tl.constexpr,
#                    BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
#     """M[z,c,tile] = U[z,c,o] @ V[z,o,tile]."""
#     tile=tl.program_id(0);g=tl.program_id(1);z=tl.program_id(2)
#     nn:tl.constexpr=(NT+BN-1)//BN
#     ci=tile//nn*BM+tl.arange(0,BM);n=tile%nn*BN+tl.arange(0,BN)
#     kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
#     for block in range((O+BK-1)//BK):
#         oc=block*BK+kk
#         a=tl.load(U+((g*36+z)*C+ci[:,None])*O+oc[None,:],
#                   (ci[:,None]<C)&(oc[None,:]<O),0)
#         b=tl.load(V+((g*36+z)*O+oc[:,None])*NT+n[None,:],
#                   (oc[:,None]<O)&(n[None,:]<NT),0)
#         acc=tl.dot(a,b,acc,input_precision="ieee")
#     tl.store(M+((g*36+z)*C+ci[:,None])*NT+n[None,:],acc,
#              (ci[:,None]<C)&(n[None,:]<NT))


# @triton.jit
# def _wg_f4_dx_output(M,DX,START,G:tl.constexpr,B:tl.constexpr,
#                      C:tl.constexpr,H:tl.constexpr,W:tl.constexpr,
#                      NT:tl.constexpr,BLOCK:tl.constexpr):
#     """A^T M A; 4x4 tiles do not overlap, so stores need no atomics."""
#     i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);g=tl.program_id(1)
#     ci,n=i//NT,i%NT;t=START+n
#     th:tl.constexpr=(H+3)//4;tw:tl.constexpr=(W+3)//4
#     batch=t//(th*tw);ty=t//tw%th;tx=t%tw
#     valid=(ci<C)&(batch<B)
#     base=M+g*36*C*NT+ci*NT+n;step:tl.constexpr=C*NT
#     m00=tl.load(base+ 0*step,ci<C,0);m01=tl.load(base+ 1*step,ci<C,0);m02=tl.load(base+ 2*step,ci<C,0);m03=tl.load(base+ 3*step,ci<C,0);m04=tl.load(base+ 4*step,ci<C,0);m05=tl.load(base+ 5*step,ci<C,0)
#     m10=tl.load(base+ 6*step,ci<C,0);m11=tl.load(base+ 7*step,ci<C,0);m12=tl.load(base+ 8*step,ci<C,0);m13=tl.load(base+ 9*step,ci<C,0);m14=tl.load(base+10*step,ci<C,0);m15=tl.load(base+11*step,ci<C,0)
#     m20=tl.load(base+12*step,ci<C,0);m21=tl.load(base+13*step,ci<C,0);m22=tl.load(base+14*step,ci<C,0);m23=tl.load(base+15*step,ci<C,0);m24=tl.load(base+16*step,ci<C,0);m25=tl.load(base+17*step,ci<C,0)
#     m30=tl.load(base+18*step,ci<C,0);m31=tl.load(base+19*step,ci<C,0);m32=tl.load(base+20*step,ci<C,0);m33=tl.load(base+21*step,ci<C,0);m34=tl.load(base+22*step,ci<C,0);m35=tl.load(base+23*step,ci<C,0)
#     m40=tl.load(base+24*step,ci<C,0);m41=tl.load(base+25*step,ci<C,0);m42=tl.load(base+26*step,ci<C,0);m43=tl.load(base+27*step,ci<C,0);m44=tl.load(base+28*step,ci<C,0);m45=tl.load(base+29*step,ci<C,0)
#     m50=tl.load(base+30*step,ci<C,0);m51=tl.load(base+31*step,ci<C,0);m52=tl.load(base+32*step,ci<C,0);m53=tl.load(base+33*step,ci<C,0);m54=tl.load(base+34*step,ci<C,0);m55=tl.load(base+35*step,ci<C,0)
#     # A^T M, then multiply by A.  Keep only sixteen final values.
#     t00=m00+m10+m20+m30+m40;t01=m01+m11+m21+m31+m41;t02=m02+m12+m22+m32+m42;t03=m03+m13+m23+m33+m43;t04=m04+m14+m24+m34+m44;t05=m05+m15+m25+m35+m45
#     t10=m10-m20+2*m30-2*m40;t11=m11-m21+2*m31-2*m41;t12=m12-m22+2*m32-2*m42;t13=m13-m23+2*m33-2*m43;t14=m14-m24+2*m34-2*m44;t15=m15-m25+2*m35-2*m45
#     t20=m10+m20+4*m30+4*m40;t21=m11+m21+4*m31+4*m41;t22=m12+m22+4*m32+4*m42;t23=m13+m23+4*m33+4*m43;t24=m14+m24+4*m34+4*m44;t25=m15+m25+4*m35+4*m45
#     t30=m10-m20+8*m30-8*m40+m50;t31=m11-m21+8*m31-8*m41+m51;t32=m12-m22+8*m32-8*m42+m52;t33=m13-m23+8*m33-8*m43+m53;t34=m14-m24+8*m34-8*m44+m54;t35=m15-m25+8*m35-8*m45+m55
#     y=ty*4;x=tx*4;dst=DX+((g*B+batch)*C+ci)*H*W+y*W+x
#     o00=t00+t01+t02+t03+t04;o01=t01-t02+2*t03-2*t04;o02=t01+t02+4*t03+4*t04;o03=t01-t02+8*t03-8*t04+t05
#     o10=t10+t11+t12+t13+t14;o11=t11-t12+2*t13-2*t14;o12=t11+t12+4*t13+4*t14;o13=t11-t12+8*t13-8*t14+t15
#     o20=t20+t21+t22+t23+t24;o21=t21-t22+2*t23-2*t24;o22=t21+t22+4*t23+4*t24;o23=t21-t22+8*t23-8*t24+t25
#     o30=t30+t31+t32+t33+t34;o31=t31-t32+2*t33-2*t34;o32=t31+t32+4*t33+4*t34;o33=t31-t32+8*t33-8*t34+t35
#     tl.store(dst+0*W+0,o00,valid&(y+0<H)&(x+0<W));tl.store(dst+0*W+1,o01,valid&(y+0<H)&(x+1<W))
#     tl.store(dst+0*W+2,o02,valid&(y+0<H)&(x+2<W));tl.store(dst+0*W+3,o03,valid&(y+0<H)&(x+3<W))
#     tl.store(dst+1*W+0,o10,valid&(y+1<H)&(x+0<W));tl.store(dst+1*W+1,o11,valid&(y+1<H)&(x+1<W))
#     tl.store(dst+1*W+2,o12,valid&(y+1<H)&(x+2<W));tl.store(dst+1*W+3,o13,valid&(y+1<H)&(x+3<W))
#     tl.store(dst+2*W+0,o20,valid&(y+2<H)&(x+0<W));tl.store(dst+2*W+1,o21,valid&(y+2<H)&(x+1<W))
#     tl.store(dst+2*W+2,o22,valid&(y+2<H)&(x+2<W));tl.store(dst+2*W+3,o23,valid&(y+2<H)&(x+3<W))
#     tl.store(dst+3*W+0,o30,valid&(y+3<H)&(x+0<W));tl.store(dst+3*W+1,o31,valid&(y+3<H)&(x+1<W))
#     tl.store(dst+3*W+2,o32,valid&(y+3<H)&(x+2<W));tl.store(dst+3*W+3,o33,valid&(y+3<H)&(x+3<W))


# @triton.jit
# def _wg_f4_gemm(V,E,U,C:tl.constexpr,O:tl.constexpr,NT:tl.constexpr,
#                 ADD:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
#     tile=tl.program_id(0);g=tl.program_id(1);z=tl.program_id(2)
#     nc:tl.constexpr=(C+BN-1)//BN
#     oc=tile//nc*BM+tl.arange(0,BM);ci=tile%nc*BN+tl.arange(0,BN)
#     kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
#     for block in range((NT+BK-1)//BK):
#         k=block*BK+kk
#         a=tl.load(E+((g*36+z)*O+oc[:,None])*NT+k[None,:],(oc[:,None]<O)&(k[None,:]<NT),0)
#         b=tl.load(V+((g*36+z)*C+ci[None,:])*NT+k[:,None],(ci[None,:]<C)&(k[:,None]<NT),0)
#         acc=tl.dot(a,b,acc,input_precision="ieee")
#     dst=U+((g*36+z)*O+oc[:,None])*C+ci[None,:]
#     mask=(oc[:,None]<O)&(ci[None,:]<C)
#     if ADD:acc+=tl.load(dst,mask,0)
#     tl.store(dst,acc,mask)


# @triton.jit
# def _wg_f4_inverse(U,DWOUT,G:tl.constexpr,C:tl.constexpr,O:tl.constexpr,
#                    BLOCK:tl.constexpr):
#     i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);g=i//(O*C);j=i%(O*C)
#     valid=i<G*O*C;base=U+g*36*O*C+j;step:tl.constexpr=O*C
#     u00=tl.load(base+ 0*step,valid,0);u01=tl.load(base+ 1*step,valid,0);u02=tl.load(base+ 2*step,valid,0);u03=tl.load(base+ 3*step,valid,0);u04=tl.load(base+ 4*step,valid,0);u05=tl.load(base+ 5*step,valid,0)
#     u10=tl.load(base+ 6*step,valid,0);u11=tl.load(base+ 7*step,valid,0);u12=tl.load(base+ 8*step,valid,0);u13=tl.load(base+ 9*step,valid,0);u14=tl.load(base+10*step,valid,0);u15=tl.load(base+11*step,valid,0)
#     u20=tl.load(base+12*step,valid,0);u21=tl.load(base+13*step,valid,0);u22=tl.load(base+14*step,valid,0);u23=tl.load(base+15*step,valid,0);u24=tl.load(base+16*step,valid,0);u25=tl.load(base+17*step,valid,0)
#     u30=tl.load(base+18*step,valid,0);u31=tl.load(base+19*step,valid,0);u32=tl.load(base+20*step,valid,0);u33=tl.load(base+21*step,valid,0);u34=tl.load(base+22*step,valid,0);u35=tl.load(base+23*step,valid,0)
#     u40=tl.load(base+24*step,valid,0);u41=tl.load(base+25*step,valid,0);u42=tl.load(base+26*step,valid,0);u43=tl.load(base+27*step,valid,0);u44=tl.load(base+28*step,valid,0);u45=tl.load(base+29*step,valid,0)
#     u50=tl.load(base+30*step,valid,0);u51=tl.load(base+31*step,valid,0);u52=tl.load(base+32*step,valid,0);u53=tl.load(base+33*step,valid,0);u54=tl.load(base+34*step,valid,0);u55=tl.load(base+35*step,valid,0)
#     t00=.25*u00-(u10+u20)/6+(u30+u40)/24;t01=.25*u01-(u11+u21)/6+(u31+u41)/24;t02=.25*u02-(u12+u22)/6+(u32+u42)/24;t03=.25*u03-(u13+u23)/6+(u33+u43)/24;t04=.25*u04-(u14+u24)/6+(u34+u44)/24;t05=.25*u05-(u15+u25)/6+(u35+u45)/24
#     t10=(-u10+u20)/6+(u30-u40)/12;t11=(-u11+u21)/6+(u31-u41)/12;t12=(-u12+u22)/6+(u32-u42)/12;t13=(-u13+u23)/6+(u33-u43)/12;t14=(-u14+u24)/6+(u34-u44)/12;t15=(-u15+u25)/6+(u35-u45)/12
#     t20=-(u10+u20)/6+(u30+u40)/6+u50;t21=-(u11+u21)/6+(u31+u41)/6+u51;t22=-(u12+u22)/6+(u32+u42)/6+u52;t23=-(u13+u23)/6+(u33+u43)/6+u53;t24=-(u14+u24)/6+(u34+u44)/6+u54;t25=-(u15+u25)/6+(u35+u45)/6+u55
#     tl.store(DWOUT+i*9+0,.25*t00-(t01+t02)/6+(t03+t04)/24,valid)
#     tl.store(DWOUT+i*9+1,(-t01+t02)/6+(t03-t04)/12,valid)
#     tl.store(DWOUT+i*9+2,-(t01+t02)/6+(t03+t04)/6+t05,valid)
#     tl.store(DWOUT+i*9+3,.25*t10-(t11+t12)/6+(t13+t14)/24,valid)
#     tl.store(DWOUT+i*9+4,(-t11+t12)/6+(t13-t14)/12,valid)
#     tl.store(DWOUT+i*9+5,-(t11+t12)/6+(t13+t14)/6+t15,valid)
#     tl.store(DWOUT+i*9+6,.25*t20-(t21+t22)/6+(t23+t24)/24,valid)
#     tl.store(DWOUT+i*9+7,(-t21+t22)/6+(t23-t24)/12,valid)
#     tl.store(DWOUT+i*9+8,-(t21+t22)/6+(t23+t24)/6+t25,valid)


# def winograd_plan(x, d, max_bytes=WORKSPACE_LIMIT_BYTES):
#     g,b,c,h,w = x.shape
#     o = d.shape[2]
#     nt = b * ((h+1)//2) * ((w+1)//2)
#     fixed = 16*o*c*4
#     per_tile = 16*(c+o)*4
#     per_group = fixed + per_tile*nt
#     if per_group <= max_bytes:
#         gc, tc = min(g, max_bytes//per_group), nt
#     else:
#         gc, tc = 1, min(nt, (max_bytes-fixed)//per_tile)
#         if tc < 1:
#             raise ValueError("Winograd fixed state exceeds the workspace budget")
#         if tc >= 16:
#             tc = tc//16*16
#     chunks_g = (g+gc-1)//gc
#     chunks_t = (nt+tc-1)//tc
#     return dict(groups=gc, tiles=tc, total_tiles=nt,
#                 group_chunks=chunks_g, tile_chunks=chunks_t,
#                 bytes=gc*(fixed+per_tile*tc),
#                 kernel_launches=chunks_g*(3*chunks_t+1)+1)


# def grad_weight_workspace(x, d, max_bytes=WORKSPACE_LIMIT_BYTES):
#     p = winograd_plan(x, d, max_bytes)
#     gc, tc = p["groups"], p["tiles"]
#     c, o = x.shape[2], d.shape[2]
#     return (x.new_empty((gc,16,c,tc)), x.new_empty((gc,16,o,tc)),
#             x.new_empty((gc,16,o,c)))


# def launch_pack_x(x, v, start=0):
#     g,b,c,h,w = x.shape
#     nt = v.shape[-1]
#     _wg_pack_x[(triton.cdiv(c*nt,128),g)](
#         x,v,start,b,c,h,w,nt,128,num_warps=4)


# def launch_pack_d(d, e, start=0):
#     g,b,o,h,w = d.shape
#     nt = e.shape[-1]
#     _wg_pack_d[(triton.cdiv(o*nt,128),g)](
#         d,e,start,b,o,h,w,nt,128,num_warps=4)


# def launch_winograd_gemm_v7(v, e, u, active_groups, add=False):
#     c, nt = v.shape[2:]
#     o = e.shape[2]
#     bm,bn,bk = WG_TILE_V7
#     _wg_gemm_v7[(triton.cdiv(o,bm)*triton.cdiv(c,bn),active_groups,16)](
#         v,e,u,c,o,nt,add,bm,bn,bk,num_warps=4,num_stages=2)


# def launch_winograd_gemm(v, e, u, active_groups, add=False):
#     c, nt = v.shape[2:]
#     o = e.shape[2]
#     bm,bn,bk = WG_TILE
#     bk = min(bk,max(16,triton.next_power_of_2(nt)))
#     _wg_gemm_v8[(triton.cdiv(o,bm)*triton.cdiv(c,bn),active_groups,16)](
#         v,e,u,c,o,nt,add,bm,bn,bk,num_warps=WG_WARPS,num_stages=2)


# def launch_gw_inverse(u, dw):
#     g,o,c = dw.shape[:3]
#     _wg_inverse[(triton.cdiv(g*o*c,128),)](u,dw,g,c,o,128,num_warps=4)


# def launch_grad_weight_bias(x, d, dw, db, workspace, gemm=None):
#     if gemm is None:
#         gemm = launch_winograd_gemm
#     v,e,u = workspace
#     g,b,_,h,w = x.shape
#     total = b*((h+1)//2)*((w+1)//2)
#     gc, tc = v.shape[0], v.shape[-1]
#     for base in range(0,g,gc):
#         end = min(g,base+gc)
#         xx,dd = x[base:end],d[base:end]
#         for start in range(0,total,tc):
#             launch_pack_x(xx,v,start)
#             launch_pack_d(dd,e,start)
#             gemm(v,e,u,end-base,add=start!=0)
#         launch_gw_inverse(u,dw[base:end])
#     launch_grad_bias(d,db)


# def winograd_f4_plan(x,d,max_bytes=WORKSPACE_LIMIT_BYTES):
#     g,b,c,h,w=x.shape;o=d.shape[2]
#     nt=b*((h+3)//4)*((w+3)//4)
#     fixed=F4_COMPONENTS*o*c*4
#     per_tile=F4_COMPONENTS*(c+o)*4
#     per_group=fixed+per_tile*nt
#     if per_group<=max_bytes:
#         gc,tc=min(g,max_bytes//per_group),nt
#     else:
#         gc,tc=1,min(nt,(max_bytes-fixed)//per_tile)
#         if tc<1:raise ValueError("F(4,3) fixed state exceeds the workspace budget")
#         if tc>=16:tc=tc//16*16
#     return dict(groups=gc,tiles=tc,total_tiles=nt,
#                 group_chunks=(g+gc-1)//gc,tile_chunks=(nt+tc-1)//tc,
#                 bytes=gc*(fixed+per_tile*tc),
#                 kernel_launches=((g+gc-1)//gc)*(3*((nt+tc-1)//tc)+1)+1)


# def grad_weight_f4_workspace(x,d,max_bytes=WORKSPACE_LIMIT_BYTES):
#     p=winograd_f4_plan(x,d,max_bytes);gc,tc=p["groups"],p["tiles"]
#     c,o=x.shape[2],d.shape[2]
#     return (x.new_empty((gc,F4_COMPONENTS,c,tc)),
#             x.new_empty((gc,F4_COMPONENTS,o,tc)),
#             x.new_empty((gc,F4_COMPONENTS,o,c)))


# def f4_dx_plan(x,d,max_bytes=WORKSPACE_LIMIT_BYTES):
#     g,b,c,h,w=x.shape;o=d.shape[2]
#     nt=b*((h+3)//4)*((w+3)//4)
#     fixed=F4_COMPONENTS*c*o*4
#     per_tile=F4_COMPONENTS*(c+o)*4
#     per_group=fixed+per_tile*nt
#     if per_group<=max_bytes:
#         gc,tc=min(g,max_bytes//per_group),nt
#     else:
#         gc,tc=1,min(nt,(max_bytes-fixed)//per_tile)
#         if tc<1:raise ValueError("F(4,3) dX fixed state exceeds the workspace budget")
#         if tc>=16:tc=tc//16*16
#     return dict(groups=gc,tiles=tc,total_tiles=nt,
#                 group_chunks=(g+gc-1)//gc,tile_chunks=(nt+tc-1)//tc,
#                 bytes=gc*(fixed+per_tile*tc),
#                 kernel_launches=((g+gc-1)//gc)*(1+3*((nt+tc-1)//tc)))


# def f4_dx_workspace(x,d,max_bytes=WORKSPACE_LIMIT_BYTES):
#     p=f4_dx_plan(x,d,max_bytes);gc,tc=p["groups"],p["tiles"]
#     c,o=x.shape[2],d.shape[2]
#     return (x.new_empty((gc,F4_COMPONENTS,o,tc)),
#             x.new_empty((gc,F4_COMPONENTS,c,tc)),
#             x.new_empty((gc,F4_COMPONENTS,c,o)))


# def joint_f4_plan(x,d,max_bytes=WORKSPACE_LIMIT_BYTES):
#     g,b,c,h,w=x.shape;o=d.shape[2]
#     nt=b*((h+3)//4)*((w+3)//4)
#     one_fixed=F4_COMPONENTS*c*o*4
#     per_tile=F4_COMPONENTS*(o+max(c,o)+c)*4
#     alias_group=one_fixed+per_tile*nt
#     if alias_group<=max_bytes:
#         gc,tc=min(g,max_bytes//alias_group),nt
#         alias_du=True;fixed=one_fixed
#     else:
#         # With streamed tile chunks the transformed weight must survive while
#         # dU accumulates, so it needs a second fixed-size buffer.
#         fixed=2*one_fixed
#         full_group=fixed+per_tile*nt
#         if full_group<=max_bytes:
#             gc,tc=min(g,max_bytes//full_group),nt
#         else:
#             gc,tc=1,min(nt,(max_bytes-fixed)//per_tile)
#             if tc<1:raise ValueError("joint F(4,3) fixed state exceeds the workspace budget")
#             if tc>=16:tc=tc//16*16
#         alias_du=False
#     cg=(g+gc-1)//gc;ct=(nt+tc-1)//tc
#     # Per group chunk: weight transform, five kernels per tile chunk, dW
#     # inverse. Bias is one final launch for the complete tensor.
#     return dict(groups=gc,tiles=tc,total_tiles=nt,alias_du=alias_du,
#                 group_chunks=cg,tile_chunks=ct,
#                 bytes=gc*(fixed+per_tile*tc),
#                 kernel_launches=cg*(2+5*ct)+1)


# def joint_f4_workspace(x,d,max_bytes=WORKSPACE_LIMIT_BYTES):
#     p=joint_f4_plan(x,d,max_bytes);gc,tc=p["groups"],p["tiles"]
#     c,o=x.shape[2],d.shape[2]
#     e=x.new_empty((gc,F4_COMPONENTS,o,tc))
#     shared=x.new_empty((gc,F4_COMPONENTS,max(c,o),tc))
#     mid=x.new_empty((gc,F4_COMPONENTS,c,tc))
#     uw=x.new_empty((gc,F4_COMPONENTS,c,o))
#     du=uw if p["alias_du"] else x.new_empty((gc,F4_COMPONENTS,o,c))
#     return e,shared,mid,uw,du


# def launch_f4_pack_x(x,v,start=0):
#     g,b,c,h,w=x.shape;nt=v.shape[-1]
#     _wg_f4_pack_x[(triton.cdiv(c*nt,64),g)](
#         x,v,v,start,b,c,h,w,nt,False,64,num_warps=4)


# def launch_f4_pack_d(d,e,start=0):
#     g,b,o,h,w=d.shape;nt=e.shape[-1]
#     _wg_f4_pack_d[(triton.cdiv(o*nt,64),g)](
#         d,e,start,b,o,h,w,nt,64,num_warps=4)


# def launch_f4_joint_pack_d(d,v,e,start=0):
#     g,b,o,h,w=d.shape;nt=e.shape[-1]
#     _wg_f4_joint_pack_d[(triton.cdiv(o*nt,64),g)](
#         d,v,e,start,b,o,h,w,nt,64,num_warps=4)


# def launch_f4_pack_weight_dx(weight,u):
#     g,o,c=weight.shape[:3]
#     _wg_f4_pack_weight_dx[(triton.cdiv(c*o,64),g)](
#         weight,u,g,c,o,64,num_warps=4)


# def launch_f4_dx_gemm(u,v,m,active_groups,c=None,o=None):
#     if c is None:c=m.shape[2]
#     if o is None:o=v.shape[2]
#     nt=m.shape[-1]
#     bm=64
#     bn=16 if nt<=16 else 32 if nt<=32 else 64
#     bk=16
#     _wg_f4_dx_gemm[(triton.cdiv(c,bm)*triton.cdiv(nt,bn),active_groups,F4_COMPONENTS)](
#         u,v,m,c,o,nt,bm,bn,bk,num_warps=4,num_stages=2)


# def launch_f4_dx_gemm_schedule(u,v,m,active_groups,c,o,bm,bn,bk,warps):
#     nt=m.shape[-1]
#     _wg_f4_dx_gemm[(triton.cdiv(c,bm)*triton.cdiv(nt,bn),active_groups,F4_COMPONENTS)](
#         u,v,m,c,o,nt,bm,bn,bk,num_warps=warps,num_stages=2)


# def launch_f4_dx_output(m,dx,start=0):
#     g,b,c,h,w=dx.shape;nt=m.shape[-1]
#     _wg_f4_dx_output[(triton.cdiv(c*nt,64),g)](
#         m,dx,start,g,b,c,h,w,nt,64,num_warps=4)


# def launch_f4_gemm(v,e,u,active_groups,add=False):
#     c,nt=v.shape[2:];o=e.shape[2]
#     bm,bn,bk=WG_TILE_V7
#     _wg_f4_gemm[(triton.cdiv(o,bm)*triton.cdiv(c,bn),active_groups,F4_COMPONENTS)](
#         v,e,u,c,o,nt,add,bm,bn,bk,num_warps=4,num_stages=2)


# def launch_f4_gemm_explicit(v,e,u,active_groups,c,o,add=False):
#     """Packed-pointer form used when v's backing storage has max(C,O)."""
#     nt=e.shape[-1];bm,bn,bk=WG_TILE_V7
#     _wg_f4_gemm[(triton.cdiv(o,bm)*triton.cdiv(c,bn),active_groups,F4_COMPONENTS)](
#         v,e,u,c,o,nt,add,bm,bn,bk,num_warps=4,num_stages=2)


# def launch_f4_gemm_schedule(v,e,u,active_groups,c,o,add,bm,bn,bk,warps):
#     nt=e.shape[-1]
#     _wg_f4_gemm[(triton.cdiv(o,bm)*triton.cdiv(c,bn),active_groups,F4_COMPONENTS)](
#         v,e,u,c,o,nt,add,bm,bn,bk,num_warps=warps,num_stages=2)


# def launch_f4_gemm_bmm(v,e,u,active_groups,c=None,o=None,add=False):
#     """IEEE batched GEMM on the native Winograd buffers (no layout copy).

#     ``v`` may use the joint path's max(C, O)-wide backing allocation, but
#     ``launch_f4_pack_x`` deliberately overwrites that storage in compact
#     [group, component, C, tile] order.  The view below therefore uses C*tile
#     (not the Python allocation's max(C,O)*tile stride) between component
#     matrices.  No compacting copy is materialized.  The output is written
#     directly into the caller-provided transformed-weight buffer.
#     """
#     if c is None:c=v.shape[2]
#     if o is None:o=e.shape[2]
#     batches=active_groups*F4_COMPONENTS
#     aa=e.as_strided((batches,o,e.shape[-1]),
#                     (e.stride(1),e.stride(2),e.stride(3)))
#     nt=v.shape[-1]
#     vv=v.as_strided((batches,c,nt),(c*nt,nt,1))
#     bb=vv.transpose(1,2)
#     # ``u`` may alias the [C,O] transformed-weight allocation.  Reinterpret
#     # its identical contiguous C*O storage as [O,C] instead of trusting the
#     # Python shape of that alias.
#     out=u.as_strided((batches,o,c),(o*c,c,1))
#     if add:
#         torch.baddbmm(out,aa,bb,beta=1,alpha=1,out=out)
#     else:
#         torch.bmm(aa,bb,out=out)


# def launch_f4_inverse(u,dw):
#     g,o,c=dw.shape[:3]
#     _wg_f4_inverse[(triton.cdiv(g*o*c,64),)](
#         u,dw,g,c,o,64,num_warps=4)


# def launch_grad_weight_bias_f4(x,d,dw,db,workspace,gemm=None):
#     if gemm is None:gemm=launch_f4_gemm
#     v,e,u=workspace;g,b,_,h,w=x.shape
#     total=b*((h+3)//4)*((w+3)//4);gc,tc=v.shape[0],v.shape[-1]
#     for base in range(0,g,gc):
#         end=min(g,base+gc);xx,dd=x[base:end],d[base:end]
#         for start in range(0,total,tc):
#             launch_f4_pack_x(xx,v,start);launch_f4_pack_d(dd,e,start)
#             gemm(v,e,u,end-base,add=start!=0)
#         launch_f4_inverse(u,dw[base:end])
#     launch_grad_bias(d,db)


# def launch_grad_x_f4(d,weight,dx,workspace):
#     v,m,u=workspace;g,b,o,h,w=d.shape;c=weight.shape[2]
#     total=b*((h+3)//4)*((w+3)//4);gc,tc=v.shape[0],v.shape[-1]
#     for base in range(0,g,gc):
#         end=min(g,base+gc);dd,ww,out=d[base:end],weight[base:end],dx[base:end]
#         launch_f4_pack_weight_dx(ww,u)
#         for start in range(0,total,tc):
#             launch_f4_pack_x(dd,v,start)
#             launch_f4_dx_gemm(u,v,m,end-base,c,o)
#             launch_f4_dx_output(m,out,start)


# def launch_joint_f4(x,weight,d,dx,dw,db,workspace,dx_gemm=None,dw_gemm=None):
#     e,shared,mid,uw,du=workspace
#     g,b,c,h,w=x.shape;o=d.shape[2]
#     total=b*((h+3)//4)*((w+3)//4);gc,tc=e.shape[0],e.shape[-1]
#     for base in range(0,g,gc):
#         end=min(g,base+gc);active=end-base
#         xx,ww,dd=x[base:end],weight[base:end],d[base:end]
#         outx,outw=dx[base:end],dw[base:end]
#         launch_f4_pack_weight_dx(ww,uw)
#         for start in range(0,total,tc):
#             # One launch exposes both dY transforms to compiler CSE.
#             launch_f4_joint_pack_d(dd,shared,e,start)
#             if dx_gemm is None:
#                 launch_f4_dx_gemm(uw,shared,mid,active,c,o)
#             else:
#                 dx_gemm(uw,shared,mid,active,c,o)
#             launch_f4_dx_output(mid,outx,start)
#             # dV is dead after output transform; overwrite its backing buffer
#             # with B^T X B before the weight-gradient GEMM.
#             launch_f4_pack_x(xx,shared,start)
#             if dw_gemm is None:
#                 launch_f4_gemm_explicit(shared,e,du,active,c,o,add=start!=0)
#             else:
#                 dw_gemm(shared,e,du,active,c,o,start!=0)
#         launch_f4_inverse(du,outw)
#     launch_grad_bias(d,db)


# def _run(x,weight,grad_output,stride,padding,dilation,groups,gemm=None,
#          old_dw=False,f4=False,f4_dx=False,joint=False):
#     g,_,_,o,_,_ = _validate(x,weight,grad_output,stride,padding,dilation,groups)
#     with torch.cuda.device(x.device):
#         x,weight,d = (t.contiguous() for t in (x,weight,grad_output))
#         dx,dw,db = torch.empty_like(x),torch.empty_like(weight),x.new_empty((g,o))
#         if joint:
#             work=joint_f4_workspace(x,d)
#             launch_joint_f4(x,weight,d,dx,dw,db,work,dw_gemm=gemm)
#             return dx,dw,db
#         if f4_dx:
#             gx_work=f4_dx_workspace(x,d)
#             launch_grad_x_f4(d,weight,dx,gx_work)
#         else:
#             gx_work = grad_x_workspace(d,dx)
#             launch_grad_x(d,weight,dx,gx_work)
#         del gx_work
#         if old_dw:
#             launch_grad_weight_bias_v5(x,d,dw,db)
#         elif f4:
#             gw_work=grad_weight_f4_workspace(x,d)
#             launch_grad_weight_bias_f4(x,d,dw,db,gw_work,gemm)
#         else:
#             gw_work = grad_weight_workspace(x,d)
#             launch_grad_weight_bias(x,d,dw,db,gw_work,gemm)
#     return dx,dw,db


# def _run_small_direct(x,weight,grad_output,stride,padding,dilation,groups,
#                       dx_kind="implicit32",workspace=None):
#     """Two-launch direct path, or three launches for the bmm+fold ablation."""
#     g,_,_,o,_,_=_validate(x,weight,grad_output,stride,padding,dilation,groups)
#     with torch.cuda.device(x.device):
#         x,weight,d=(t.contiguous() for t in (x,weight,grad_output))
#         dx,dw,db=torch.empty_like(x),torch.empty_like(weight),x.new_empty((g,o))
#         if dx_kind=="implicit16":
#             launch_grad_x_implicit(d,weight,dx,16,32,16,4)
#         elif dx_kind=="implicit16sq":
#             launch_grad_x_implicit(d,weight,dx,16,16,16,4)
#         elif dx_kind=="implicit32":
#             launch_grad_x_implicit(d,weight,dx,32,32,16,4)
#         elif dx_kind=="implicit64":
#             launch_grad_x_implicit(d,weight,dx,32,64,16,4)
#         elif dx_kind=="implicit_bk32":
#             launch_grad_x_implicit(d,weight,dx,32,32,32,4)
#         elif dx_kind=="bmm":
#             if workspace is None:workspace=grad_x_workspace(d,dx)
#             launch_grad_x_bmm(d,weight,dx,workspace)
#         else:
#             raise ValueError("unknown small direct dX schedule")
#         launch_grad_weight_bias_v5(x,d,dw,db)
#     return dx,dw,db


# def _run_rgb(x,weight,grad_output,stride,padding,dilation,groups,
#              dx_kind="direct3_16",force_split=None):
#     """Specialized ConvNet RGB-input path with compact C=3 computation."""
#     g,_,c,o,_,_=_validate(x,weight,grad_output,stride,padding,dilation,groups)
#     if c!=3:
#         raise ValueError("v13 RGB candidates require C=3")
#     with torch.cuda.device(x.device):
#         x,weight,d=(t.contiguous() for t in (x,weight,grad_output))
#         dx,dw,db=torch.empty_like(x),torch.empty_like(weight),x.new_empty((g,o))
#         if dx_kind=="direct3_16":
#             launch_rgb_grad_x_direct3(d,weight,dx,16,16,4)
#         elif dx_kind=="direct3_32":
#             launch_rgb_grad_x_direct3(d,weight,dx,32,16,4)
#         elif dx_kind=="implicit16sq":
#             launch_grad_x_implicit(d,weight,dx,16,16,16,4)
#         elif dx_kind=="bmm":
#             workspace=grad_x_workspace(d,dx)
#             launch_grad_x_bmm(d,weight,dx,workspace)
#         else:
#             raise ValueError("unknown v13 RGB dX schedule")
#         launch_rgb_grad_weight_bias(x,d,dw,db,force_split)
#     return dx,dw,db


# def conv_bwd(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
#     """v11: joint F(4,3), Triton dX and native-layout batched-GEMM dW."""
#     return _run(x,weight,grad_output,stride,padding,dilation,groups,
#                 gemm=launch_f4_gemm_bmm,joint=True)


# def conv_bwd_v13_rgb(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
#     """v13 primary RGB candidate: scalar C=3 dX plus occupancy-aware dW/dB."""
#     return _run_rgb(x,weight,grad_output,stride,padding,dilation,groups)


# def conv_bwd_v13_rgb32(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
#     return _run_rgb(x,weight,grad_output,stride,padding,dilation,groups,"direct3_32")


# def conv_bwd_v13_rgb_implicit(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
#     """Ablation: padded implicit dX with the same compact RGB dW/dB."""
#     return _run_rgb(x,weight,grad_output,stride,padding,dilation,groups,"implicit16sq")


# def conv_bwd_v13_rgb_bmm(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
#     """Ablation: bmm+fold dX with the same compact RGB dW/dB."""
#     return _run_rgb(x,weight,grad_output,stride,padding,dilation,groups,"bmm")


# def conv_bwd_v13_rgb_nosplit(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
#     """Ablation for tiny cases: primary dX with one unsplit dW CTA per O tile."""
#     return _run_rgb(x,weight,grad_output,stride,padding,dilation,groups,
#                     "direct3_16",1)


# def conv_bwd_v12_direct(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
#     """v12 small-work candidate: one implicit dX kernel plus fused direct dW/dB."""
#     return _run_small_direct(x,weight,grad_output,stride,padding,dilation,groups)


# def conv_bwd_v12_direct16(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
#     return _run_small_direct(x,weight,grad_output,stride,padding,dilation,groups,"implicit16")


# def conv_bwd_v12_direct16sq(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
#     """Higher-grid small-N schedule: 16x16x16 implicit dX tiles."""
#     return _run_small_direct(x,weight,grad_output,stride,padding,dilation,groups,"implicit16sq")


# def conv_bwd_v12_direct64(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
#     return _run_small_direct(x,weight,grad_output,stride,padding,dilation,groups,"implicit64")


# def conv_bwd_v12_direct_bk32(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
#     return _run_small_direct(x,weight,grad_output,stride,padding,dilation,groups,"implicit_bk32")


# def conv_bwd_v12_bmm_dx(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
#     """Ablation: bmm dX columns + fold, with the same fused direct dW/dB."""
#     return _run_small_direct(x,weight,grad_output,stride,padding,dilation,groups,"bmm")


# def conv_bwd_v10(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
#     """v10 control: joint F(4,3) with Triton GEMMs for both contractions."""
#     return _run(x,weight,grad_output,stride,padding,dilation,groups,joint=True)


# def conv_bwd_separate_bmm(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
#     """Ablation: F(4,3) dX and batched-GEMM dW, recomputing dY transform."""
#     return _run(x,weight,grad_output,stride,padding,dilation,groups,
#                 gemm=launch_f4_gemm_bmm,f4=True,f4_dx=True)


# def conv_bwd_separate_f4(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
#     """Ablation: F(4,3) dX and dW, but transform dY separately."""
#     return _run(x,weight,grad_output,stride,padding,dilation,groups,f4=True,f4_dx=True)


# def conv_bwd_v9(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
#     """v9 baseline: direct GEMM+fold dX and F(4,3) dW."""
#     return _run(x,weight,grad_output,stride,padding,dilation,groups,f4=True)


# def conv_bwd_v7(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
#     """Explicit v7 baseline with its original GEMM and identical buffers."""
#     return _run(x,weight,grad_output,stride,padding,dilation,groups,launch_winograd_gemm_v7)


# def conv_bwd_v8(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
#     """Failed v8 block-pointer F(2,3) ablation, retained for diagnostics."""
#     return _run(x,weight,grad_output,stride,padding,dilation,groups,launch_winograd_gemm)


# def conv_bwd_v5(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
#     """Explicit v5 baseline: identical dX, original three-tap dW/dB."""
#     return _run(x,weight,grad_output,stride,padding,dilation,groups,old_dw=True)













# ############################
# ############################
# ############################
# ############################
# ############################
# fwd 如下
# 注意，没有经过正式测试。后面需要修改。
# ############################
# ############################
# ############################
# ############################
# ############################
# ############################

def configs():
    # Search both CRS-flattened and spatial-loop/channel-blocked iterators.
    return [triton.Config(dict(TM=m, TN=n, TK=k, RS_LOOP=rs),
                          num_warps=warps, num_stages=stages)
            for rs in (False, True)
            for m,n,k,warps,stages in (
                (16,64,32,4,2), (32,64,32,4,3),
                (32,128,32,4,3), (64,64,32,4,3),
                (32,64,64,4,3), (64,128,32,8,3))]


@triton.autotune(configs=configs(), key=[
    'G','B','C','H','W','O','R','S','P','Q','SH','SW','PH','PW','DH','DW',
    'BIAS','PRECISION'])
@triton.jit
def _conv(X, Weight, Bias, Y, G:tl.constexpr, B:tl.constexpr,
          C:tl.constexpr,H:tl.constexpr,W:tl.constexpr,O:tl.constexpr,
          R:tl.constexpr,S:tl.constexpr,P:tl.constexpr,Q:tl.constexpr,
          SH:tl.constexpr,SW:tl.constexpr,PH:tl.constexpr,PW:tl.constexpr,
          DH:tl.constexpr,DW:tl.constexpr,BIAS:tl.constexpr,
          PRECISION:tl.constexpr,TM:tl.constexpr,TN:tl.constexpr,
          TK:tl.constexpr,RS_LOOP:tl.constexpr):
    # Each program owns one [output-channel, output-spatial] tile.
    # Batch stays outside GEMM to preserve contiguous NCHW spatial stores.
    pid = tl.program_id(0)
    gb = tl.program_id(1)
    g, b = gb // B, gb % B
    nm, nn = tl.cdiv(O,TM), tl.cdiv(P*Q,TN)
    # Reuse activation tiles across up to four output-channel tiles.
    chunk = pid // (4*nn)
    first = chunk*4
    actual = tl.minimum(nm-first,4)
    mi = first + (pid % (4*nn)) % actual
    ni = (pid % (4*nn)) // actual
    oc = mi*TM + tl.arange(0,TM)
    n = ni*TN + tl.arange(0,TN)
    ih0, iw0 = n//Q*SH-PH, n%Q*SW-PW
    kk = tl.arange(0,TK)
    xp = X + (g*B+b)*C*H*W
    wp = Weight + g*O*C*R*S
    acc = tl.zeros((TM,TN),tl.float32)
    if RS_LOOP:
        # A fixed filter position makes adjacent spatial loads contiguous.
        # Useful for large C; for C=3 the flattened variant wastes less work.
        for cb in range(tl.cdiv(C,TK)):
            ci = cb*TK+kk
            for r in tl.static_range(R):
                for s in tl.static_range(S):
                    ih, iw = ih0+r*DH, iw0+s*DW
                    a = tl.load(wp+oc[:,None]*(C*R*S)+ci[None,:]*(R*S)+r*S+s,
                                (oc[:,None]<O)&(ci[None,:]<C),0)
                    v = tl.load(xp+ci[:,None]*H*W+ih[None,:]*W+iw[None,:],
                                (ci[:,None]<C)&(n[None,:]<P*Q)&
                                (ih[None,:]>=0)&(ih[None,:]<H)&
                                (iw[None,:]>=0)&(iw[None,:]<W),0)
                    acc = tl.dot(a,v,acc,input_precision=PRECISION)
    else:
        for kb in range(tl.cdiv(C*R*S,TK)):
            k = kb*TK+kk
            ci, r, s = k//(R*S), (k//S)%R, k%S
            ih = ih0[None,:]+r[:,None]*DH
            iw = iw0[None,:]+s[:,None]*DW
            a = tl.load(wp+oc[:,None]*(C*R*S)+k[None,:],
                        (oc[:,None]<O)&(k[None,:]<C*R*S),0)
            v = tl.load(xp+ci[:,None]*H*W+ih*W+iw,
                        (k[:,None]<C*R*S)&(n[None,:]<P*Q)&
                        (ih>=0)&(ih<H)&(iw>=0)&(iw<W),0)
            acc = tl.dot(a,v,acc,input_precision=PRECISION)
    if BIAS:
        acc += tl.load(Bias+g*O+oc,oc<O,0)[:,None]
    tl.store(Y+((g*B+b)*O+oc[:,None])*(P*Q)+n[None,:],acc,
             (oc[:,None]<O)&(n[None,:]<P*Q))


def pair(v):
    if isinstance(v,int):
        return v,v
    if len(v)!=2 or any(not isinstance(i,int) for i in v):
        raise ValueError('Expected an int or pair of ints')
    return tuple(v)


def grouped_conv2d(x, weight, bias=None, stride=1, padding=0,
                   dilation=1, groups=1, precision='ieee'):
    """X [G*B,C,H,W], W [G,O,C,R,S], bias [G,O] -> [G*B,O,P,Q].

    Leading dimension uses group-major order: index = group * B + batch.
    Forward only. Use no_grad for manual-backward integrations.
    """
    if x.ndim != 4 or weight.ndim != 5:
        raise ValueError('Expected x=[G*B,C,H,W], weight=[G,O,C,R,S]')
    if not isinstance(groups, int) or groups <= 0:
        raise ValueError('groups must be a positive integer')

    n, c, h, w = x.shape
    gw, o, cw, r, s = weight.shape
    g = groups
    if n % g != 0:
        raise ValueError('x.shape[0] must be divisible by groups')
    b = n // g

    if g != gw or c != cw or min(*x.shape, *weight.shape) <= 0:
        raise ValueError('Invalid shapes or mismatched group/channel dimensions')

    sh, sw = pair(stride)
    ph, pw = pair(padding)
    dh, dw = pair(dilation)
    if min(sh, sw, dh, dw) <= 0 or min(ph, pw) < 0:
        raise ValueError('Invalid convolution parameters')

    p = (h + 2*ph - dh*(r-1) - 1) // sh + 1
    q = (w + 2*pw - dw*(s-1) - 1) // sw + 1
    if min(p, q) <= 0 or n > 65535:
        raise ValueError('Invalid output size or G*B exceeds CUDA grid.y limit')

    tensors = [x, weight] + ([] if bias is None else [bias])
    if bias is not None and tuple(bias.shape) != (g, o):
        raise ValueError('Bias must have shape [G,O]')
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError('Supported dtypes: float16, bfloat16, float32')

    for t in tensors:
        if (not t.is_cuda or t.device != x.device
                or t.dtype != x.dtype or not t.is_contiguous()):
            raise ValueError(
                'All tensors must be contiguous CUDA tensors on same device/dtype'
            )
        if torch.is_grad_enabled() and t.requires_grad:
            raise RuntimeError('Forward-only kernel: autograd is not implemented')

    if precision not in ('ieee', 'tf32', 'tf32x3'):
        raise ValueError('Invalid precision')
    if max(x.numel(), weight.numel(), n*o*p*q) >= 2**31:
        raise ValueError('Tensor too large for int32 indexing')

    with torch.cuda.device(x.device):
        y = torch.empty((n, o, p, q), device=x.device, dtype=x.dtype)
        grid = lambda meta: (
            triton.cdiv(o, meta['TM']) * triton.cdiv(p*q, meta['TN']),
            n,
        )
        _conv[grid](
            x, weight, bias if bias is not None else x, y,
            g, b, c, h, w, o, r, s, p, q,
            sh, sw, ph, pw, dh, dw, bias is not None, precision,
        )
    return y