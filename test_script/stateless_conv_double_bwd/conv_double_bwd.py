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
