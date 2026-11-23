"""Bounded-size spatial tiles for group-first norm planes larger than 4096.

Internal helpers shared by fwd/bwd/double-bwd. Small-spatial kernels are kept
in their original modules. No activation packing or whole-activation scratch.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _stats(X, D, OUT, H, ST, PG, PB, B: tl.constexpr, C: tl.constexpr,
           ROWS: tl.constexpr, M: tl.constexpr, EPS: tl.constexpr,
           MODE: tl.constexpr, HAS_H: tl.constexpr, HAS_G: tl.constexpr,
           S: tl.constexpr):
    row = tl.program_id(0)
    base = row.to(tl.int64)*M
    col = tl.arange(0, S)
    anchor = tl.load(X+base).to(tl.float32)
    acc = tl.zeros((S,),tl.float32)
    for block in range(tl.cdiv(M,S)):
        j = block*S+col
        x = tl.load(X+base+j,j<M,other=0.).to(tl.float32)
        acc += tl.where(j<M,x-anchor,0.)
    mu = tl.sum(acc,0)/M
    vv = tl.zeros((S,),tl.float32)
    sd = tl.zeros((S,),tl.float32)
    sdx = tl.zeros((S,),tl.float32)
    sh = tl.zeros((S,),tl.float32)
    shx = tl.zeros((S,),tl.float32)
    for block in range(tl.cdiv(M,S)):
        j = block*S+col
        x = tl.load(X+base+j,j<M,other=0.).to(tl.float32)
        xc = tl.where(j<M,(x-anchor)-mu,0.)
        vv += xc*xc
        if MODE != 0:
            d = tl.load(D+base+j,j<M,other=0.).to(tl.float32)
            out = tl.load(OUT+base+j,j<M,other=0.).to(tl.float32)
            d = tl.where(out>0.,d,0.)
            sd += d
            sdx += d*xc
            if HAS_H:
                h = tl.load(H+base+j,j<M,other=0.).to(tl.float32)
                sh += h
                shx += h*xc
    inv = tl.rsqrt(tl.sum(vv,0)/M+EPS)
    tl.store(ST+row,anchor)
    tl.store(ST+ROWS+row,mu)
    tl.store(ST+2*ROWS+row,inv)
    if MODE != 0:
        sum_d = tl.sum(sd,0)
        sum_dz = tl.sum(sdx,0)*inv
        d0 = sum_d/M
        dz = sum_dz/M
        tl.store(ST+3*ROWS+row,d0)
        tl.store(ST+4*ROWS+row,dz)
        if MODE == 1:
            # PG/PB are affine outputs directly at B=1, per-row partials otherwise.
            tl.store(PG+row,sum_dz)
            tl.store(PB+row,sum_d)
        if HAS_H:
            h0 = tl.sum(sh,0)/M
            hz = tl.sum(shx,0)*inv/M
            # Match the small kernel's projected dot instead of subtracting
            # three large, nearly equal scalar products for the Hessian term.
            dot_acc = tl.zeros((S,),tl.float32)
            for block in range(tl.cdiv(M,S)):
                j = block*S+col
                x = tl.load(X+base+j,j<M,other=0.).to(tl.float32)
                z = ((x-anchor)-mu)*inv
                d = tl.load(D+base+j,j<M,other=0.).to(tl.float32)
                out = tl.load(OUT+base+j,j<M,other=0.).to(tl.float32)
                d = tl.where(out>0.,d,0.)
                h = tl.load(H+base+j,j<M,other=0.).to(tl.float32)
                dot_acc += tl.where(j<M,h*(d-d0-z*dz),0.)
            dot = tl.sum(dot_acc,0)
            tl.store(ST+5*ROWS+row,h0)
            tl.store(ST+6*ROWS+row,hz)
            tl.store(ST+7*ROWS+row,dot/M)
            if HAS_G:
                tl.store(PG+row,dot*inv)


@triton.jit
def _apply(X, GAMMA, BETA, D, OUT, H, V, T, ST, Y, GX,
           B: tl.constexpr, C: tl.constexpr, ROWS: tl.constexpr,
           M: tl.constexpr, MODE: tl.constexpr, HAS_G: tl.constexpr,
           HAS_BETA: tl.constexpr, HAS_H: tl.constexpr,
           HAS_V: tl.constexpr, HAS_T: tl.constexpr, S: tl.constexpr):
    row = tl.program_id(0)
    j = tl.program_id(1)*S+tl.arange(0,S)
    off = row.to(tl.int64)*M+j
    affine = (row//(B*C))*C+row%C
    anchor = tl.load(ST+row)
    mu = tl.load(ST+ROWS+row)
    inv = tl.load(ST+2*ROWS+row)
    x = tl.load(X+off,j<M,other=0.).to(tl.float32)
    z = ((x-anchor)-mu)*inv
    gamma = tl.full((),1.,tl.float32)
    if HAS_G:gamma = tl.load(GAMMA+affine).to(tl.float32)
    if MODE == 0:
        y = z*gamma
        if HAS_BETA:y += tl.load(BETA+affine).to(tl.float32)
        y = tl.maximum(y,0.,propagate_nan=tl.PropagateNan.ALL)
        tl.store(Y+off,y,j<M)
    else:
        out = tl.load(OUT+off,j<M,other=0.).to(tl.float32)
        d = tl.load(D+off,j<M,other=0.).to(tl.float32)
        d = tl.where(out>0.,d,0.)
        d0 = tl.load(ST+3*ROWS+row)
        dz = tl.load(ST+4*ROWS+row)
        pd = d-d0-z*dz
        if MODE == 1:
            tl.store(Y+off,pd*(gamma*inv),j<M)
        else:
            gx = tl.zeros((S,),tl.float32)
            ggo = tl.zeros((S,),tl.float32)
            v = tl.full((),0.,tl.float32)
            if HAS_V:v = tl.load(V+affine).to(tl.float32)
            if HAS_H:
                h = tl.load(H+off,j<M,other=0.).to(tl.float32)
                h0 = tl.load(ST+5*ROWS+row)
                hz = tl.load(ST+6*ROWS+row)
                q = tl.load(ST+7*ROWS+row)
                ph = h-h0-z*hz
                scale = gamma*inv*inv
                gx = pd*(v*inv-scale*hz)-scale*(dz*ph+z*q)
                ggo = ph*(gamma*inv)
            elif HAS_V:
                gx = pd*(v*inv)
            if HAS_V:ggo += z*v
            if HAS_T:ggo += tl.load(T+affine).to(tl.float32)
            tl.store(GX+off,gx,j<M)
            tl.store(Y+off,tl.where(out>0.,ggo,0.),j<M)


def _launch(x, weight, bias, d, out, h, v, t, y, gx, pg, pb,
            b, c, m, eps, mode):
    rows = x.shape[0]*c
    if rows >= 2**31:
        raise ValueError('Too many spatial planes for int32 row indexing')
    st = torch.empty((3 if mode==0 else 5 if mode==1 else 8,rows),device=x.device,dtype=torch.float32)
    ptr = lambda a:x if a is None else a
    _stats[(rows,)](x,ptr(d),ptr(out),ptr(h),st,ptr(pg),ptr(pb),
        b,c,rows,m,eps,mode,h is not None,weight is not None,1024,num_warps=4)
    _apply[(rows,triton.cdiv(m,1024))](x,ptr(weight),ptr(bias),ptr(d),ptr(out),
        ptr(h),ptr(v),ptr(t),st,y,ptr(gx),b,c,rows,m,mode,
        weight is not None,bias is not None,h is not None,v is not None,
        t is not None,1024,num_warps=4)


def forward(x, weight, bias, b, c, m, eps):
    y = torch.empty_like(x)
    _launch(x,weight,bias,None,None,None,None,None,y,None,None,None,b,c,m,eps,0)
    return y


def backward(x, weight, output, grad_output, buffers, b, c, m, eps):
    dx,dg,db,pg,pb = buffers
    _launch(x,weight,None,grad_output,output,None,None,None,dx,None,pg,pb,b,c,m,eps,1)


def double_backward(x, weight, h, v, t, d, out, buffers, b, c, m, eps):
    ggo,gx,gg,part = buffers
    _launch(x,weight,None,d,out,h,v,t,ggo,gx,part,None,b,c,m,eps,2)
