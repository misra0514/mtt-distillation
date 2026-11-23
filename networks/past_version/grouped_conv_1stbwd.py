"""Group-first ConvNet 3x3 stride=padding=dilation=1 convolution backward.

conv_bwd(x, weight, grad_output, stride=1, padding=1, dilation=1, groups=1)
returns grad_x [G,B,C,H,W], grad_weight [G,O,C,3,3], grad_bias [G,O].

The public ``conv_bwd`` entry point applies the measured v13 selector.  v14
adds a batch-chunked direct dX contraction: one launch handles several complete
batches, while a small planner jointly chunks G and B under the 128 MiB
workspace limit.  This removes the one-GEMM-plus-one-fold launch pair per
batch paid by the v13 bmm path.  v14 also tests 32- and 64-output-channel RGB
dW tiles, plus an F(4,3) dW alternative.  These v14 paths remain explicit
candidates until benchmark data justifies changing the public selector.

v13 adds compact RGB-input kernels for ConvNet's first convolution.  Its dX
kernel reuses each dY load across exactly three scalar input-channel
accumulators, while its dW/dB kernel contracts the compact C*3*3=27 axis and
uses split reduction only when needed for GPU occupancy.  v12 adds a
two-launch, zero-workspace direct path for small dense workloads: one
implicit-GEMM kernel writes dX directly and the existing direct row kernel
computes dW/dB together.  A bmm+fold dX ablation is also exposed.
v11: Winograd F(4x4,3x3) for both dX and dW.  The joint candidate
loads each 6x6 dY tile once and emits both transforms needed by dX and dW.
dX uses the Triton contraction; dW uses batched GEMM directly on native
Winograd buffers.  v10 retains the all-Triton joint contraction as a control.
v9 retains v7 dX, Winograd F(4x4,3x3) dW, and separate dB.  F(4,3)
uses 36 transform components for sixteen output positions rather than v7's
16 components for four positions.  The dW GEMM deliberately retains the
proven manual-pointer 64x64x16/4-warp schedule after v8's block-pointer
64x64x32/8-warp experiment regressed on RTX A5000.
Triton contractions use IEEE FP32.  Batched GEMM follows PyTorch's matmul
precision setting; the benchmark explicitly disables TF32 for both sides.
Temporary buffers are reused within this call and limited to 128 MiB per
phase. No transformed values or tensors are cached across calls.
conv_bwd_v7 and conv_bwd_v5 retain the previous explicit baselines.  The v8
GEMM remains available only as a diagnostic launch helper.
"""
import functools
import operator
import torch
import triton
import triton.language as tl

WORKSPACE_LIMIT_BYTES = 128 * 1024**2
WG_TILE_V7 = (64, 64, 16)
WG_TILE = (64, 64, 32)
WG_WARPS = 8
F4_COMPONENTS = 36

def _pair(value):
    try:
        n = operator.index(value)
    except TypeError:
        values = tuple(map(operator.index, value))
        if len(values) != 2:
            raise ValueError("expected an integer or a pair of integers")
        return values
    return n, n

@triton.jit
def _dw_db_row_tile(X,D,DWOUT,DBOUT,C:tl.constexpr,O:tl.constexpr,
                    H:tl.constexpr,W:tl.constexpr,B:tl.constexpr,ROW:tl.constexpr,
                    BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    tile=tl.program_id(0)
    g=tl.program_id(1)
    no:tl.constexpr=(O+BN-1)//BN
    ci=tile//no*BM+tl.arange(0,BM)
    oc=tile%no*BN+tl.arange(0,BN)
    vh:tl.constexpr=H-(1 if ROW!=1 else 0)
    iy0:tl.constexpr=1 if ROW==2 else 0
    dy0:tl.constexpr=1 if ROW==0 else 0
    kk=tl.arange(0,BK)
    a0=tl.zeros((BM,BN),tl.float32)
    a1=tl.zeros((BM,BN),tl.float32)
    a2=tl.zeros((BM,BN),tl.float32)
    bias=tl.full((BN,),0,tl.float32)
    for block in range((B*vh*W+BK-1)//BK):
        k=block*BK+kk
        batch=k//(vh*W)
        pos=k%(vh*W)
        y=pos//W
        x=pos%W
        valid=k<B*vh*W
        dv=tl.load(D+(g*B+batch[:,None])*O*H*W+oc[None,:]*H*W+(y[:,None]+dy0)*W+x[:,None],
                   (oc[None,:]<O)&valid[:,None],0)
        base=(g*B+batch[None,:])*C*H*W+ci[:,None]*H*W+(y[None,:]+iy0)*W+x[None,:]
        mask=(ci[:,None]<C)&valid[None,:]
        x0=tl.load(X+base-1,mask&(x[None,:]>0),0)
        a0=tl.dot(x0,dv,a0,input_precision="ieee")
        x1=tl.load(X+base,mask,0)
        a1=tl.dot(x1,dv,a1,input_precision="ieee")
        x2=tl.load(X+base+1,mask&(x[None,:]+1<W),0)
        a2=tl.dot(x2,dv,a2,input_precision="ieee")
        if ROW==1:
            if tile//no==0:
                bias+=tl.sum(dv,0)
    target=DWOUT+g*O*C*9+oc[None,:]*C*9+ci[:,None]*9+ROW*3
    mask=(ci[:,None]<C)&(oc[None,:]<O)
    tl.store(target,a0,mask)
    tl.store(target+1,a1,mask)
    tl.store(target+2,a2,mask)
    if ROW==1:
        if tile//no==0:
            tl.store(DBOUT+g*O+oc,bias,oc<O)

@triton.jit
def _dw_db_row(X,D,DWOUT,DBOUT,C:tl.constexpr,O:tl.constexpr,
               H:tl.constexpr,W:tl.constexpr,B:tl.constexpr,
               BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    for row in tl.static_range(3):
        if tl.program_id(2)==row:
            _dw_db_row_tile(X,D,DWOUT,DBOUT,C,O,H,W,B,row,BM,BN,BK)

@triton.jit
def _db_reduce(D, DBOUT, B: tl.constexpr, O: tl.constexpr,
               HW: tl.constexpr, BLOCK: tl.constexpr):
    oc = tl.program_id(0)
    g = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    # Bounded per-CTA state even as B*HW grows; every dY is read once.
    for block in range((B * HW + BLOCK - 1) // BLOCK):
        k = block * BLOCK + offsets
        batch = k // HW
        pos = k % HW
        value = tl.load(D + ((g * B + batch) * O + oc) * HW + pos,
                        k < B * HW, 0)
        acc += value
    tl.store(DBOUT + g * O + oc, tl.sum(acc, 0))

@triton.jit
def _gx_gemm(D,W,Z,B:tl.constexpr,BATCH,
             C:tl.constexpr,O:tl.constexpr,HW:tl.constexpr,
             BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    tile=tl.program_id(0)
    g=tl.program_id(1)
    nn:tl.constexpr=(HW+BN-1)//BN
    crs=tile//nn*BM+tl.arange(0,BM)
    n=tile%nn*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for block in range((O+BK-1)//BK):
        oc=block*BK+kk
        weights=tl.load(W+(g*O+oc[None,:])*C*9+crs[:,None],
                        (crs[:,None]<C*9)&(oc[None,:]<O),0)
        dy=tl.load(D+((g*B+BATCH)*O+oc[:,None])*HW+n[None,:],
                   (oc[:,None]<O)&(n[None,:]<HW),0)
        acc=tl.dot(weights,dy,acc,input_precision="ieee")
    tl.store(Z+(g*C*9+crs[:,None])*HW+n[None,:],acc,
             (crs[:,None]<C*9)&(n[None,:]<HW))

@triton.jit
def _gx_fold(Z,DX,G:tl.constexpr,B:tl.constexpr,BATCH,
             C:tl.constexpr,H:tl.constexpr,W:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    pos=i%(H*W)
    ci=i//(H*W)%C
    g=i//(C*H*W)
    iy=pos//W
    ix=pos%W
    acc=tl.full((BLOCK,),0,tl.float32)
    for rr in tl.static_range(3):
        for ss in tl.static_range(3):
            py=iy+1-rr
            px=ix+1-ss
            offset=((g*C+ci)*9+rr*3+ss)*H*W+py*W+px
            value=tl.load(Z+offset,(i<G*C*H*W)&(py>=0)&(py<H)&(px>=0)&(px<W),0)
            acc+=value
    tl.store(DX+((g*B+BATCH)*C+ci)*H*W+pos,acc,i<G*C*H*W)


@triton.jit
def _gx_gemm_batch_chunk(D,W,Z,B:tl.constexpr,BSTART,
                         ACTIVE_B:tl.constexpr,MAX_B:tl.constexpr,
                         C:tl.constexpr,O:tl.constexpr,HW:tl.constexpr,
                         BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    """Z[c,r,s,b,p] = sum_o W[o,c,r,s] * dY[b,o,p].

    Unlike ``_gx_gemm``, N contains several whole batches.  Keeping whole
    images in a chunk lets the following fold read every spatial neighbour
    without halos or atomics.
    """
    tile=tl.program_id(0);g=tl.program_id(1)
    active_n:tl.constexpr=ACTIVE_B*HW
    max_n:tl.constexpr=MAX_B*HW
    nn:tl.constexpr=(active_n+BN-1)//BN
    crs=tile//nn*BM+tl.arange(0,BM)
    n=tile%nn*BN+tl.arange(0,BN)
    local_batch=n//HW;pos=n%HW;batch=BSTART+local_batch
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for block in range((O+BK-1)//BK):
        oc=block*BK+kk
        weights=tl.load(W+(g*O+oc[None,:])*C*9+crs[:,None],
                        (crs[:,None]<C*9)&(oc[None,:]<O),0.0)
        dy=tl.load(D+((g*B+batch[None,:])*O+oc[:,None])*HW+pos[None,:],
                   (oc[:,None]<O)&(n[None,:]<active_n),0.0)
        acc=tl.dot(weights,dy,acc,input_precision="ieee")
    tl.store(Z+(g*C*9+crs[:,None])*max_n+n[None,:],acc,
             (crs[:,None]<C*9)&(n[None,:]<active_n))


@triton.jit
def _gx_fold_batch_chunk(Z,DX,B:tl.constexpr,BSTART,
                         ACTIVE_G:tl.constexpr,ACTIVE_B:tl.constexpr,
                         MAX_B:tl.constexpr,
                         C:tl.constexpr,H:tl.constexpr,W:tl.constexpr,
                         BLOCK:tl.constexpr):
    """Fold one complete batch chunk directly to group-first dX."""
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    hw:tl.constexpr=H*W
    max_n:tl.constexpr=MAX_B*hw
    pos=i%hw;ci=i//hw%C;local_batch=i//(C*hw)%ACTIVE_B
    g=i//(ACTIVE_B*C*hw)
    iy=pos//W;ix=pos%W
    valid=i<ACTIVE_G*ACTIVE_B*C*hw
    acc=tl.full((BLOCK,),0.0,tl.float32)
    for rr in tl.static_range(3):
        for ss in tl.static_range(3):
            py=iy+1-rr;px=ix+1-ss
            zoff=((g*C+ci)*9+rr*3+ss)*max_n+local_batch*hw+py*W+px
            acc+=tl.load(Z+zoff,valid&(py>=0)&(py<H)&(px>=0)&(px<W),0.0)
    tl.store(DX+((g*B+(BSTART+local_batch))*C+ci)*hw+pos,acc,valid)


@triton.jit
def _gx_implicit_direct(D,WEIGHT,DX,B:tl.constexpr,C:tl.constexpr,
                        O:tl.constexpr,H:tl.constexpr,WI:tl.constexpr,
                        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    """Write dX directly: reduce [O,3,3] without materialized columns/fold."""
    tile=tl.program_id(0);g=tl.program_id(1)
    hw:tl.constexpr=H*WI
    nall:tl.constexpr=B*hw
    nn:tl.constexpr=(nall+BN-1)//BN
    ci=tile//nn*BM+tl.arange(0,BM)
    n=tile%nn*BN+tl.arange(0,BN)
    batch=n//hw;pos=n%hw;iy=pos//WI;ix=pos%WI
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    # k = (filter position)*O + output channel.  With O=128 and BK=16/32,
    # each dot block normally keeps one spatial offset and contiguous dY
    # channels while still reducing the complete O*3*3 dimension.
    for block in range((O*9+BK-1)//BK):
        k=block*BK+kk;rs=k//O;oc=k%O
        rr=rs//3;ss=rs%3
        weight=tl.load(
            WEIGHT+((g*O+oc[None,:])*C+ci[:,None])*9+rs[None,:],
            (ci[:,None]<C)&(k[None,:]<O*9),0)
        py=iy[None,:]+1-rr[:,None]
        px=ix[None,:]+1-ss[:,None]
        dy=tl.load(
            D+((g*B+batch[None,:])*O+oc[:,None])*hw+py*WI+px,
            (k[:,None]<O*9)&(n[None,:]<nall)&
            (py>=0)&(py<H)&(px>=0)&(px<WI),0)
        acc=tl.dot(weight,dy,acc,input_precision="ieee")
    tl.store(DX+((g*B+batch[None,:])*C+ci[:,None])*hw+pos[None,:],acc,
             (ci[:,None]<C)&(n[None,:]<nall))


@triton.jit
def _rgb_dx_direct3(D, WEIGHT, DX, B:tl.constexpr, O:tl.constexpr,
                    H:tl.constexpr, WI:tl.constexpr,
                    BLOCK_N:tl.constexpr, BLOCK_K:tl.constexpr):
    """C=3 dX without padded channel tiles or a materialized im2col buffer."""
    n=tl.program_id(0)*BLOCK_N+tl.arange(0,BLOCK_N)
    g=tl.program_id(1)
    hw:tl.constexpr=H*WI
    nall:tl.constexpr=B*hw
    batch=n//hw;pos=n%hw;iy=pos//WI;ix=pos%WI
    kk=tl.arange(0,BLOCK_K)
    a0=tl.zeros((BLOCK_N,),tl.float32)
    a1=tl.zeros((BLOCK_N,),tl.float32)
    a2=tl.zeros((BLOCK_N,),tl.float32)
    # A dY tile is shared by the three real input channels.  This avoids the
    # 16/32-channel padding paid by a conventional small-M dot product.
    for rs in tl.static_range(9):
        # Keep the static expressions inline. Triton 3.1 rejects a constexpr
        # local that is rebound by successive unrolled iterations.
        py=iy+1-(rs//3);px=ix+1-(rs%3)
        valid_pos=(n<nall)&(py>=0)&(py<H)&(px>=0)&(px<WI)
        for ob in range((O+BLOCK_K-1)//BLOCK_K):
            oc=ob*BLOCK_K+kk
            dy=tl.load(D+((g*B+batch[None,:])*O+oc[:,None])*hw+
                       py[None,:]*WI+px[None,:],
                       (oc[:,None]<O)&valid_pos[None,:],0.0)
            wbase=WEIGHT+(g*O+oc)*27+rs
            w0=tl.load(wbase+0*9,oc<O,0.0)
            w1=tl.load(wbase+1*9,oc<O,0.0)
            w2=tl.load(wbase+2*9,oc<O,0.0)
            a0+=tl.sum(dy*w0[:,None],axis=0)
            a1+=tl.sum(dy*w1[:,None],axis=0)
            a2+=tl.sum(dy*w2[:,None],axis=0)
    out=DX+(g*B+batch)*3*hw+pos
    tl.store(out+0*hw,a0,n<nall)
    tl.store(out+1*hw,a1,n<nall)
    tl.store(out+2*hw,a2,n<nall)


@triton.jit
def _rgb_dw_db_flat(X,D,DW,DB,PDW,PDB,B:tl.constexpr,O:tl.constexpr,
                    H:tl.constexpr,WI:tl.constexpr,SPLIT:tl.constexpr,
                    PARTIAL:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,
                    BK:tl.constexpr):
    """[O,K]@[K,27] dW and dB; optional contiguous split-K partials."""
    pid=tl.program_id(0);gs=tl.program_id(1)
    g=gs//SPLIT;part=gs%SPLIT
    oc=pid*BM+tl.arange(0,BM)
    crs=tl.arange(0,BN);ci=crs//9;rs=crs%9;rr=rs//3;ss=rs%3
    hw:tl.constexpr=H*WI
    total:tl.constexpr=B*hw
    chunk:tl.constexpr=((total+SPLIT-1)//SPLIT+BK-1)//BK*BK
    kk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    bias=tl.zeros((BM,),tl.float32)
    for block in range((chunk+BK-1)//BK):
        k=part*chunk+block*BK+kk
        batch=k//hw;pos=k%hw;y=pos//WI;xpos=pos%WI
        valid=k<total
        dv=tl.load(D+((g*B+batch[None,:])*O+oc[:,None])*hw+pos[None,:],
                   (oc[:,None]<O)&valid[None,:],0.0)
        iy=y[:,None]+rr[None,:]-1
        ix=xpos[:,None]+ss[None,:]-1
        xv=tl.load(X+((g*B+batch[:,None])*3+ci[None,:])*hw+iy*WI+ix,
                   valid[:,None]&(crs[None,:]<27)&
                   (iy>=0)&(iy<H)&(ix>=0)&(ix<WI),0.0)
        acc=tl.dot(dv,xv,acc,input_precision="ieee")
        bias+=tl.sum(dv,axis=1)
    mask=(oc[:,None]<O)&(crs[None,:]<27)
    if PARTIAL:
        tl.store(PDW+((g*SPLIT+part)*O+oc[:,None])*27+crs[None,:],acc,mask)
        tl.store(PDB+(g*SPLIT+part)*O+oc,bias,oc<O)
    else:
        tl.store(DW+(g*O+oc[:,None])*27+crs[None,:],acc,mask)
        tl.store(DB+g*O+oc,bias,oc<O)


@triton.jit
def _rgb_dw_db_finish(PDW,PDB,DW,DB,O:tl.constexpr,SPLIT:tl.constexpr,
                      BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);g=tl.program_id(1)
    oc=i//28;col=i%28
    acc=tl.zeros((BLOCK,),tl.float32)
    for part in range(SPLIT):
        w=tl.load(PDW+((g*SPLIT+part)*O+oc)*27+col,
                  (oc<O)&(col<27),0.0)
        b=tl.load(PDB+(g*SPLIT+part)*O+oc,(oc<O)&(col==27),0.0)
        acc+=w+b
    tl.store(DW+(g*O+oc)*27+col,acc,(oc<O)&(col<27))
    tl.store(DB+g*O+oc,acc,(oc<O)&(col==27))

def _validate(x, weight, d, stride, padding, dilation, groups):
    if x.ndim != 5 or weight.ndim != 5 or d.ndim != 5:
        raise ValueError("expected 5D group-first tensors")
    g, b, c, h, w = x.shape
    o = weight.shape[1]
    if (min(g, b, c, o) < 1 or min(h, w) < 2 or
        tuple(weight.shape) != (g, o, c, 3, 3) or
        tuple(d.shape) != (g, b, o, h, w)):
        raise ValueError("expected x=[G,B,C,H,W], weight=[G,O,C,3,3], dY=[G,B,O,H,W]; H,W>=2")
    if (_pair(stride) != (1, 1) or _pair(padding) != (1, 1) or
        _pair(dilation) != (1, 1) or operator.index(groups) not in (1, g)):
        raise ValueError("this family requires stride=padding=dilation=1; groups=1 or G")
    if (g > 65535 or max(g*b*c*h*w, g*b*o*h*w, g*o*c*9) >= 2**31):
        raise ValueError("tensor indexing or CUDA grid limit exceeded")
    if c * 9 * h * w * 4 > WORKSPACE_LIMIT_BYTES:
        raise ValueError("one group's dX workspace exceeds the 128 MiB budget")
    if any(not t.is_cuda or t.dtype != torch.float32 or t.device != x.device
           for t in (x, weight, d)):
        raise ValueError("expected FP32 tensors on the same CUDA device")
    return g, b, c, o, h, w

def grad_x_workspace(d, dx):
    """Reuse one batch's buffer across B; chunk G to keep it <=128 MiB."""
    g, _, c, h, w = dx.shape
    per_group = c * 9 * h * w * dx.element_size()
    if per_group > WORKSPACE_LIMIT_BYTES:
        raise ValueError("one group's dX workspace exceeds the 128 MiB budget")
    chunk = min(g, max(1, WORKSPACE_LIMIT_BYTES // per_group))
    return d.new_empty((chunk, c * 9, h * w))


@functools.lru_cache(maxsize=256)
def _grad_x_batch_chunk_shape(g,b,c,h,w,max_bytes):
    """Choose G/B chunk sizes that minimize launch pairs under max_bytes."""
    per_group_batch=c*9*h*w*4
    capacity=max_bytes//per_group_batch
    if capacity<1:
        raise ValueError("one group/batch dX workspace exceeds the workspace budget")
    upper=min(g,capacity)
    if upper<=1024:
        group_choices=range(1,upper+1)
    else:
        # Exact enumeration is unnecessary for extreme synthetic G.  Include
        # both endpoints and all quotient breakpoints that can alter a chunk
        # count; normal ConvNet shapes use the exact path above.
        choices={1,upper}
        for q in range(1,1025):
            choices.add(min(upper,max(1,g//q)))
            choices.add(min(upper,max(1,capacity//q)))
        group_choices=sorted(choices)
    best=None
    for gc in group_choices:
        bc=min(b,capacity//gc)
        if bc<1:
            continue
        chunks=((g+gc-1)//gc)*((b+bc-1)//bc)
        score=(chunks,-gc*bc,-bc,-gc)
        if best is None or score<best[0]:
            best=(score,gc,bc)
    _,gc,bc=best
    return gc,bc,per_group_batch*gc*bc


def grad_x_batch_chunk_plan(x,max_bytes=WORKSPACE_LIMIT_BYTES):
    g,b,c,h,w=x.shape
    gc,bc,nbytes=_grad_x_batch_chunk_shape(
        g,b,c,h,w,operator.index(max_bytes))
    return dict(groups=gc,batches=bc,bytes=nbytes,
                group_chunks=(g+gc-1)//gc,
                batch_chunks=(b+bc-1)//bc,
                kernel_launches=2*((g+gc-1)//gc)*((b+bc-1)//bc))


def grad_x_batch_chunk_workspace(x,max_bytes=WORKSPACE_LIMIT_BYTES):
    plan=grad_x_batch_chunk_plan(x,max_bytes)
    return x.new_empty((plan["groups"],x.shape[2]*9,
                        plan["batches"]*x.shape[3]*x.shape[4]))


def launch_grad_x_batch_chunked(d,weight,dx,workspace,
                                bm=32,bn=64,bk=16,warps=4):
    """Direct dX with one contraction/fold pair per G/B workspace chunk."""
    g,b,o,h,w=d.shape;c=weight.shape[2]
    gc=workspace.shape[0]
    max_b=workspace.shape[-1]//(h*w)
    if (workspace.shape[1]!=c*9 or max_b<1 or
            workspace.shape[-1]!=max_b*h*w):
        raise ValueError("invalid batch-chunked dX workspace")
    for base in range(0,g,gc):
        end=min(g,base+gc);active_g=end-base
        dd,ww,out=d[base:end],weight[base:end],dx[base:end]
        for bstart in range(0,b,max_b):
            active_b=min(max_b,b-bstart)
            active_n=active_b*h*w
            grid=(triton.cdiv(c*9,bm)*triton.cdiv(active_n,bn),active_g)
            _gx_gemm_batch_chunk[grid](
                dd,ww,workspace,b,bstart,active_b,max_b,c,o,h*w,
                bm,bn,bk,num_warps=warps,num_stages=2)
            count=active_g*active_b*c*h*w
            _gx_fold_batch_chunk[(triton.cdiv(count,128),)](
                workspace,out,b,bstart,active_g,active_b,max_b,c,h,w,128,
                num_warps=4)

def launch_gemm_one_batch(d, weight, z, batch):
    g, b, o, h, w = d.shape
    c = weight.shape[2]
    grid = (triton.cdiv(c * 9, 32) * triton.cdiv(h * w, 64), g)
    _gx_gemm[grid](d, weight, z, b, batch, c, o, h*w, 32, 64, 16,
                   num_warps=4, num_stages=2)

def launch_fold_one_batch(z, dx, batch):
    g, b, c, h, w = dx.shape
    _gx_fold[(triton.cdiv(g*c*h*w, 128),)](
        z, dx, g, b, batch, c, h, w, 128, num_warps=4)

def launch_grad_x(d, weight, dx, workspace):
    g, b = d.shape[:2]
    chunk = workspace.shape[0]
    if chunk < 1:
        raise ValueError("empty workspace")
    for base in range(0, g, chunk):
        end = min(g, base + chunk)
        dd, ww, out = d[base:end], weight[base:end], dx[base:end]
        for batch in range(b):
            launch_gemm_one_batch(dd, ww, workspace, batch)
            launch_fold_one_batch(workspace, out, batch)


def launch_grad_x_implicit(d,weight,dx,bm=32,bn=32,bk=16,warps=4):
    """One-kernel direct dX; intended for small B*H*W workloads."""
    g,b,o,h,w=d.shape;c=weight.shape[2]
    grid=(triton.cdiv(c,bm)*triton.cdiv(b*h*w,bn),g)
    _gx_implicit_direct[grid](d,weight,dx,b,c,o,h,w,bm,bn,bk,
                              num_warps=warps,num_stages=2)


def launch_rgb_grad_x_direct3(d,weight,dx,block_n=16,block_k=16,warps=4):
    """Direct C=3 dX, sharing dY across all three input channels."""
    g,b,o,h,w=d.shape
    if weight.shape[2:] != (3,3,3):
        raise ValueError("RGB dX requires weight=[G,O,3,3,3]")
    _rgb_dx_direct3[(triton.cdiv(b*h*w,block_n),g)](
        d,weight,dx,b,o,h,w,block_n,block_k,
        num_warps=warps,num_stages=2)


def rgb_dw_split_count(x,d,force_split=None,bm=32):
    """Occupancy/work rule used only inside the RGB candidate family."""
    g,b,_,h,w=x.shape;o=d.shape[2]
    if force_split is not None:
        split=operator.index(force_split)
        if split<1 or split>64:
            raise ValueError("RGB split must be in [1,64]")
        return split
    bm=operator.index(bm)
    if bm not in (32,64):
        raise ValueError("RGB dW BM must be 32 or 64")
    base_ctas=g*triton.cdiv(o,bm)
    occupancy=triton.cdiv(64,max(1,base_ctas))
    reduction=triton.cdiv(b*h*w,8192)
    return min(64,max(1,occupancy,reduction))


def launch_rgb_grad_weight_bias(x,d,dw,db,force_split=None,bm=32):
    """Compact [O,K]@[K,27] RGB dW; dB shares the same dY reads."""
    g,b,c,h,w=x.shape;o=d.shape[2]
    if c!=3 or tuple(dw.shape)!=(g,o,3,3,3):
        raise ValueError("RGB dW requires x channels=3 and dw=[G,O,3,3,3]")
    bm=operator.index(bm)
    split=rgb_dw_split_count(x,d,force_split,bm)
    grid=(triton.cdiv(o,bm),g*split)
    warps=4 if bm==32 else 8
    if split==1:
        # Dummy partial pointers are never accessed under PARTIAL=False.
        _rgb_dw_db_flat[grid](x,d,dw,db,dw,db,b,o,h,w,split,False,bm,32,16,
                              num_warps=warps,num_stages=2)
        return split
    pdw=x.new_empty((g,split,o,27))
    pdb=x.new_empty((g,split,o))
    _rgb_dw_db_flat[grid](x,d,dw,db,pdw,pdb,b,o,h,w,split,True,bm,32,16,
                          num_warps=warps,num_stages=2)
    _rgb_dw_db_finish[(triton.cdiv(o*28,256),g)](
        pdw,pdb,dw,db,o,split,256,num_warps=4)
    return split


def launch_grad_x_bmm(d,weight,dx,workspace):
    """Direct dX GEMM through bmm(out=...), followed by the existing fold."""
    g,b,o,h,w=d.shape;c=weight.shape[2];chunk=workspace.shape[0]
    for base in range(0,g,chunk):
        end=min(g,base+chunk);active=end-base
        ww=weight[base:end].reshape(active,o,c*9).transpose(1,2)
        for batch in range(b):
            dd=d[base:end,batch].reshape(active,o,h*w)
            torch.bmm(ww,dd,out=workspace[:active])
            launch_fold_one_batch(workspace,dx[base:end],batch)

def launch_grad_weight_bias_v5(x, d, dw, db):
    g, b, c, h, w = x.shape
    o = d.shape[2]
    _dw_db_row[(triton.cdiv(c, 32) * triton.cdiv(o, 32), g, 3)](
        x, d, dw, db, c, o, h, w, b, 32, 32, 16,
        num_warps=4, num_stages=2)

def launch_grad_bias(d, db):
    g, b, o, h, w = d.shape
    _db_reduce[(o, g)](d, db, b, o, h*w, 1024, num_warps=4)


@triton.jit
def _wg_pack_x(X, V, START, B: tl.constexpr, C: tl.constexpr,
               H: tl.constexpr, W: tl.constexpr, NT: tl.constexpr,
               BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.program_id(1)
    ci, n = i // NT, i % NT
    t = START + n
    th: tl.constexpr = (H + 1) // 2
    tw: tl.constexpr = (W + 1) // 2
    batch = t // (th * tw)
    y = t // tw % th * 2 - 1
    x = t % tw * 2 - 1
    base = ((g * B + batch) * C + ci) * H * W
    valid = (ci < C) & (batch < B)
    x00 = tl.load(X+base+y*W+x, valid&(y>=0)&(y<H)&(x>=0)&(x<W), 0)
    x01 = tl.load(X+base+y*W+x+1, valid&(y>=0)&(y<H)&(x+1>=0)&(x+1<W), 0)
    x02 = tl.load(X+base+y*W+x+2, valid&(y>=0)&(y<H)&(x+2>=0)&(x+2<W), 0)
    x03 = tl.load(X+base+y*W+x+3, valid&(y>=0)&(y<H)&(x+3>=0)&(x+3<W), 0)
    x10 = tl.load(X+base+(y+1)*W+x, valid&(y+1>=0)&(y+1<H)&(x>=0)&(x<W), 0)
    x11 = tl.load(X+base+(y+1)*W+x+1, valid&(y+1>=0)&(y+1<H)&(x+1>=0)&(x+1<W), 0)
    x12 = tl.load(X+base+(y+1)*W+x+2, valid&(y+1>=0)&(y+1<H)&(x+2>=0)&(x+2<W), 0)
    x13 = tl.load(X+base+(y+1)*W+x+3, valid&(y+1>=0)&(y+1<H)&(x+3>=0)&(x+3<W), 0)
    x20 = tl.load(X+base+(y+2)*W+x, valid&(y+2>=0)&(y+2<H)&(x>=0)&(x<W), 0)
    x21 = tl.load(X+base+(y+2)*W+x+1, valid&(y+2>=0)&(y+2<H)&(x+1>=0)&(x+1<W), 0)
    x22 = tl.load(X+base+(y+2)*W+x+2, valid&(y+2>=0)&(y+2<H)&(x+2>=0)&(x+2<W), 0)
    x23 = tl.load(X+base+(y+2)*W+x+3, valid&(y+2>=0)&(y+2<H)&(x+3>=0)&(x+3<W), 0)
    x30 = tl.load(X+base+(y+3)*W+x, valid&(y+3>=0)&(y+3<H)&(x>=0)&(x<W), 0)
    x31 = tl.load(X+base+(y+3)*W+x+1, valid&(y+3>=0)&(y+3<H)&(x+1>=0)&(x+1<W), 0)
    x32 = tl.load(X+base+(y+3)*W+x+2, valid&(y+3>=0)&(y+3<H)&(x+2>=0)&(x+2<W), 0)
    x33 = tl.load(X+base+(y+3)*W+x+3, valid&(y+3>=0)&(y+3<H)&(x+3>=0)&(x+3<W), 0)
    a00, a01, a02, a03 = x00-x20, x01-x21, x02-x22, x03-x23
    a10, a11, a12, a13 = x10+x20, x11+x21, x12+x22, x13+x23
    a20, a21, a22, a23 = x20-x10, x21-x11, x22-x12, x23-x13
    a30, a31, a32, a33 = x30-x10, x31-x11, x32-x12, x33-x13
    dst = V + g*16*C*NT + ci*NT + n
    step: tl.constexpr = C*NT
    # Fill padding with zero, including a partial final tile chunk.
    mask = ci < C
    tl.store(dst+0*step, a00-a02, mask)
    tl.store(dst+1*step, a01+a02, mask)
    tl.store(dst+2*step, a02-a01, mask)
    tl.store(dst+3*step, a03-a01, mask)
    tl.store(dst+4*step, a10-a12, mask)
    tl.store(dst+5*step, a11+a12, mask)
    tl.store(dst+6*step, a12-a11, mask)
    tl.store(dst+7*step, a13-a11, mask)
    tl.store(dst+8*step, a20-a22, mask)
    tl.store(dst+9*step, a21+a22, mask)
    tl.store(dst+10*step, a22-a21, mask)
    tl.store(dst+11*step, a23-a21, mask)
    tl.store(dst+12*step, a30-a32, mask)
    tl.store(dst+13*step, a31+a32, mask)
    tl.store(dst+14*step, a32-a31, mask)
    tl.store(dst+15*step, a33-a31, mask)


@triton.jit
def _wg_pack_d(D, E, START, B: tl.constexpr, O: tl.constexpr,
               H: tl.constexpr, W: tl.constexpr, NT: tl.constexpr,
               BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.program_id(1)
    oc, n = i // NT, i % NT
    t = START + n
    th: tl.constexpr = (H + 1) // 2
    tw: tl.constexpr = (W + 1) // 2
    batch = t // (th * tw)
    y = t // tw % th * 2
    x = t % tw * 2
    base = ((g * B + batch) * O + oc) * H * W + y*W+x
    valid = (oc < O) & (batch < B)
    a = tl.load(D+base, valid&(y<H)&(x<W), 0)
    b = tl.load(D+base+1, valid&(y<H)&(x+1<W), 0)
    c = tl.load(D+base+W, valid&(y+1<H)&(x<W), 0)
    d = tl.load(D+base+W+1, valid&(y+1<H)&(x+1<W), 0)
    dst = E + g*16*O*NT + oc*NT + n
    step: tl.constexpr = O*NT
    mask = oc < O
    tl.store(dst+0*step, a, mask)
    tl.store(dst+1*step, a+b, mask)
    tl.store(dst+2*step, a-b, mask)
    tl.store(dst+3*step, b, mask)
    tl.store(dst+4*step, a+c, mask)
    tl.store(dst+5*step, (a+c)+(b+d), mask)
    tl.store(dst+6*step, (a+c)-(b+d), mask)
    tl.store(dst+7*step, b+d, mask)
    tl.store(dst+8*step, a-c, mask)
    tl.store(dst+9*step, (a-c)+(b-d), mask)
    tl.store(dst+10*step, (a-c)-(b-d), mask)
    tl.store(dst+11*step, b-d, mask)
    tl.store(dst+12*step, c, mask)
    tl.store(dst+13*step, c+d, mask)
    tl.store(dst+14*step, c-d, mask)
    tl.store(dst+15*step, d, mask)


@triton.jit
def _wg_gemm_v7(V, E, U, C: tl.constexpr, O: tl.constexpr, NT: tl.constexpr,
             ADD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    g = tl.program_id(1)
    z = tl.program_id(2)
    nc: tl.constexpr = (C + BN - 1) // BN
    oc = tile // nc * BM + tl.arange(0, BM)
    ci = tile % nc * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range((NT + BK - 1) // BK):
        k = block * BK + kk
        a = tl.load(E+((g*16+z)*O+oc[:,None])*NT+k[None,:],
                    (oc[:,None]<O)&(k[None,:]<NT), 0)
        b = tl.load(V+((g*16+z)*C+ci[None,:])*NT+k[:,None],
                    (ci[None,:]<C)&(k[:,None]<NT), 0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    dst = U + ((g*16+z)*O+oc[:,None])*C+ci[None,:]
    valid = (oc[:,None]<O)&(ci[None,:]<C)
    if ADD:
        acc += tl.load(dst, valid, 0)
    tl.store(dst, acc, valid)


@triton.jit
def _wg_gemm_v8(V, E, U, C: tl.constexpr, O: tl.constexpr, NT: tl.constexpr,
                ADD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # The same packed operands as v7. Block pointers describe their exact
    # contiguous axes, and advance by a K tile rather than rebuilding a
    # full pointer tensor on each iteration. FP32 SIMT, no TF32 conversion.
    tile = tl.program_id(0)
    g = tl.program_id(1)
    z = tl.program_id(2)
    nc: tl.constexpr = (C + BN - 1) // BN
    om, cn = tile // nc * BM, tile % nc * BN
    a_ptr = tl.make_block_ptr(
        base=E+(g*16+z)*O*NT, shape=(O,NT), strides=(NT,1),
        offsets=(om,0), block_shape=(BM,BK), order=(1,0))
    b_ptr = tl.make_block_ptr(
        base=V+(g*16+z)*C*NT, shape=(NT,C), strides=(1,NT),
        offsets=(0,cn), block_shape=(BK,BN), order=(0,1))
    acc = tl.zeros((BM,BN),tl.float32)
    for block in range((NT+BK-1)//BK):
        a = tl.load(a_ptr,boundary_check=(0,1),padding_option="zero")
        b = tl.load(b_ptr,boundary_check=(0,1),padding_option="zero")
        acc = tl.dot(a,b,acc,input_precision="ieee")
        a_ptr = tl.advance(a_ptr,(0,BK))
        b_ptr = tl.advance(b_ptr,(BK,0))
    oc = om+tl.arange(0,BM)
    ci = cn+tl.arange(0,BN)
    dst = U+((g*16+z)*O+oc[:,None])*C+ci[None,:]
    valid = (oc[:,None]<O)&(ci[None,:]<C)
    if ADD:
        acc += tl.load(dst,valid,0)
    tl.store(dst,acc,valid)


@triton.jit
def _wg_inverse(U, DWOUT, G: tl.constexpr, C: tl.constexpr,
                O: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    g, j = i//(O*C), i%(O*C)
    valid = i < G*O*C
    base = U + g*16*O*C+j
    step: tl.constexpr = O*C
    u0=tl.load(base+0*step,valid,0)
    u1=tl.load(base+1*step,valid,0)
    u2=tl.load(base+2*step,valid,0)
    u3=tl.load(base+3*step,valid,0)
    u4=tl.load(base+4*step,valid,0)
    u5=tl.load(base+5*step,valid,0)
    u6=tl.load(base+6*step,valid,0)
    u7=tl.load(base+7*step,valid,0)
    u8=tl.load(base+8*step,valid,0)
    u9=tl.load(base+9*step,valid,0)
    u10=tl.load(base+10*step,valid,0)
    u11=tl.load(base+11*step,valid,0)
    u12=tl.load(base+12*step,valid,0)
    u13=tl.load(base+13*step,valid,0)
    u14=tl.load(base+14*step,valid,0)
    u15=tl.load(base+15*step,valid,0)
    a0=u0+0.5*(u4+u8)
    a1=u1+0.5*(u5+u9)
    a2=u2+0.5*(u6+u10)
    a3=u3+0.5*(u7+u11)
    b0=0.5*(u4-u8)
    b1=0.5*(u5-u9)
    b2=0.5*(u6-u10)
    b3=0.5*(u7-u11)
    c0=0.5*(u4+u8)+u12
    c1=0.5*(u5+u9)+u13
    c2=0.5*(u6+u10)+u14
    c3=0.5*(u7+u11)+u15
    tl.store(DWOUT+i*9+0,a0+0.5*(a1+a2),valid)
    tl.store(DWOUT+i*9+1,0.5*(a1-a2),valid)
    tl.store(DWOUT+i*9+2,0.5*(a1+a2)+a3,valid)
    tl.store(DWOUT+i*9+3,b0+0.5*(b1+b2),valid)
    tl.store(DWOUT+i*9+4,0.5*(b1-b2),valid)
    tl.store(DWOUT+i*9+5,0.5*(b1+b2)+b3,valid)
    tl.store(DWOUT+i*9+6,c0+0.5*(c1+c2),valid)
    tl.store(DWOUT+i*9+7,0.5*(c1-c2),valid)
    tl.store(DWOUT+i*9+8,0.5*(c1+c2)+c3,valid)


# Winograd F(4x4,3x3) weight-gradient path.  For one spatial tile,
# V = B^T x B, E = A dY A^T, dU = E @ V^T, and dW = G^T dU G.
# The matrices are the standard real-valued minimal-filtering transforms:
#
# B^T = [[4,0,-5,0,1,0], [0,-4,-4,1,1,0],
#        [0,4,-4,-1,1,0], [0,-2,-1,2,1,0],
#        [0,2,-1,-2,1,0], [0,4,0,-5,0,1]]
# A^T = [[1,1,1,1,1,0], [0,1,-1,2,-2,0],
#        [0,1,1,4,4,0], [0,1,-1,8,-8,1]]
# G   = [[1/4,0,0], [-1/6,-1/6,-1/6], [-1/6,1/6,-1/6],
#        [1/24,1/12,1/6], [1/24,-1/12,1/6], [0,0,1]]

@triton.jit
def _wg_f4_pack_x(X,V,E,START,B:tl.constexpr,C:tl.constexpr,
                  H:tl.constexpr,W:tl.constexpr,NT:tl.constexpr,
                  WRITE_E:tl.constexpr,BLOCK:tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.program_id(1)
    ci, n = i // NT, i % NT
    th: tl.constexpr = (H + 3) // 4
    tw: tl.constexpr = (W + 3) // 4
    t = START + n
    batch = t // (th * tw)
    y = t // tw % th * 4 - 1
    x = t % tw * 4 - 1
    base = ((g * B + batch) * C + ci) * H * W + y * W + x
    valid = (ci < C) & (batch < B)

    d00=tl.load(X+base+0*W+0,valid&(y+0>=0)&(y+0<H)&(x+0>=0)&(x+0<W),0)
    d01=tl.load(X+base+0*W+1,valid&(y+0>=0)&(y+0<H)&(x+1>=0)&(x+1<W),0)
    d02=tl.load(X+base+0*W+2,valid&(y+0>=0)&(y+0<H)&(x+2>=0)&(x+2<W),0)
    d03=tl.load(X+base+0*W+3,valid&(y+0>=0)&(y+0<H)&(x+3>=0)&(x+3<W),0)
    d04=tl.load(X+base+0*W+4,valid&(y+0>=0)&(y+0<H)&(x+4>=0)&(x+4<W),0)
    d05=tl.load(X+base+0*W+5,valid&(y+0>=0)&(y+0<H)&(x+5>=0)&(x+5<W),0)
    d10=tl.load(X+base+1*W+0,valid&(y+1>=0)&(y+1<H)&(x+0>=0)&(x+0<W),0)
    d11=tl.load(X+base+1*W+1,valid&(y+1>=0)&(y+1<H)&(x+1>=0)&(x+1<W),0)
    d12=tl.load(X+base+1*W+2,valid&(y+1>=0)&(y+1<H)&(x+2>=0)&(x+2<W),0)
    d13=tl.load(X+base+1*W+3,valid&(y+1>=0)&(y+1<H)&(x+3>=0)&(x+3<W),0)
    d14=tl.load(X+base+1*W+4,valid&(y+1>=0)&(y+1<H)&(x+4>=0)&(x+4<W),0)
    d15=tl.load(X+base+1*W+5,valid&(y+1>=0)&(y+1<H)&(x+5>=0)&(x+5<W),0)
    d20=tl.load(X+base+2*W+0,valid&(y+2>=0)&(y+2<H)&(x+0>=0)&(x+0<W),0)
    d21=tl.load(X+base+2*W+1,valid&(y+2>=0)&(y+2<H)&(x+1>=0)&(x+1<W),0)
    d22=tl.load(X+base+2*W+2,valid&(y+2>=0)&(y+2<H)&(x+2>=0)&(x+2<W),0)
    d23=tl.load(X+base+2*W+3,valid&(y+2>=0)&(y+2<H)&(x+3>=0)&(x+3<W),0)
    d24=tl.load(X+base+2*W+4,valid&(y+2>=0)&(y+2<H)&(x+4>=0)&(x+4<W),0)
    d25=tl.load(X+base+2*W+5,valid&(y+2>=0)&(y+2<H)&(x+5>=0)&(x+5<W),0)
    d30=tl.load(X+base+3*W+0,valid&(y+3>=0)&(y+3<H)&(x+0>=0)&(x+0<W),0)
    d31=tl.load(X+base+3*W+1,valid&(y+3>=0)&(y+3<H)&(x+1>=0)&(x+1<W),0)
    d32=tl.load(X+base+3*W+2,valid&(y+3>=0)&(y+3<H)&(x+2>=0)&(x+2<W),0)
    d33=tl.load(X+base+3*W+3,valid&(y+3>=0)&(y+3<H)&(x+3>=0)&(x+3<W),0)
    d34=tl.load(X+base+3*W+4,valid&(y+3>=0)&(y+3<H)&(x+4>=0)&(x+4<W),0)
    d35=tl.load(X+base+3*W+5,valid&(y+3>=0)&(y+3<H)&(x+5>=0)&(x+5<W),0)
    d40=tl.load(X+base+4*W+0,valid&(y+4>=0)&(y+4<H)&(x+0>=0)&(x+0<W),0)
    d41=tl.load(X+base+4*W+1,valid&(y+4>=0)&(y+4<H)&(x+1>=0)&(x+1<W),0)
    d42=tl.load(X+base+4*W+2,valid&(y+4>=0)&(y+4<H)&(x+2>=0)&(x+2<W),0)
    d43=tl.load(X+base+4*W+3,valid&(y+4>=0)&(y+4<H)&(x+3>=0)&(x+3<W),0)
    d44=tl.load(X+base+4*W+4,valid&(y+4>=0)&(y+4<H)&(x+4>=0)&(x+4<W),0)
    d45=tl.load(X+base+4*W+5,valid&(y+4>=0)&(y+4<H)&(x+5>=0)&(x+5<W),0)
    d50=tl.load(X+base+5*W+0,valid&(y+5>=0)&(y+5<H)&(x+0>=0)&(x+0<W),0)
    d51=tl.load(X+base+5*W+1,valid&(y+5>=0)&(y+5<H)&(x+1>=0)&(x+1<W),0)
    d52=tl.load(X+base+5*W+2,valid&(y+5>=0)&(y+5<H)&(x+2>=0)&(x+2<W),0)
    d53=tl.load(X+base+5*W+3,valid&(y+5>=0)&(y+5<H)&(x+3>=0)&(x+3<W),0)
    d54=tl.load(X+base+5*W+4,valid&(y+5>=0)&(y+5<H)&(x+4>=0)&(x+4<W),0)
    d55=tl.load(X+base+5*W+5,valid&(y+5>=0)&(y+5<H)&(x+5>=0)&(x+5<W),0)

    # Left multiply by B^T, one column at a time.
    t00=4*d00-5*d20+d40; t01=4*d01-5*d21+d41; t02=4*d02-5*d22+d42
    t03=4*d03-5*d23+d43; t04=4*d04-5*d24+d44; t05=4*d05-5*d25+d45
    t10=-4*d10-4*d20+d30+d40; t11=-4*d11-4*d21+d31+d41
    t12=-4*d12-4*d22+d32+d42; t13=-4*d13-4*d23+d33+d43
    t14=-4*d14-4*d24+d34+d44; t15=-4*d15-4*d25+d35+d45
    t20=4*d10-4*d20-d30+d40; t21=4*d11-4*d21-d31+d41
    t22=4*d12-4*d22-d32+d42; t23=4*d13-4*d23-d33+d43
    t24=4*d14-4*d24-d34+d44; t25=4*d15-4*d25-d35+d45
    t30=-2*d10-d20+2*d30+d40; t31=-2*d11-d21+2*d31+d41
    t32=-2*d12-d22+2*d32+d42; t33=-2*d13-d23+2*d33+d43
    t34=-2*d14-d24+2*d34+d44; t35=-2*d15-d25+2*d35+d45
    t40=2*d10-d20-2*d30+d40; t41=2*d11-d21-2*d31+d41
    t42=2*d12-d22-2*d32+d42; t43=2*d13-d23-2*d33+d43
    t44=2*d14-d24-2*d34+d44; t45=2*d15-d25-2*d35+d45
    t50=4*d10-5*d30+d50; t51=4*d11-5*d31+d51; t52=4*d12-5*d32+d52
    t53=4*d13-5*d33+d53; t54=4*d14-5*d34+d54; t55=4*d15-5*d35+d55

    dst=V+g*36*C*NT+ci*NT+n
    step: tl.constexpr=C*NT
    mask=ci<C
    # Right multiply by B, equivalently apply B^T to every transformed row.
    tl.store(dst+ 0*step,4*t00-5*t02+t04,mask); tl.store(dst+ 1*step,-4*t01-4*t02+t03+t04,mask)
    tl.store(dst+ 2*step,4*t01-4*t02-t03+t04,mask); tl.store(dst+ 3*step,-2*t01-t02+2*t03+t04,mask)
    tl.store(dst+ 4*step,2*t01-t02-2*t03+t04,mask); tl.store(dst+ 5*step,4*t01-5*t03+t05,mask)
    tl.store(dst+ 6*step,4*t10-5*t12+t14,mask); tl.store(dst+ 7*step,-4*t11-4*t12+t13+t14,mask)
    tl.store(dst+ 8*step,4*t11-4*t12-t13+t14,mask); tl.store(dst+ 9*step,-2*t11-t12+2*t13+t14,mask)
    tl.store(dst+10*step,2*t11-t12-2*t13+t14,mask); tl.store(dst+11*step,4*t11-5*t13+t15,mask)
    tl.store(dst+12*step,4*t20-5*t22+t24,mask); tl.store(dst+13*step,-4*t21-4*t22+t23+t24,mask)
    tl.store(dst+14*step,4*t21-4*t22-t23+t24,mask); tl.store(dst+15*step,-2*t21-t22+2*t23+t24,mask)
    tl.store(dst+16*step,2*t21-t22-2*t23+t24,mask); tl.store(dst+17*step,4*t21-5*t23+t25,mask)
    tl.store(dst+18*step,4*t30-5*t32+t34,mask); tl.store(dst+19*step,-4*t31-4*t32+t33+t34,mask)
    tl.store(dst+20*step,4*t31-4*t32-t33+t34,mask); tl.store(dst+21*step,-2*t31-t32+2*t33+t34,mask)
    tl.store(dst+22*step,2*t31-t32-2*t33+t34,mask); tl.store(dst+23*step,4*t31-5*t33+t35,mask)
    tl.store(dst+24*step,4*t40-5*t42+t44,mask); tl.store(dst+25*step,-4*t41-4*t42+t43+t44,mask)
    tl.store(dst+26*step,4*t41-4*t42-t43+t44,mask); tl.store(dst+27*step,-2*t41-t42+2*t43+t44,mask)
    tl.store(dst+28*step,2*t41-t42-2*t43+t44,mask); tl.store(dst+29*step,4*t41-5*t43+t45,mask)
    tl.store(dst+30*step,4*t50-5*t52+t54,mask); tl.store(dst+31*step,-4*t51-4*t52+t53+t54,mask)
    tl.store(dst+32*step,4*t51-4*t52-t53+t54,mask); tl.store(dst+33*step,-2*t51-t52+2*t53+t54,mask)
    tl.store(dst+34*step,2*t51-t52-2*t53+t54,mask); tl.store(dst+35*step,4*t51-5*t53+t55,mask)
    if WRITE_E:
        # The central 4x4 dY region is exactly the tile used by A dY A^T.
        # Produce that transform from the already loaded 6x6 tile.
        p00=d11;p01=d12;p02=d13;p03=d14
        p10=d11+d21+d31+d41;p11=d12+d22+d32+d42;p12=d13+d23+d33+d43;p13=d14+d24+d34+d44
        p20=d11-d21+d31-d41;p21=d12-d22+d32-d42;p22=d13-d23+d33-d43;p23=d14-d24+d34-d44
        p30=d11+2*d21+4*d31+8*d41;p31=d12+2*d22+4*d32+8*d42;p32=d13+2*d23+4*d33+8*d43;p33=d14+2*d24+4*d34+8*d44
        p40=d11-2*d21+4*d31-8*d41;p41=d12-2*d22+4*d32-8*d42;p42=d13-2*d23+4*d33-8*d43;p43=d14-2*d24+4*d34-8*d44
        p50=d41;p51=d42;p52=d43;p53=d44
        edst=E+g*36*C*NT+ci*NT+n
        tl.store(edst+ 0*step,p00,mask);tl.store(edst+ 1*step,p00+p01+p02+p03,mask);tl.store(edst+ 2*step,p00-p01+p02-p03,mask)
        tl.store(edst+ 3*step,p00+2*p01+4*p02+8*p03,mask);tl.store(edst+ 4*step,p00-2*p01+4*p02-8*p03,mask);tl.store(edst+ 5*step,p03,mask)
        tl.store(edst+ 6*step,p10,mask);tl.store(edst+ 7*step,p10+p11+p12+p13,mask);tl.store(edst+ 8*step,p10-p11+p12-p13,mask)
        tl.store(edst+ 9*step,p10+2*p11+4*p12+8*p13,mask);tl.store(edst+10*step,p10-2*p11+4*p12-8*p13,mask);tl.store(edst+11*step,p13,mask)
        tl.store(edst+12*step,p20,mask);tl.store(edst+13*step,p20+p21+p22+p23,mask);tl.store(edst+14*step,p20-p21+p22-p23,mask)
        tl.store(edst+15*step,p20+2*p21+4*p22+8*p23,mask);tl.store(edst+16*step,p20-2*p21+4*p22-8*p23,mask);tl.store(edst+17*step,p23,mask)
        tl.store(edst+18*step,p30,mask);tl.store(edst+19*step,p30+p31+p32+p33,mask);tl.store(edst+20*step,p30-p31+p32-p33,mask)
        tl.store(edst+21*step,p30+2*p31+4*p32+8*p33,mask);tl.store(edst+22*step,p30-2*p31+4*p32-8*p33,mask);tl.store(edst+23*step,p33,mask)
        tl.store(edst+24*step,p40,mask);tl.store(edst+25*step,p40+p41+p42+p43,mask);tl.store(edst+26*step,p40-p41+p42-p43,mask)
        tl.store(edst+27*step,p40+2*p41+4*p42+8*p43,mask);tl.store(edst+28*step,p40-2*p41+4*p42-8*p43,mask);tl.store(edst+29*step,p43,mask)
        tl.store(edst+30*step,p50,mask);tl.store(edst+31*step,p50+p51+p52+p53,mask);tl.store(edst+32*step,p50-p51+p52-p53,mask)
        tl.store(edst+33*step,p50+2*p51+4*p52+8*p53,mask);tl.store(edst+34*step,p50-2*p51+4*p52-8*p53,mask);tl.store(edst+35*step,p53,mask)


@triton.jit
def _wg_f4_pack_d(D,E,START,B:tl.constexpr,O:tl.constexpr,
                  H:tl.constexpr,W:tl.constexpr,NT:tl.constexpr,
                  BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);g=tl.program_id(1)
    oc,n=i//NT,i%NT
    th:tl.constexpr=(H+3)//4;tw:tl.constexpr=(W+3)//4
    t=START+n;batch=t//(th*tw);y=t//tw%th*4;x=t%tw*4
    base=((g*B+batch)*O+oc)*H*W+y*W+x
    valid=(oc<O)&(batch<B)
    d00=tl.load(D+base+0*W+0,valid&(y+0<H)&(x+0<W),0);d01=tl.load(D+base+0*W+1,valid&(y+0<H)&(x+1<W),0)
    d02=tl.load(D+base+0*W+2,valid&(y+0<H)&(x+2<W),0);d03=tl.load(D+base+0*W+3,valid&(y+0<H)&(x+3<W),0)
    d10=tl.load(D+base+1*W+0,valid&(y+1<H)&(x+0<W),0);d11=tl.load(D+base+1*W+1,valid&(y+1<H)&(x+1<W),0)
    d12=tl.load(D+base+1*W+2,valid&(y+1<H)&(x+2<W),0);d13=tl.load(D+base+1*W+3,valid&(y+1<H)&(x+3<W),0)
    d20=tl.load(D+base+2*W+0,valid&(y+2<H)&(x+0<W),0);d21=tl.load(D+base+2*W+1,valid&(y+2<H)&(x+1<W),0)
    d22=tl.load(D+base+2*W+2,valid&(y+2<H)&(x+2<W),0);d23=tl.load(D+base+2*W+3,valid&(y+2<H)&(x+3<W),0)
    d30=tl.load(D+base+3*W+0,valid&(y+3<H)&(x+0<W),0);d31=tl.load(D+base+3*W+1,valid&(y+3<H)&(x+1<W),0)
    d32=tl.load(D+base+3*W+2,valid&(y+3<H)&(x+2<W),0);d33=tl.load(D+base+3*W+3,valid&(y+3<H)&(x+3<W),0)
    t00=d00;t01=d01;t02=d02;t03=d03
    t10=d00+d10+d20+d30;t11=d01+d11+d21+d31;t12=d02+d12+d22+d32;t13=d03+d13+d23+d33
    t20=d00-d10+d20-d30;t21=d01-d11+d21-d31;t22=d02-d12+d22-d32;t23=d03-d13+d23-d33
    t30=d00+2*d10+4*d20+8*d30;t31=d01+2*d11+4*d21+8*d31;t32=d02+2*d12+4*d22+8*d32;t33=d03+2*d13+4*d23+8*d33
    t40=d00-2*d10+4*d20-8*d30;t41=d01-2*d11+4*d21-8*d31;t42=d02-2*d12+4*d22-8*d32;t43=d03-2*d13+4*d23-8*d33
    t50=d30;t51=d31;t52=d32;t53=d33
    dst=E+g*36*O*NT+oc*NT+n;step:tl.constexpr=O*NT;mask=oc<O
    tl.store(dst+ 0*step,t00,mask);tl.store(dst+ 1*step,t00+t01+t02+t03,mask);tl.store(dst+ 2*step,t00-t01+t02-t03,mask)
    tl.store(dst+ 3*step,t00+2*t01+4*t02+8*t03,mask);tl.store(dst+ 4*step,t00-2*t01+4*t02-8*t03,mask);tl.store(dst+ 5*step,t03,mask)
    tl.store(dst+ 6*step,t10,mask);tl.store(dst+ 7*step,t10+t11+t12+t13,mask);tl.store(dst+ 8*step,t10-t11+t12-t13,mask)
    tl.store(dst+ 9*step,t10+2*t11+4*t12+8*t13,mask);tl.store(dst+10*step,t10-2*t11+4*t12-8*t13,mask);tl.store(dst+11*step,t13,mask)
    tl.store(dst+12*step,t20,mask);tl.store(dst+13*step,t20+t21+t22+t23,mask);tl.store(dst+14*step,t20-t21+t22-t23,mask)
    tl.store(dst+15*step,t20+2*t21+4*t22+8*t23,mask);tl.store(dst+16*step,t20-2*t21+4*t22-8*t23,mask);tl.store(dst+17*step,t23,mask)
    tl.store(dst+18*step,t30,mask);tl.store(dst+19*step,t30+t31+t32+t33,mask);tl.store(dst+20*step,t30-t31+t32-t33,mask)
    tl.store(dst+21*step,t30+2*t31+4*t32+8*t33,mask);tl.store(dst+22*step,t30-2*t31+4*t32-8*t33,mask);tl.store(dst+23*step,t33,mask)
    tl.store(dst+24*step,t40,mask);tl.store(dst+25*step,t40+t41+t42+t43,mask);tl.store(dst+26*step,t40-t41+t42-t43,mask)
    tl.store(dst+27*step,t40+2*t41+4*t42+8*t43,mask);tl.store(dst+28*step,t40-2*t41+4*t42-8*t43,mask);tl.store(dst+29*step,t43,mask)
    tl.store(dst+30*step,t50,mask);tl.store(dst+31*step,t50+t51+t52+t53,mask);tl.store(dst+32*step,t50-t51+t52-t53,mask)
    tl.store(dst+33*step,t50+2*t51+4*t52+8*t53,mask);tl.store(dst+34*step,t50-2*t51+4*t52-8*t53,mask);tl.store(dst+35*step,t53,mask)


@triton.jit
def _wg_f4_joint_pack_d(D,V,E,START,B:tl.constexpr,O:tl.constexpr,
                        H:tl.constexpr,W:tl.constexpr,NT:tl.constexpr,
                        BLOCK:tl.constexpr):
    # One set of 6x6 loads emits both B^T dY B and A dY A^T.
    _wg_f4_pack_x(D,V,E,START,B,O,H,W,NT,True,BLOCK)


@triton.jit
def _wg_f4_pack_weight_dx(W,U,G:tl.constexpr,C:tl.constexpr,O:tl.constexpr,
                          BLOCK:tl.constexpr):
    """U[z,c,o] = G * rot180(W[o,c]) * G^T for forward-form dX."""
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);g=tl.program_id(1)
    ci=i//O;oc=i%O;valid=ci<C
    base=W+(g*O+oc)*C*9+ci*9
    # conv_transpose(dY,W) == conv(dY, rot180(W)^T), padding one.
    w00=tl.load(base+8,valid,0);w01=tl.load(base+7,valid,0);w02=tl.load(base+6,valid,0)
    w10=tl.load(base+5,valid,0);w11=tl.load(base+4,valid,0);w12=tl.load(base+3,valid,0)
    w20=tl.load(base+2,valid,0);w21=tl.load(base+1,valid,0);w22=tl.load(base+0,valid,0)
    # Left multiply by G.
    t00=.25*w00;t01=.25*w01;t02=.25*w02
    t10=-(w00+w10+w20)/6;t11=-(w01+w11+w21)/6;t12=-(w02+w12+w22)/6
    t20=(-w00+w10-w20)/6;t21=(-w01+w11-w21)/6;t22=(-w02+w12-w22)/6
    t30=w00/24+w10/12+w20/6;t31=w01/24+w11/12+w21/6;t32=w02/24+w12/12+w22/6
    t40=w00/24-w10/12+w20/6;t41=w01/24-w11/12+w21/6;t42=w02/24-w12/12+w22/6
    t50=w20;t51=w21;t52=w22
    dst=U+g*36*C*O+ci*O+oc;step:tl.constexpr=C*O
    # Right multiply by G^T.
    tl.store(dst+ 0*step,.25*t00,valid);tl.store(dst+ 1*step,-(t00+t01+t02)/6,valid)
    tl.store(dst+ 2*step,(-t00+t01-t02)/6,valid);tl.store(dst+ 3*step,t00/24+t01/12+t02/6,valid)
    tl.store(dst+ 4*step,t00/24-t01/12+t02/6,valid);tl.store(dst+ 5*step,t02,valid)
    tl.store(dst+ 6*step,.25*t10,valid);tl.store(dst+ 7*step,-(t10+t11+t12)/6,valid)
    tl.store(dst+ 8*step,(-t10+t11-t12)/6,valid);tl.store(dst+ 9*step,t10/24+t11/12+t12/6,valid)
    tl.store(dst+10*step,t10/24-t11/12+t12/6,valid);tl.store(dst+11*step,t12,valid)
    tl.store(dst+12*step,.25*t20,valid);tl.store(dst+13*step,-(t20+t21+t22)/6,valid)
    tl.store(dst+14*step,(-t20+t21-t22)/6,valid);tl.store(dst+15*step,t20/24+t21/12+t22/6,valid)
    tl.store(dst+16*step,t20/24-t21/12+t22/6,valid);tl.store(dst+17*step,t22,valid)
    tl.store(dst+18*step,.25*t30,valid);tl.store(dst+19*step,-(t30+t31+t32)/6,valid)
    tl.store(dst+20*step,(-t30+t31-t32)/6,valid);tl.store(dst+21*step,t30/24+t31/12+t32/6,valid)
    tl.store(dst+22*step,t30/24-t31/12+t32/6,valid);tl.store(dst+23*step,t32,valid)
    tl.store(dst+24*step,.25*t40,valid);tl.store(dst+25*step,-(t40+t41+t42)/6,valid)
    tl.store(dst+26*step,(-t40+t41-t42)/6,valid);tl.store(dst+27*step,t40/24+t41/12+t42/6,valid)
    tl.store(dst+28*step,t40/24-t41/12+t42/6,valid);tl.store(dst+29*step,t42,valid)
    tl.store(dst+30*step,.25*t50,valid);tl.store(dst+31*step,-(t50+t51+t52)/6,valid)
    tl.store(dst+32*step,(-t50+t51-t52)/6,valid);tl.store(dst+33*step,t50/24+t51/12+t52/6,valid)
    tl.store(dst+34*step,t50/24-t51/12+t52/6,valid);tl.store(dst+35*step,t52,valid)


@triton.jit
def _wg_f4_dx_gemm(U,V,M,C:tl.constexpr,O:tl.constexpr,NT:tl.constexpr,
                   BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    """M[z,c,tile] = U[z,c,o] @ V[z,o,tile]."""
    tile=tl.program_id(0);g=tl.program_id(1);z=tl.program_id(2)
    nn:tl.constexpr=(NT+BN-1)//BN
    ci=tile//nn*BM+tl.arange(0,BM);n=tile%nn*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for block in range((O+BK-1)//BK):
        oc=block*BK+kk
        a=tl.load(U+((g*36+z)*C+ci[:,None])*O+oc[None,:],
                  (ci[:,None]<C)&(oc[None,:]<O),0)
        b=tl.load(V+((g*36+z)*O+oc[:,None])*NT+n[None,:],
                  (oc[:,None]<O)&(n[None,:]<NT),0)
        acc=tl.dot(a,b,acc,input_precision="ieee")
    tl.store(M+((g*36+z)*C+ci[:,None])*NT+n[None,:],acc,
             (ci[:,None]<C)&(n[None,:]<NT))


@triton.jit
def _wg_f4_dx_output(M,DX,START,G:tl.constexpr,B:tl.constexpr,
                     C:tl.constexpr,H:tl.constexpr,W:tl.constexpr,
                     NT:tl.constexpr,BLOCK:tl.constexpr):
    """A^T M A; 4x4 tiles do not overlap, so stores need no atomics."""
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);g=tl.program_id(1)
    ci,n=i//NT,i%NT;t=START+n
    th:tl.constexpr=(H+3)//4;tw:tl.constexpr=(W+3)//4
    batch=t//(th*tw);ty=t//tw%th;tx=t%tw
    valid=(ci<C)&(batch<B)
    base=M+g*36*C*NT+ci*NT+n;step:tl.constexpr=C*NT
    m00=tl.load(base+ 0*step,ci<C,0);m01=tl.load(base+ 1*step,ci<C,0);m02=tl.load(base+ 2*step,ci<C,0);m03=tl.load(base+ 3*step,ci<C,0);m04=tl.load(base+ 4*step,ci<C,0);m05=tl.load(base+ 5*step,ci<C,0)
    m10=tl.load(base+ 6*step,ci<C,0);m11=tl.load(base+ 7*step,ci<C,0);m12=tl.load(base+ 8*step,ci<C,0);m13=tl.load(base+ 9*step,ci<C,0);m14=tl.load(base+10*step,ci<C,0);m15=tl.load(base+11*step,ci<C,0)
    m20=tl.load(base+12*step,ci<C,0);m21=tl.load(base+13*step,ci<C,0);m22=tl.load(base+14*step,ci<C,0);m23=tl.load(base+15*step,ci<C,0);m24=tl.load(base+16*step,ci<C,0);m25=tl.load(base+17*step,ci<C,0)
    m30=tl.load(base+18*step,ci<C,0);m31=tl.load(base+19*step,ci<C,0);m32=tl.load(base+20*step,ci<C,0);m33=tl.load(base+21*step,ci<C,0);m34=tl.load(base+22*step,ci<C,0);m35=tl.load(base+23*step,ci<C,0)
    m40=tl.load(base+24*step,ci<C,0);m41=tl.load(base+25*step,ci<C,0);m42=tl.load(base+26*step,ci<C,0);m43=tl.load(base+27*step,ci<C,0);m44=tl.load(base+28*step,ci<C,0);m45=tl.load(base+29*step,ci<C,0)
    m50=tl.load(base+30*step,ci<C,0);m51=tl.load(base+31*step,ci<C,0);m52=tl.load(base+32*step,ci<C,0);m53=tl.load(base+33*step,ci<C,0);m54=tl.load(base+34*step,ci<C,0);m55=tl.load(base+35*step,ci<C,0)
    # A^T M, then multiply by A.  Keep only sixteen final values.
    t00=m00+m10+m20+m30+m40;t01=m01+m11+m21+m31+m41;t02=m02+m12+m22+m32+m42;t03=m03+m13+m23+m33+m43;t04=m04+m14+m24+m34+m44;t05=m05+m15+m25+m35+m45
    t10=m10-m20+2*m30-2*m40;t11=m11-m21+2*m31-2*m41;t12=m12-m22+2*m32-2*m42;t13=m13-m23+2*m33-2*m43;t14=m14-m24+2*m34-2*m44;t15=m15-m25+2*m35-2*m45
    t20=m10+m20+4*m30+4*m40;t21=m11+m21+4*m31+4*m41;t22=m12+m22+4*m32+4*m42;t23=m13+m23+4*m33+4*m43;t24=m14+m24+4*m34+4*m44;t25=m15+m25+4*m35+4*m45
    t30=m10-m20+8*m30-8*m40+m50;t31=m11-m21+8*m31-8*m41+m51;t32=m12-m22+8*m32-8*m42+m52;t33=m13-m23+8*m33-8*m43+m53;t34=m14-m24+8*m34-8*m44+m54;t35=m15-m25+8*m35-8*m45+m55
    y=ty*4;x=tx*4;dst=DX+((g*B+batch)*C+ci)*H*W+y*W+x
    o00=t00+t01+t02+t03+t04;o01=t01-t02+2*t03-2*t04;o02=t01+t02+4*t03+4*t04;o03=t01-t02+8*t03-8*t04+t05
    o10=t10+t11+t12+t13+t14;o11=t11-t12+2*t13-2*t14;o12=t11+t12+4*t13+4*t14;o13=t11-t12+8*t13-8*t14+t15
    o20=t20+t21+t22+t23+t24;o21=t21-t22+2*t23-2*t24;o22=t21+t22+4*t23+4*t24;o23=t21-t22+8*t23-8*t24+t25
    o30=t30+t31+t32+t33+t34;o31=t31-t32+2*t33-2*t34;o32=t31+t32+4*t33+4*t34;o33=t31-t32+8*t33-8*t34+t35
    tl.store(dst+0*W+0,o00,valid&(y+0<H)&(x+0<W));tl.store(dst+0*W+1,o01,valid&(y+0<H)&(x+1<W))
    tl.store(dst+0*W+2,o02,valid&(y+0<H)&(x+2<W));tl.store(dst+0*W+3,o03,valid&(y+0<H)&(x+3<W))
    tl.store(dst+1*W+0,o10,valid&(y+1<H)&(x+0<W));tl.store(dst+1*W+1,o11,valid&(y+1<H)&(x+1<W))
    tl.store(dst+1*W+2,o12,valid&(y+1<H)&(x+2<W));tl.store(dst+1*W+3,o13,valid&(y+1<H)&(x+3<W))
    tl.store(dst+2*W+0,o20,valid&(y+2<H)&(x+0<W));tl.store(dst+2*W+1,o21,valid&(y+2<H)&(x+1<W))
    tl.store(dst+2*W+2,o22,valid&(y+2<H)&(x+2<W));tl.store(dst+2*W+3,o23,valid&(y+2<H)&(x+3<W))
    tl.store(dst+3*W+0,o30,valid&(y+3<H)&(x+0<W));tl.store(dst+3*W+1,o31,valid&(y+3<H)&(x+1<W))
    tl.store(dst+3*W+2,o32,valid&(y+3<H)&(x+2<W));tl.store(dst+3*W+3,o33,valid&(y+3<H)&(x+3<W))


@triton.jit
def _wg_f4_gemm(V,E,U,C:tl.constexpr,O:tl.constexpr,NT:tl.constexpr,
                ADD:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    tile=tl.program_id(0);g=tl.program_id(1);z=tl.program_id(2)
    nc:tl.constexpr=(C+BN-1)//BN
    oc=tile//nc*BM+tl.arange(0,BM);ci=tile%nc*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for block in range((NT+BK-1)//BK):
        k=block*BK+kk
        a=tl.load(E+((g*36+z)*O+oc[:,None])*NT+k[None,:],(oc[:,None]<O)&(k[None,:]<NT),0)
        b=tl.load(V+((g*36+z)*C+ci[None,:])*NT+k[:,None],(ci[None,:]<C)&(k[:,None]<NT),0)
        acc=tl.dot(a,b,acc,input_precision="ieee")
    dst=U+((g*36+z)*O+oc[:,None])*C+ci[None,:]
    mask=(oc[:,None]<O)&(ci[None,:]<C)
    if ADD:acc+=tl.load(dst,mask,0)
    tl.store(dst,acc,mask)


@triton.jit
def _wg_f4_inverse(U,DWOUT,G:tl.constexpr,C:tl.constexpr,O:tl.constexpr,
                   BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);g=i//(O*C);j=i%(O*C)
    valid=i<G*O*C;base=U+g*36*O*C+j;step:tl.constexpr=O*C
    u00=tl.load(base+ 0*step,valid,0);u01=tl.load(base+ 1*step,valid,0);u02=tl.load(base+ 2*step,valid,0);u03=tl.load(base+ 3*step,valid,0);u04=tl.load(base+ 4*step,valid,0);u05=tl.load(base+ 5*step,valid,0)
    u10=tl.load(base+ 6*step,valid,0);u11=tl.load(base+ 7*step,valid,0);u12=tl.load(base+ 8*step,valid,0);u13=tl.load(base+ 9*step,valid,0);u14=tl.load(base+10*step,valid,0);u15=tl.load(base+11*step,valid,0)
    u20=tl.load(base+12*step,valid,0);u21=tl.load(base+13*step,valid,0);u22=tl.load(base+14*step,valid,0);u23=tl.load(base+15*step,valid,0);u24=tl.load(base+16*step,valid,0);u25=tl.load(base+17*step,valid,0)
    u30=tl.load(base+18*step,valid,0);u31=tl.load(base+19*step,valid,0);u32=tl.load(base+20*step,valid,0);u33=tl.load(base+21*step,valid,0);u34=tl.load(base+22*step,valid,0);u35=tl.load(base+23*step,valid,0)
    u40=tl.load(base+24*step,valid,0);u41=tl.load(base+25*step,valid,0);u42=tl.load(base+26*step,valid,0);u43=tl.load(base+27*step,valid,0);u44=tl.load(base+28*step,valid,0);u45=tl.load(base+29*step,valid,0)
    u50=tl.load(base+30*step,valid,0);u51=tl.load(base+31*step,valid,0);u52=tl.load(base+32*step,valid,0);u53=tl.load(base+33*step,valid,0);u54=tl.load(base+34*step,valid,0);u55=tl.load(base+35*step,valid,0)
    t00=.25*u00-(u10+u20)/6+(u30+u40)/24;t01=.25*u01-(u11+u21)/6+(u31+u41)/24;t02=.25*u02-(u12+u22)/6+(u32+u42)/24;t03=.25*u03-(u13+u23)/6+(u33+u43)/24;t04=.25*u04-(u14+u24)/6+(u34+u44)/24;t05=.25*u05-(u15+u25)/6+(u35+u45)/24
    t10=(-u10+u20)/6+(u30-u40)/12;t11=(-u11+u21)/6+(u31-u41)/12;t12=(-u12+u22)/6+(u32-u42)/12;t13=(-u13+u23)/6+(u33-u43)/12;t14=(-u14+u24)/6+(u34-u44)/12;t15=(-u15+u25)/6+(u35-u45)/12
    t20=-(u10+u20)/6+(u30+u40)/6+u50;t21=-(u11+u21)/6+(u31+u41)/6+u51;t22=-(u12+u22)/6+(u32+u42)/6+u52;t23=-(u13+u23)/6+(u33+u43)/6+u53;t24=-(u14+u24)/6+(u34+u44)/6+u54;t25=-(u15+u25)/6+(u35+u45)/6+u55
    tl.store(DWOUT+i*9+0,.25*t00-(t01+t02)/6+(t03+t04)/24,valid)
    tl.store(DWOUT+i*9+1,(-t01+t02)/6+(t03-t04)/12,valid)
    tl.store(DWOUT+i*9+2,-(t01+t02)/6+(t03+t04)/6+t05,valid)
    tl.store(DWOUT+i*9+3,.25*t10-(t11+t12)/6+(t13+t14)/24,valid)
    tl.store(DWOUT+i*9+4,(-t11+t12)/6+(t13-t14)/12,valid)
    tl.store(DWOUT+i*9+5,-(t11+t12)/6+(t13+t14)/6+t15,valid)
    tl.store(DWOUT+i*9+6,.25*t20-(t21+t22)/6+(t23+t24)/24,valid)
    tl.store(DWOUT+i*9+7,(-t21+t22)/6+(t23-t24)/12,valid)
    tl.store(DWOUT+i*9+8,-(t21+t22)/6+(t23+t24)/6+t25,valid)


def winograd_plan(x, d, max_bytes=WORKSPACE_LIMIT_BYTES):
    g,b,c,h,w = x.shape
    o = d.shape[2]
    nt = b * ((h+1)//2) * ((w+1)//2)
    fixed = 16*o*c*4
    per_tile = 16*(c+o)*4
    per_group = fixed + per_tile*nt
    if per_group <= max_bytes:
        gc, tc = min(g, max_bytes//per_group), nt
    else:
        gc, tc = 1, min(nt, (max_bytes-fixed)//per_tile)
        if tc < 1:
            raise ValueError("Winograd fixed state exceeds the workspace budget")
        if tc >= 16:
            tc = tc//16*16
    chunks_g = (g+gc-1)//gc
    chunks_t = (nt+tc-1)//tc
    return dict(groups=gc, tiles=tc, total_tiles=nt,
                group_chunks=chunks_g, tile_chunks=chunks_t,
                bytes=gc*(fixed+per_tile*tc),
                kernel_launches=chunks_g*(3*chunks_t+1)+1)


def grad_weight_workspace(x, d, max_bytes=WORKSPACE_LIMIT_BYTES):
    p = winograd_plan(x, d, max_bytes)
    gc, tc = p["groups"], p["tiles"]
    c, o = x.shape[2], d.shape[2]
    return (x.new_empty((gc,16,c,tc)), x.new_empty((gc,16,o,tc)),
            x.new_empty((gc,16,o,c)))


def launch_pack_x(x, v, start=0):
    g,b,c,h,w = x.shape
    nt = v.shape[-1]
    _wg_pack_x[(triton.cdiv(c*nt,128),g)](
        x,v,start,b,c,h,w,nt,128,num_warps=4)


def launch_pack_d(d, e, start=0):
    g,b,o,h,w = d.shape
    nt = e.shape[-1]
    _wg_pack_d[(triton.cdiv(o*nt,128),g)](
        d,e,start,b,o,h,w,nt,128,num_warps=4)


def launch_winograd_gemm_v7(v, e, u, active_groups, add=False):
    c, nt = v.shape[2:]
    o = e.shape[2]
    bm,bn,bk = WG_TILE_V7
    _wg_gemm_v7[(triton.cdiv(o,bm)*triton.cdiv(c,bn),active_groups,16)](
        v,e,u,c,o,nt,add,bm,bn,bk,num_warps=4,num_stages=2)


def launch_winograd_gemm(v, e, u, active_groups, add=False):
    c, nt = v.shape[2:]
    o = e.shape[2]
    bm,bn,bk = WG_TILE
    bk = min(bk,max(16,triton.next_power_of_2(nt)))
    _wg_gemm_v8[(triton.cdiv(o,bm)*triton.cdiv(c,bn),active_groups,16)](
        v,e,u,c,o,nt,add,bm,bn,bk,num_warps=WG_WARPS,num_stages=2)


def launch_gw_inverse(u, dw):
    g,o,c = dw.shape[:3]
    _wg_inverse[(triton.cdiv(g*o*c,128),)](u,dw,g,c,o,128,num_warps=4)


def launch_grad_weight_bias(x, d, dw, db, workspace, gemm=None):
    if gemm is None:
        gemm = launch_winograd_gemm
    v,e,u = workspace
    g,b,_,h,w = x.shape
    total = b*((h+1)//2)*((w+1)//2)
    gc, tc = v.shape[0], v.shape[-1]
    for base in range(0,g,gc):
        end = min(g,base+gc)
        xx,dd = x[base:end],d[base:end]
        for start in range(0,total,tc):
            launch_pack_x(xx,v,start)
            launch_pack_d(dd,e,start)
            gemm(v,e,u,end-base,add=start!=0)
        launch_gw_inverse(u,dw[base:end])
    launch_grad_bias(d,db)


def winograd_f4_plan(x,d,max_bytes=WORKSPACE_LIMIT_BYTES):
    g,b,c,h,w=x.shape;o=d.shape[2]
    nt=b*((h+3)//4)*((w+3)//4)
    fixed=F4_COMPONENTS*o*c*4
    per_tile=F4_COMPONENTS*(c+o)*4
    per_group=fixed+per_tile*nt
    if per_group<=max_bytes:
        gc,tc=min(g,max_bytes//per_group),nt
    else:
        gc,tc=1,min(nt,(max_bytes-fixed)//per_tile)
        if tc<1:raise ValueError("F(4,3) fixed state exceeds the workspace budget")
        if tc>=16:tc=tc//16*16
    return dict(groups=gc,tiles=tc,total_tiles=nt,
                group_chunks=(g+gc-1)//gc,tile_chunks=(nt+tc-1)//tc,
                bytes=gc*(fixed+per_tile*tc),
                kernel_launches=((g+gc-1)//gc)*(3*((nt+tc-1)//tc)+1)+1)


def grad_weight_f4_workspace(x,d,max_bytes=WORKSPACE_LIMIT_BYTES):
    p=winograd_f4_plan(x,d,max_bytes);gc,tc=p["groups"],p["tiles"]
    c,o=x.shape[2],d.shape[2]
    return (x.new_empty((gc,F4_COMPONENTS,c,tc)),
            x.new_empty((gc,F4_COMPONENTS,o,tc)),
            x.new_empty((gc,F4_COMPONENTS,o,c)))


def f4_dx_plan(x,d,max_bytes=WORKSPACE_LIMIT_BYTES):
    g,b,c,h,w=x.shape;o=d.shape[2]
    nt=b*((h+3)//4)*((w+3)//4)
    fixed=F4_COMPONENTS*c*o*4
    per_tile=F4_COMPONENTS*(c+o)*4
    per_group=fixed+per_tile*nt
    if per_group<=max_bytes:
        gc,tc=min(g,max_bytes//per_group),nt
    else:
        gc,tc=1,min(nt,(max_bytes-fixed)//per_tile)
        if tc<1:raise ValueError("F(4,3) dX fixed state exceeds the workspace budget")
        if tc>=16:tc=tc//16*16
    return dict(groups=gc,tiles=tc,total_tiles=nt,
                group_chunks=(g+gc-1)//gc,tile_chunks=(nt+tc-1)//tc,
                bytes=gc*(fixed+per_tile*tc),
                kernel_launches=((g+gc-1)//gc)*(1+3*((nt+tc-1)//tc)))


def f4_dx_workspace(x,d,max_bytes=WORKSPACE_LIMIT_BYTES):
    p=f4_dx_plan(x,d,max_bytes);gc,tc=p["groups"],p["tiles"]
    c,o=x.shape[2],d.shape[2]
    return (x.new_empty((gc,F4_COMPONENTS,o,tc)),
            x.new_empty((gc,F4_COMPONENTS,c,tc)),
            x.new_empty((gc,F4_COMPONENTS,c,o)))


def joint_f4_plan(x,d,max_bytes=WORKSPACE_LIMIT_BYTES):
    g,b,c,h,w=x.shape;o=d.shape[2]
    nt=b*((h+3)//4)*((w+3)//4)
    one_fixed=F4_COMPONENTS*c*o*4
    per_tile=F4_COMPONENTS*(o+max(c,o)+c)*4
    alias_group=one_fixed+per_tile*nt
    if alias_group<=max_bytes:
        gc,tc=min(g,max_bytes//alias_group),nt
        alias_du=True;fixed=one_fixed
    else:
        # With streamed tile chunks the transformed weight must survive while
        # dU accumulates, so it needs a second fixed-size buffer.
        fixed=2*one_fixed
        full_group=fixed+per_tile*nt
        if full_group<=max_bytes:
            gc,tc=min(g,max_bytes//full_group),nt
        else:
            gc,tc=1,min(nt,(max_bytes-fixed)//per_tile)
            if tc<1:raise ValueError("joint F(4,3) fixed state exceeds the workspace budget")
            if tc>=16:tc=tc//16*16
        alias_du=False
    cg=(g+gc-1)//gc;ct=(nt+tc-1)//tc
    # Per group chunk: weight transform, five kernels per tile chunk, dW
    # inverse. Bias is one final launch for the complete tensor.
    return dict(groups=gc,tiles=tc,total_tiles=nt,alias_du=alias_du,
                group_chunks=cg,tile_chunks=ct,
                bytes=gc*(fixed+per_tile*tc),
                kernel_launches=cg*(2+5*ct)+1)


def joint_f4_workspace(x,d,max_bytes=WORKSPACE_LIMIT_BYTES):
    p=joint_f4_plan(x,d,max_bytes);gc,tc=p["groups"],p["tiles"]
    c,o=x.shape[2],d.shape[2]
    e=x.new_empty((gc,F4_COMPONENTS,o,tc))
    shared=x.new_empty((gc,F4_COMPONENTS,max(c,o),tc))
    mid=x.new_empty((gc,F4_COMPONENTS,c,tc))
    uw=x.new_empty((gc,F4_COMPONENTS,c,o))
    du=uw if p["alias_du"] else x.new_empty((gc,F4_COMPONENTS,o,c))
    return e,shared,mid,uw,du


def launch_f4_pack_x(x,v,start=0):
    g,b,c,h,w=x.shape;nt=v.shape[-1]
    _wg_f4_pack_x[(triton.cdiv(c*nt,64),g)](
        x,v,v,start,b,c,h,w,nt,False,64,num_warps=4)


def launch_f4_pack_d(d,e,start=0):
    g,b,o,h,w=d.shape;nt=e.shape[-1]
    _wg_f4_pack_d[(triton.cdiv(o*nt,64),g)](
        d,e,start,b,o,h,w,nt,64,num_warps=4)


def launch_f4_joint_pack_d(d,v,e,start=0):
    g,b,o,h,w=d.shape;nt=e.shape[-1]
    _wg_f4_joint_pack_d[(triton.cdiv(o*nt,64),g)](
        d,v,e,start,b,o,h,w,nt,64,num_warps=4)


def launch_f4_pack_weight_dx(weight,u):
    g,o,c=weight.shape[:3]
    _wg_f4_pack_weight_dx[(triton.cdiv(c*o,64),g)](
        weight,u,g,c,o,64,num_warps=4)


def launch_f4_dx_gemm(u,v,m,active_groups,c=None,o=None):
    if c is None:c=m.shape[2]
    if o is None:o=v.shape[2]
    nt=m.shape[-1]
    bm=64
    bn=16 if nt<=16 else 32 if nt<=32 else 64
    bk=16
    _wg_f4_dx_gemm[(triton.cdiv(c,bm)*triton.cdiv(nt,bn),active_groups,F4_COMPONENTS)](
        u,v,m,c,o,nt,bm,bn,bk,num_warps=4,num_stages=2)


def launch_f4_dx_gemm_schedule(u,v,m,active_groups,c,o,bm,bn,bk,warps):
    nt=m.shape[-1]
    _wg_f4_dx_gemm[(triton.cdiv(c,bm)*triton.cdiv(nt,bn),active_groups,F4_COMPONENTS)](
        u,v,m,c,o,nt,bm,bn,bk,num_warps=warps,num_stages=2)


def launch_f4_dx_output(m,dx,start=0):
    g,b,c,h,w=dx.shape;nt=m.shape[-1]
    _wg_f4_dx_output[(triton.cdiv(c*nt,64),g)](
        m,dx,start,g,b,c,h,w,nt,64,num_warps=4)


def launch_f4_gemm(v,e,u,active_groups,add=False):
    c,nt=v.shape[2:];o=e.shape[2]
    bm,bn,bk=WG_TILE_V7
    _wg_f4_gemm[(triton.cdiv(o,bm)*triton.cdiv(c,bn),active_groups,F4_COMPONENTS)](
        v,e,u,c,o,nt,add,bm,bn,bk,num_warps=4,num_stages=2)


def launch_f4_gemm_explicit(v,e,u,active_groups,c,o,add=False):
    """Packed-pointer form used when v's backing storage has max(C,O)."""
    nt=e.shape[-1];bm,bn,bk=WG_TILE_V7
    _wg_f4_gemm[(triton.cdiv(o,bm)*triton.cdiv(c,bn),active_groups,F4_COMPONENTS)](
        v,e,u,c,o,nt,add,bm,bn,bk,num_warps=4,num_stages=2)


def launch_f4_gemm_schedule(v,e,u,active_groups,c,o,add,bm,bn,bk,warps):
    nt=e.shape[-1]
    _wg_f4_gemm[(triton.cdiv(o,bm)*triton.cdiv(c,bn),active_groups,F4_COMPONENTS)](
        v,e,u,c,o,nt,add,bm,bn,bk,num_warps=warps,num_stages=2)


def launch_f4_gemm_bmm(v,e,u,active_groups,c=None,o=None,add=False):
    """IEEE batched GEMM on the native Winograd buffers (no layout copy).

    ``v`` may use the joint path's max(C, O)-wide backing allocation, but
    ``launch_f4_pack_x`` deliberately overwrites that storage in compact
    [group, component, C, tile] order.  The view below therefore uses C*tile
    (not the Python allocation's max(C,O)*tile stride) between component
    matrices.  No compacting copy is materialized.  The output is written
    directly into the caller-provided transformed-weight buffer.
    """
    if c is None:c=v.shape[2]
    if o is None:o=e.shape[2]
    batches=active_groups*F4_COMPONENTS
    aa=e.as_strided((batches,o,e.shape[-1]),
                    (e.stride(1),e.stride(2),e.stride(3)))
    nt=v.shape[-1]
    vv=v.as_strided((batches,c,nt),(c*nt,nt,1))
    bb=vv.transpose(1,2)
    # ``u`` may alias the [C,O] transformed-weight allocation.  Reinterpret
    # its identical contiguous C*O storage as [O,C] instead of trusting the
    # Python shape of that alias.
    out=u.as_strided((batches,o,c),(o*c,c,1))
    if add:
        torch.baddbmm(out,aa,bb,beta=1,alpha=1,out=out)
    else:
        torch.bmm(aa,bb,out=out)


def launch_f4_inverse(u,dw):
    g,o,c=dw.shape[:3]
    _wg_f4_inverse[(triton.cdiv(g*o*c,64),)](
        u,dw,g,c,o,64,num_warps=4)


def launch_grad_weight_bias_f4(x,d,dw,db,workspace,gemm=None):
    if gemm is None:gemm=launch_f4_gemm
    v,e,u=workspace;g,b,_,h,w=x.shape
    total=b*((h+3)//4)*((w+3)//4);gc,tc=v.shape[0],v.shape[-1]
    for base in range(0,g,gc):
        end=min(g,base+gc);xx,dd=x[base:end],d[base:end]
        for start in range(0,total,tc):
            launch_f4_pack_x(xx,v,start);launch_f4_pack_d(dd,e,start)
            gemm(v,e,u,end-base,add=start!=0)
        launch_f4_inverse(u,dw[base:end])
    launch_grad_bias(d,db)


def launch_grad_x_f4(d,weight,dx,workspace):
    v,m,u=workspace;g,b,o,h,w=d.shape;c=weight.shape[2]
    total=b*((h+3)//4)*((w+3)//4);gc,tc=v.shape[0],v.shape[-1]
    for base in range(0,g,gc):
        end=min(g,base+gc);dd,ww,out=d[base:end],weight[base:end],dx[base:end]
        launch_f4_pack_weight_dx(ww,u)
        for start in range(0,total,tc):
            launch_f4_pack_x(dd,v,start)
            launch_f4_dx_gemm(u,v,m,end-base,c,o)
            launch_f4_dx_output(m,out,start)


def launch_joint_f4(x,weight,d,dx,dw,db,workspace,dx_gemm=None,dw_gemm=None):
    e,shared,mid,uw,du=workspace
    g,b,c,h,w=x.shape;o=d.shape[2]
    total=b*((h+3)//4)*((w+3)//4);gc,tc=e.shape[0],e.shape[-1]
    for base in range(0,g,gc):
        end=min(g,base+gc);active=end-base
        xx,ww,dd=x[base:end],weight[base:end],d[base:end]
        outx,outw=dx[base:end],dw[base:end]
        launch_f4_pack_weight_dx(ww,uw)
        for start in range(0,total,tc):
            # One launch exposes both dY transforms to compiler CSE.
            launch_f4_joint_pack_d(dd,shared,e,start)
            if dx_gemm is None:
                launch_f4_dx_gemm(uw,shared,mid,active,c,o)
            else:
                dx_gemm(uw,shared,mid,active,c,o)
            launch_f4_dx_output(mid,outx,start)
            # dV is dead after output transform; overwrite its backing buffer
            # with B^T X B before the weight-gradient GEMM.
            launch_f4_pack_x(xx,shared,start)
            if dw_gemm is None:
                launch_f4_gemm_explicit(shared,e,du,active,c,o,add=start!=0)
            else:
                dw_gemm(shared,e,du,active,c,o,start!=0)
        launch_f4_inverse(du,outw)
    launch_grad_bias(d,db)


def _run(x,weight,grad_output,stride,padding,dilation,groups,gemm=None,
         old_dw=False,f4=False,f4_dx=False,joint=False):
    g,_,_,o,_,_ = _validate(x,weight,grad_output,stride,padding,dilation,groups)
    with torch.cuda.device(x.device):
        x,weight,d = (t.contiguous() for t in (x,weight,grad_output))
        dx,dw,db = torch.empty_like(x),torch.empty_like(weight),x.new_empty((g,o))
        if joint:
            work=joint_f4_workspace(x,d)
            launch_joint_f4(x,weight,d,dx,dw,db,work,dw_gemm=gemm)
            return dx,dw,db
        if f4_dx:
            gx_work=f4_dx_workspace(x,d)
            launch_grad_x_f4(d,weight,dx,gx_work)
        else:
            gx_work = grad_x_workspace(d,dx)
            launch_grad_x(d,weight,dx,gx_work)
        del gx_work
        if old_dw:
            launch_grad_weight_bias_v5(x,d,dw,db)
        elif f4:
            gw_work=grad_weight_f4_workspace(x,d)
            launch_grad_weight_bias_f4(x,d,dw,db,gw_work,gemm)
        else:
            gw_work = grad_weight_workspace(x,d)
            launch_grad_weight_bias(x,d,dw,db,gw_work,gemm)
    return dx,dw,db


def _run_small_direct(x,weight,grad_output,stride,padding,dilation,groups,
                      dx_kind="implicit32",workspace=None):
    """Two-launch direct path, or three launches for the bmm+fold ablation."""
    g,_,_,o,_,_=_validate(x,weight,grad_output,stride,padding,dilation,groups)
    with torch.cuda.device(x.device):
        x,weight,d=(t.contiguous() for t in (x,weight,grad_output))
        dx,dw,db=torch.empty_like(x),torch.empty_like(weight),x.new_empty((g,o))
        if dx_kind=="implicit16":
            launch_grad_x_implicit(d,weight,dx,16,32,16,4)
        elif dx_kind=="implicit16sq":
            launch_grad_x_implicit(d,weight,dx,16,16,16,4)
        elif dx_kind=="implicit32":
            launch_grad_x_implicit(d,weight,dx,32,32,16,4)
        elif dx_kind=="implicit64":
            launch_grad_x_implicit(d,weight,dx,32,64,16,4)
        elif dx_kind=="implicit_bk32":
            launch_grad_x_implicit(d,weight,dx,32,32,32,4)
        elif dx_kind=="bmm":
            if workspace is None:workspace=grad_x_workspace(d,dx)
            launch_grad_x_bmm(d,weight,dx,workspace)
        else:
            raise ValueError("unknown small direct dX schedule")
        launch_grad_weight_bias_v5(x,d,dw,db)
    return dx,dw,db


def _run_rgb(x,weight,grad_output,stride,padding,dilation,groups,
             dx_kind="direct3_16",force_split=None):
    """Specialized ConvNet RGB-input path with compact C=3 computation."""
    g,_,c,o,_,_=_validate(x,weight,grad_output,stride,padding,dilation,groups)
    if c!=3:
        raise ValueError("v13 RGB candidates require C=3")
    with torch.cuda.device(x.device):
        x,weight,d=(t.contiguous() for t in (x,weight,grad_output))
        dx,dw,db=torch.empty_like(x),torch.empty_like(weight),x.new_empty((g,o))
        if dx_kind=="direct3_16":
            launch_rgb_grad_x_direct3(d,weight,dx,16,16,4)
        elif dx_kind=="direct3_32":
            launch_rgb_grad_x_direct3(d,weight,dx,32,16,4)
        elif dx_kind=="implicit16sq":
            launch_grad_x_implicit(d,weight,dx,16,16,16,4)
        elif dx_kind=="bmm":
            workspace=grad_x_workspace(d,dx)
            launch_grad_x_bmm(d,weight,dx,workspace)
        else:
            raise ValueError("unknown v13 RGB dX schedule")
        launch_rgb_grad_weight_bias(x,d,dw,db,force_split)
    return dx,dw,db


def _run_v14_rgb(x,weight,grad_output,stride,padding,dilation,groups,
                 dw_kind="compact32"):
    """Batch-chunked dX plus one of the experimental RGB dW families."""
    g,_,c,o,_,_=_validate(x,weight,grad_output,stride,padding,dilation,groups)
    if c!=3:
        raise ValueError("v14 RGB candidates require C=3")
    with torch.cuda.device(x.device):
        x,weight,d=(t.contiguous() for t in (x,weight,grad_output))
        dx,dw,db=torch.empty_like(x),torch.empty_like(weight),x.new_empty((g,o))
        gx_work=grad_x_batch_chunk_workspace(x)
        launch_grad_x_batch_chunked(d,weight,dx,gx_work)
        del gx_work
        if dw_kind=="compact32":
            launch_rgb_grad_weight_bias(x,d,dw,db,bm=32)
        elif dw_kind=="compact64":
            launch_rgb_grad_weight_bias(x,d,dw,db,bm=64)
        elif dw_kind=="f4_bmm":
            gw_work=grad_weight_f4_workspace(x,d)
            launch_grad_weight_bias_f4(
                x,d,dw,db,gw_work,gemm=launch_f4_gemm_bmm)
        else:
            raise ValueError("unknown v14 RGB dW family")
    return dx,dw,db


def conv_bwd_v11_joint_bmm(x,weight,grad_output,stride=1,padding=1,
                           dilation=1,groups=1):
    """v11: joint F(4,3), Triton dX and native-layout batched-GEMM dW."""
    return _run(x,weight,grad_output,stride,padding,dilation,groups,
                gemm=launch_f4_gemm_bmm,joint=True)


def conv_bwd_v13_rgb(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
    """v13 primary RGB candidate: scalar C=3 dX plus occupancy-aware dW/dB."""
    return _run_rgb(x,weight,grad_output,stride,padding,dilation,groups)


def conv_bwd_v13_rgb32(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
    return _run_rgb(x,weight,grad_output,stride,padding,dilation,groups,"direct3_32")


def conv_bwd_v13_rgb_implicit(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
    """Ablation: padded implicit dX with the same compact RGB dW/dB."""
    return _run_rgb(x,weight,grad_output,stride,padding,dilation,groups,"implicit16sq")


def conv_bwd_v13_rgb_bmm(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
    """Ablation: bmm+fold dX with the same compact RGB dW/dB."""
    return _run_rgb(x,weight,grad_output,stride,padding,dilation,groups,"bmm")


def conv_bwd_v13_rgb_nosplit(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
    """Ablation for tiny cases: primary dX with one unsplit dW CTA per O tile."""
    return _run_rgb(x,weight,grad_output,stride,padding,dilation,groups,
                    "direct3_16",1)


def conv_bwd_v14_rgb_chunk32(x,weight,grad_output,stride=1,padding=1,
                             dilation=1,groups=1):
    """v14 candidate: batch-chunked dX plus BM=32 compact RGB dW/dB."""
    return _run_v14_rgb(x,weight,grad_output,stride,padding,dilation,groups,
                        "compact32")


def conv_bwd_v14_rgb_chunk64(x,weight,grad_output,stride=1,padding=1,
                             dilation=1,groups=1):
    """v14 candidate: batch-chunked dX plus BM=64 compact RGB dW/dB."""
    return _run_v14_rgb(x,weight,grad_output,stride,padding,dilation,groups,
                        "compact64")


def conv_bwd_v14_rgb_chunk_f4(x,weight,grad_output,stride=1,padding=1,
                              dilation=1,groups=1):
    """v14 candidate: batch-chunked dX plus native-buffer F(4,3) bmm dW."""
    return _run_v14_rgb(x,weight,grad_output,stride,padding,dilation,groups,
                        "f4_bmm")


def conv_bwd_v12_direct(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
    """v12 small-work candidate: one implicit dX kernel plus fused direct dW/dB."""
    return _run_small_direct(x,weight,grad_output,stride,padding,dilation,groups)


def conv_bwd_v12_direct16(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
    return _run_small_direct(x,weight,grad_output,stride,padding,dilation,groups,"implicit16")


def conv_bwd_v12_direct16sq(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
    """Higher-grid small-N schedule: 16x16x16 implicit dX tiles."""
    return _run_small_direct(x,weight,grad_output,stride,padding,dilation,groups,"implicit16sq")


def conv_bwd_v12_direct64(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
    return _run_small_direct(x,weight,grad_output,stride,padding,dilation,groups,"implicit64")


def conv_bwd_v12_direct_bk32(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
    return _run_small_direct(x,weight,grad_output,stride,padding,dilation,groups,"implicit_bk32")


def conv_bwd_v12_bmm_dx(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
    """Ablation: bmm dX columns + fold, with the same fused direct dW/dB."""
    return _run_small_direct(x,weight,grad_output,stride,padding,dilation,groups,"bmm")


def conv_bwd_v10(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
    """v10 control: joint F(4,3) with Triton GEMMs for both contractions."""
    return _run(x,weight,grad_output,stride,padding,dilation,groups,joint=True)


def conv_bwd_separate_bmm(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
    """Ablation: F(4,3) dX and batched-GEMM dW, recomputing dY transform."""
    return _run(x,weight,grad_output,stride,padding,dilation,groups,
                gemm=launch_f4_gemm_bmm,f4=True,f4_dx=True)


def conv_bwd_separate_f4(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
    """Ablation: F(4,3) dX and dW, but transform dY separately."""
    return _run(x,weight,grad_output,stride,padding,dilation,groups,f4=True,f4_dx=True)


def conv_bwd_v9(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
    """v9 baseline: direct GEMM+fold dX and F(4,3) dW."""
    return _run(x,weight,grad_output,stride,padding,dilation,groups,f4=True)


def conv_bwd_v7(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
    """Explicit v7 baseline with its original GEMM and identical buffers."""
    return _run(x,weight,grad_output,stride,padding,dilation,groups,launch_winograd_gemm_v7)


def conv_bwd_v8(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
    """Failed v8 block-pointer F(2,3) ablation, retained for diagnostics."""
    return _run(x,weight,grad_output,stride,padding,dilation,groups,launch_winograd_gemm)


def conv_bwd_v5(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
    """Explicit v5 baseline: identical dX, original three-tap dW/dB."""
    return _run(x,weight,grad_output,stride,padding,dilation,groups,old_dw=True)


def select_algorithm(x,weight,grad_output,stride=1,padding=1,dilation=1,
                     groups=1):
    """Return the measured v13 rule used by :func:`conv_bwd`.

    v14 candidates are deliberately excluded until the new sweep validates
    both correctness and performance.  Unknown 3x3 shapes use the robust v11
    joint Winograd implementation.
    """
    g,b,c,o,h,w=_validate(
        x,weight,grad_output,stride,padding,dilation,groups)
    if c==3:
        if o==128 and h==32 and w==32 and g==1:
            return "v13_rgb_implicit" if b==1 else "v11_joint_bmm"
        return "v13_rgb_bmm"
    if c==128 and o==128 and b==1 and h==w:
        if h==8:
            if g==1:
                return "v12_direct16"
            if g<=4:
                return "v12_bmm_dx"
            return "v7_f2"
        if h==16:
            return "v12_bmm_dx" if g<=4 else "v9_f4dw"
        if h==32 and g==1:
            return "v7_f2"
    return "v11_joint_bmm"


def conv_bwd(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1):
    """Direct-call group-first convolution backward with automatic selection.

    Inputs and outputs use ``[G,B,C,H,W]``, ``[G,O,C,3,3]`` and
    ``[G,B,O,H,W]`` / ``[G,B,C,H,W]``, ``[G,O,C,3,3]``, ``[G,O]``.
    The signature intentionally matches the user-facing baseline.
    """
    name=select_algorithm(
        x,weight,grad_output,stride,padding,dilation,groups)
    table={
        "v7_f2":conv_bwd_v7,
        "v9_f4dw":conv_bwd_v9,
        "v11_joint_bmm":conv_bwd_v11_joint_bmm,
        "v12_direct16":conv_bwd_v12_direct16,
        "v12_bmm_dx":conv_bwd_v12_bmm_dx,
        "v13_rgb_implicit":conv_bwd_v13_rgb_implicit,
        "v13_rgb_bmm":conv_bwd_v13_rgb_bmm,
    }
    return table[name](x,weight,grad_output,stride,padding,dilation,groups)

