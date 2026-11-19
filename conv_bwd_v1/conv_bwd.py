"""Group-first dense 3x3, stride=padding=dilation=1, FP32 backward.

x: [G,B,C,H,W], weight: [G,O,C,3,3], grad_output: [G,B,O,H,W].
Return grad_x, grad_weight, grad_bias ([G,O]). G counts independent
convolutions; groups=1 or groups=G both refer to this outer group axis.

v6 uses a single-accumulator implicit GEMM for dW, tile 64x32x16,
without a global im2col buffer or split-K. dB is an independent reduction
to keep the GEMM mainloop branch-free. dX is v5 GEMM+fold, generalized
to the spatial/channel sizes.
conv_bwd_v5 is an explicit comparison entry point, not an auto selector.
"""

import operator
import torch
import triton
import triton.language as tl

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
def _dw_implicit(X, D, DWOUT,
                    B: tl.constexpr, C: tl.constexpr, O: tl.constexpr,
                    H: tl.constexpr, W: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Virtual GEMM [r,s,c; b,p,q] @ [b,p,q; o]. Tiles never cross a
    # filter tap, so the tap offset is uniform across the whole CTA.
    # The patch matrix is formed on load; no global im2col is written.
    tile = tl.program_id(0)
    g = tl.program_id(1)
    nc: tl.constexpr = (C + BM - 1) // BM
    no: tl.constexpr = (O + BN - 1) // BN
    tap = tile // (nc * no)
    ctile = tile // no % nc
    ci = ctile * BM + tl.arange(0, BM)
    oc = tile % no * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    # Use the full HW reduction. In the target family, HW is a power
    # of two, avoiding the H-1 row kernel's batch/position division.
    for block in range((B * H * W + BK - 1) // BK):
        k = block * BK + kk
        batch = k // (H * W)
        pos = k % (H * W)
        iy = pos // W + tap // 3 - 1
        ix = pos % W + tap % 3 - 1
        valid = k < B * H * W
        inp = tl.load(
            X + ((g * B + batch[None, :]) * C + ci[:, None]) * H * W
              + iy[None, :] * W + ix[None, :],
            (ci[:, None] < C) & valid[None, :]
            & (iy[None, :] >= 0) & (iy[None, :] < H)
            & (ix[None, :] >= 0) & (ix[None, :] < W), 0)
        dy = tl.load(
            D + ((g * B + batch[:, None]) * O + oc[None, :]) * H * W
              + pos[:, None],
            valid[:, None] & (oc[None, :] < O), 0)
        acc = tl.dot(inp, dy, acc, input_precision="ieee")
    tl.store(DWOUT + (g * O + oc[None, :]) * C * 9
             + ci[:, None] * 9 + tap, acc,
             (ci[:, None] < C) & (oc[None, :] < O))


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


WORKSPACE_LIMIT_BYTES = 128 * 1024**2


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


def launch_grad_weight_bias_v5(x, d, dw, db):
    g, b, c, h, w = x.shape
    o = d.shape[2]
    _dw_db_row[(triton.cdiv(c, 32) * triton.cdiv(o, 32), g, 3)](
        x, d, dw, db, c, o, h, w, b, 32, 32, 16,
        num_warps=4, num_stages=2)


def launch_grad_weight(x, d, dw):
    g, b, c, h, w = x.shape
    o = d.shape[2]
    _dw_implicit[(9 * triton.cdiv(c, 64) * triton.cdiv(o, 32), g)](
        x, d, dw, b, c, o, h, w, 64, 32, 16,
        num_warps=4, num_stages=2)


def launch_grad_bias(d, db):
    g, b, o, h, w = d.shape
    _db_reduce[(o, g)](d, db, b, o, h*w, 1024, num_warps=4)


def launch_grad_weight_bias(x, d, dw, db):
    launch_grad_weight(x, d, dw)
    launch_grad_bias(d, db)


def _run(x, weight, grad_output, stride, padding, dilation, groups, old_dw):
    g, _, _, o, _, _ = _validate(x, weight, grad_output, stride, padding, dilation, groups)
    with torch.cuda.device(x.device):
        x, weight, d = (t.contiguous() for t in (x, weight, grad_output))
        dx, dw, db = torch.empty_like(x), torch.empty_like(weight), x.new_empty((g, o))
        workspace = grad_x_workspace(d, dx)
        launch_grad_x(d, weight, dx, workspace)
        del workspace
        if old_dw:
            launch_grad_weight_bias_v5(x, d, dw, db)
        else:
            launch_grad_weight_bias(x, d, dw, db)
    return dx, dw, db


def conv_bwd(x, weight, grad_output, stride=1, padding=1, dilation=1, groups=1):
    """v6 candidate: v5 dX, single-accumulator implicit dW, reduced dB."""
    return _run(x, weight, grad_output, stride, padding, dilation, groups, False)


def conv_bwd_v5(x, weight, grad_output, stride=1, padding=1, dilation=1, groups=1):
    """Explicit v5 comparison: same dX, original three-tap dW/dB."""
    return _run(x, weight, grad_output, stride, padding, dilation, groups, True)


