"""Group-first ConvNet and ResNet convolution backward.

conv_bwd(x, weight, grad_output, stride=1, padding=1, dilation=1, groups=1)
returns grad_x [G,B,C,H,W], grad_weight [G,O,C,R,S], grad_bias [G,O].

This single-file build contains the layout adapters, measured selector
rules, and width-only Winograd implementation. It requires PyTorch and Triton.
v27 adds opt-in first-backward candidates for ResNet stride-2 convolutions.
v28 adds opt-in packed 3x3/s1 BMM and lower-workspace 3x3/s2 schedules.
v29 adds an opt-in direct dW for B=1 and 4x4 stride-1 activations.
v30 adds opt-in fused and spatial-major BMM dX for 4x4 stride-1 activations.
v31 adds opt-in RGB 7x7/s2/p3 dX and dW/dB schedules for ResNet stems.
v32 adds packed BMM and a compact dW kernel for the 2x2 ResNet layer4.

v26 uses measured group-first Torch, per-group Torch, and small dense
overrides. "custom_auto" exposes the v24 selector.
v23 optionally uses torch.bmm for the F(4,3) dW contraction with BMM dX.
v22 adds an opt-in F(4,3) dW path with six parallel row transforms and BMM dX.
v21 adds B=1 small-spatial fused Winograd weight gradients and a group-batched dX ablation.
v20 adds opt-in dense 3x3/s1 providers. v20_balanced_f4 equalizes
streamed transform chunks. v20_split_direct and v20_joint_direct
use the same implicit dX and flat-filter dW/dB tiles in two or one launch.

v19 retains the measured v18 selector and adds a small-B direct-dX probe that
batches every image of one group into a single ``torch.bmm(out=...)`` call.
The group-first dY layout is already [B,O,H,W] inside one group, so this path
does not transpose or copy dY.  A batch-major fold writes dX directly.  The
probe is intentionally not selected before its measurements are available.
v16's 3x3/s2
dX uses four fixed-parity contractions, so each input coordinate evaluates
only the 1/2/2/4 filter taps that can actually reach it.  The corresponding
dW and the 1x1/s2 shortcut use native group-first direct contractions with a
bounded split-K workspace.  Several 32/64-channel tile schedules are exposed
for measurement before the provisional defaults are finalized.

The public ``conv_bwd`` entry point applies the measured v14 selector.  v14
adds a batch-chunked direct dX contraction: one launch handles several complete
batches, while a small planner jointly chunks G and B under the 128 MiB
workspace limit.  This removes the one-GEMM-plus-one-fold launch pair per
batch paid by the v13 bmm path.  v14 also tests 32- and 64-output-channel RGB
dW tiles, plus an F(4,3) dW alternative.  The validated BM32/BM64 paths now
form the RGB branch of the public selector.

v15 keeps the proven batch-chunked dX unchanged and probes only the remaining
RGB dW bottleneck: half/double split counts and BK=32 schedules for BM32/BM64.
The public selector does not use these probes before measurement.

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
import math
import operator
import torch
import triton
import triton.language as tl

# Width-only Winograd dX

@triton.jit
def _pack_weight(Weight, U, C: tl.constexpr, O: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    g = tl.program_id(1)
    ci = i % C
    kr = i // C
    oc, r = kr // 3, kr % 3
    valid = kr < 3*O
    src = Weight + ((g*O + oc)*C + ci)*9 + 3*r
    # Horizontal reversal converts the transposed convolution to correlation.
    a = tl.load(src+2, valid, 0)
    b = tl.load(src+1, valid, 0)
    c = tl.load(src, valid, 0)
    step: tl.constexpr = 3*O*C
    out = U + g*6*step + i
    tl.store(out, a*0.25, valid)
    tl.store(out+step, -(a+b+c)/6., valid)
    tl.store(out+2*step, (-a+b-c)/6., valid)
    tl.store(out+3*step, (a+2*b+4*c)/24., valid)
    tl.store(out+4*step, (a-2*b+4*c)/24., valid)
    tl.store(out+5*step, c, valid)


@triton.jit
def _pack_dy(D, V, O: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
             TW: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    g = tl.program_id(1)
    tx = i % TW
    sy = i // TW % (H+2) - 1
    oc = i // ((H+2)*TW)
    left = 4*tx - 1
    row_valid = (oc < O) & (sy >= 0) & (sy < H)
    src = D + ((g*O+oc)*H+sy)*W+left
    a = tl.load(src,   row_valid & (left>=0) & (left<W), 0)
    b = tl.load(src+1, row_valid & (left+1<W), 0)
    c = tl.load(src+2, row_valid & (left+2<W), 0)
    d = tl.load(src+3, row_valid & (left+3<W), 0)
    e = tl.load(src+4, row_valid & (left+4<W), 0)
    f = tl.load(src+5, row_valid & (left+5<W), 0)
    v0 = 4*a-5*c+e
    v1 = -4*(b+c)+d+e
    v2 = 4*(b-c)-d+e
    v3 = -2*b-c+2*d+e
    v4 = 2*b-c-2*d+e
    v5 = 4*b-5*d+f

    # Each (oc,r,output row,tile) has exactly one writer, including the
    # zero halo. Transform each real source row once, then fan out to r.
    step: tl.constexpr = 3*O*H*TW
    for r in tl.static_range(3):
        y = sy - 1 + r
        keep = (oc < O) & (y >= 0) & (y < H)
        out = V + (g*6*step) + ((oc*3+r)*H+y)*TW+tx
        tl.store(out, v0, keep)
        tl.store(out+step, v1, keep)
        tl.store(out+2*step, v2, keep)
        tl.store(out+3*step, v3, keep)
        tl.store(out+4*step, v4, keep)
        tl.store(out+5*step, v5, keep)


@triton.jit
def _write_dx(M, DX, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
              TW: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    g = tl.program_id(1)
    x = 4*(i % TW)
    step: tl.constexpr = C*H*TW
    valid = i < step
    src = M + g*6*step + i
    a = tl.load(src, valid, 0)
    b = tl.load(src+step, valid, 0)
    c = tl.load(src+2*step, valid, 0)
    d = tl.load(src+3*step, valid, 0)
    e = tl.load(src+4*step, valid, 0)
    f = tl.load(src+5*step, valid, 0)
    out = DX + g*C*H*W + (i // TW)*W+x
    tl.store(out, a+(b+c)+(d+e), valid)
    tl.store(out+1, (b-c)+2*(d-e), valid & (x+1<W))
    tl.store(out+2, (b+c)+4*(d+e), valid & (x+2<W))
    tl.store(out+3, (b-c)+8*(d-e)+f, valid & (x+3<W))


def _width_f4_workspace(dy, dx, max_bytes, group_cap=16):
    g, _, c, h, w = dx.shape
    o = dy.shape[2]
    nt = h*triton.cdiv(w, 4)
    per_group = 6*(3*o*c + 3*o*nt + c*nt)*dx.element_size()
    active = min(g, group_cap, max_bytes//per_group)
    if active < 1:
        raise ValueError("one group of width F(4,3) exceeds the workspace budget")
    return (dx.new_empty((active, 6, 3*o, c)), dx.new_empty((active, 6, 3*o, nt)),
            dx.new_empty((active, 6, c, nt)))


def _width_f4_pack_weight(wt, u):
    g, o, c = wt.shape[:3]
    _pack_weight[(triton.cdiv(3*o*c, 128), g)](wt, u, c, o, 128, num_warps=4)


def _width_f4_pack_dy(dy, v):
    g, _, o, h, w = dy.shape
    tw = triton.cdiv(w, 4)
    _pack_dy[(triton.cdiv(o*(h+2)*tw, 128), g)](dy, v, o, h, w, tw, 128, num_warps=4)


def _width_f4_multiply(u, v, m):
    # U is stored transposed to keep the weight pack contiguous along C.
    torch.bmm(u.flatten(0, 1).transpose(1, 2), v.flatten(0, 1), out=m.flatten(0, 1))


def _width_f4_write_dx(m, dx):
    g, _, c, h, w = dx.shape
    tw = triton.cdiv(w, 4)
    _write_dx[(triton.cdiv(c*h*tw, 128), g)](m, dx, c, h, w, tw, 128, num_warps=4)


def _launch_width_f4(dy, wt, dx, buffers):
    u, v, m = buffers
    cap = u.shape[0]
    for start in range(0, dx.shape[0], cap):
        sl = slice(start, start+cap)
        active = dx[sl].shape[0]
        ua, va, ma = u[:active], v[:active], m[:active]
        _width_f4_pack_weight(wt[sl], ua)
        _width_f4_pack_dy(dy[sl], va)
        _width_f4_multiply(ua, va, ma)
        _width_f4_write_dx(ma, dx[sl])

# Torch backward and group-first layout adapters

def _torch_conv_bwd(x, weight, grad_output, stride=1, padding=1,
                   dilation=1, groups=1):
    """The user-supplied baseline, unchanged except for formatting."""
    input_shape = x.shape
    weight_size = weight.shape
    grad_x = torch.nn.grad.conv2d_input(
        input_shape, weight, grad_output,
        stride, padding, dilation, groups,
    )
    grad_weight = torch.nn.grad.conv2d_weight(
        x, weight_size, grad_output,
        stride, padding, dilation, groups,
    )
    grad_bias = grad_output.sum(dim=(0, 2, 3))
    return grad_x, grad_weight, grad_bias


def _pack_torch_activation(t):
    g, b, c, h, w = t.shape
    return t.permute(1, 0, 2, 3, 4).reshape(b, g * c, h, w).contiguous()


def _prepare_torch_native(x, weight, grad_output):
    return (_pack_torch_activation(x), weight.flatten(0, 1).contiguous(),
            _pack_torch_activation(grad_output))


def _unpack_torch_native(outputs, g, c, o):
    dx, dw, db = outputs
    b, _, h, w = dx.shape
    return (
        dx.reshape(b, g, c, h, w).permute(1, 0, 2, 3, 4).contiguous(),
        dw.reshape(g, o, c, *dw.shape[-2:]).contiguous(),
        db.reshape(g, o).contiguous(),
    )


def _torch_group_first(x, weight, grad_output, stride, padding, dilation):
    g, _, c = x.shape[:3]
    o = weight.shape[1]
    xp, wp, dp = _prepare_torch_native(x, weight, grad_output)
    return _unpack_torch_native(
        _torch_conv_bwd(xp, wp, dp, stride, padding, dilation, g),
        g, c, o,
    )


def _torch_group_first_chunked(x, weight, grad_output, stride, padding, dilation,
                              groups_per_call):
    """Experimental group chunks, each containing complete independent groups."""
    if groups_per_call < 1:
        raise ValueError("groups_per_call must be positive")
    g, b, c, h, w = x.shape
    if groups_per_call >= g:
        return _torch_group_first(x, weight, grad_output, stride, padding, dilation)

    o = weight.shape[1]
    dx = x.new_empty(x.shape)
    dw = weight.new_empty(weight.shape)
    db = grad_output.new_empty((g, o))
    for first in range(0, g, groups_per_call):
        last = min(g, first + groups_per_call)
        chunk = slice(first, last)
        gc = last - first
        xp, wp, dp = _prepare_torch_native(x[chunk], weight[chunk], grad_output[chunk])
        gx, gw, gb = _torch_conv_bwd(xp, wp, dp, stride, padding, dilation, gc)
        del xp, wp, dp
        dx[chunk].copy_(gx.reshape(b, gc, c, h, w).transpose(0, 1))
        dw[chunk].copy_(gw.reshape_as(weight[chunk]))
        db[chunk].copy_(gb.reshape(gc, o))
        del gx, gw, gb
    return dx, dw, db

# Measured Conv3 overrides for contiguous group-first FP32
# Exact Conv3 rules measured in the v24 four-shard sweep.
#
# Each (B, G, C, H) point used square inputs, 128 output channels per
# group and 3x3/s1/p1/d1 in contiguous group-first FP32. Torch with both
# layout conversions was at least 3% faster in event and wall medians
# than the fastest tested custom candidate, won at least 5/7 paired CUDA
# event rounds, stayed within 5% CV and had lower peak extra allocation.

_TORCH_CASES_BY_GPU = {
    "NVIDIA RTX 5000 Ada Generation": frozenset({
        (1, 2, 3, 32),
        (1, 2, 3, 64),
        (1, 2, 3, 128),
        (1, 2, 128, 32),
        (1, 8, 3, 32),
        (1, 8, 3, 128),
        (1, 8, 128, 8),
        (1, 8, 128, 16),
        (100, 1, 3, 32),
        (100, 1, 3, 128),
        (100, 1, 128, 32),
    }),
    "NVIDIA RTX A5000": frozenset({
        (1, 1, 3, 64),
        (1, 1, 3, 128),
        (1, 1, 128, 16),
        (1, 1, 128, 32),
        (1, 16, 3, 128),
        (1, 32, 3, 64),
        (1, 32, 3, 128),
        (1, 32, 128, 8),
        (1, 32, 128, 16),
        (1, 32, 128, 32),
        (1, 64, 3, 128),
        (1, 64, 128, 8),
        (1, 64, 128, 16),
        (1, 64, 128, 32),
        (10, 2, 128, 8),
        (1000, 1, 3, 64),
        (1000, 1, 3, 128),
    }),
}
# v25's single-group ablation won all seven rounds at each of these four points.
_TORCH_G1_CASES_BY_GPU = {
    "NVIDIA RTX A5000": frozenset((10, g, 128, 64) for g in (2, 4, 16, 64)),
}
_CUSTOM_CASES_BY_GPU = {
    "NVIDIA RTX A5000": {(1, 16, 128, 16): "v21_bmm_dx"},
}

_MEASURED_CASES = frozenset().union(
    *_TORCH_CASES_BY_GPU.values(), *_TORCH_G1_CASES_BY_GPU.values(),
    *_CUSTOM_CASES_BY_GPU.values(),
)


def _measured_provider(gpu, case):
    if case in _TORCH_G1_CASES_BY_GPU.get(gpu, ()):
        return "torch_g1"
    if case in _TORCH_CASES_BY_GPU.get(gpu, ()):
        return "torch_group_first"
    return _CUSTOM_CASES_BY_GPU.get(gpu, {}).get(case)


WORKSPACE_LIMIT_BYTES = 128 * 1024**2
WG_TILE_V7 = (64, 64, 16)
WG_TILE = (64, 64, 32)
WG_WARPS = 8
F4_COMPONENTS = 36


def _output_hw(h,w,r,s,stride,padding,dilation):
    sh,sw=_pair(stride);ph,pw=_pair(padding);dh,dw=_pair(dilation)
    return ((h+2*ph-dh*(r-1)-1)//sh+1,
            (w+2*pw-dw*(s-1)-1)//sw+1)

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
def _gx_fold_group_bmm(Z,DX,GROUP,B:tl.constexpr,BSTART,
                       ACTIVE_B:tl.constexpr,C:tl.constexpr,
                       H:tl.constexpr,W:tl.constexpr,BLOCK:tl.constexpr):
    """Fold batch-major [B,C*9,H*W] columns for one group."""
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    hw:tl.constexpr=H*W
    pos=i%hw;ci=i//hw%C;local_batch=i//(C*hw)
    iy=pos//W;ix=pos%W
    valid=i<ACTIVE_B*C*hw
    acc=tl.full((BLOCK,),0.0,tl.float32)
    for rr in tl.static_range(3):
        for ss in tl.static_range(3):
            py=iy+1-rr;px=ix+1-ss
            zoff=((local_batch*C+ci)*9+rr*3+ss)*hw+py*W+px
            acc+=tl.load(Z+zoff,valid&(py>=0)&(py<H)&(px>=0)&(px<W),0.0)
    out=((GROUP*B+(BSTART+local_batch))*C+ci)*hw+pos
    tl.store(DX+out,acc,valid)


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
def _v20_dw_tile(X, D, DW, DB, tile, group, B:tl.constexpr, C:tl.constexpr,
                 O:tl.constexpr, H:tl.constexpr, W:tl.constexpr):
    # Each CTA owns 32 output channels and 32 consecutive filter coefficients.
    # The first coefficient tile also owns the 32 corresponding bias entries.
    cols:tl.constexpr = (C * 9 + 31) // 32
    oc = tile // cols * 32 + tl.arange(0, 32)
    crs = tile % cols * 32 + tl.arange(0, 32)
    ci, tap = crs // 9, crs % 9
    acc = tl.zeros((32, 32), tl.float32)
    bias = tl.zeros((32,), tl.float32)
    for block in range((B * H * W + 31) // 32):
        k = block * 32 + tl.arange(0, 32)
        batch, pos = k // (H * W), k % (H * W)
        valid = k < B * H * W
        dy = tl.load(D + ((group*B + batch[None,:])*O + oc[:,None])*H*W
                     + pos[None,:], (oc[:,None] < O) & valid[None,:], 0.)
        iy = (pos // W)[:,None] + (tap // 3)[None,:] - 1
        ix = (pos % W)[:,None] + (tap % 3)[None,:] - 1
        mask = (valid[:,None] & (ci[None,:] < C) & (iy >= 0) &
                (iy < H) & (ix >= 0) & (ix < W))
        xval = tl.load(X + ((group*B + batch[:,None])*C + ci[None,:])*H*W
                       + iy*W + ix, mask, 0.)
        acc = tl.dot(dy, xval, acc, input_precision="ieee")
        if tile % cols == 0:
            bias += tl.sum(dy, 1)
    tl.store(DW + (group*O + oc[:,None])*C*9 + crs[None,:], acc,
             (oc[:,None] < O) & (crs[None,:] < C*9))
    if tile % cols == 0:
        tl.store(DB + group*O + oc, bias, oc < O)


@triton.jit
def _v20_direct_grid(X, WEIGHT, D, DX, DW, DB, B:tl.constexpr, C:tl.constexpr,
                     O:tl.constexpr, H:tl.constexpr, W:tl.constexpr,
                     NX:tl.constexpr):
    tile = tl.program_id(0)
    if tile < NX:
        _gx_implicit_direct(D, WEIGHT, DX, B, C, O, H, W, 16, 32, 32)
    else:
        _v20_dw_tile(X, D, DW, DB, tile-NX, tl.program_id(1), B, C, O, H, W)


def launch_v20_direct(x, weight, d, dx, dw, db, joint=True):
    g, b, c, h, w = x.shape
    o = weight.shape[1]
    dx_tiles = triton.cdiv(c, 16) * triton.cdiv(b*h*w, 32) if joint else 0
    dw_tiles = triton.cdiv(o, 32) * triton.cdiv(c*9, 32)
    _v20_direct_grid[(dx_tiles + dw_tiles, g)](
        x, weight, d, dx, dw, db, b, c, o, h, w, dx_tiles,
        num_warps=4, num_stages=2)


def _v20_direct(x, weight, grad_output, stride, padding, dilation, groups, joint):
    g, _, _, o, _, _ = _validate(x, weight, grad_output, stride, padding, dilation, groups)
    with torch.cuda.device(x.device):
        x, weight, d = (t.contiguous() for t in (x, weight, grad_output))
        dx, dw, db = torch.empty_like(x), torch.empty_like(weight), x.new_empty((g,o))
        if not joint:
            launch_grad_x_implicit(d, weight, dx, 16, 32, 32, 4)
        launch_v20_direct(x, weight, d, dx, dw, db, joint)
    return dx, dw, db


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


# ResNet stride-2 kernels.  dX for a 3x3/pad1 convolution is split by the
# parity of the input coordinate.  Each launch consequently has a fixed set
# of contributing filter taps: 1, 2, 2, or 4 instead of nine masked taps.
@triton.jit
def _s2_gx_parity(D,WGT,DX,B:tl.constexpr,C:tl.constexpr,O:tl.constexpr,
                  H:tl.constexpr,WI:tl.constexpr,P:tl.constexpr,Q:tl.constexpr,
                  YPAR:tl.constexpr,XPAR:tl.constexpr,
                  BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    tile=tl.program_id(0);g=tl.program_id(1)
    hh:tl.constexpr=(H-YPAR+1)//2
    ww:tl.constexpr=(WI-XPAR+1)//2
    nall:tl.constexpr=B*hh*ww
    nn:tl.constexpr=(nall+BN-1)//BN
    ci=tile//nn*BM+tl.arange(0,BM)
    n=tile%nn*BN+tl.arange(0,BN)
    batch=n//(hh*ww);pos=n%(hh*ww)
    iy=(pos//ww)*2+YPAR;ix=(pos%ww)*2+XPAR
    kk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for rr in tl.static_range(3):
        if rr%2==(YPAR+1)%2:
            for ss in tl.static_range(3):
                if ss%2==(XPAR+1)%2:
                    py=(iy+1-rr)//2;px=(ix+1-ss)//2
                    for ob in range((O+BK-1)//BK):
                        oc=ob*BK+kk
                        weight=tl.load(
                            WGT+((g*O+oc[None,:])*C+ci[:,None])*9+(rr*3+ss),
                            (ci[:,None]<C)&(oc[None,:]<O),0.0)
                        dy=tl.load(
                            D+((g*B+batch[None,:])*O+oc[:,None])*(P*Q)+
                            py[None,:]*Q+px[None,:],
                            (oc[:,None]<O)&(n[None,:]<nall)&
                            (py[None,:]>=0)&(py[None,:]<P)&
                            (px[None,:]>=0)&(px[None,:]<Q),0.0)
                        acc=tl.dot(weight,dy,acc,input_precision="ieee")
    tl.store(DX+((g*B+batch[None,:])*C+ci[:,None])*(H*WI)+
             iy[None,:]*WI+ix[None,:],acc,
             (ci[:,None]<C)&(n[None,:]<nall))


@triton.jit
def _pw_s2_gx(D,WGT,DX,B:tl.constexpr,C:tl.constexpr,O:tl.constexpr,
              H:tl.constexpr,WI:tl.constexpr,P:tl.constexpr,Q:tl.constexpr,
              BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
              FILL_HOLES:tl.constexpr=False):
    """Compute only sampled dX locations; the launcher zeros the holes."""
    tile=tl.program_id(0);g=tl.program_id(1)
    nall:tl.constexpr=B*P*Q
    nn:tl.constexpr=(nall+BN-1)//BN
    ci=tile//nn*BM+tl.arange(0,BM)
    n=tile%nn*BN+tl.arange(0,BN)
    batch=n//(P*Q);pos=n%(P*Q);py=pos//Q;px=pos%Q
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for ob in range((O+BK-1)//BK):
        oc=ob*BK+kk
        weight=tl.load(WGT+(g*O+oc[None,:])*C+ci[:,None],
                       (ci[:,None]<C)&(oc[None,:]<O),0.0)
        dy=tl.load(D+((g*B+batch[None,:])*O+oc[:,None])*(P*Q)+pos[None,:],
                   (oc[:,None]<O)&(n[None,:]<nall),0.0)
        acc=tl.dot(weight,dy,acc,input_precision="ieee")
    out = DX+((g*B+batch[None,:])*C+ci[:,None])*(H*WI)+(py[None,:]*2)*WI+px[None,:]*2
    valid = (ci[:,None]<C)&(n[None,:]<nall)
    tl.store(out,acc,valid)
    if FILL_HOLES:
        has_x = px[None,:]*2+1 < WI
        has_y = py[None,:]*2+1 < H
        tl.store(out+1,0.0,valid&has_x)
        tl.store(out+WI,0.0,valid&has_y)
        tl.store(out+WI+1,0.0,valid&has_y&has_x)


@triton.jit
def _resnet_pack_dy(DY, PACKED, FIRST, B:tl.constexpr, O:tl.constexpr,
                    SPATIAL:tl.constexpr, K:tl.constexpr, BLOCK:tl.constexpr):
    group = tl.program_id(1)
    i = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    channel = i//K
    batch_pos = i%K
    batch = FIRST+batch_pos//SPATIAL
    pos = batch_pos%SPATIAL
    value = tl.load(DY+((group*B+batch)*O+channel)*SPATIAL+pos,channel<O,0.0)
    tl.store(PACKED+group*O*K+i,value,channel<O)


@triton.jit
def _resnet_pack_patches(X, COLS, FIRST, B:tl.constexpr, C:tl.constexpr,
                         H:tl.constexpr, W:tl.constexpr, P:tl.constexpr,
                         Q:tl.constexpr, R:tl.constexpr, PAD:tl.constexpr,
                         K:tl.constexpr, BLOCK:tl.constexpr,
                         STRIDE:tl.constexpr=2, CHANNELS:tl.constexpr=0,
                         CHANNEL_FIRST=0):
    NC:tl.constexpr = C if CHANNELS == 0 else CHANNELS
    group = tl.program_id(1)
    i = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    col = i//K
    channel = col//(R*R)
    pos = i%K
    batch = FIRST+pos//(P*Q)
    py = pos//Q%P
    px = pos%Q
    iy = STRIDE*py+col//R%R-PAD
    ix = STRIDE*px+col%R-PAD
    value = tl.load(X+((group*B+batch)*C+CHANNEL_FIRST+channel)*H*W+iy*W+ix,
                    (channel<NC)&(iy>=0)&(iy<H)&(ix>=0)&(ix<W),0.0)
    tl.store(COLS+group*NC*R*R*K+i,value,channel<NC)


@triton.jit
def _resnet_fold_dx(COLS, DX, FIRST, B:tl.constexpr, C:tl.constexpr,
                    H:tl.constexpr, W:tl.constexpr, P:tl.constexpr,
                    Q:tl.constexpr, R:tl.constexpr, PAD:tl.constexpr,
                    K:tl.constexpr, ACTIVE_B:tl.constexpr, BLOCK:tl.constexpr,
                    STRIDE:tl.constexpr=2, CHANNELS:tl.constexpr=0,
                    CHANNEL_FIRST=0, SPATIAL_MAJOR:tl.constexpr=False):
    # One thread owns each input position. Visit just the taps that can
    # reach it, including the stride-2 holes when R=1.
    NC:tl.constexpr = C if CHANNELS == 0 else CHANNELS
    group = tl.program_id(1)
    i = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    local_batch = i//(NC*H*W)
    channel = i//(H*W)%NC
    iy = i//W%H
    ix = i%W
    valid = local_batch<ACTIVE_B
    row_offset = (iy+PAD)%STRIDE
    col_offset = (ix+PAD)%STRIDE
    acc = tl.full((BLOCK,),0.0,tl.float32)
    for ry in tl.static_range((R+STRIDE-1)//STRIDE):
        rr = STRIDE*ry+row_offset
        py = (iy+PAD-rr)//STRIDE
        for rx in tl.static_range((R+STRIDE-1)//STRIDE):
            ss = STRIDE*rx+col_offset
            px = (ix+PAD-ss)//STRIDE
            row = channel*R*R+rr*R+ss
            col = local_batch*P*Q+py*Q+px
            if SPATIAL_MAJOR:
                offset = (group*K+col)*(NC*R*R)+row
            else:
                offset = (group*NC*R*R+row)*K+col
            mask = valid&(rr<R)&(ss<R)&(py>=0)&(py<P)&(px>=0)&(px<Q)
            acc += tl.load(COLS+offset,mask,0.0)
    out = ((group*B+FIRST+local_batch)*C+CHANNEL_FIRST+channel)*H*W+iy*W+ix
    tl.store(DX+out,acc,valid)


@triton.jit
def _layer4_fused_dx(DY, WEIGHT, DX, PARTIAL, B:tl.constexpr, C:tl.constexpr,
                     O:tl.constexpr, SPLIT:tl.constexpr, BM:tl.constexpr,
                     BN:tl.constexpr, BK:tl.constexpr):
    spatial_tile = tl.program_id(0)
    ci = tl.program_id(1)*BM + tl.arange(0,BM)
    group = tl.program_id(2)//SPLIT
    split = tl.program_id(2)%SPLIT
    n = spatial_tile*BN + tl.arange(0,BN)
    batch, pos = n//16, n%16
    iy, ix = pos//4, pos%4
    k = tl.arange(0,BK)
    blocks_per_split:tl.constexpr = (O+SPLIT*BK-1)//(SPLIT*BK)
    acc = tl.zeros((BM,BN),tl.float32)
    # Keep the nine taps for an output-channel block together. The dY values
    # and neighboring native OIHW coefficients can be reused through cache.
    for block in range(blocks_per_split*9):
        oc = (split*blocks_per_split+block//9)*BK+k
        tap = block%9
        py, px = iy+1-tap//3, ix+1-tap%3
        valid = (n<B*16)&(py>=0)&(py<4)&(px>=0)&(px<4)
        d = tl.load(DY+((group*B+batch[None,:])*O+oc[:,None])*16
                    +py[None,:]*4+px[None,:],
                    (oc[:,None]<O)&valid[None,:],0.0)
        w = tl.load(WEIGHT+((group*O+oc[None,:])*C+ci[:,None])*9+tap,
                    (ci[:,None]<C)&(oc[None,:]<O),0.0)
        acc = tl.dot(w,d,acc,input_precision="ieee")
    mask = (ci[:,None]<C)&(n[None,:]<B*16)
    off = (batch[None,:]*C+ci[:,None])*16+pos[None,:]
    if SPLIT == 1:
        tl.store(DX+group*B*C*16+off,acc,mask)
    else:
        tl.store(PARTIAL+(group*SPLIT+split)*B*C*16+off,acc,mask)


@triton.jit
def _layer4_finish_dx(PARTIAL, DX, COUNT:tl.constexpr, SPLIT:tl.constexpr,
                      BLOCK:tl.constexpr):
    group = tl.program_id(1)
    i = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    acc = tl.zeros((BLOCK,),tl.float32)
    for split in tl.static_range(SPLIT):
        acc += tl.load(PARTIAL+(group*SPLIT+split)*COUNT+i,i<COUNT,0.0)
    tl.store(DX+group*COUNT+i,acc,i<COUNT)


@triton.jit
def _tiny2_split_positions(tile):
    # Turn the last two spatial axes (row, column) into four tensors.
    left,right = tl.split(tile)
    top_left,bottom_left = tl.split(left)
    top_right,bottom_right = tl.split(right)
    return top_left,top_right,bottom_left,bottom_right


@triton.jit
def _tiny2_valid_dw(X, DY, DW, B:tl.constexpr, C:tl.constexpr, O:tl.constexpr,
                    MO:tl.constexpr, NI:tl.constexpr, BK:tl.constexpr):
    group = tl.program_id(1)
    nci:tl.constexpr = (C+NI-1)//NI
    first_oc = tl.program_id(0)//nci*MO
    first_ci = tl.program_id(0)%nci*NI
    out_idx = tl.arange(0,MO*4)
    in_idx = tl.arange(0,NI*4)
    out_channel = first_oc+out_idx//4
    in_channel = first_ci+in_idx//4
    kk = tl.arange(0,BK)
    acc = tl.zeros((MO*4,NI*4),tl.float32)
    # Four positions per channel become adjacent GEMM rows/columns. Only
    # real input/output position pairs are computed, with K=B.
    for block in range((B+BK-1)//BK):
        batch = block*BK+kk
        dout = tl.load(DY+((group*B+batch[None,:])*O+out_channel[:,None])*4
                       +out_idx[:,None]%4,
                       (out_channel[:,None]<O)&(batch[None,:]<B),0.0)
        xin = tl.load(X+((group*B+batch[:,None])*C+in_channel[None,:])*4
                      +in_idx[None,:]%4,
                      (in_channel[None,:]<C)&(batch[:,None]<B),0.0)
        acc = tl.dot(dout,xin,acc,input_precision="ieee")

    # [out_channel, in_channel, dy_row, dy_col, x_row, x_col].
    pairs = tl.permute(tl.reshape(acc,(MO,2,2,NI,2,2)),(0,3,1,2,4,5))
    x0,x1,x2,x3 = _tiny2_split_positions(pairs)
    d0x0,d1x0,d2x0,d3x0 = _tiny2_split_positions(x0)
    d0x1,d1x1,d2x1,d3x1 = _tiny2_split_positions(x1)
    d0x2,d1x2,d2x2,d3x2 = _tiny2_split_positions(x2)
    d0x3,d1x3,d2x3,d3x3 = _tiny2_split_positions(x3)

    # A pair maps to (input position - output position + 1). Reduce the
    # pairs that share a tap before writing OIHW, without a global 16-plane
    # intermediate. Interleave eight taps to coalesce the output stores.
    center = (d0x0+d1x1)+(d2x2+d3x3)
    taps_0246 = tl.join(tl.join(d3x0,center),tl.join(d2x1,d1x2))
    taps_1357 = tl.join(tl.join(d2x0+d3x1,d0x1+d2x3),
                        tl.join(d1x0+d3x2,d0x2+d1x3))
    taps = tl.reshape(tl.join(taps_0246,taps_1357),(MO,NI,8))
    oc = first_oc+tl.arange(0,MO)
    ci = first_ci+tl.arange(0,NI)
    off = ((group*O+oc[:,None])*C+ci[None,:])*9
    valid = (oc[:,None]<O)&(ci[None,:]<C)
    tl.store(DW+off[:,:,None]+tl.arange(0,8)[None,None,:],taps,valid[:,:,None])
    tl.store(DW+off+8,d0x3,valid)


@triton.jit
def _dense_4x4_dw(X, DY, DW, C:tl.constexpr, O:tl.constexpr,
                  BLOCK_O:tl.constexpr, BLOCK_CRS:tl.constexpr):
    """One output tile reduces all 16 positions and writes final OIHW dW."""
    group = tl.program_id(1)
    tile = tl.program_id(0)
    tiles_o:tl.constexpr = (O+BLOCK_O-1)//BLOCK_O
    oc = (tile%tiles_o)*BLOCK_O+tl.arange(0,BLOCK_O)
    crs = (tile//tiles_o)*BLOCK_CRS+tl.arange(0,BLOCK_CRS)
    pos = tl.arange(0,16)

    ci = crs//9
    tap = crs%9
    iy = pos[:,None]//4+tap[None,:]//3-1
    ix = pos[:,None]%4+tap[None,:]%3-1
    x_offset = (group*C+ci[None,:])*16+iy*4+ix
    x_mask = (ci[None,:]<C)&(iy>=0)&(iy<4)&(ix>=0)&(ix<4)
    x_values = tl.load(X+x_offset,x_mask,0.0)
    dy_values = tl.load(DY+(group*O+oc[:,None])*16+pos[None,:],
                        oc[:,None]<O,0.0)
    dw_values = tl.dot(dy_values,x_values,input_precision="ieee")
    dw_offset = (group*O+oc[:,None])*(C*9)+crs[None,:]
    tl.store(DW+dw_offset,dw_values,(oc[:,None]<O)&(ci[None,:]<C))


@triton.jit
def _direct_dw_split(X,D,DW,PDW,B:tl.constexpr,C:tl.constexpr,
                     O:tl.constexpr,H:tl.constexpr,WI:tl.constexpr,
                     P:tl.constexpr,Q:tl.constexpr,R:tl.constexpr,
                     S:tl.constexpr,STRIDE:tl.constexpr,PAD:tl.constexpr,
                     SPLIT:tl.constexpr,PARTIAL:tl.constexpr,
                     BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    """Native group-first dW = dY[O,K] @ sampled-X[K,C*R*S]."""
    pid=tl.program_id(0);gs=tl.program_id(1)
    g=gs//SPLIT;part=gs%SPLIT
    ncols:tl.constexpr=C*R*S
    nn:tl.constexpr=(ncols+BN-1)//BN
    oc=pid//nn*BM+tl.arange(0,BM)
    crs=pid%nn*BN+tl.arange(0,BN)
    ci=crs//(R*S);rs=crs%(R*S);rr=rs//S;ss=rs%S
    total:tl.constexpr=B*P*Q
    chunk:tl.constexpr=((total+SPLIT-1)//SPLIT+BK-1)//BK*BK
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for kb in range((chunk+BK-1)//BK):
        k=part*chunk+kb*BK+kk
        batch=k//(P*Q);pos=k%(P*Q);py=pos//Q;px=pos%Q
        valid=k<total
        dy=tl.load(D+((g*B+batch[None,:])*O+oc[:,None])*(P*Q)+pos[None,:],
                   (oc[:,None]<O)&valid[None,:],0.0)
        iy=py[:,None]*STRIDE+rr[None,:]-PAD
        ix=px[:,None]*STRIDE+ss[None,:]-PAD
        xv=tl.load(X+((g*B+batch[:,None])*C+ci[None,:])*(H*WI)+iy*WI+ix,
                   valid[:,None]&(crs[None,:]<ncols)&
                   (iy>=0)&(iy<H)&(ix>=0)&(ix<WI),0.0)
        acc=tl.dot(dy,xv,acc,input_precision="ieee")
    mask=(oc[:,None]<O)&(crs[None,:]<ncols)
    if PARTIAL:
        tl.store(PDW+((g*SPLIT+part)*O+oc[:,None])*ncols+crs[None,:],
                 acc,mask)
    else:
        tl.store(DW+(g*O+oc[:,None])*ncols+crs[None,:],acc,mask)


@triton.jit
def _direct_dw_finish(PDW,DW,O:tl.constexpr,NCOLS:tl.constexpr,
                      SPLIT:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);g=tl.program_id(1)
    valid=i<O*NCOLS;acc=tl.zeros((BLOCK,),tl.float32)
    for part in range(SPLIT):
        acc+=tl.load(PDW+((g*SPLIT+part)*O*NCOLS)+i,valid,0.0)
    tl.store(DW+g*O*NCOLS+i,acc,valid)


@triton.jit
def _stem7_pack_image_weights(W, PACKED, FIRST, N:tl.constexpr,
                              B:tl.constexpr, O:tl.constexpr,
                              BLOCK:tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    per_image:tl.constexpr = O*147
    group = (FIRST+i//per_image)//B
    valid = i<N*per_image
    v = tl.load(W+group*per_image+i%per_image,valid,0.0)
    tl.store(PACKED+i,v,valid)


@triton.jit
def _stem7_dx_phase(D, W, DX, B:tl.constexpr, O:tl.constexpr,
                    H:tl.constexpr, WI:tl.constexpr, P:tl.constexpr,
                    Q:tl.constexpr, Y_PHASE:tl.constexpr, X_PHASE:tl.constexpr,
                    BN:tl.constexpr, BK:tl.constexpr):
    # One thread tile owns a parity sub-grid. Each dY read feeds the three
    # RGB accumulators, with no channel padding or materialized tap columns.
    hh:tl.constexpr = (H+1-Y_PHASE)//2
    ww:tl.constexpr = (WI+1-X_PHASE)//2
    y0:tl.constexpr = 1-Y_PHASE
    x0:tl.constexpr = 1-X_PHASE
    rows:tl.constexpr = 4-y0
    cols:tl.constexpr = 4-x0
    oc_blocks:tl.constexpr = (O+BK-1)//BK
    g = tl.program_id(1)
    n = tl.program_id(0)*BN+tl.arange(0,BN)
    batch, pos = n//(hh*ww), n%(hh*ww)
    iy, ix = 2*(pos//ww)+Y_PHASE, 2*(pos%ww)+X_PHASE
    dy_base = ((g*B+batch[None,:])*O)*(P*Q)
    k = tl.arange(0,BK)
    a0 = tl.zeros((BK,BN),tl.float32)
    a1 = tl.zeros((BK,BN),tl.float32)
    a2 = tl.zeros((BK,BN),tl.float32)
    for block in range(rows*cols*oc_blocks):
        tap = block//oc_blocks
        rr = y0+2*(tap//cols)
        ss = x0+2*(tap%cols)
        oc = (block%oc_blocks)*BK+k
        py, px = (iy+3-rr)//2, (ix+3-ss)//2
        valid = (batch<B)&(py>=0)&(py<P)&(px>=0)&(px<Q)
        dy = tl.load(D+dy_base+oc[:,None]*(P*Q)+py[None,:]*Q+px[None,:],
                     (oc[:,None]<O)&valid[None,:],0.0)
        off = (g*O+oc)*147+rr*7+ss
        w0 = tl.load(W+off,oc<O,0.0)
        w1 = tl.load(W+off+49,oc<O,0.0)
        w2 = tl.load(W+off+98,oc<O,0.0)
        a0 = tl.fma(dy,w0[:,None],a0)
        a1 = tl.fma(dy,w1[:,None],a1)
        a2 = tl.fma(dy,w2[:,None],a2)
    out = DX+(g*B+batch)*3*H*WI+iy*WI+ix
    tl.store(out,tl.sum(a0,axis=0),batch<B)
    tl.store(out+H*WI,tl.sum(a1,axis=0),batch<B)
    tl.store(out+2*H*WI,tl.sum(a2,axis=0),batch<B)


@triton.jit
def _stem7_dx_parity(D, W, DX, B:tl.constexpr, O:tl.constexpr, H:tl.constexpr,
                     WI:tl.constexpr, P:tl.constexpr, Q:tl.constexpr,
                     BN:tl.constexpr, BK:tl.constexpr):
    phase = tl.program_id(2)
    if phase==0:
        _stem7_dx_phase(D,W,DX,B,O,H,WI,P,Q,0,0,BN,BK)
    elif phase==1:
        if WI>1:
            _stem7_dx_phase(D,W,DX,B,O,H,WI,P,Q,0,1,BN,BK)
    elif phase==2:
        if H>1:
            _stem7_dx_phase(D,W,DX,B,O,H,WI,P,Q,1,0,BN,BK)
    else:
        if H>1 and WI>1:
            _stem7_dx_phase(D,W,DX,B,O,H,WI,P,Q,1,1,BN,BK)


@triton.jit
def _stem7_dw_db_split(X, DY, DW, DB, PARTIAL, B:tl.constexpr,
                        O:tl.constexpr, H:tl.constexpr, WI:tl.constexpr,
                        P:tl.constexpr, Q:tl.constexpr, SPLIT:tl.constexpr,
                        BM:tl.constexpr, BN:tl.constexpr, BK:tl.constexpr):
    tiles_n:tl.constexpr = (147+BN-1)//BN
    blocks:tl.constexpr = (B*P*Q+BK-1)//BK
    tile = tl.program_id(0)
    gs = tl.program_id(1)
    g, part = gs//SPLIT, gs%SPLIT
    # Partition integral BK blocks so no split is empty, including odd tails.
    each:tl.constexpr = blocks//SPLIT
    extra:tl.constexpr = blocks%SPLIT
    lo = part*each+tl.minimum(part,extra)
    hi = lo+each+tl.where(part<extra,1,0)
    oc = tile//tiles_n*BM+tl.arange(0,BM)
    crs = tile%tiles_n*BN+tl.arange(0,BN)
    ci = crs//49
    ry, rx = (crs%49)//7, crs%7
    kk = tl.arange(0,BK)
    acc = tl.zeros((BM,BN),tl.float32)
    bias = tl.zeros((BM,),tl.float32)
    for block in range(lo,hi):
        k = block*BK+kk
        batch, pos = k//(P*Q), k%(P*Q)
        dy = tl.load(DY+((g*B+batch[None,:])*O+oc[:,None])*(P*Q)+pos[None,:],
                     (oc[:,None]<O)&(batch[None,:]<B),0.0)
        iy = 2*(pos[:,None]//Q)+ry[None,:]-3
        ix = 2*(pos[:,None]%Q)+rx[None,:]-3
        x = tl.load(X+((g*B+batch[:,None])*3+ci[None,:])*H*WI+iy*WI+ix,
                    (batch[:,None]<B)&(crs[None,:]<147)&
                    (iy>=0)&(iy<H)&(ix>=0)&(ix<WI),0.0)
        acc = tl.dot(dy,x,acc,input_precision="ieee")
        if tile%tiles_n==0:
            bias += tl.sum(dy,axis=1)
    valid = (oc[:,None]<O)&(crs[None,:]<147)
    if SPLIT==1:
        tl.store(DW+(g*O+oc[:,None])*147+crs[None,:],acc,valid)
        if tile%tiles_n==0:
            tl.store(DB+g*O+oc,bias,oc<O)
    else:
        tl.store(PARTIAL+(gs*O+oc[:,None])*148+crs[None,:],acc,valid)
        if tile%tiles_n==0:
            tl.store(PARTIAL+(gs*O+oc)*148+147,bias,oc<O)


@triton.jit
def _stem7_dw_db_finish(PARTIAL, DW, DB, O:tl.constexpr, SPLIT:tl.constexpr,
                         BLOCK:tl.constexpr, REDUCE:tl.constexpr):
    g = tl.program_id(1)
    i = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    acc = tl.zeros((REDUCE,BLOCK),tl.float32)
    rr = tl.arange(0,REDUCE)
    for base in range((SPLIT+REDUCE-1)//REDUCE):
        part = base*REDUCE+rr
        off = (g*SPLIT+part[:,None])*O*148+i[None,:]
        acc += tl.load(PARTIAL+off,(part[:,None]<SPLIT)&(i[None,:]<O*148),0.0)
    value = tl.sum(acc,axis=0)
    oc, col = i//148, i%148
    tl.store(DW+(g*O+oc)*147+col,value,(oc<O)&(col<147))
    tl.store(DB+g*O+oc,value,(oc<O)&(col==147))


def _validate_general(x,weight,d,stride,padding,dilation,groups):
    if x.ndim!=5 or weight.ndim!=5 or d.ndim!=5:
        raise ValueError("expected 5D group-first tensors")
    g,b,c,h,w=x.shape;o=weight.shape[1];r,s=weight.shape[-2:]
    sh,sw=_pair(stride);ph,pw=_pair(padding);dh,dw=_pair(dilation)
    p,q=_output_hw(h,w,r,s,(sh,sw),(ph,pw),(dh,dw))
    if (min(g,b,c,o,h,w,r,s,p,q)<1 or
            tuple(weight.shape)!=(g,o,c,r,s) or
            tuple(d.shape)!=(g,b,o,p,q)):
        raise ValueError("inconsistent group-first convolution shapes")
    if min(sh,sw,dh,dw)<1 or min(ph,pw)<0:
        raise ValueError("invalid stride, padding or dilation")
    if operator.index(groups) not in (1,g):
        raise ValueError("groups must be 1 or the group-first G dimension")
    if (g>65535 or max(x.numel(),weight.numel(),d.numel())>=2**31):
        raise ValueError("tensor indexing or CUDA grid limit exceeded")
    if any(not t.is_cuda or t.dtype!=torch.float32 or t.device!=x.device
           for t in (x,weight,d)):
        raise ValueError("expected FP32 tensors on the same CUDA device")
    return g,b,c,o,h,w,p,q,r,s

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


def launch_rgb_grad_weight_bias(x,d,dw,db,force_split=None,bm=32,bk=16):
    """Compact [O,K]@[K,27] RGB dW; dB shares the same dY reads."""
    g,b,c,h,w=x.shape;o=d.shape[2]
    if c!=3 or tuple(dw.shape)!=(g,o,3,3,3):
        raise ValueError("RGB dW requires x channels=3 and dw=[G,O,3,3,3]")
    bm=operator.index(bm)
    bk=operator.index(bk)
    if bk not in (16,32):
        raise ValueError("RGB dW BK must be 16 or 32")
    split=rgb_dw_split_count(x,d,force_split,bm)
    grid=(triton.cdiv(o,bm),g*split)
    warps=4 if bm==32 else 8
    if split==1:
        # Dummy partial pointers are never accessed under PARTIAL=False.
        _rgb_dw_db_flat[grid](x,d,dw,db,dw,db,b,o,h,w,split,False,bm,32,bk,
                              num_warps=warps,num_stages=2)
        return split
    pdw=x.new_empty((g,split,o,27))
    pdb=x.new_empty((g,split,o))
    _rgb_dw_db_flat[grid](x,d,dw,db,pdw,pdb,b,o,h,w,split,True,bm,32,bk,
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


def grad_x_group_bmm_workspace(x,max_bytes=WORKSPACE_LIMIT_BYTES):
    """Batch-major dX columns reused by one group and one B chunk."""
    _,b,c,h,w=x.shape
    per_batch=c*9*h*w*x.element_size()
    max_b=min(b,operator.index(max_bytes)//per_batch)
    if max_b<1:
        raise ValueError("one batch of group-BMM dX exceeds the workspace budget")
    return x.new_empty((max_b,c*9,h*w))


def launch_grad_x_group_bmm(d,weight,dx,workspace):
    """One strided-batched GEMM per group/B chunk, then one fused fold.

    B is the BMM batch dimension instead of a Python launch loop.  The
    transposed weight is broadcast with batch stride zero; dY is already
    contiguous as [B,O,H*W] inside each group, so no packing copy is needed.
    """
    g,b,o,h,w=d.shape;c=weight.shape[2];max_b=workspace.shape[0]
    if tuple(workspace.shape[1:])!=(c*9,h*w):
        raise ValueError("invalid group-BMM dX workspace")
    for group in range(g):
        ww=weight[group].reshape(o,c*9).transpose(0,1).unsqueeze(0)
        for bstart in range(0,b,max_b):
            active=min(max_b,b-bstart)
            aa=ww.expand(active,-1,-1)
            dd=d[group,bstart:bstart+active].reshape(active,o,h*w)
            torch.bmm(aa,dd,out=workspace[:active])
            count=active*c*h*w
            _gx_fold_group_bmm[(triton.cdiv(count,128),)](
                workspace,dx,group,b,bstart,active,c,h,w,128,num_warps=4)

def launch_grad_weight_bias_direct(x,d,dw,db,bm=32,bn=32,bk=16,warps=4):
    """Schedule-exposed direct dense 3x3 dW+dB launcher.

    v5 fixed all three tiles at 32x32x16.  The remaining Conv3 misses are
    small reductions where changing the output/reduction aspect ratio can be
    more important than changing the mathematical algorithm, so v17 exposes
    only a few fixed schedules for measurement (there is no autotune here).
    """
    g, b, c, h, w = x.shape
    o = d.shape[2]
    _dw_db_row[(triton.cdiv(c,bm)*triton.cdiv(o,bn),g,3)](
        x,d,dw,db,c,o,h,w,b,bm,bn,bk,
        num_warps=warps,num_stages=2)


def launch_grad_weight_bias_v5(x, d, dw, db):
    launch_grad_weight_bias_direct(x,d,dw,db,32,32,16,4)

def launch_grad_bias(d, db):
    g, b, o, h, w = d.shape
    _db_reduce[(o, g)](d, db, b, o, h*w, 1024, num_warps=4)


def launch_s2_grad_x_parity(d,weight,dx,bm=32,bn=64,bk=16,warps=4):
    """Four fixed-parity contractions for 3x3/stride2/pad1 dX."""
    g,b,o,p,q=d.shape;_,_,c,h,w=dx.shape
    if tuple(weight.shape)!=(g,o,c,3,3):
        raise ValueError("stride-2 dX requires weight=[G,O,C,3,3]")
    for yp in range(2):
        hh=(h-yp+1)//2
        for xp in range(2):
            ww=(w-xp+1)//2
            n=b*hh*ww
            grid=(triton.cdiv(c,bm)*triton.cdiv(n,bn),g)
            _s2_gx_parity[grid](
                d,weight,dx,b,c,o,h,w,p,q,yp,xp,bm,bn,bk,
                num_warps=warps,num_stages=2)


def launch_pointwise_s2_grad_x(d,weight,dx,bm=32,bn=64,bk=16,warps=4,*,fill_holes=False):
    """Zero unsampled positions, then contract and scatter sampled dX."""
    g,b,o,p,q=d.shape;_,_,c,h,w=dx.shape
    if tuple(weight.shape)!=(g,o,c,1,1):
        raise ValueError("pointwise dX requires weight=[G,O,C,1,1]")
    if not fill_holes:
        dx.zero_()
    grid=(triton.cdiv(c,bm)*triton.cdiv(b*p*q,bn),g)
    _pw_s2_gx[grid](d,weight,dx,b,c,o,h,w,p,q,bm,bn,bk,fill_holes,
                    num_warps=warps,num_stages=2)


def direct_dw_split_count(x,d,r,s,bm=32,bn=32,force_split=None,
                          max_bytes=WORKSPACE_LIMIT_BYTES,bk=16):
    """Conservative split-K rule bounded by the 128 MiB partial buffer."""
    g,b,c=x.shape[:3];o=d.shape[2];p,q=d.shape[-2:]
    one=g*o*c*r*s*x.element_size()
    # Never create a split with no BK=16 reduction block.  Besides wasting
    # launches, empty parts make tiny odd-shape correctness cases misleading.
    useful=triton.cdiv(b*p*q,operator.index(bk))
    max_split=max(1,min(16,useful,
                        operator.index(max_bytes)//max(1,one)))
    if force_split is not None:
        split=operator.index(force_split)
        if split<1 or split>max_split:
            raise ValueError(f"direct dW split must be in [1,{max_split}]")
        return split
    base=g*triton.cdiv(o,bm)*triton.cdiv(c*r*s,bn)
    occupancy=triton.cdiv(128,max(1,base))
    reduction=triton.cdiv(b*p*q,8192)
    return min(max_split,max(1,occupancy,reduction))


def launch_direct_grad_weight(x,d,dw,force_split=None,
                              bm=32,bn=32,bk=16):
    """Direct dW for the ResNet 3x3/s2 and 1x1/s2 families."""
    g,b,c,h,w=x.shape;o=d.shape[2];p,q=d.shape[-2:];r,s=dw.shape[-2:]
    if r!=s or r not in (1,3):
        raise ValueError("direct ResNet dW supports square 1x1 or 3x3")
    pad=1 if r==3 else 0
    split=direct_dw_split_count(x,d,r,s,bm,bn,force_split,bk=bk)
    ncols=c*r*s
    grid=(triton.cdiv(o,bm)*triton.cdiv(ncols,bn),g*split)
    warps=4 if bm==32 else 8
    if split==1:
        _direct_dw_split[grid](
            x,d,dw,dw,b,c,o,h,w,p,q,r,s,2,pad,split,False,
            bm,bn,bk,num_warps=warps,num_stages=2)
        return split
    partial=x.new_empty((g,split,o,ncols))
    _direct_dw_split[grid](
        x,d,dw,partial,b,c,o,h,w,p,q,r,s,2,pad,split,True,
        bm,bn,bk,num_warps=warps,num_stages=2)
    _direct_dw_finish[(triton.cdiv(o*ncols,256),g)](
        partial,dw,o,ncols,split,256,num_warps=4)
    return split


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


def joint_f4_plan_grouped(x,d,max_bytes=WORKSPACE_LIMIT_BYTES,groups=None):
    """Joint plan with an explicit number of concurrently packed groups.

    The original planner correctly minimizes memory, but once a full group no
    longer fits it always falls back to one group at a time.  On group-first
    workloads this can leave the GPU under-filled.  This diagnostic planner
    divides the same byte budget over ``groups`` groups so the benchmark can
    determine whether fewer host-side group chunks outweigh smaller tile
    chunks.  It never exceeds ``max_bytes``.
    """
    g,b,c,h,w=x.shape;o=d.shape[2]
    nt=b*((h+3)//4)*((w+3)//4)
    gc=min(g,g if groups is None else max(1,int(groups)))
    one_fixed=F4_COMPONENTS*c*o*4
    per_tile=F4_COMPONENTS*(o+max(c,o)+c)*4
    alias_bytes=gc*(one_fixed+per_tile*nt)
    if alias_bytes<=max_bytes:
        tc=nt;alias_du=True;fixed=one_fixed
    else:
        alias_du=False;fixed=2*one_fixed
        per_group=max_bytes//gc
        tc=min(nt,(per_group-fixed)//per_tile)
        if tc<1:
            raise ValueError("grouped joint F(4,3) state exceeds the workspace budget")
        if tc>=16:tc=tc//16*16
    cg=(g+gc-1)//gc;ct=(nt+tc-1)//tc
    return dict(groups=gc,tiles=tc,total_tiles=nt,alias_du=alias_du,
                group_chunks=cg,tile_chunks=ct,
                bytes=gc*(fixed+per_tile*tc),
                kernel_launches=cg*(2+5*ct)+1)


def balanced_f4_plan(x, d, max_bytes=WORKSPACE_LIMIT_BYTES):
    plan = joint_f4_plan(x, d, max_bytes)
    count = plan["tile_chunks"]
    if count == 1:
        return plan
    # Keep the same chunk count. Both GEMMs operate on the allocated width,
    # including masked tail tiles, so minimize that width before allocation.
    old = plan["tiles"]
    target = (plan["total_tiles"] + count - 1) // count
    width = min(old, (target + 15) // 16 * 16)
    bytes_per_tile = F4_COMPONENTS * (d.shape[2] + max(x.shape[2], d.shape[2]) + x.shape[2]) * 4
    return dict(plan, tiles=width,
                bytes=plan["bytes"] - plan["groups"]*(old-width)*bytes_per_tile)


def joint_f4_workspace_flat(x,d,max_bytes=WORKSPACE_LIMIT_BYTES,groups=None,*,plan=None):
    """Allocate every live joint-F4 buffer from one caching-allocator block.

    The views preserve exactly the compact layouts consumed by the existing
    kernels and BMM calls.  Compared with ``joint_f4_workspace`` this changes
    neither arithmetic nor workspace bytes; it removes four allocator calls
    from the latency-sensitive public API.
    """
    p=plan if plan is not None else (joint_f4_plan(x,d,max_bytes) if groups is None else
                                    joint_f4_plan_grouped(x,d,max_bytes,groups))
    gc,tc=p["groups"],p["tiles"]
    c,o=x.shape[2],d.shape[2]
    shapes=[(gc,F4_COMPONENTS,o,tc),
            (gc,F4_COMPONENTS,max(c,o),tc),
            (gc,F4_COMPONENTS,c,tc),
            (gc,F4_COMPONENTS,c,o)]
    if not p["alias_du"]:
        shapes.append((gc,F4_COMPONENTS,o,c))
    sizes=[math.prod(shape) for shape in shapes]
    flat=x.new_empty((sum(sizes),))
    views=[];offset=0
    for shape,size in zip(shapes,sizes):
        views.append(flat[offset:offset+size].view(shape));offset+=size
    e,shared,mid,uw=views[:4]
    du=uw if p["alias_du"] else views[4]
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


def launch_f4_dx_gemm_bmm(u,v,m,active_groups,c=None,o=None):
    """Batched-GEMM dX contraction directly on the native F4 buffers."""
    if c is None:c=m.shape[2]
    if o is None:o=v.shape[2]
    nt=m.shape[-1];batches=active_groups*F4_COMPONENTS
    aa=u.as_strided((batches,c,o),(c*o,o,1))
    bb=v.as_strided((batches,o,nt),(o*nt,nt,1))
    out=m.as_strided((batches,c,nt),(c*nt,nt,1))
    torch.bmm(aa,bb,out=out)


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


def launch_grad_weight_f4(x,d,dw,workspace,gemm=None):
    if gemm is None:gemm=launch_f4_gemm
    v,e,u=workspace;g,b,_,h,w=x.shape
    total=b*((h+3)//4)*((w+3)//4);gc,tc=v.shape[0],v.shape[-1]
    for base in range(0,g,gc):
        end=min(g,base+gc);xx,dd=x[base:end],d[base:end]
        for start in range(0,total,tc):
            launch_f4_pack_x(xx,v,start);launch_f4_pack_d(dd,e,start)
            gemm(v,e,u,end-base,add=start!=0)
        launch_f4_inverse(u,dw[base:end])


def launch_grad_weight_bias_f4(x,d,dw,db,workspace,gemm=None):
    launch_grad_weight_f4(x,d,dw,workspace,gemm)
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


def launch_joint_f4(x,weight,d,dx,dw,db,workspace,dx_gemm=None,dw_gemm=None,*,compact_tail=False):
    e,shared,mid,uw,du=workspace
    g,b,c,h,w=x.shape;o=d.shape[2]
    total=b*((h+3)//4)*((w+3)//4);gc,tc=e.shape[0],e.shape[-1]
    full = (e, shared, mid)
    tail = full
    if compact_tail:
        tail_width = min(tc, triton.cdiv(total % tc, 16) * 16)
        if 0 < tail_width < tc:
            # Reinterpret contiguous prefixes with compact component strides.
            # Slicing the last axis would retain the original component strides.
            tail = tuple(t.view(-1)[:t.numel() // tc * tail_width]
                         .view(*t.shape[:-1], tail_width) for t in full)
    for base in range(0,g,gc):
        end=min(g,base+gc);active=end-base
        xx,ww,dd=x[base:end],weight[base:end],d[base:end]
        outx,outw=dx[base:end],dw[base:end]
        launch_f4_pack_weight_dx(ww,uw)
        for start in range(0,total,tc):
            chunk_e, chunk_shared, chunk_mid = tail if total - start < tc else full
            # One launch exposes both dY transforms to compiler CSE.
            launch_f4_joint_pack_d(dd,chunk_shared,chunk_e,start)
            if dx_gemm is None:
                launch_f4_dx_gemm(uw,chunk_shared,chunk_mid,active,c,o)
            else:
                dx_gemm(uw,chunk_shared,chunk_mid,active,c,o)
            launch_f4_dx_output(chunk_mid,outx,start)
            # dV is dead after output transform; overwrite its backing buffer
            # with B^T X B before the weight-gradient GEMM.
            launch_f4_pack_x(xx,chunk_shared,start)
            if dw_gemm is None:
                launch_f4_gemm_explicit(chunk_shared,chunk_e,du,active,c,o,add=start!=0)
            else:
                dw_gemm(chunk_shared,chunk_e,du,active,c,o,start!=0)
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


# With B=1 the target layers have exactly 16 spatial tiles. Each dot
# contracts the complete tile dimension and each CTA owns its final weights.
@triton.jit
def _v21_product(E, V, group, component: tl.constexpr, components: tl.constexpr,
                 oc, ci, O: tl.constexpr, C: tl.constexpr, NT: tl.constexpr):
    k = tl.arange(0, 16)
    ebase = E + (group * components + component) * O * NT
    vbase = V + (group * components + component) * C * NT
    left = tl.load(ebase + oc[:, None] * NT + k[None, :],
                   (oc[:, None] < O) & (k[None, :] < NT), 0.)
    right = tl.load(vbase + ci[None, :] * NT + k[:, None],
                    (ci[None, :] < C) & (k[:, None] < NT), 0.)
    return tl.dot(left, right, input_precision="ieee")


@triton.jit
def _v21_inverse_f2(a, b, c, d):
    mid = 0.5 * (b + c)
    return a + mid, 0.5 * (b - c), mid + d


@triton.jit
def _v21_inverse_f4(a, b, c, d, e, f):
    bc = (b + c) / 6.
    de = d + e
    return 0.25 * a - bc + de / 24., (c - b) / 6. + (d - e) / 12., de / 6. - bc + f


@triton.jit
def _v21_filter_row(E, V, group, row: tl.constexpr, alpha: tl.constexpr,
                    oc, ci, O: tl.constexpr, C: tl.constexpr, NT: tl.constexpr):
    base: tl.constexpr = row * alpha
    count: tl.constexpr = alpha * alpha
    a = _v21_product(E, V, group, base, count, oc, ci, O, C, NT)
    b = _v21_product(E, V, group, base+1, count, oc, ci, O, C, NT)
    c = _v21_product(E, V, group, base+2, count, oc, ci, O, C, NT)
    d = _v21_product(E, V, group, base+3, count, oc, ci, O, C, NT)
    if alpha == 4:
        return _v21_inverse_f2(a, b, c, d)
    else:
        e = _v21_product(E, V, group, base+4, count, oc, ci, O, C, NT)
        f = _v21_product(E, V, group, base+5, count, oc, ci, O, C, NT)
        return _v21_inverse_f4(a, b, c, d, e, f)


@triton.jit
def _v21_fused_dw(V, E, DW, C: tl.constexpr, O: tl.constexpr, NT: tl.constexpr,
                  ALPHA: tl.constexpr, BC: tl.constexpr, BO: tl.constexpr):
    tile = tl.program_id(0)
    group = tl.program_id(1)
    nc: tl.constexpr = (C + BC - 1) // BC
    ci = (tile % nc) * BC + tl.arange(0, BC)
    oc = (tile // nc) * BO + tl.arange(0, BO)
    a0, a1, a2 = _v21_filter_row(E, V, group, 0, ALPHA, oc, ci, O, C, NT)
    b0, b1, b2 = _v21_filter_row(E, V, group, 1, ALPHA, oc, ci, O, C, NT)
    c0, c1, c2 = _v21_filter_row(E, V, group, 2, ALPHA, oc, ci, O, C, NT)
    d0, d1, d2 = _v21_filter_row(E, V, group, 3, ALPHA, oc, ci, O, C, NT)
    if ALPHA == 4:
        w00, w10, w20 = _v21_inverse_f2(a0, b0, c0, d0)
        w01, w11, w21 = _v21_inverse_f2(a1, b1, c1, d1)
        w02, w12, w22 = _v21_inverse_f2(a2, b2, c2, d2)
    else:
        e0, e1, e2 = _v21_filter_row(E, V, group, 4, ALPHA, oc, ci, O, C, NT)
        f0, f1, f2 = _v21_filter_row(E, V, group, 5, ALPHA, oc, ci, O, C, NT)
        w00, w10, w20 = _v21_inverse_f4(a0, b0, c0, d0, e0, f0)
        w01, w11, w21 = _v21_inverse_f4(a1, b1, c1, d1, e1, f1)
        w02, w12, w22 = _v21_inverse_f4(a2, b2, c2, d2, e2, f2)
    out = DW + ((group * O + oc[:, None]) * C + ci[None, :]) * 9
    valid = (oc[:, None] < O) & (ci[None, :] < C)
    tl.store(out + 0, w00, valid)
    tl.store(out + 1, w01, valid)
    tl.store(out + 2, w02, valid)
    tl.store(out + 3, w10, valid)
    tl.store(out + 4, w11, valid)
    tl.store(out + 5, w12, valid)
    tl.store(out + 6, w20, valid)
    tl.store(out + 7, w21, valid)
    tl.store(out + 8, w22, valid)


def v21_tile_size(x):
    _, b, _, h, w = x.shape
    if b != 1 or max(h, w) > 16:
        raise ValueError("v21 small Winograd requires B=1 and 2 <= H,W <= 16")
    return 2 if max(h, w) <= 8 else 4


def v21_dw_workspace(x, d, max_bytes=WORKSPACE_LIMIT_BYTES):
    g, _, c, h, w = x.shape
    o = d.shape[2]
    tile = v21_tile_size(x)
    count = (tile + 2)**2
    nt = triton.cdiv(h, tile) * triton.cdiv(w, tile)
    gc = min(g, max_bytes // (count * (c+o) * nt * x.element_size()))
    if gc < 1:
        raise ValueError("one group's packed Winograd inputs exceed the workspace budget")
    return x.new_empty((gc, count, c, nt)), d.new_empty((gc, count, o, nt))


def launch_v21_fused_dw(v, e, dw):
    active, o, c = dw.shape[:3]
    alpha = 4 if v.shape[1] == 16 else 6
    _v21_fused_dw[(triton.cdiv(c, 16) * triton.cdiv(o, 16), active)](
        v, e, dw, c, o, v.shape[-1], alpha, 16, 16, num_warps=4, num_stages=1)


def launch_v21_weight(x, d, dw, workspace):
    v, e = workspace
    pack_x, pack_d = ((launch_pack_x, launch_pack_d) if v.shape[1] == 16
                      else (launch_f4_pack_x, launch_f4_pack_d))
    for start in range(0, x.shape[0], v.shape[0]):
        groups = slice(start, start + v.shape[0])
        pack_x(x[groups], v)
        pack_d(d[groups], e)
        launch_v21_fused_dw(v, e, dw[groups])


def launch_v21_weight_bias(x, d, dw, db, workspace):
    launch_v21_weight(x, d, dw, workspace)
    launch_grad_bias(d, db)


@triton.jit
def _v22_f4_rows(V, E, T, C: tl.constexpr, O: tl.constexpr, NT: tl.constexpr,
                 BC: tl.constexpr, BO: tl.constexpr):
    tile = tl.program_id(0)
    group = tl.program_id(1)
    row = tl.program_id(2)
    nc: tl.constexpr = (C + BC - 1) // BC
    oc = (tile // nc) * BO + tl.arange(0, BO)
    ci = (tile % nc) * BC + tl.arange(0, BC)
    # Rebase both inputs to one row of six [channels, NT] planes.
    # The v21 helper then addresses components 0..5 relative to these bases.
    first = (group * 6 + row) * 6
    t0, t1, t2 = _v21_filter_row(E + first*O*NT, V + first*C*NT,
                                  0, 0, 6, oc, ci, O, C, NT)
    # [group, row, out_channel, in_channel, filter_column]
    dst = T + (((group*6 + row)*O + oc[:, None])*C + ci[None, :])*3
    valid = (oc[:, None] < O) & (ci[None, :] < C)
    tl.store(dst, t0, valid)
    tl.store(dst+1, t1, valid)
    tl.store(dst+2, t2, valid)


@triton.jit
def _v22_f4_finish(T, DW, C: tl.constexpr, O: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(1)
    j = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    plane: tl.constexpr = O*C*3
    valid = j < plane
    src = T + group*6*plane + j
    a = tl.load(src, valid, 0.)
    b = tl.load(src+plane, valid, 0.)
    c = tl.load(src+2*plane, valid, 0.)
    d = tl.load(src+3*plane, valid, 0.)
    e = tl.load(src+4*plane, valid, 0.)
    f = tl.load(src+5*plane, valid, 0.)
    top, mid, bottom = _v21_inverse_f4(a, b, c, d, e, f)
    # j spans [out_channel, in_channel, filter_column].
    dst = DW + (group*O*C + j//3)*9 + j%3
    tl.store(dst, top, valid)
    tl.store(dst+3, mid, valid)
    tl.store(dst+6, bottom, valid)


def v22_dw_plan(x, d, max_bytes=WORKSPACE_LIMIT_BYTES):
    g, b, c, h, w = x.shape
    o = d.shape[2]
    nt = triton.cdiv(h, 4) * triton.cdiv(w, 4)
    if b != 1 or min(h, w) < 2 or max(h, w) > 16:
        raise ValueError("v22 F(4,3) requires B=1 and 2 <= H,W <= 16")
    per_group = (36*(c+o)*nt + 18*o*c) * x.element_size()
    gc = min(g, max_bytes // per_group)
    if gc < 1:
        raise ValueError("one group's packed inputs and row products exceed the workspace budget")
    return dict(groups=gc, tiles=nt, group_chunks=triton.cdiv(g, gc),
                bytes=gc*per_group, intermediate_bytes=gc*18*o*c*x.element_size())


def v22_dw_workspace(x, d, max_bytes=WORKSPACE_LIMIT_BYTES):
    plan = v22_dw_plan(x, d, max_bytes)
    gc, nt = plan["groups"], plan["tiles"]
    c, o = x.shape[2], d.shape[2]
    return (x.new_empty((gc, 36, c, nt)), d.new_empty((gc, 36, o, nt)),
            x.new_empty((gc, 6, o, c, 3)))


def launch_v22_rows(v, e, t, active_groups):
    _, _, o, c, _ = t.shape
    grid = (triton.cdiv(c, 32) * triton.cdiv(o, 32), active_groups, 6)
    _v22_f4_rows[grid](v, e, t, c, o, v.shape[-1], 32, 32, num_warps=4, num_stages=1)


def launch_v22_finish(t, dw):
    g, o, c = dw.shape[:3]
    _v22_f4_finish[(triton.cdiv(o*c*3, 128), g)](t, dw, c, o, 128, num_warps=4)


def launch_v22_weight_bias(x, d, dw, db, workspace):
    v, e, t = workspace
    gc = v.shape[0]
    for start in range(0, x.shape[0], gc):
        groups = slice(start, start + gc)
        out = dw[groups]
        launch_f4_pack_x(x[groups], v)
        launch_f4_pack_d(d[groups], e)
        launch_v22_rows(v, e, t, out.shape[0])
        launch_v22_finish(t, out)
    launch_grad_bias(d, db)


def _run_v21(x, weight, d, stride, padding, dilation, groups, algorithm, dx_schedule=None):
    g, _, _, o, _, _ = _validate(x, weight, d, stride, padding, dilation, groups)
    tile = v21_tile_size(x)
    if algorithm == "v23_f4_bmm_dw" and tile != 4:
        raise ValueError("v23_f4_bmm_dw requires the small F(4,3) family (8 < max(H,W) <= 16)")
    with torch.cuda.device(x.device):
        x, weight, d = (t.contiguous() for t in (x, weight, d))
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)
        db = x.new_empty((g, o))
        if dx_schedule == "width_f4":
            _launch_width_f4(d, weight, dx, _width_f4_workspace(d, dx, WORKSPACE_LIMIT_BYTES))
        else:
            if dx_schedule == "stream4":
                # Allocate only four groups of columns. The producer and fold
                # consume each chunk before the next group reuses this storage.
                cols = grad_x_workspace(d[:4], dx[:4])
            else:
                cols = grad_x_workspace(d, dx)
            dx_fn = launch_grad_x if algorithm == "v21_fused_dw" else launch_grad_x_bmm
            dx_fn(d, weight, dx, cols)
            del cols
        if algorithm == "v22_row_dw":
            work = v22_dw_workspace(x, d)
            launch_v22_weight_bias(x, d, dw, db, work)
        elif algorithm in ("v21_fused_dw", "v21_bmm_dx_fused_dw"):
            work = v21_dw_workspace(x, d)
            launch_v21_weight_bias(x, d, dw, db, work)
        elif tile == 2:
            work = grad_weight_workspace(x, d)
            launch_grad_weight_bias(x, d, dw, db, work, launch_winograd_gemm_v7)
        else:
            work = grad_weight_f4_workspace(x, d)
            gemm = launch_f4_gemm_bmm if algorithm == "v23_f4_bmm_dw" else launch_f4_gemm
            launch_grad_weight_bias_f4(x, d, dw, db, work, gemm)
    return dx, dw, db


def _run_joint_variant(x,weight,grad_output,stride,padding,dilation,groups,
                       max_bytes=WORKSPACE_LIMIT_BYTES,packed_groups=None,
                       dx_bmm=False,balanced=False,compact_tail=False):
    """Joint F4 runner with a flat allocation and optional compact tail."""
    g,_,_,o,_,_=_validate(
        x,weight,grad_output,stride,padding,dilation,groups)
    with torch.cuda.device(x.device):
        x,weight,d=(t.contiguous() for t in (x,weight,grad_output))
        dx,dw,db=torch.empty_like(x),torch.empty_like(weight),x.new_empty((g,o))
        plan=balanced_f4_plan(x,d,max_bytes) if balanced else None
        work=joint_f4_workspace_flat(x,d,max_bytes,packed_groups,plan=plan)
        launch_joint_f4(
            x,weight,d,dx,dw,db,work,
            dx_gemm=launch_f4_dx_gemm_bmm if dx_bmm else None,
            dw_gemm=launch_f4_gemm_bmm,compact_tail=compact_tail)
    return dx,dw,db


def _run_small_direct(x,weight,grad_output,stride,padding,dilation,groups,
                      dx_kind="implicit32",workspace=None,
                      dw_schedule=(32,32,16,4)):
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
        elif dx_kind=="implicit16_bk32":
            launch_grad_x_implicit(d,weight,dx,16,32,32,4)
        elif dx_kind=="implicit16x64":
            launch_grad_x_implicit(d,weight,dx,16,64,16,4)
        elif dx_kind=="implicit16x64_bk32":
            launch_grad_x_implicit(d,weight,dx,16,64,32,4)
        elif dx_kind=="chunk32":
            if workspace is None:workspace=grad_x_batch_chunk_workspace(x)
            launch_grad_x_batch_chunked(d,weight,dx,workspace,32,64,16,4)
        elif dx_kind=="chunk16":
            if workspace is None:workspace=grad_x_batch_chunk_workspace(x)
            launch_grad_x_batch_chunked(d,weight,dx,workspace,16,64,16,4)
        elif dx_kind=="bmm":
            if workspace is None:workspace=grad_x_workspace(d,dx)
            launch_grad_x_bmm(d,weight,dx,workspace)
        elif dx_kind=="group_bmm":
            if workspace is None:workspace=grad_x_group_bmm_workspace(x)
            launch_grad_x_group_bmm(d,weight,dx,workspace)
        else:
            raise ValueError("unknown small direct dX schedule")
        launch_grad_weight_bias_direct(x,d,dw,db,*dw_schedule)
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


def _run_v17_rgb_direct(x,weight,grad_output,stride,padding,dilation,groups,
                        block_n=64,block_k=32,dw_bm=32):
    """No-workspace RGB probe for the launch/allocator-bound Conv3 points."""
    g,_,c,o,_,_=_validate(x,weight,grad_output,stride,padding,dilation,groups)
    if c!=3:
        raise ValueError("v17 RGB direct candidates require C=3")
    with torch.cuda.device(x.device):
        x,weight,d=(t.contiguous() for t in (x,weight,grad_output))
        dx,dw,db=torch.empty_like(x),torch.empty_like(weight),x.new_empty((g,o))
        launch_rgb_grad_x_direct3(
            d,weight,dx,block_n,block_k,8 if block_n>=128 else 4)
        launch_rgb_grad_weight_bias(x,d,dw,db,None,dw_bm,32)
    return dx,dw,db


def _run_v14_rgb(x,weight,grad_output,stride,padding,dilation,groups,
                 dw_kind="compact32",force_split=None,bk=16,*,group_bmm_dx=False):
    """Batch-chunked dX plus one of the experimental RGB dW families."""
    g,b,c,o,_,_=_validate(x,weight,grad_output,stride,padding,dilation,groups)
    if c!=3:
        raise ValueError("v14 RGB candidates require C=3")
    with torch.cuda.device(x.device):
        x,weight,d=(t.contiguous() for t in (x,weight,grad_output))
        dx,dw,db=torch.empty_like(x),torch.empty_like(weight),x.new_empty((g,o))
        if group_bmm_dx and b == 1:
            gx_work=grad_x_workspace(d,dx)
            launch_grad_x_bmm(d,weight,dx,gx_work)
        elif group_bmm_dx:
            gx_work=grad_x_group_bmm_workspace(x)
            launch_grad_x_group_bmm(d,weight,dx,gx_work)
        else:
            gx_work=grad_x_batch_chunk_workspace(x)
            launch_grad_x_batch_chunked(d,weight,dx,gx_work)
        del gx_work
        if dw_kind=="compact32":
            launch_rgb_grad_weight_bias(x,d,dw,db,force_split,bm=32,bk=bk)
        elif dw_kind=="compact64":
            launch_rgb_grad_weight_bias(x,d,dw,db,force_split,bm=64,bk=bk)
        elif dw_kind=="f4_bmm":
            gw_work=grad_weight_f4_workspace(x,d)
            launch_grad_weight_bias_f4(
                x,d,dw,db,gw_work,gemm=launch_f4_gemm_bmm)
        else:
            raise ValueError("unknown v14 RGB dW family")
    return dx,dw,db


def _run_resnet_s2(x,weight,grad_output,stride,padding,dilation,groups,
                   dx_bm=32,dx_bn=64,dw_bm=64,dw_bn=32,dw_bk=16,
                   force_split=None,*,shortcut_fill_holes=False):
    """Direct native-layout first backward for ResNet downsampling layers."""
    g,b,c,o,h,w,p,q,r,s=_validate_general(
        x,weight,grad_output,stride,padding,dilation,groups)
    expected_pad=1 if r==3 else 0
    if (r!=s or r not in (1,3) or _pair(stride)!=(2,2) or
            _pair(padding)!=(expected_pad,expected_pad) or
            _pair(dilation)!=(1,1) or (shortcut_fill_holes and r!=1)):
        raise ValueError("v16 ResNet family requires 3x3/s2/p1 or 1x1/s2/p0")
    with torch.cuda.device(x.device):
        x,weight,d=(t.contiguous() for t in (x,weight,grad_output))
        dx,dw,db=torch.empty_like(x),torch.empty_like(weight),x.new_empty((g,o))
        if r==3:
            launch_s2_grad_x_parity(
                d,weight,dx,dx_bm,dx_bn,16,4 if dx_bm==32 else 8)
        else:
            launch_pointwise_s2_grad_x(
                d,weight,dx,dx_bm,dx_bn,16,4 if dx_bm==32 else 8,
                fill_holes=shortcut_fill_holes)
        launch_direct_grad_weight(
            x,d,dw,force_split,dw_bm,dw_bn,dw_bk)
        launch_grad_bias(d,db)
    return dx,dw,db


def _resnet_bmm_chunk_shape(g,b,columns,o,p,q,element_size,max_bytes,*,balanced=False):
    spatial_bytes = p*q*element_size
    # A one-image chunk reads dY through a view, including the case where a
    # single image's columns fit but allocating a dY copy would not.
    capacity = max_bytes//((columns+o)*spatial_bytes)
    if capacity < 1:
        if not balanced or max_bytes < columns*spatial_bytes:
            raise ValueError("one image's packed columns exceed the workspace budget")
        capacity = 1
    groups = min(g,max(1,capacity//b))
    batches = min(b,capacity//groups)
    if balanced:
        # Shrink a max-sized chunk when it would leave a very short tail.
        # Keep the number of B/G chunks unchanged.
        groups = triton.cdiv(g,triton.cdiv(g,groups))
        batches = triton.cdiv(b,triton.cdiv(b,batches))
        if batches == 1:
            groups = min(g,max_bytes//(columns*spatial_bytes))
    return groups,batches


def _layer4_dx_schedule(g,b,o):
    # B=1 has just 16 spatial positions. Split its channel reduction to
    # expose parallel work without creating per-tap output columns.
    split = (4 if g<=4 else 2) if b==1 and o>=128 else 1
    return split,32,16 if b==1 else 32,32


def _launch_layer4_fused_dx(weight,d,dx,max_bytes,stage_end):
    g,b,c = dx.shape[:3]
    o = d.shape[2]
    split,bm,bn,bk = _layer4_dx_schedule(g,b,o)
    # CUDA grid Y/Z have a 65535 limit. Keep both the partial buffer and
    # its offsets bounded for callers beyond the benchmark shapes.
    if (g*split>65535 or triton.cdiv(c,bm)>65535 or
            (split>1 and dx.numel()*split*dx.element_size()>max_bytes)):
        raise ValueError("fused 4x4 dX exceeds the CUDA grid or split workspace budget")
    partial = dx.new_empty((g,split,b,c,16)) if split>1 else dx
    if stage_end is not None:
        stage_end("dx_workspace")
    _layer4_fused_dx[(triton.cdiv(b*16,bn),triton.cdiv(c,bm),g*split)](
        d,weight,dx,partial,b,c,o,split,bm,bn,bk,num_warps=4,num_stages=2)
    if stage_end is not None:
        stage_end("dx_fused")
    if split>1:
        block = 256
        _layer4_finish_dx[(triton.cdiv(b*c*16,block),g)](
            partial,dx,b*c*16,split,block,num_warps=4)
        if stage_end is not None:
            stage_end("dx_reduce")


def _launch_resnet_bmm(x,weight,d,dx,dw,*,packed_dw,max_bytes=WORKSPACE_LIMIT_BYTES,
                       stride=2,channel_block=None,balanced=False,stage_end=None,
                       dx_mode="columns"):
    """Share packed dY and reuse one column buffer across dX and dW."""
    g,b,c,h,w = x.shape
    o,p,q = d.shape[2:]
    r = weight.shape[-1]
    pad = r//2
    stripe = c if channel_block is None else min(c,channel_block)
    taps = r*r
    columns = c*taps
    spatial = p*q
    gc,bc = _resnet_bmm_chunk_shape(g,b,stripe*taps,o,p,q,x.element_size(),max_bytes,
                                   balanced=balanced)
    cols_storage = x.new_empty((gc*stripe*taps*bc*spatial,))
    dy_storage = d.new_empty((gc*o*bc*spatial,)) if bc>1 else None
    if stage_end is not None:
        stage_end("workspace")
    for group_start in range(0,g,gc):
        group_slice = slice(group_start,group_start+gc)
        xs,ds,dxs = x[group_slice],d[group_slice],dx[group_slice]
        active_g = xs.shape[0]
        w_flat = weight[group_slice].view(active_g,o,columns)
        dw_flat = dw[group_slice].view(active_g,o,columns)
        for first in range(0,b,bc):
            active_b = min(bc,b-first)
            k = active_b*spatial
            if active_b == 1:
                dy = ds[:,first].view(active_g,o,spatial)
            else:
                dy = dy_storage[:active_g*o*k].view(active_g,o,k)
                _resnet_pack_dy[(triton.cdiv(o*k,256),active_g)](
                    ds,dy,first,b,o,spatial,k,256)
                if stage_end is not None:
                    stage_end("pack_dy")
            for channel_first in range(0,c,stripe):
                nc = min(stripe,c-channel_first)
                count = nc*taps
                span = slice(channel_first*taps,channel_first*taps+count)
                storage = cols_storage[:active_g*count*k]
                cols = storage.view(active_g,count,k)
                if dx_mode != "fused":
                    spatial_major = dx_mode == "spatial_major"
                    if spatial_major:
                        torch.bmm(dy.transpose(1,2),w_flat[...,span],
                                  out=storage.view(active_g,k,count))
                    else:
                        torch.bmm(w_flat[...,span].transpose(1,2),dy,out=cols)
                    if stage_end is not None:
                        stage_end("dx_bmm")
                    _resnet_fold_dx[(triton.cdiv(active_b*nc*h*w,128),active_g)](
                        storage,dxs,first,b,c,h,w,p,q,r,pad,k,active_b,128,
                        stride,nc,channel_first,SPATIAL_MAJOR=spatial_major)
                    if stage_end is not None:
                        stage_end("fold_dx")
                if packed_dw:
                    _resnet_pack_patches[(triton.cdiv(count*k,256),active_g)](
                        xs,cols,first,b,c,h,w,p,q,r,pad,k,256,
                        stride,nc,channel_first)
                    if stage_end is not None:
                        stage_end("pack_x")
                    dw_block = dw_flat[...,span]
                    if first == 0:
                        torch.bmm(dy,cols.transpose(1,2),out=dw_block)
                    else:
                        torch.baddbmm(dw_block,dy,cols.transpose(1,2),out=dw_block)
                    if stage_end is not None:
                        stage_end("dw_bmm")


def _launch_resnet_bmm_backward(x,weight,d,dx,dw,db,*,packed_dw=True,step=2,
                                max_bytes=WORKSPACE_LIMIT_BYTES,channel_block=None,
                                balanced=False,stage_end=None,dx_mode="columns"):
    if dx_mode == "fused":
        _launch_layer4_fused_dx(weight,d,dx,max_bytes,stage_end)
    _launch_resnet_bmm(x,weight,d,dx,dw,packed_dw=packed_dw,max_bytes=max_bytes,
                       stride=step,channel_block=channel_block,balanced=balanced,
                       stage_end=stage_end,dx_mode=dx_mode)
    if not packed_dw:
        if step==1:
            g,_,c,_,_ = x.shape
            o = d.shape[2]
            block_o,block_crs = 32,64
            tiles = triton.cdiv(o,block_o)*triton.cdiv(c*9,block_crs)
            _dense_4x4_dw[(tiles,g)](x,d,dw,c,o,block_o,block_crs,num_warps=4)
        else:
            launch_direct_grad_weight(x,d,dw,bm=64,bn=32,bk=16)
        if stage_end is not None:
            stage_end("dw_direct")
    launch_grad_bias(d,db)
    if stage_end is not None:
        stage_end("db")


def _run_resnet_bmm(x,weight,grad_output,stride,padding,dilation,groups,*,packed_dw=True,
                    step=2,max_bytes=WORKSPACE_LIMIT_BYTES,channel_block=None,balanced=False,
                    layer4=False,dx_mode="columns"):
    g,b,_,o,h,w,_,_,r,s = _validate_general(
        x,weight,grad_output,stride,padding,dilation,groups)
    if (r!=s or r not in (1,3,7) or _pair(stride)!=(step,step) or
            _pair(padding)!=(r//2,r//2) or _pair(dilation)!=(1,1) or
            (not packed_dw and r==7)):
        raise ValueError("packed ResNet path requires a supported stride and square odd filter with same padding")
    if layer4:
        if (h,w,r,step)!=(4,4,3,1):
            raise ValueError("v30 layer4 candidates require 4x4 input and 3x3/s1/p1")
    elif step==1 and (r!=3 or b!=1):
        raise ValueError("packed stride-1 candidate requires B=1 and a 3x3 filter")
    if step==1 and not packed_dw and (h,w)!=(4,4):
        raise ValueError("direct stride-1 dW requires 4x4 activations")
    if (torch.backends.cuda.matmul.allow_tf32 or torch.is_autocast_enabled() or
            (torch.is_grad_enabled() and any(t.requires_grad for t in (x,weight,grad_output)))):
        raise ValueError("packed ResNet path requires FP32 matmul, autocast off and no autograd recording")
    with torch.cuda.device(x.device):
        x,weight,d = (t.contiguous() for t in (x,weight,grad_output))
        dx,dw,db = torch.empty_like(x),torch.empty_like(weight),x.new_empty((g,o))
        _launch_resnet_bmm_backward(x,weight,d,dx,dw,db,packed_dw=packed_dw,step=step,
                                    max_bytes=max_bytes,channel_block=channel_block,
                                    balanced=balanced,dx_mode=dx_mode)
    return dx,dw,db


TINY2_WORKSPACE_BYTES = 32*2**20
TINY2_DW_TILE = (16,16,16)


def _launch_tiny2_backward(x,weight,d,dx,dw,db,*,valid_dw=False,
                           max_bytes=TINY2_WORKSPACE_BYTES,stage_end=None):
    _launch_resnet_bmm(x,weight,d,dx,dw,packed_dw=not valid_dw,stride=1,balanced=True,
                       max_bytes=max_bytes,stage_end=stage_end)
    if valid_dw:
        g,b,c = x.shape[:3]
        o = d.shape[2]
        mo,ni,bk = TINY2_DW_TILE
        grid = (triton.cdiv(o,mo)*triton.cdiv(c,ni),g)
        _tiny2_valid_dw[grid](x,d,dw,b,c,o,mo,ni,bk,num_warps=4,num_stages=2)
        if stage_end is not None:
            stage_end("dw_valid_pairs")
    launch_grad_bias(d,db)
    if stage_end is not None:
        stage_end("db")


def _run_tiny2(x,weight,grad_output,stride,padding,dilation,groups,*,valid_dw=False):
    g,_,_,o,h,w,_,_,r,s = _validate_general(
        x,weight,grad_output,stride,padding,dilation,groups)
    if ((h,w,r,s)!=(2,2,3,3) or _pair(stride)!=(1,1) or
            _pair(padding)!=(1,1) or _pair(dilation)!=(1,1)):
        raise ValueError("v32 TinyImageNet candidates require 2x2 inputs and 3x3/s1/p1/d1")
    if (torch.backends.cuda.matmul.allow_tf32 or torch.is_autocast_enabled() or
            (torch.is_grad_enabled() and any(t.requires_grad for t in (x,weight,grad_output)))):
        raise ValueError("v32 TinyImageNet candidates require FP32 IEEE, autocast off and no autograd recording")
    with torch.cuda.device(x.device):
        x,weight,d = (t.contiguous() for t in (x,weight,grad_output))
        dx,dw,db = torch.empty_like(x),torch.empty_like(weight),x.new_empty((g,o))
        _launch_tiny2_backward(x,weight,d,dx,dw,db,valid_dw=valid_dw)
    return dx,dw,db


STEM7_WORKSPACE_BYTES = 64*2**20


def _stem7_bmm_images(g,b,o,p,q,element_size,budget):
    per_image = 147*(p*q+(o if g>1 and b>1 else 0))*element_size
    capacity = min(65535,budget//per_image)
    if capacity<1:
        raise ValueError("one RGB stem image exceeds the dX columns workspace budget")
    return triton.cdiv(g*b,triton.cdiv(g*b,capacity))


def _launch_stem7_bmm_dx(weight,d,dx,max_bytes,stage_end):
    g,b,_,h,w = dx.shape
    o,p,q = d.shape[2:]
    images = g*b
    chunk = _stem7_bmm_images(g,b,o,p,q,d.element_size(),max_bytes)
    dy = d.view(images,o,p*q)
    output = dx.view(images,1,3,h,w)
    weights = weight.view(g,o,147)
    columns = d.new_empty((chunk,147,p*q))
    # Consecutive image batches stay native. Only image chunks straddling a
    # group boundary need a weight gather (36.75 KiB per image for O=64).
    packed = None
    if stage_end is not None:
        stage_end("dx_workspace")
    for first in range(0,images,chunk):
        count = min(chunk,images-first)
        if b==1:
            wi = weights[first:first+count]
        elif first//b==(first+count-1)//b:
            gi = first//b
            wi = weights[gi:gi+1].expand(count,-1,-1)
        else:
            if packed is None:
                packed = weight.new_empty((chunk,o,147))
            wi = packed[:count]
            _stem7_pack_image_weights[(triton.cdiv(count*o*147,256),)](
                weights,wi,first,count,b,o,256,num_warps=4)
            if stage_end is not None:
                stage_end("pack_weights")
        torch.bmm(wi.transpose(1,2),dy[first:first+count],out=columns[:count])
        if stage_end is not None:
            stage_end("dx_bmm")
        _resnet_fold_dx[(triton.cdiv(3*h*w,128),count)](
            columns,output[first:first+count],0,1,3,h,w,p,q,7,3,p*q,1,128,
            num_warps=4)
        if stage_end is not None:
            stage_end("fold_dx")


def _stem7_dw_schedule(g,b,o,p,q,element_size,budget):
    # The narrower N tile reduces live FP32 accumulators and shared memory.
    bm,bn,bk = (32,32,32)
    blocks = triton.cdiv(b*p*q,bk)
    tiles = triton.cdiv(o,bm)*triton.cdiv(147,bn)
    # Aim for two waves on a 64-SM A5000 and bounded reductions for large B.
    desired = max(triton.cdiv(128,g*tiles),triton.cdiv(b*p*q,2048))
    limit = min(blocks,65535//g,budget//(g*o*148*element_size))
    return max(1,min(desired,limit)),bm,bn,bk


def _launch_stem7_dw_db(x,d,dw,db,max_bytes,stage_end):
    g,b,_,h,w = x.shape
    o,p,q = d.shape[2:]
    split,bm,bn,bk = _stem7_dw_schedule(g,b,o,p,q,x.element_size(),max_bytes)
    partial = x.new_empty((g,split,o,148)) if split>1 else dw
    grid = (triton.cdiv(o,bm)*triton.cdiv(147,bn),g*split)
    _stem7_dw_db_split[grid](x,d,dw,db,partial,b,o,h,w,p,q,split,bm,bn,bk,
                            num_warps=4,num_stages=2)
    if stage_end is not None:
        stage_end("dw_db_gemm")
    if split>1:
        _stem7_dw_db_finish[(triton.cdiv(o*148,64),g)](
            partial,dw,db,o,split,64,8,num_warps=4)
        if stage_end is not None:
            stage_end("dw_db_reduce")


def _launch_stem7_backward(x,weight,d,dx,dw,db,*,dx_mode="bmm",packed_dw=False,
                           max_bytes=STEM7_WORKSPACE_BYTES,stage_end=None):
    g,b,_,h,w = x.shape
    o,p,q = d.shape[2:]
    if dx_mode=="bmm":
        _launch_stem7_bmm_dx(weight,d,dx,max_bytes,stage_end)
    else:
        positions = b*((h+1)//2)*((w+1)//2)
        _stem7_dx_parity[(triton.cdiv(positions,32),g,4)](
            d,weight,dx,b,o,h,w,p,q,32,16,num_warps=4)
        if stage_end is not None:
            stage_end("dx_parity")
    if packed_dw:
        _launch_resnet_bmm(x,weight,d,dx,dw,packed_dw=True,max_bytes=max_bytes,
                           balanced=True,stage_end=stage_end,dx_mode="fused")
        torch.sum(d,dim=(1,3,4),out=db)
        if stage_end is not None:
            stage_end("db")
    else:
        _launch_stem7_dw_db(x,d,dw,db,max_bytes,stage_end)


def _run_stem7(x,weight,grad_output,stride,padding,dilation,groups,**schedule):
    g,_,c,o,_,_,_,_,r,s = _validate_general(
        x,weight,grad_output,stride,padding,dilation,groups)
    if ((c,r,s)!=(3,7,7) or _pair(stride)!=(2,2) or
            _pair(padding)!=(3,3) or _pair(dilation)!=(1,1)):
        raise ValueError("v31 RGB stem requires C=3 and 7x7/s2/p3/d1")
    uses_bmm = schedule.get("dx_mode","bmm")=="bmm" or schedule.get("packed_dw",False)
    if (torch.is_autocast_enabled() or
            (uses_bmm and torch.backends.cuda.matmul.allow_tf32) or
            (torch.is_grad_enabled() and any(t.requires_grad for t in (x,weight,grad_output)))):
        raise ValueError("v31 stem7 requires FP32 IEEE, autocast off and no autograd recording")
    with torch.cuda.device(x.device):
        x,weight,d = (t.contiguous() for t in (x,weight,grad_output))
        dx,dw,db = torch.empty_like(x),torch.empty_like(weight),x.new_empty((g,o))
        _launch_stem7_backward(x,weight,d,dx,dw,db,**schedule)
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


def _scaled_rgb_split(x,grad_output,bm,numerator,denominator=1):
    base=rgb_dw_split_count(x,grad_output,bm=bm)
    return min(64,max(1,(base*numerator+denominator-1)//denominator))


def conv_bwd_v15_rgb64_half_split(x,weight,grad_output,stride=1,padding=1,
                                  dilation=1,groups=1):
    """Ablation: BM64 with half of the v14 split count."""
    split=_scaled_rgb_split(x,grad_output,64,1,2)
    return _run_v14_rgb(x,weight,grad_output,stride,padding,dilation,groups,
                        "compact64",split,16)


def conv_bwd_v15_rgb64_double_split(x,weight,grad_output,stride=1,padding=1,
                                    dilation=1,groups=1):
    """Ablation: BM64 with twice the v14 split count, capped at 64."""
    split=_scaled_rgb_split(x,grad_output,64,2)
    return _run_v14_rgb(x,weight,grad_output,stride,padding,dilation,groups,
                        "compact64",split,16)


def conv_bwd_v15_rgb64_bk32(x,weight,grad_output,stride=1,padding=1,
                            dilation=1,groups=1):
    """Ablation: BM64 with a wider reduction tile."""
    return _run_v14_rgb(x,weight,grad_output,stride,padding,dilation,groups,
                        "compact64",None,32)


def conv_bwd_v15_rgb32_bk32(x,weight,grad_output,stride=1,padding=1,
                            dilation=1,groups=1):
    """Ablation: BM32 with a wider reduction tile."""
    return _run_v14_rgb(x,weight,grad_output,stride,padding,dilation,groups,
                        "compact32",None,32)


def conv_bwd_v17_rgb64k32(x,weight,grad_output,stride=1,padding=1,
                          dilation=1,groups=1):
    """RGB small-work probe: 64 positions, 32 output channels per step."""
    return _run_v17_rgb_direct(
        x,weight,grad_output,stride,padding,dilation,groups,64,32,32)


def conv_bwd_v17_rgb128k32(x,weight,grad_output,stride=1,padding=1,
                           dilation=1,groups=1):
    """RGB small-work probe with a wider spatial tile and BM64 dW."""
    return _run_v17_rgb_direct(
        x,weight,grad_output,stride,padding,dilation,groups,128,32,64)


def _conv_bwd_v26_rgb_bmm(x,weight,grad_output,stride,padding,dilation,groups):
    g,b,_,h,w=x.shape
    o=grad_output.shape[2]
    current = _select_algorithm_metadata(g,b,3,o,h,w,h,w,3,3,(1,1),(1,1),(1,1))
    bm = 32 if current in (
        "v13_rgb_implicit", "v17_rgb64k32", "v14_rgb_chunk32", "v15_rgb32_bk32"
    ) else 64
    bk = 32 if current in ("v17_rgb64k32", "v15_rgb32_bk32", "v15_rgb64_bk32") else 16
    split = _scaled_rgb_split(x,grad_output,bm,2) if current == "v15_rgb64_double" else None
    return _run_v14_rgb(x,weight,grad_output,stride,padding,dilation,groups,
                        f"compact{bm}",split,bk,group_bmm_dx=True)


def conv_bwd_v17_direct16_bk32(x,weight,grad_output,stride=1,padding=1,
                               dilation=1,groups=1):
    """Dense small-work probe: preserve BM16 and halve the dX K loops."""
    return _run_small_direct(
        x,weight,grad_output,stride,padding,dilation,groups,
        "implicit16_bk32",dw_schedule=(32,32,32,4))


def conv_bwd_v17_direct16x64(x,weight,grad_output,stride=1,padding=1,
                             dilation=1,groups=1):
    """Dense small-work probe: fewer N tiles without increasing dX BM."""
    return _run_small_direct(
        x,weight,grad_output,stride,padding,dilation,groups,
        "implicit16x64",dw_schedule=(64,32,16,8))


def conv_bwd_v17_direct16x64_bk32(x,weight,grad_output,stride=1,padding=1,
                                  dilation=1,groups=1):
    """Dense small-work probe combining the wider N and K schedules."""
    return _run_small_direct(
        x,weight,grad_output,stride,padding,dilation,groups,
        "implicit16x64_bk32",dw_schedule=(64,32,32,8))


def conv_bwd_v17_chunk32(x,weight,grad_output,stride=1,padding=1,
                         dilation=1,groups=1):
    """Dense probe: whole-batch chunked dX plus the proven direct dW."""
    return _run_small_direct(
        x,weight,grad_output,stride,padding,dilation,groups,
        "chunk32",dw_schedule=(32,32,16,4))


def conv_bwd_v17_chunk16_bk32(x,weight,grad_output,stride=1,padding=1,
                              dilation=1,groups=1):
    """Dense chunked dX with more M occupancy and a BK32 dW reduction."""
    return _run_small_direct(
        x,weight,grad_output,stride,padding,dilation,groups,
        "chunk16",dw_schedule=(32,32,32,4))


def conv_bwd_v17_separate_bmm(x,weight,grad_output,stride=1,padding=1,
                              dilation=1,groups=1):
    """Large-work probe: separate 128-MiB F4 phases reduce tile chunking."""
    return conv_bwd_separate_bmm(
        x,weight,grad_output,stride,padding,dilation,groups)


def conv_bwd_v18_joint_flat128(x,weight,grad_output,stride=1,padding=1,
                               dilation=1,groups=1):
    """Measured v11 math with a single 128-MiB workspace allocation."""
    return _run_joint_variant(
        x,weight,grad_output,stride,padding,dilation,groups,
        max_bytes=128*1024**2)


def conv_bwd_v18_joint_flat128_allg(x,weight,grad_output,stride=1,padding=1,
                                    dilation=1,groups=1):
    """Use the same 128-MiB budget but process all G groups concurrently."""
    return _run_joint_variant(
        x,weight,grad_output,stride,padding,dilation,groups,
        max_bytes=128*1024**2,packed_groups=x.shape[0])


def conv_bwd_v18_joint_flat256(x,weight,grad_output,stride=1,padding=1,
                               dilation=1,groups=1):
    """Larger-workspace probe; still below ATen's measured large-B peaks."""
    return _run_joint_variant(
        x,weight,grad_output,stride,padding,dilation,groups,
        max_bytes=256*1024**2)


def conv_bwd_v18_joint_flat512(x,weight,grad_output,stride=1,padding=1,
                               dilation=1,groups=1):
    """512-MiB upper-bound probe for eliminating streamed tile chunks."""
    return _run_joint_variant(
        x,weight,grad_output,stride,padding,dilation,groups,
        max_bytes=512*1024**2)


def conv_bwd_v18_joint_dxbmm128(x,weight,grad_output,stride=1,padding=1,
                                dilation=1,groups=1):
    """Flat 128-MiB joint path with torch.bmm for the F4 dX contraction."""
    return _run_joint_variant(
        x,weight,grad_output,stride,padding,dilation,groups,
        max_bytes=128*1024**2,dx_bmm=True)


def conv_bwd_v19_group_bmm_dx_bk32(x,weight,grad_output,stride=1,padding=1,
                                    dilation=1,groups=1):
    """Small-B probe: batch dX over B, retain the proven BK32 direct dW."""
    return _run_small_direct(
        x,weight,grad_output,stride,padding,dilation,groups,
        "group_bmm",dw_schedule=(32,32,32,4))


def conv_bwd_v16_s2_32(x,weight,grad_output,stride=2,padding=1,
                       dilation=1,groups=1):
    """3x3/s2 candidate: 32x64 dX and BM32 direct dW."""
    return _run_resnet_s2(
        x,weight,grad_output,stride,padding,dilation,groups,
        dx_bm=32,dx_bn=64,dw_bm=32,dw_bn=32)


def conv_bwd_v16_s2_64(x,weight,grad_output,stride=2,padding=1,
                       dilation=1,groups=1):
    """3x3/s2 candidate: wider channel tiles in both contractions."""
    return _run_resnet_s2(
        x,weight,grad_output,stride,padding,dilation,groups,
        dx_bm=64,dx_bn=32,dw_bm=64,dw_bn=32)


def conv_bwd_v16_s2_mixed64(x,weight,grad_output,stride=2,padding=1,
                            dilation=1,groups=1):
    """3x3/s2 candidate: high-N dX schedule plus BM64 dW."""
    return _run_resnet_s2(
        x,weight,grad_output,stride,padding,dilation,groups,
        dx_bm=32,dx_bn=64,dw_bm=64,dw_bn=32)


def conv_bwd_v16_s2_mixed64_nosplit(x,weight,grad_output,stride=2,padding=1,
                                    dilation=1,groups=1):
    """Ablation: mixed schedule with one dW reduction CTA per output tile."""
    return _run_resnet_s2(
        x,weight,grad_output,stride,padding,dilation,groups,
        dx_bm=32,dx_bn=64,dw_bm=64,dw_bn=32,force_split=1)


def conv_bwd_v16_s2_mixed64_bk32(x,weight,grad_output,stride=2,padding=1,
                                 dilation=1,groups=1):
    """Ablation: mixed schedule with a wider dW reduction tile."""
    return _run_resnet_s2(
        x,weight,grad_output,stride,padding,dilation,groups,
        dx_bm=32,dx_bn=64,dw_bm=64,dw_bn=32,dw_bk=32)


def conv_bwd_v16_pointwise32(x,weight,grad_output,stride=2,padding=0,
                             dilation=1,groups=1):
    """1x1/s2 shortcut candidate with BM32 dW."""
    return _run_resnet_s2(
        x,weight,grad_output,stride,padding,dilation,groups,
        dx_bm=32,dx_bn=64,dw_bm=32,dw_bn=32)


def conv_bwd_v16_pointwise64(x,weight,grad_output,stride=2,padding=0,
                             dilation=1,groups=1):
    """1x1/s2 shortcut candidate with wide dX and dW tiles."""
    return _run_resnet_s2(
        x,weight,grad_output,stride,padding,dilation,groups,
        dx_bm=64,dx_bn=32,dw_bm=64,dw_bn=32)


def conv_bwd_v16_pointwise_mixed64(x,weight,grad_output,stride=2,padding=0,
                                   dilation=1,groups=1):
    """1x1/s2 shortcut candidate with high-N dX and BM64 dW."""
    return _run_resnet_s2(
        x,weight,grad_output,stride,padding,dilation,groups,
        dx_bm=32,dx_bn=64,dw_bm=64,dw_bn=32)


def conv_bwd_v16_pointwise_mixed64_nosplit(
        x,weight,grad_output,stride=2,padding=0,dilation=1,groups=1):
    """Shortcut ablation with one dW reduction CTA per output tile."""
    return _run_resnet_s2(
        x,weight,grad_output,stride,padding,dilation,groups,
        dx_bm=32,dx_bn=64,dw_bm=64,dw_bn=32,force_split=1)


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


@functools.lru_cache(maxsize=256)
def _select_algorithm_metadata(g,b,c,o,h,w,p,q,r,s,stride,padding,dilation):
    """Pure metadata selector; every returned provider validates the inputs."""
    geometry=(r,s,stride,padding,dilation)
    if geometry==(3,3,(2,2),(1,1),(1,1)):
        return "v16_s2_mixed64"
    if geometry==(1,1,(2,2),(0,0),(1,1)):
        return "v16_pointwise_mixed64"
    if geometry!=(3,3,(1,1),(1,1),(1,1)):
        raise ValueError("auto currently supports 3x3/s1/p1, 3x3/s2/p1 and 1x1/s2/p0")
    if c==3:
        if o==128 and h==32 and w==32 and g==1:
            if b==1:
                return "v13_rgb_implicit"
        # v17's no-workspace direct path is a clear winner only at the tested
        # B=1,G=4 point.  Do not extrapolate it to B=10, where it regressed.
        if o==128 and h==32 and w==32 and b==1 and g==4:
            return "v17_rgb64k32"
        if o==128 and h==32 and w==32:
            if b==1:
                return "v15_rgb32_bk32"
            if b==10 and g==4:
                return "v15_rgb64_double"
            if b==100 and g==2:
                return "v15_rgb64_double"
            if b==1000 and g==1:
                return "v15_rgb32_bk32"
            return "v15_rgb64_bk32"
        if h==64 and w==64:
            if (b==10 and g==16) or (b==100 and g in (2,4)):
                return "v14_rgb_chunk32"
        if h==128 and w==128 and b==10 and g==4:
            return "v14_rgb_chunk32"
        return "v14_rgb_chunk64"
    if c==128 and o==128 and b==1 and h==w:
        if h==8:
            if g==1:
                return "v17_direct16_bk32"
            if g==4:
                return "v17_chunk16_bk32"
            if g<=4:
                return "v12_bmm_dx"
            return "v7_f2"
        if h==16:
            if g==4:
                return "v17_chunk32"
            return "v12_bmm_dx" if g<=4 else "v9_f4dw"
        if h==32 and g==1:
            return "v7_f2"
    if c==128 and o==128 and h==w==8 and b==10 and g==1:
        return "v17_chunk16_bk32"
    return "v11_joint_bmm"


@functools.lru_cache(maxsize=32)
def _cuda_device_name(device):
    return torch.cuda.get_device_name(device)


def _select_auto_algorithm(x, weight, grad_output, metadata):
    g,b,c,o,h,w,p,q,r,s,stride,padding,dilation = metadata
    case = (b,g,c,h)
    if (case in _MEASURED_CASES and o == 128 and h == w == p == q
            and (r,s,stride,padding,dilation) == (3,3,(1,1),(1,1),(1,1))
            and tuple(weight.shape) == (g,o,c,3,3)
            and tuple(grad_output.shape) == (g,b,o,p,q)
            and torch.backends.cudnn.enabled and not torch.backends.cudnn.allow_tf32
            and not torch.is_autocast_enabled()):
        tensors = (x, weight, grad_output)
        if (all(t.is_cuda and t.device == x.device and t.dtype == torch.float32
                and t.is_contiguous() for t in tensors)
                and not (torch.is_grad_enabled() and any(t.requires_grad for t in tensors))):
            provider = _measured_provider(_cuda_device_name(x.device), case)
            if provider is not None:
                return provider
    return _select_algorithm_metadata(*metadata)


def select_algorithm(x,weight,grad_output,stride=1,padding=1,dilation=1,
                     groups=1):
    """Return the provider that the default group-first call will select."""
    shape = _validate_general(x,weight,grad_output,stride,padding,dilation,groups)
    metadata = (*shape,_pair(stride),_pair(padding),_pair(dilation))
    return _select_auto_algorithm(x,weight,grad_output,metadata)


def _conv_bwd_torch_group_first(x, weight, grad_output, stride, padding, dilation, groups,
                               *, per_group=False):
    if x.ndim != 5 or weight.ndim != 5 or grad_output.ndim != 5:
        raise ValueError("expected 5D group-first inputs")
    g, b, c = x.shape[:3]
    if (operator.index(groups) not in (1,g) or weight.shape[0] != g or weight.shape[2] != c
            or grad_output.shape[:3] != (g,b,weight.shape[1])):
        raise ValueError("inconsistent group-first input, weight, grad_output or groups")
    if per_group:
        return _torch_group_first_chunked(
            x, weight, grad_output, stride, padding, dilation, groups_per_call=1)
    return _torch_group_first(x, weight, grad_output, stride, padding, dilation)


_ALGORITHM_TABLE={
    "torch_group_first":_conv_bwd_torch_group_first,
    "torch_g1":functools.partial(_conv_bwd_torch_group_first,per_group=True),
    "v7_f2":conv_bwd_v7,
    "v9_f4dw":conv_bwd_v9,
    "v11_joint_bmm":conv_bwd_v11_joint_bmm,
    "v12_direct16":conv_bwd_v12_direct16,
    "v12_bmm_dx":conv_bwd_v12_bmm_dx,
    "v13_rgb_implicit":conv_bwd_v13_rgb_implicit,
    "v13_rgb_bmm":conv_bwd_v13_rgb_bmm,
    "v14_rgb_chunk32":conv_bwd_v14_rgb_chunk32,
    "v14_rgb_chunk64":conv_bwd_v14_rgb_chunk64,
    "v15_rgb64_double":conv_bwd_v15_rgb64_double_split,
    "v15_rgb64_bk32":conv_bwd_v15_rgb64_bk32,
    "v15_rgb32_bk32":conv_bwd_v15_rgb32_bk32,
    "v17_rgb64k32":conv_bwd_v17_rgb64k32,
    "v17_rgb128k32":conv_bwd_v17_rgb128k32,
    "v17_direct16_bk32":conv_bwd_v17_direct16_bk32,
    "v17_chunk32":conv_bwd_v17_chunk32,
    "v17_chunk16_bk32":conv_bwd_v17_chunk16_bk32,
    "v16_s2_mixed64":conv_bwd_v16_s2_mixed64,
    "v16_pointwise_mixed64":conv_bwd_v16_pointwise_mixed64,
    "v27_s2_bmm":_run_resnet_bmm,
    "v27_s2_bmm_dx":functools.partial(_run_resnet_bmm,packed_dw=False),
    "v27_shortcut_fused_zero":functools.partial(_run_resnet_s2,shortcut_fill_holes=True),
    "v28_dense_bmm":functools.partial(_run_resnet_bmm,step=1,balanced=True),
    "v29_dense_direct_dw":functools.partial(
        _run_resnet_bmm,step=1,balanced=True,packed_dw=False),
    "v30_layer4_bmm":functools.partial(
        _run_resnet_bmm,step=1,balanced=True,max_bytes=32*2**20,layer4=True),
    "v30_layer4_spatial_bmm":functools.partial(
        _run_resnet_bmm,step=1,balanced=True,max_bytes=32*2**20,layer4=True,
        dx_mode="spatial_major"),
    "v30_layer4_fused_dx":functools.partial(
        _run_resnet_bmm,step=1,balanced=True,max_bytes=32*2**20,layer4=True,
        dx_mode="fused"),
    "v31_stem7_bmm":_run_stem7,
    "v31_stem7_direct":functools.partial(_run_stem7,dx_mode="direct"),
    "v31_stem7_packed_dw":functools.partial(_run_stem7,packed_dw=True),
    "v32_tiny2_bmm":_run_tiny2,
    "v32_tiny2_valid_dw":functools.partial(_run_tiny2,valid_dw=True),
    "v28_s2_bmm_32m":functools.partial(_run_resnet_bmm,max_bytes=32*2**20,balanced=True),
    "v28_s2_bmm_64m":functools.partial(_run_resnet_bmm,max_bytes=64*2**20,balanced=True),
    "v28_s2_bmm_c64":functools.partial(
        _run_resnet_bmm,max_bytes=64*2**20,channel_block=64,balanced=True),
    "v18_joint_flat128":conv_bwd_v18_joint_flat128,
    "v19_group_bmm_dx_bk32":conv_bwd_v19_group_bmm_dx_bk32,
    "v26_rgb_bmm_dx":_conv_bwd_v26_rgb_bmm,
    "v26_small_bmm_dx_bk32":functools.partial(
        _run_small_direct,dx_kind="bmm",dw_schedule=(32,32,32,4)),
    "v26_joint_compact":functools.partial(_run_joint_variant,compact_tail=True),
    "v26_joint_compact_dxbmm":functools.partial(_run_joint_variant,compact_tail=True,dx_bmm=True),
}

_V21_ALGORITHMS = (
    "v21_bmm_dx", "v21_fused_dw", "v21_bmm_dx_fused_dw", "v22_row_dw", "v23_f4_bmm_dw",
)


def conv_bwd(x,weight,grad_output,stride=1,padding=1,dilation=1,groups=1,*,algorithm="auto",dx_schedule=None):
    """Direct-call group-first convolution backward with automatic selection.

    Inputs and outputs use ``[G,B,C,H,W]``, ``[G,O,C,R,S]`` and
    ``[G,B,O,P,Q]`` / ``[G,B,C,H,W]``, ``[G,O,C,R,S]``, ``[G,O]``.
    The signature intentionally matches the user-facing baseline.
    "auto" uses v26 measured overrides. "custom_auto" selects a custom kernel
    by the v24 rules. "torch_group_first" and "torch_g1" explicitly use Torch.
    For the explicit small-B BMM-dX controls, dx_schedule="width_f4" or
    "stream4" replaces only dX. None uses the selected algorithm as supplied.
    """
    if dx_schedule is not None:
        if dx_schedule not in ("width_f4", "stream4") or algorithm not in (
                "v21_bmm_dx", "v21_bmm_dx_fused_dw", "v23_f4_bmm_dw"):
            raise ValueError("dx_schedule requires an explicit small-B BMM-dX control and width_f4 or stream4")
    if algorithm == "v18_allg":
        return conv_bwd_v18_joint_flat128_allg(x, weight, grad_output, stride, padding, dilation, groups)
    if algorithm in ("v20_joint_direct", "v20_split_direct"):
        return _v20_direct(x, weight, grad_output, stride, padding, dilation, groups,
                           joint=algorithm=="v20_joint_direct")
    if algorithm=="v20_balanced_f4":
        return _run_joint_variant(x, weight, grad_output, stride, padding, dilation,
                                  groups, balanced=True)
    if algorithm in ("auto", "custom_auto"):
        # Full tensor validation belongs to the chosen provider.
        g,b,c,h,w=x.shape;_,o,_,r,s=weight.shape;_,_,_,p,q=grad_output.shape
        metadata = (g,b,c,o,h,w,p,q,r,s,_pair(stride),_pair(padding),_pair(dilation))
        if algorithm == "auto":
            algorithm = _select_auto_algorithm(x,weight,grad_output,metadata)
        else:
            algorithm = _select_algorithm_metadata(*metadata)
    if algorithm in _V21_ALGORITHMS:
        return _run_v21(x, weight, grad_output, stride, padding, dilation, groups, algorithm, dx_schedule)
    provider = _ALGORITHM_TABLE.get(algorithm)
    if provider is None:
        raise ValueError(f"unknown algorithm {algorithm!r}")
    return provider(x,weight,grad_output,stride,padding,dilation,groups)

