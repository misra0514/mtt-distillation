"""FP32 fused InstanceNorm+ReLU backward for group-major [G*B,C,H,W].

No packing, atomics, global cache, or dependency on forward statistics.
Only fused first backward is implemented. No autograd/double-backward binding.
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _row_backward(X, DY, OUT, GAMMA, DX, DG, DB, PG, PB,
                  B: tl.constexpr, C: tl.constexpr, ROWS: tl.constexpr,
                  HW: tl.constexpr, EPS: tl.constexpr,
                  R: tl.constexpr, S: tl.constexpr):
    row = tl.program_id(0) * R + tl.arange(0, R)
    col = tl.arange(0, S)
    valid = (row[:, None] < ROWS) & (col[None, :] < HW)
    offset = row[:, None].to(tl.int64) * HW + col[None, :]
    x = tl.load(X + offset, valid, other=0).to(tl.float32)
    # Shift before the mean reduction to retain accuracy for large common offsets.
    anchor = tl.load(X + row.to(tl.int64) * HW, row < ROWS, other=0.).to(tl.float32)
    shifted = tl.where(col[None, :] < HW, x - anchor[:, None], 0.)
    mean_shift = tl.sum(shifted, 1) / HW
    centered = tl.where(col[None, :] < HW, shifted - mean_shift[:, None], 0.)
    var = tl.sum(centered * centered, 1) / HW
    rstd = tl.rsqrt(var + EPS)
    xhat = centered * rstd[:, None]

    dy = tl.load(DY + offset, valid, other=0).to(tl.float32)
    out = tl.load(OUT + offset, valid, other=0).to(tl.float32)
    dy = tl.where(out > 0., dy, 0.)
    sb = tl.sum(dy, 1)
    sg = tl.sum(dy * xhat, 1)
    affine = (row // (B * C)) * C + row % C
    gamma = tl.load(GAMMA + affine, row < ROWS, other=0).to(tl.float32)
    dx = (dy - sb[:, None] / HW - xhat * (sg[:, None] / HW))
    dx = dx * (gamma * rstd)[:, None]
    tl.store(DX + offset, dx, valid)
    if B == 1:
        tl.store(DG + affine, sg, row < ROWS)
        tl.store(DB + affine, sb, row < ROWS)
    else:
        tl.store(PG + row, sg, row < ROWS)
        tl.store(PB + row, sb, row < ROWS)


@triton.jit
def _reduce_batch(PG, PB, DG, DB, B: tl.constexpr, C: tl.constexpr,
                  BC: tl.constexpr, BB: tl.constexpr):
    # Reduce the full batch tile in one tree for B<=1024. Eight channels per
    # CTA bound the register footprint at B=1000 without a loop-carried sum.
    c = tl.program_id(0) * BC + tl.arange(0, BC)
    g = tl.program_id(1)
    b = tl.arange(0, BB)
    sg = tl.zeros((BC,), tl.float32)
    sb = tl.zeros((BC,), tl.float32)
    for block in tl.static_range((B + BB - 1) // BB):
        bi = block * BB + b
        offset = (g.to(tl.int64) * B + bi[:, None]) * C + c[None, :]
        mask = (bi[:, None] < B) & (c[None, :] < C)
        sg += tl.sum(tl.load(PG + offset, mask, other=0.), 0)
        sb += tl.sum(tl.load(PB + offset, mask, other=0.), 0)
    tl.store(DG + g * C + c, sg, c < C)
    tl.store(DB + g * C + c, sb, c < C)


# v2 reducer retained ONLY for short diagnostic comparisons.
@triton.jit
def _reduce_batch_v2(PG, PB, DG, DB, B: tl.constexpr, C: tl.constexpr,
                  BC: tl.constexpr, BB: tl.constexpr):
    c = tl.program_id(0) * BC + tl.arange(0, BC)
    g = tl.program_id(1)
    b = tl.arange(0, BB)
    sg = tl.zeros((BC,), tl.float32)
    sb = tl.zeros((BC,), tl.float32)
    for block in range(tl.cdiv(B, BB)):
        bi = block * BB + b
        offset = (g.to(tl.int64) * B + bi[:, None]) * C + c[None, :]
        mask = (bi[:, None] < B) & (c[None, :] < C)
        sg += tl.sum(tl.load(PG + offset, mask, other=0.), 0)
        sb += tl.sum(tl.load(PB + offset, mask, other=0.), 0)
    tl.store(DG + g * C + c, sg, c < C)
    tl.store(DB + g * C + c, sb, c < C)


def _validate(x, weight, output, grad_output, eps):
    if x.ndim != 4 or weight.ndim != 1 or x.shape != output.shape or x.shape != grad_output.shape:
        raise ValueError('Expected x/output/dY=[G*B,C,H,W], weight=[G*C]')
    gb, c, h, w = x.shape
    if min(gb, c, h, w) <= 0 or h * w <= 1:
        raise ValueError('Positive dimensions and H*W>1 are required')
    if weight.numel() % c or weight.numel() == 0:
        raise ValueError('weight must contain G*C elements')
    g = weight.numel() // c
    if gb % g:
        raise ValueError('Leading input dimension must be divisible by G')
    for t in (x, weight, output, grad_output):
        if not t.is_cuda or t.device != x.device or t.dtype != torch.float32:
            raise ValueError('All inputs must be FP32 CUDA tensors on the same device')
        if not t.is_contiguous():
            raise ValueError('Inputs must be contiguous; this interface does not pack/copy them')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')
    if h * w > 4096:
        raise ValueError('v3 targets Conv3 spatial sizes up to 64x64 (H*W<=4096)')
    return g, gb // g, c, h * w


def _allocate(x, weight, g, b, c):
    dx = torch.empty_like(x)
    dg = torch.empty_like(weight)
    db = torch.empty_like(weight)
    pg = torch.empty((g * b * c,), device=x.device, dtype=torch.float32) if b > 1 else dg
    pb = torch.empty_like(pg) if b > 1 else db
    return dx, dg, db, pg, pb


def _launch_rows(x, weight, output, grad_output, buffers, g, b, c, hw, eps, rows, num_warps=4):
    dx, dg, db, pg, pb = buffers
    _row_backward[(triton.cdiv(g * b * c, rows),)](
        x, grad_output, output, weight, dx, dg, db, pg, pb,
        b, c, g * b * c, hw, eps, rows, triton.next_power_of_2(hw),
        num_warps=num_warps,
    )


def _launch_reduce(buffers, g, b, c):
    if b > 1:
        _, dg, db, pg, pb = buffers
        bb = min(triton.next_power_of_2(b), 1024)
        _reduce_batch[(triton.cdiv(c, 8), g)](pg, pb, dg, db, b, c, 8, bb,
                                              num_warps=4)


def _run(x, weight, output, grad_output, eps, rows, num_warps=4):
    g, b, c, hw = _validate(x, weight, output, grad_output, eps)
    if rows not in (1, 4, 8) or num_warps not in (1, 2, 4):
        raise ValueError('rows must be 1/4/8 and num_warps must be 1/2/4')
    if (rows, num_warps) != (1, 4) and hw > 256:
        raise ValueError('Explicit tiling candidates require H*W<=256')
    with torch.cuda.device(x.device):
        buffers = _allocate(x, weight, g, b, c)
        _launch_rows(x, weight, output, grad_output, buffers, g, b, c, hw, eps, rows, num_warps)
        _launch_reduce(buffers, g, b, c)
    return buffers[:3]


def grouped_instancenorm_bwd(x, weight, output, grad_output, v_fuse=True, eps=1e-5,
                              *, rows=1, num_warps=4):
    """Stateless fused backward. Returns dX=[G*B,C,H,W], dweight/dbias=[G*C].

    G is inferred from weight.numel()/C; the leading input index is g*B+b.
    output is the post-ReLU tensor. No statistics cache or layout conversion.
    FP32 contiguous CUDA, H*W in [2,4096]; fused first backward only.
    Default row computation (rows=1, num_warps=4) preserves v1.
    v3 changes the batch reduction; it requires on-device verification. Explicit small-spatial
    candidates pack 4/8 planes into one CTA with 1/2/4 warps; no selector.
    """
    if not v_fuse:
        raise NotImplementedError('v3 implements only the fused path (v_fuse=True)')
    return _run(x, weight, output, grad_output, eps, rows, num_warps)


def grouped_instancenorm_bwd_rows4(x, weight, output, grad_output, eps=1e-5):
    """Explicit candidate: same algorithm, four spatial planes per program."""
    return _run(x, weight, output, grad_output, eps, rows=4)


# User-provided two-stage/atomic reference. Kept as a separate old-layout path.
@triton.jit
def _reference_stats(X, mean_ptr, rstd_ptr, stride_n, stride_c, C, HW,
                     eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    n = tl.program_id(0)
    c = tl.program_id(1)
    x_ptr = X + n * stride_n + c * stride_c
    sum_x = tl.zeros([], dtype=tl.float32)
    sum_x2 = tl.zeros([], dtype=tl.float32)
    for off in range(0, HW, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + idx, idx < HW, other=0.).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)
    invN = 1.0 / tl.full([], HW, tl.float32)
    mean = sum_x * invN
    var = sum_x2 * invN - mean * mean
    rstd = tl.rsqrt(var + eps)
    tl.store(mean_ptr + n * C + c, mean)
    tl.store(rstd_ptr + n * C + c, rstd)


@triton.jit
def _reference_backward(dY, X, gamma, out_ptr, mean_ptr, rstd_ptr,
                        dX, dgamma, dbeta, stride_n, stride_c, C, HW,
                        BLOCK_SIZE: tl.constexpr):
    n = tl.program_id(0)
    c = tl.program_id(1)
    idx_offset = n * stride_n + c * stride_c
    g = tl.load(gamma + c).to(tl.float32)
    mean = tl.load(mean_ptr + n * C + c).to(tl.float32)
    rstd = tl.load(rstd_ptr + n * C + c).to(tl.float32)
    sum_dy = tl.zeros([], dtype=tl.float32)
    sum_dy_xhat = tl.zeros([], dtype=tl.float32)
    for off in range(0, HW, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < HW
        x = tl.load(X + idx_offset + idx, mask, other=0.).to(tl.float32)
        dy = tl.load(dY + idx_offset + idx, mask, other=0.).to(tl.float32)
        outv = tl.load(out_ptr + idx_offset + idx, mask, other=0.).to(tl.float32)
        dy = tl.where(outv > 0., dy, 0.)
        xhat = (x - mean) * rstd
        sum_dy += tl.sum(dy, axis=0)
        sum_dy_xhat += tl.sum(dy * xhat, axis=0)
    tl.atomic_add(dbeta + c, sum_dy)
    tl.atomic_add(dgamma + c, sum_dy_xhat)
    invN = 1.0 / tl.full([], HW, tl.float32)
    term2 = (g * sum_dy) * invN
    term3 = (g * sum_dy_xhat) * invN
    for off in range(0, HW, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < HW
        x = tl.load(X + idx_offset + idx, mask, other=0.).to(tl.float32)
        dy = tl.load(dY + idx_offset + idx, mask, other=0.).to(tl.float32)
        outv = tl.load(out_ptr + idx_offset + idx, mask, other=0.).to(tl.float32)
        dy = tl.where(outv > 0., dy, 0.)
        xhat = (x - mean) * rstd
        dx = rstd * (g * dy - term2 - term3 * xhat)
        tl.store(dX + idx_offset + idx, dx, mask=mask)


def reference_instancenorm_relu_bwd(x, gamma, grad_output, out, eps=1e-5):
    """Original signature/layout: x=[B,G*C,H,W], gamma=[G*C]."""
    n, c, h, w = x.shape
    hw = h * w
    x_flat = x.contiguous().view(n, c, hw)
    dy_flat = grad_output.contiguous().view(n, c, hw)
    out_flat = out.contiguous().view(n, c, hw)
    with torch.cuda.device(x.device):
        mean = torch.empty((n * c,), device=x.device, dtype=torch.float32)
        rstd = torch.empty_like(mean)
        dx = torch.empty_like(x_flat, dtype=torch.float32)
        dg = torch.zeros((c,), device=x.device, dtype=torch.float32)
        db = torch.zeros_like(dg)
        _reference_stats[(n, c)](x_flat, mean, rstd, c * hw, hw, c, hw,
                                eps=eps, BLOCK_SIZE=256)
        _reference_backward[(n, c)](dy_flat, x_flat, gamma, out_flat, mean, rstd,
                                    dx, dg, db, c * hw, hw, c, hw, BLOCK_SIZE=256)
    return dx.view(n, c, h, w), dg, db


def to_old(x, g):
    gb, c, h, w = x.shape
    return x.view(g, gb // g, c, h, w).transpose(0, 1).reshape(gb // g, g * c, h, w)


def to_new(x, g):
    b, gc, h, w = x.shape
    return x.view(b, g, gc // g, h, w).transpose(0, 1).reshape(g * b, gc // g, h, w)


def reference_grouped_instancenorm_bwd(x, weight, output, grad_output, eps=1e-5):
    g = weight.numel() // x.shape[1]
    dx, dg, db = reference_instancenorm_relu_bwd(
        to_old(x, g), weight, to_old(grad_output, g), to_old(output, g), eps)
    return to_new(dx, g), dg, db
