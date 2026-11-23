"""Fused group-first FP32 InstanceNorm + ReLU, for manual-backward use.

x/y: [G*B,C,H,W], with leading index g*B+b; affine vectors: [G*C].
No activation packing, atomics, statistics cache, or autotuning.
Large spatial planes use a small per-row statistics workspace.
"""
import math
import operator

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

if __package__:
    from . import _grouped_norm_large as _large
else:
    import _grouped_norm_large as _large


@triton.jit
def _fwd_rows(X, GAMMA, BETA, Y,
              B: tl.constexpr, C: tl.constexpr, ROWS: tl.constexpr,
              M: tl.constexpr, EPS: tl.constexpr,
              HAS_GAMMA: tl.constexpr, HAS_BETA: tl.constexpr,
              R: tl.constexpr, S: tl.constexpr):
    row = tl.program_id(0) * R + tl.arange(0, R)
    col = tl.arange(0, S)
    base = row.to(tl.int64) * M
    valid = (row[:, None] < ROWS) & (col[None, :] < M)
    x = tl.load(X + base[:, None] + col[None, :], valid, other=0).to(tl.float32)
    # Center around an exactly representable input value before summing. This
    # avoids E[x*x]-E[x]**2 cancellation and loss of a small mean atop a large DC.
    anchor = tl.load(X + base, row < ROWS, other=0).to(tl.float32)
    shifted = tl.where(valid, x - anchor[:, None], 0.)
    mean_shift = tl.sum(shifted, 1) / M
    xc = tl.where(valid, shifted - mean_shift[:, None], 0.)
    variance = tl.sum(xc * xc, 1) / M
    inv_std = tl.rsqrt(variance + EPS)
    affine = (row // (B * C)) * C + row % C
    scale = inv_std
    if HAS_GAMMA:
        gamma = tl.load(GAMMA + affine, row < ROWS, other=0).to(tl.float32)
        scale = scale * gamma
    value = xc * scale[:, None]
    if HAS_BETA:
        beta = tl.load(BETA + affine, row < ROWS, other=0).to(tl.float32)
        value = value + beta[:, None]
    value = tl.maximum(value, 0., propagate_nan=tl.PropagateNan.ALL)
    tl.store(Y + base[:, None] + col[None, :], value, valid)


def _validate(x, weight, bias, eps, groups, rows, num_warps):
    if x.ndim != 4 or min(x.shape) <= 0:
        raise ValueError('x must be nonempty [G*B,C,H,W]')
    gb, c, h, w = x.shape
    if h*w < 2:
        raise ValueError('Spatial size H*W >= 2 is required')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be positive and finite')
    affine = weight if weight is not None else bias
    g = (affine.numel() // c if affine is not None else 1) if groups is None else operator.index(groups)
    if gb*c >= 2**31:
        raise ValueError('Too many spatial planes for int32 row indexing')
    if g <= 0 or gb % g:
        raise ValueError('G must be positive and divide x.shape[0]')
    for name, t in (('weight', weight), ('bias', bias)):
        if t is not None and (t.ndim != 1 or t.numel() != g*c):
            raise ValueError(f'{name} must be flat [G*C]')
    for t in (x, weight, bias):
        if t is None:
            continue
        if not t.is_cuda or t.dtype != torch.float32 or t.device != x.device or not t.is_contiguous():
            raise ValueError('All inputs must be contiguous FP32 CUDA tensors on the same device')
        if torch.is_grad_enabled() and t.requires_grad:
            raise RuntimeError('Forward-only API: call inside your custom autograd.Function.forward or torch.no_grad()')
    if rows not in (1, 4) or num_warps not in (1, 4, 8):
        raise ValueError('rows must be 1 or 4; num_warps must be 1, 4, or 8')
    if rows != 1 and h*w > 256:
        raise ValueError('rows=4 is restricted to H*W <= 256')
    return g, gb//g, c, h*w


def _launch(x, weight, bias, y, b, c, m, eps, rows, num_warps):
    nrows = x.shape[0] * c
    _fwd_rows[(triton.cdiv(nrows, rows),)](
        x, weight if weight is not None else x,
        bias if bias is not None else x, y,
        b, c, nrows, m, eps, weight is not None, bias is not None,
        rows, triton.next_power_of_2(m), num_warps=num_warps)


def grouped_instancenorm_fwd(x, weight=None, bias=None, eps=1e-5,
                             *, groups=None, rows=1, num_warps=4):
    """Return ReLU(InstanceNorm(x)), without changing group-first layout.

    FP32 contiguous CUDA only. Each (g,b,c) is normalized over H*W using
    population variance (unbiased=False), with independent [g,c] affine values.
    H*W>4096 uses bounded spatial tiles; <=4096 retains the tested kernel.
    Intended for the existing manually implemented backward/double backward.
    When both affine inputs are None, groups defaults to 1 (same result for
    every partition); callers may still pass groups explicitly.
    """
    _, b, c, m = _validate(x, weight, bias, eps, groups, rows, num_warps)
    with torch.cuda.device(x.device):
        if m > 4096:
            return _large.forward(x, weight, bias, b, c, m, eps)
        y = torch.empty_like(x)
        _launch(x, weight, bias, y, b, c, m, eps, rows, num_warps)
    return y


def to_old(x, g):
    gb, c, h, w = x.shape
    return x.view(g, gb//g, c, h, w).transpose(0, 1).reshape(gb//g, g*c, h, w)


def to_new(x, g):
    b, gc, h, w = x.shape
    return x.view(b, g, gc//g, h, w).transpose(0, 1).reshape(g*b, gc//g, h, w)


def torch_native_old(x_old, weight, bias, eps=1e-5):
    """Literal supplied baseline; old [B,G*C,H,W] input/output."""
    return F.relu(F.instance_norm(x_old, weight=weight, bias=bias, eps=eps))


def torch_old_adapter(x, weight, bias, g, eps=1e-5):
    return to_new(torch_native_old(to_old(x, g), weight, bias, eps), g)


def torch_flat_rows(x, weight, bias, g, eps=1e-5):
    """Strong Torch baseline: activation views only; affine expansion is timed.

    View every (g,b,c) plane as one channel of a single sample. InstanceNorm
    still normalizes exactly the same independent spatial planes. This
    avoids an activation transpose at the cost of materializing affine rows.
    """
    gb, c, h, w = x.shape
    b = gb//g
    def expand(t):
        return None if t is None else t.view(g, 1, c).expand(g, b, c).reshape(-1)
    return F.relu(F.instance_norm(x.view(1, gb*c, h, w),
                  weight=expand(weight), bias=expand(bias), eps=eps)).view_as(x)
