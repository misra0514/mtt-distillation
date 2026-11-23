"""Fused group-first FP32 InstanceNorm + ReLU, for manual-backward use.

x/y: [G*B,C,H,W], with leading index g*B+b; affine vectors: [G*C].
No activation packing, atomics, saved statistics cache, or autotuning.
Spatial planes >4096 use a bounded-tile fallback and small statistics buffer.
"""
import math
import operator
from collections import OrderedDict
from functools import lru_cache

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Only immutable launch metadata / compiled binaries are retained, never
# tensors, tensor addresses, outputs, streams, or CUDA graphs. Calls remain
# independent. The direct CompiledKernel ABI is checked against Triton 3.1;
# other versions use the regular JIT launch path.
_DIRECT_SUPPORTED = triton.__version__.split('.')[:2] == ['3', '1']
_LAUNCHERS = OrderedDict()
_MAX_LAUNCHERS = 256


@lru_cache(maxsize=16)
def _sm_count(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def _default_config(nrows, m, sms):
    """Small set of spatial/parallelism rules, with no per-case lookup table."""
    if m <= 64:
        # R=16 needs enough independent planes for >=8 CTAs per SM.
        return (8 if nrows < sms*128 else 16), 4
    if m <= 256:
        if nrows <= sms*32:
            return 4, 1
        if nrows < sms*256:
            return 8, 4
    return 1, 4


def _resolve_config(nrows, m, device, rows, num_warps):
    if rows is None:
        rows, suggested_warps = _default_config(nrows,m,_sm_count(device))
    else:
        suggested_warps = 4
    return rows, suggested_warps if num_warps is None else num_warps


@triton.jit
def _fwd_rows(X, GAMMA, BETA, Y,
              B: tl.constexpr, C: tl.constexpr, ROWS: tl.constexpr,
              M: tl.constexpr, EPS: tl.constexpr,
              HAS_GAMMA: tl.constexpr, HAS_BETA: tl.constexpr,
              R: tl.constexpr, S: tl.constexpr):
    row = tl.program_id(0) * R + tl.arange(0, R)
    col = tl.arange(0, S)
    # Main Conv3 cases have no row tail: eliminate redundant predicates.
    if ROWS % R == 0:
        valid_row = tl.full((R,), True, tl.int1)
    else:
        valid_row = row < ROWS
    if ROWS * M < 2147483648:
        base = row * M
    else:
        base = row.to(tl.int64) * M
    valid = valid_row[:, None] & (col[None, :] < M)
    x = tl.load(X + base[:, None] + col[None, :], valid, other=0).to(tl.float32)
    # Center around an exactly representable input value before summing. This
    # avoids E[x*x]-E[x]**2 cancellation and loss of a small mean atop a large DC.
    anchor = tl.load(X + base, valid_row, other=0).to(tl.float32)
    shifted = tl.where(valid, x - anchor[:, None], 0.)
    mean_shift = tl.sum(shifted, 1) / M
    xc = tl.where(valid, shifted - mean_shift[:, None], 0.)
    variance = tl.sum(xc * xc, 1) / M
    inv_std = tl.rsqrt(variance + EPS)
    if B == 1:
        affine = row
    elif ROWS == B * C:
        affine = row % C
    else:
        affine = (row // (B * C)) * C + row % C
    scale = inv_std
    if HAS_GAMMA:
        gamma = tl.load(GAMMA + affine, valid_row, other=0).to(tl.float32)
        scale = scale * gamma
    value = xc * scale[:, None]
    if HAS_BETA:
        beta = tl.load(BETA + affine, valid_row, other=0).to(tl.float32)
        value = value + beta[:, None]
    value = tl.maximum(value, 0., propagate_nan=tl.PropagateNan.ALL)
    tl.store(Y + base[:, None] + col[None, :], value, valid)


@triton.jit
def _large_stats(X, ST, ROWS: tl.constexpr, M: tl.constexpr,
                 EPS: tl.constexpr, S: tl.constexpr):
    row = tl.program_id(0)
    base = row.to(tl.int64)*M
    col = tl.arange(0,S)
    anchor = tl.load(X+base).to(tl.float32)
    acc = tl.zeros((S,),tl.float32)
    for block in range(tl.cdiv(M,S)):
        j = block*S+col
        x = tl.load(X+base+j,j<M,other=0.).to(tl.float32)
        acc += tl.where(j<M,x-anchor,0.)
    mu = tl.sum(acc,0)/M
    vv = tl.zeros((S,),tl.float32)
    for block in range(tl.cdiv(M,S)):
        j = block*S+col
        x = tl.load(X+base+j,j<M,other=0.).to(tl.float32)
        xc = tl.where(j<M,(x-anchor)-mu,0.)
        vv += xc*xc
    inv = tl.rsqrt(tl.sum(vv,0)/M+EPS)
    tl.store(ST+row,anchor)
    tl.store(ST+ROWS+row,mu)
    tl.store(ST+2*ROWS+row,inv)


@triton.jit
def _large_apply(X, GAMMA, BETA, ST, Y, B: tl.constexpr, C: tl.constexpr,
                 ROWS: tl.constexpr, M: tl.constexpr, HAS_GAMMA: tl.constexpr,
                 HAS_BETA: tl.constexpr, S: tl.constexpr):
    row = tl.program_id(0)
    j = tl.program_id(1)*S+tl.arange(0,S)
    off = row.to(tl.int64)*M+j
    if B == 1:
        affine = row
    elif ROWS == B*C:
        affine = row%C
    else:
        affine = (row//(B*C))*C+row%C
    x = tl.load(X+off,j<M,other=0.).to(tl.float32)
    z = ((x-tl.load(ST+row))-tl.load(ST+ROWS+row))*tl.load(ST+2*ROWS+row)
    if HAS_GAMMA:z *= tl.load(GAMMA+affine).to(tl.float32)
    if HAS_BETA:z += tl.load(BETA+affine).to(tl.float32)
    tl.store(Y+off,tl.maximum(z,0.,propagate_nan=tl.PropagateNan.ALL),j<M)


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
    device, dtype, grad_enabled = x.device, torch.float32, torch.is_grad_enabled()
    for t in (x, weight, bias):
        if t is None:
            continue
        if not t.is_cuda or t.dtype != dtype or t.device != device or not t.is_contiguous():
            raise ValueError('All inputs must be contiguous FP32 CUDA tensors on the same device')
        if grad_enabled and t.requires_grad:
            raise RuntimeError('Forward-only API: call inside your custom autograd.Function.forward or torch.no_grad()')
    if (rows is not None and rows not in (1, 4, 8, 16)) or (num_warps is not None and num_warps not in (1, 4, 8)):
        raise ValueError('rows must be 1/4/8/16; num_warps must be 1/4/8')
    if rows not in (None,1) and h*w > 256:
        raise ValueError('Multi-row configurations require H*W <= 256')
    if h*w > 4096 and num_warps not in (None,4):
        raise ValueError('Large spatial planes use fixed num_warps=4')
    return g, gb//g, c, h*w


def _launch(x, weight, bias, y, b, c, m, eps, rows, num_warps, workspace=None,
            fast_launch=True):
    nrows = x.shape[0] * c
    if m > 4096:
        st = workspace if workspace is not None else torch.empty((3,nrows),device=x.device,dtype=torch.float32)
        _large_stats[(nrows,)](x,st,nrows,m,eps,1024,num_warps=4)
        _large_apply[(nrows,triton.cdiv(m,1024))](x,weight if weight is not None else x,
            bias if bias is not None else x,st,y,b,c,nrows,m,weight is not None,
            bias is not None,1024,num_warps=4)
        return
    gamma = weight if weight is not None else x
    beta = bias if bias is not None else x
    grid = (triton.cdiv(nrows,rows),1,1)
    direct = fast_launch and _DIRECT_SUPPORTED and not _fwd_rows.pre_run_hooks
    if direct:
        # Match the JIT's pointer-divisibility specialization. An offset but
        # contiguous tensor may not have the allocator's usual 16B alignment.
        alignment = ((x.data_ptr()%16==0) | ((gamma.data_ptr()%16==0)<<1)
                     | ((beta.data_ptr()%16==0)<<2) | ((y.data_ptr()%16==0)<<3))
        key = (x.device.index,b,c,nrows,m,eps,rows,num_warps,
               weight is not None,bias is not None,alignment)
        runner = _LAUNCHERS.get(key)
        if runner is not None:
            _LAUNCHERS.move_to_end(key)
            # CompiledKernel obtains the CURRENT stream on each invocation and
            # retains its standard launch hooks. Only four runtime tensors are
            # passed; all shape/affine flags were compiled as constexpr values.
            runner(x,gamma,beta,y)
            return
    compiled = _fwd_rows.run(
        x, gamma, beta, y,
        b, c, nrows, m, eps, weight is not None, bias is not None,
        rows, triton.next_power_of_2(m), num_warps=num_warps,
        grid=grid,warmup=False)
    if direct and compiled is not None:
        _LAUNCHERS[key] = compiled[grid]
        _LAUNCHERS.move_to_end(key)
        if len(_LAUNCHERS)>_MAX_LAUNCHERS:
            _LAUNCHERS.popitem(last=False)


def grouped_instancenorm_fwd(x, weight=None, bias=None, eps=1e-5,
                             *, groups=None, rows=None, num_warps=None,
                             fast_launch=True):
    """Return ReLU(InstanceNorm(x)), without changing group-first layout.

    FP32 contiguous CUDA only. Each (g,b,c) is normalized over H*W using
    population variance (unbiased=False), with independent [g,c] affine values.
    Small planes use one fused launch; >4096 elements use fixed spatial tiles
    (rows=1, num_warps=4) and a three-FP32-scalars-per-plane workspace.
    Intended for the existing manually implemented backward/double backward.
    Omitted rows/num_warps use spatial-size/parallelism rules. Explicit values
    remain available. fast_launch=False uses the regular Triton JIT launcher.
    Fast launch caches compiled code/immutable metadata only, not tensor data.
    When both affine inputs are None, groups defaults to 1 (same result for
    every partition); callers may still pass groups explicitly.
    """
    _, b, c, m = _validate(x, weight, bias, eps, groups, rows, num_warps)
    rows,num_warps = _resolve_config(x.shape[0]*c,m,x.device.index,rows,num_warps)
    # Avoid entering/exiting a CUDA device context on the common current-device
    # path. Explicitly guard the other path for multi-GPU callers.
    y = torch.empty_like(x)
    if torch.cuda.current_device() == x.device.index:
        _launch(x, weight, bias, y, b, c, m, eps, rows, num_warps,fast_launch=fast_launch)
    else:
        with torch.cuda.device(x.device):
            _launch(x, weight, bias, y, b, c, m, eps, rows, num_warps,fast_launch=fast_launch)
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
