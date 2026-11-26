"""Stateless FP32 group-first InstanceNorm+ReLU double backward.

Activation order: [G*B,C,H,W], leading index g*B+b. Affine vectors: [G*C].
Return order follows the project's manual double backward: (ggO, gX, gG).
No autograd binding, saved statistics cache, input packing, or atomic reduction.
vfuse=True applies ReLU gates inside norm kernels; False runs them separately.
grouped_instancenorm_only_double_bwd differentiates norm alone.
"""
import math
import torch
import triton
import triton.language as tl

if __package__:
    from . import _grouped_norm_large as _large
else:
    import _grouped_norm_large as _large


@triton.jit
def _relu_backward(DY, OUT, D_IN, N: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0).to(tl.int64)*BLOCK + tl.arange(0, BLOCK)
    valid = off < N
    dy = tl.load(DY+off, valid, other=0.).to(tl.float32)
    out = tl.load(OUT+off, valid, other=0.).to(tl.float32)
    tl.store(D_IN+off, tl.where(out > 0., dy, 0.), valid)


def _mask_relu(output, grad_output, destination=None):
    """Validated tensors; destination may alias dY for our newly allocated ggO."""
    result = torch.empty_like(grad_output) if destination is None else destination
    _relu_backward[(triton.cdiv(grad_output.numel(), 1024),)](
        grad_output, output, result, grad_output.numel(), 1024, num_warps=4)
    return result


@triton.jit
def _double_rows(X, GAMMA, H, V, T, D, OUT, GX, GG, GGO, PART,
                 B: tl.constexpr, C: tl.constexpr, ROWS: tl.constexpr,
                 M: tl.constexpr, EPS: tl.constexpr, R: tl.constexpr,
                 S: tl.constexpr, HAS_GAMMA: tl.constexpr,
                 HAS_H: tl.constexpr, HAS_V: tl.constexpr, HAS_T: tl.constexpr,
                 VFUSE: tl.constexpr = True):
    row = tl.program_id(0) * R + tl.arange(0, R)
    col = tl.arange(0, S)
    valid_row = row < ROWS
    valid = valid_row[:, None] & (col[None, :] < M)
    off = row[:, None].to(tl.int64) * M + col[None, :]
    affine = (row // (B * C)) * C + row % C
    gx = tl.full((R, S), 0., tl.float32)
    ggo = tl.full((R, S), 0., tl.float32)
    if VFUSE and ((HAS_H or HAS_V) or HAS_T):
        gate = tl.load(OUT + off, valid, other=0.).to(tl.float32) > 0.
    if HAS_H or HAS_V:
        x = tl.load(X + off, valid, other=0.).to(tl.float32)
        anchor = tl.load(X + row.to(tl.int64)*M, valid_row, other=0.).to(tl.float32)
        shifted = tl.where(col[None, :] < M, x - anchor[:, None], 0.)
        mean_shift = tl.sum(shifted, 1) / M
        centered = tl.where(col[None, :] < M, shifted - mean_shift[:, None], 0.)
        inv = tl.rsqrt(tl.sum(centered*centered, 1)/M + EPS)
        z = centered * inv[:, None]
        d = tl.load(D + off, valid, other=0.).to(tl.float32)
        if VFUSE:
            d = tl.where(gate, d, 0.)
        d0 = tl.sum(d, 1)/M
        dz = tl.sum(d*z, 1)/M
        pd = d - d0[:, None] - z*dz[:, None]
        v = tl.full((R,), 0., tl.float32)
        if HAS_V:
            v = tl.load(V + affine, valid_row, other=0.).to(tl.float32)
        if HAS_H:
            h = tl.load(H + off, valid, other=0.).to(tl.float32)
            h0 = tl.sum(h, 1)/M
            hz = tl.sum(h*z, 1)/M
            # Project first, then take the dot. This both avoids cancellation in
            # E[h*d]-E[h]E[d]-E[h*z]E[d*z] and reuses the dgamma reduction.
            dot = tl.sum(h*pd, 1)
            q = dot/M
            ph = h-h0[:, None]-z*hz[:, None]
            gamma = tl.full((R,), 1., tl.float32)
            if HAS_GAMMA:
                gamma = tl.load(GAMMA + affine, valid_row, other=0.).to(tl.float32)
            scale = gamma*inv*inv
            gx = pd*(v*inv-scale*hz)[:, None] - scale[:, None]*(dz[:, None]*ph+z*q[:, None])
            ggo = ph*(gamma*inv)[:, None]
            if HAS_GAMMA:
                part = dot*inv
                if B == 1:
                    tl.store(GG + affine, part, valid_row)
                else:
                    tl.store(PART + row, part, valid_row)
        else:
            gx = pd*(v*inv)[:, None]
        if HAS_V:
            ggo += z*v[:, None]

    if HAS_T:
        t = tl.load(T + affine, valid_row, other=0.).to(tl.float32)
        ggo += t[:, None]
    if VFUSE and ((HAS_H or HAS_V) or HAS_T):
        ggo = tl.where(gate, ggo, 0.)
    # Write explicit zeros even when all optional inputs are None.
    tl.store(GX + off, gx, valid)
    tl.store(GGO + off, ggo, valid)


@triton.jit
def _double_reduce(PART, GG, B: tl.constexpr, C: tl.constexpr,
                   BC: tl.constexpr, BB: tl.constexpr):
    c = tl.program_id(0)*BC + tl.arange(0, BC)
    g = tl.program_id(1)
    b = tl.arange(0, BB)
    acc = tl.full((BC,), 0., tl.float32)
    for block in tl.static_range((B + BB - 1)//BB):
        bi = b + block*BB
        off = (g.to(tl.int64)*B + bi[:, None])*C + c[None, :]
        mask = (bi[:, None] < B) & (c[None, :] < C)
        acc += tl.sum(tl.load(PART + off, mask, other=0.), 0)
    tl.store(GG + g*C + c, acc, c < C)


def _validate(x, weight, h, v, t, d, out, eps, groups, *, require_output=True):
    if require_output and out is None:
        raise ValueError('Both vfuse modes require the saved post-ReLU output')
    if (x.ndim != 4 or x.shape != d.shape
            or (out is not None and x.shape != out.shape)):
        raise ValueError('x/grad_output/output must be [G*B,C,H,W]')
    gb, c, hi, wi = x.shape
    if min(gb, c, hi, wi) <= 0 or hi*wi <= 1:
        raise ValueError('Positive dimensions and H*W>=2 required')
    vectors = [a for a in (weight, v, t) if a is not None]
    inferred = vectors[0].numel()//c if vectors else 1
    g = inferred if groups is None else groups
    if not isinstance(g, int) or g < 1 or gb % g:
        raise ValueError('groups must be a positive integer dividing G*B')
    for a in vectors:
        if a.ndim != 1 or a.numel() != g*c:
            raise ValueError('weight/ggw/ggb must be flat [G*C] vectors')
    if h is not None and h.shape != x.shape:
        raise ValueError('ggI must have the same shape as x')
    for a in (x, d, out, h, weight, v, t):
        if a is None: continue
        if not a.is_cuda or a.device != x.device or a.dtype != torch.float32:
            raise ValueError('All tensors must be FP32 CUDA tensors on the same device')
        if not a.is_contiguous():
            raise ValueError('All tensors must be contiguous; no implicit packing')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')
    return g, gb//g, c, hi*wi


def _allocate(x, g, b, c, has_g):
    gx, ggo = torch.empty_like(x), torch.empty_like(x)
    gg = torch.empty((g*c,), device=x.device, dtype=torch.float32) if has_g else None
    part = torch.empty((g*b*c,), device=x.device, dtype=torch.float32) if has_g and b > 1 else gg
    return ggo, gx, gg, part


def _launch_rows(x, weight, h, v, t, d, out, buffers, g, b, c, m, eps, rows, warps,
                 vfuse=True):
    if m > 4096:
        _large.double_backward(x, weight, h, v, t, d, out, buffers, b, c, m, eps,
                               vfuse=vfuse)
        return
    ggo, gx, gg, part = buffers
    # Absent arguments are compile-time eliminated; use x as a dummy pointer.
    ptr = lambda a: x if a is None else a
    _double_rows[(triton.cdiv(g*b*c, rows),)](
        x, ptr(weight), ptr(h), ptr(v), ptr(t), d, ptr(out),
        gx, ptr(gg), ggo, ptr(part), b, c, g*b*c, m, eps,
        rows, triton.next_power_of_2(m), weight is not None,
        h is not None, v is not None, t is not None, VFUSE=vfuse, num_warps=warps)


def _launch_reduce(buffers, g, b, c):
    _, _, gg, part = buffers
    if gg is not None and b > 1:
        bb = min(triton.next_power_of_2(b), 1024)
        _double_reduce[(triton.cdiv(c, 8), g)](part, gg, b, c, 8, bb, num_warps=4)


def _validate_config(rows, num_warps, m):
    if rows not in (1, 4) or num_warps not in (4, 8):
        raise ValueError('rows must be 1/4 and num_warps must be 4/8')
    if rows != 1 and m > 256:
        raise ValueError('rows4 is an explicit candidate for H*W<=256')


def _run(ggI, ggw, ggb, grad_output, output, weight, x, eps, groups,
          rows, num_warps, fuse_relu):
    g, b, c, m = _validate(x, weight, ggI, ggw, ggb, grad_output, output, eps,
                           groups, require_output=fuse_relu)
    _validate_config(rows, num_warps, m)
    with torch.cuda.device(x.device):
        buffers = _allocate(x, g, b, c, weight is not None and ggI is not None)
        _launch_rows(x, weight, ggI, ggw, ggb, grad_output, output,
                     buffers, g, b, c, m, eps, rows, num_warps, fuse_relu)
        _launch_reduce(buffers, g, b, c)
    return buffers[:3]


def grouped_instancenorm_double_bwd(ggI, ggw, ggb, grad_output, output, weight, x,
                                    vfuse=True, eps=1e-5, *, groups=None,
                                    rows=1, num_warps=4, v_fuse=None):
    """InstanceNorm+ReLU double backward -> (ggO, gX, gG) in both modes.

    x/ggI/grad_output/output: [G*B,C,H,W], leading index g*B+b.
    weight/ggw/ggb: [G*C]; ggI/ggw/ggb and weight may be None.
    vfuse=True: gate grad_output and ggO inside the norm kernels.
    vfuse=False: gate grad_output separately, call norm-only double backward,
    then gate ggO separately. output must be saved post-ReLU in both modes.
    Inputs are never modified. Only our new ggO is gated in place.
    The legacy v_fuse keyword, when supplied, overrides vfuse.
    G is inferred from affine vectors; pass groups if none exist and G>1.
    gG is None if weight or ggI is None. H*W>4096 uses bounded spatial tiles.
    Explicit tiling only; no selector or autotuning.
    """
    if v_fuse is not None:
        vfuse = v_fuse
    if not isinstance(vfuse, bool):
        raise TypeError('vfuse must be bool')
    if vfuse:
        return _run(ggI, ggw, ggb, grad_output, output, weight, x, eps, groups,
                     rows, num_warps, True)
    _, _, _, m = _validate(x, weight, ggI, ggw, ggb, grad_output, output, eps, groups)
    _validate_config(rows, num_warps, m)
    with torch.cuda.device(x.device):
        grad_norm = _mask_relu(output, grad_output)
        ggO, gX, gG = grouped_instancenorm_only_double_bwd(
            ggI, ggw, ggb, grad_norm, weight, x, eps,
            groups=groups, rows=rows, num_warps=num_warps)
        # Chain back through ReLU a second time. gX/gG must NOT be gated here.
        _mask_relu(output, ggO, destination=ggO)
    return ggO, gX, gG


def grouped_instancenorm_only_double_bwd(ggI, ggw, ggb, grad_output, weight, x,
                                         eps=1e-5, *, groups=None,
                                         rows=1, num_warps=4):
    """Pure InstanceNorm double backward -> (ggO, gX, gG), with no ReLU gates.

    No saved output is needed. grad_output is the upstream gradient at the norm
    output. Shapes, optional inputs, groups and tiling match the combined API.
    gG is None if weight or ggI is None. Supports small and ImageNet spatial sizes.
    """
    return _run(ggI, ggw, ggb, grad_output, None, weight, x, eps, groups,
                 rows, num_warps, False)


def to_old(x, g):
    if x is None: return None
    gb, c, h, w = x.shape
    return x.view(g, gb//g, c, h, w).transpose(0, 1).reshape(gb//g, g*c, h, w)


def to_new(x, g):
    if x is None: return None
    b, gc, h, w = x.shape
    return x.view(b, g, gc//g, h, w).transpose(0, 1).reshape(g*b, gc//g, h, w)


# Supplied baseline: Torch preprocessing + blocked Triton + atomic dgamma.
# Only correctness fix: explicitly zero gX when both ggX and ggG are absent;
# the supplied empty() result was unwritten in that optional-input case.
def _baseline_prepare(x, gamma, ggX, ggG, ggB, gO, out, eps=1e-5):
    n, c, h, w = x.shape
    m = h*w
    xf = x.contiguous().view(n, c, m)
    of = out.contiguous().view(n, c, m)
    df = gO.contiguous().view(n, c, m)
    relu_mask = of > 0
    di = (df*relu_mask).to(torch.float32)
    xx = xf.to(torch.float32)
    mean = xx.mean(2)
    var = xx.var(2, unbiased=False)
    inv = torch.rsqrt(var+eps)
    xc = xx-mean.unsqueeze(2)
    sd = di.sum(2)
    sdx = (di*xc).sum(2)
    empty = lambda: torch.empty(1, device=x.device, dtype=torch.float32)
    if ggX is not None:
        hh = ggX.contiguous().view(n, c, m).to(torch.float32)
        sh = hh.sum(2)
        shx = (hh*xc).sum(2)
        hd = (hh*di).sum(2)
    else:
        hh, sh, shx, hd = empty(), empty(), empty(), empty()
    gf = gamma.to(torch.float32) if gamma is not None else empty()
    vf = ggG.to(torch.float32) if ggG is not None else empty()
    tf = ggB.to(torch.float32) if ggB is not None else empty()
    return dict(args=(di, hh, of, gf, vf, tf, var.contiguous(), inv.contiguous(), xc.contiguous(),
                     sd.contiguous(), sdx.contiguous(), sh.contiguous(), shx.contiguous(), hd.contiguous()),
                shape=x.shape, x=x, eps=eps,
                flags=(gamma is not None, ggX is not None, ggG is not None, ggB is not None),
                retained=(relu_mask, mean))


def _baseline_allocate(p):
    n, c, h, w = p['shape']
    x = p['x']
    hg, hh, hv, ht = p['flags']
    gx = torch.empty((n, c, h*w), device=x.device, dtype=torch.float32)
    if not hh and not hv: gx.zero_()
    ggo = torch.empty_like(gx)
    gg = torch.zeros((c,), device=x.device, dtype=torch.float32) if hg and hh else None
    return gx, gg, ggo


def _baseline_launch(p, buffers, BLOCK_M=256, zero_gg=False):
    n, c, h, w = p['shape']
    gx, gg, ggo = buffers
    if zero_gg and gg is not None: gg.zero_()
    dummy = p['x']
    instance_norm_double_backward_kernel_blocked[(n*c, triton.cdiv(h*w, BLOCK_M))](
        *p['args'], gx, gg if gg is not None else dummy, ggo, h*w, p['eps'],
        *p['flags'], c, BLOCK_M, num_warps=4)


@torch.no_grad()
def instanceNorm_double_backwards_triton(x, gamma, ggX, ggG, ggB, gO, out,
                                         eps=1e-5, BLOCK_M=256):
    p = _baseline_prepare(x, gamma, ggX, ggG, ggB, gO, out, eps)
    buffers = _baseline_allocate(p)
    _baseline_launch(p, buffers, BLOCK_M)
    gx, gg, ggo = buffers
    return gx.view_as(x), gg, ggo.view_as(x)


@triton.jit
def instance_norm_double_backward_kernel_blocked(
    gO_ptr, ggX_ptr, out_ptr, gamma_ptr, ggG_ptr, ggB_ptr,
    var_ptr, inv_std_ptr, xcm_ptr, sum_gO_ptr, sum_gO_xmu_ptr,
    sum_ggX_ptr, sum_ggX_xmu_ptr, dot_ggX_gO_ptr, gX_ptr, gG_ptr, ggO_ptr,
    M, eps: tl.constexpr, has_gamma: tl.constexpr, has_ggX: tl.constexpr,
    has_ggG: tl.constexpr, has_ggB: tl.constexpr, C: tl.constexpr, BLOCK_M: tl.constexpr):
    nc = tl.program_id(0)
    c = nc % C
    j = tl.program_id(1)*BLOCK_M + tl.arange(0, BLOCK_M)
    mask = j < M
    off = nc.to(tl.int64)*M + j
    var = tl.load(var_ptr+nc).to(tl.float32)
    r = tl.load(inv_std_ptr+nc).to(tl.float32)
    r2, r3 = r*r, r*r*r
    mf = tl.full([], M, tl.float32)
    im = 1./mf
    sd = tl.load(sum_gO_ptr+nc).to(tl.float32)
    sdx = tl.load(sum_gO_xmu_ptr+nc).to(tl.float32)
    gamma = tl.full([], 1., tl.float32)
    if has_gamma: gamma = tl.load(gamma_ptr+c).to(tl.float32)
    v = tl.full([], 0., tl.float32)
    if has_ggG: v = tl.load(ggG_ptr+c).to(tl.float32)
    d = tl.load(gO_ptr+off, mask, other=0.).to(tl.float32)
    xc = tl.load(xcm_ptr+off, mask, other=0.).to(tl.float32)
    h = tl.full((BLOCK_M,), 0., tl.float32)
    sh, shx, hd = tl.full([], 0., tl.float32), tl.full([], 0., tl.float32), tl.full([], 0., tl.float32)
    if has_ggX:
        h = tl.load(ggX_ptr+off, mask, other=0.).to(tl.float32)
        sh = tl.load(sum_ggX_ptr+nc).to(tl.float32)
        shx = tl.load(sum_ggX_xmu_ptr+nc).to(tl.float32)
        hd = tl.load(dot_ggX_gO_ptr+nc).to(tl.float32)
    gx = tl.full((BLOCK_M,), 0., tl.float32)
    if has_ggX:
        a = sh*sd*im-hd+(3.*r2*sdx*shx)*im
        t0 = xc*r3*a*im
        t1 = shx*r3*(sd*im-d)*im
        t2 = sdx*r3*(sh*im-h)*im
        value = t0+t1+t2
        if has_gamma: value = value*gamma
        gx += value
    if has_ggG:
        t0 = d*r
        t1 = -(r*sd)*im
        t2 = -(xc*r3*sdx)*im
        gx += v*(t0+t1+t2)
    if has_ggX or has_ggG: tl.store(gX_ptr+off, gx, mask)
    if has_gamma and has_ggX:
        fb = (r*im)*(mf*d-sd-xc*r2*sdx)
        tl.atomic_add(gG_ptr+c, tl.sum(h*fb, 0))
    ggo = tl.full((BLOCK_M,), 0., tl.float32)
    if has_ggX: ggo += (gamma*r*im)*(mf*h-sh-xc*r2*shx)
    if has_ggG: ggo += v*xc*r
    if has_ggB: ggo += tl.load(ggB_ptr+c).to(tl.float32)
    ov = tl.load(out_ptr+off, mask, other=0.).to(tl.float32)
    tl.store(ggO_ptr+off, tl.where(ov > 0., ggo, 0.), mask)


def baseline_grouped_instancenorm_double_bwd(ggI, ggw, ggb, grad_output, output,
                                             weight, x, eps=1e-5, *, groups=None):
    g, _, _, _ = _validate(x, weight, ggI, ggw, ggb, grad_output, output, eps, groups)
    gx, gg, ggo = instanceNorm_double_backwards_triton(
        to_old(x, g), weight, to_old(ggI, g), ggw, ggb,
        to_old(grad_output, g), to_old(output, g), eps)
    return to_new(ggo, g), to_new(gx, g), gg
