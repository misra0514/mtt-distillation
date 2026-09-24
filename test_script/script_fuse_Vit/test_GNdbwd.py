import math
import argparse
import torch


def grouped_layernorm_forward(x, weight, bias, Fuse, eps=1e-5):
    """Forward that matches grouped_layernorm_backward: LN over D, affine per Fuse,D."""
    B, Fs, N, D = x.shape
    assert Fs == Fuse
    z, mean, rstd = torch.ops.aten.native_layer_norm.default(x, [D], None, None, eps)
    weight_view = weight.view(1, Fuse, 1, D)
    bias_view = bias.view(1, Fuse, 1, D)
    return z * weight_view + bias_view


def grouped_layernorm_backward(
    x,             # original input, [B, Fuse, N, D]
    weight,        # [Fuse * D]
    Fuse,
    grad_output,   # [B, Fuse, N, D]
    eps=1e-5,
):
    B, Fs, N, D = x.shape
    assert Fs == Fuse
    normalized_shape = [D]
    z, mean, rstd = torch.ops.aten.native_layer_norm.default(x, [D], None, None, eps)
    weight_view = weight.view(1, Fuse, 1, D)
    dweight = (grad_output * z).sum(dim=(0, 2))      # [Fuse, D]
    dbias = grad_output.sum(dim=(0, 2))              # [Fuse, D]
    dweight = dweight.reshape(Fuse * D)
    dbias = dbias.reshape(Fuse * D)
    dz = grad_output * weight_view                   # [B, Fuse, N, D]
    dx, _, _ = torch.ops.aten.native_layer_norm_backward.default(
        dz,                 # grad_out for LN output
        x,                  # original LN input
        normalized_shape,
        mean,
        rstd,
        None,               # LN weight was None
        None,               # LN bias was None
        [True, False, False] # only need dx
    )
    return dx, dweight, dbias


# Paste/import your stateless LayerNorm double backward here.
def layerNorm_double_backwards_fn(
    x,
    gamma,
    ggX,
    ggG,
    ggB,
    gO,
    normalized_shape,
    eps=1e-5,
):
    """
    LayerNorm double backward.

    Corresponds to first backward:
        dX, dGamma, dBeta = layer_norm_backward(gO, x, gamma, beta)

    Returns:
        gX:   grad wrt x
        gG:   grad wrt gamma
        ggO:  grad wrt gO
    """
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    else:
        normalized_shape = tuple(normalized_shape)

    assert tuple(x.shape[-len(normalized_shape):]) == normalized_shape, \
        f"x.shape={tuple(x.shape)} does not end with normalized_shape={normalized_shape}"

    M = math.prod(normalized_shape)
    K = x.numel() // M

    x_flat = x.reshape(K, M)
    gO_flat = gO.reshape(K, M)

    with torch.no_grad():
        mean = x_flat.mean(dim=1, keepdim=True)
        var = x_flat.var(dim=1, unbiased=False, keepdim=True)

        inv_std = torch.rsqrt(var + eps)
        x_centered = x_flat - mean
        x_hat = x_centered * inv_std

        inv_std3 = inv_std / (var + eps)

    if gamma is not None:
        gamma_flat = gamma.reshape(1, M)
    else:
        gamma_flat = None

    def first_back_no_weight(g):
        return inv_std * (
            g
            - g.mean(dim=1, keepdim=True)
            - x_hat * (g * x_hat).mean(dim=1, keepdim=True)
        )

    gX_flat = None

    if ggX is not None:
        ggX_flat = ggX.reshape(K, M)

        with torch.no_grad():
            if gamma_flat is not None:
                b = gO_flat * gamma_flat
            else:
                b = gO_flat

            a = ggX_flat

            sum_a = a.sum(dim=1, keepdim=True)
            sum_b = b.sum(dim=1, keepdim=True)

            sum_a_xmu = (a * x_centered).sum(dim=1, keepdim=True)
            sum_b_xmu = (b * x_centered).sum(dim=1, keepdim=True)

            dot_ab = (a * b).sum(dim=1, keepdim=True)

            A = (
                (sum_a * sum_b) / M
                - dot_ab
                + 3.0 * (inv_std ** 2) * sum_a_xmu * sum_b_xmu / M
            )

            term0 = x_centered * inv_std3 * A / M
            term1 = sum_a_xmu * inv_std3 * (sum_b / M - b) / M
            term2 = sum_b_xmu * inv_std3 * (sum_a / M - a) / M

            gX_flat = term0 + term1 + term2

    if ggG is not None:
        ggG_flat = ggG.reshape(1, M)

        with torch.no_grad():
            gX_G = first_back_no_weight(gO_flat * ggG_flat)

        gX_flat = gX_G if gX_flat is None else gX_flat + gX_G

    gG = None

    if gamma is not None and ggX is not None:
        ggX_flat = ggX.reshape(K, M)

        with torch.no_grad():
            rP_ggX = first_back_no_weight(ggX_flat)
            gG_flat = (gO_flat * rP_ggX).sum(dim=0)

        gG = gG_flat.reshape(normalized_shape)

    ggO_flat = None

    if ggX is not None:
        ggX_flat = ggX.reshape(K, M)

        with torch.no_grad():
            rP_ggX = first_back_no_weight(ggX_flat)

            if gamma_flat is not None:
                ggO_X = rP_ggX * gamma_flat
            else:
                ggO_X = rP_ggX

        ggO_flat = ggO_X

    if ggG is not None:
        ggG_flat = ggG.reshape(1, M)

        with torch.no_grad():
            ggO_G = ggG_flat * x_hat

        ggO_flat = ggO_G if ggO_flat is None else ggO_flat + ggO_G

    if ggB is not None:
        ggB_flat = ggB.reshape(1, M)

        with torch.no_grad():
            ggO_B = ggB_flat.expand(K, M)

        ggO_flat = ggO_B if ggO_flat is None else ggO_flat + ggO_B

    gX = gX_flat.reshape_as(x) if gX_flat is not None else None
    ggO = ggO_flat.reshape_as(x) if ggO_flat is not None else None

    return gX, gG, ggO


def grouped_layernorm_double_backward_manual(x, weight, Fuse, ggX, ggW, ggB, gO, eps=1e-5):
    """
    Manual double backward for grouped_layernorm_backward.

    Because grouped weight is [Fuse, D], call the ordinary LN double backward once
    per fuse group with gamma = weight[f].

    Inputs are cotangents of first backward outputs:
        ggX: cotangent for dx,      [B, Fuse, N, D]
        ggW: cotangent for dweight, [Fuse * D]
        ggB: cotangent for dbias,   [Fuse * D]

    Returns gradients wrt first backward inputs:
        gX:  grad wrt x,           [B, Fuse, N, D]
        gW:  grad wrt weight,      [Fuse * D]
        ggO: grad wrt grad_output, [B, Fuse, N, D]
    """
    B, Fs, N, D = x.shape
    assert Fs == Fuse

    gX = torch.zeros_like(x) if ggX is not None or ggW is not None else None
    gW = torch.zeros_like(weight)
    ggO = torch.zeros_like(gO) if ggX is not None or ggW is not None or ggB is not None else None

    ggW_view = ggW.view(Fuse, D) if ggW is not None else None
    ggB_view = ggB.view(Fuse, D) if ggB is not None else None
    weight_view = weight.view(Fuse, D)

    for f in range(Fuse):
        x_f = x[:, f, :, :]                  # [B, N, D]
        gamma_f = weight_view[f]             # [D]
        gO_f = gO[:, f, :, :]                # [B, N, D]
        ggX_f = ggX[:, f, :, :] if ggX is not None else None
        ggW_f = ggW_view[f] if ggW is not None else None
        ggB_f = ggB_view[f] if ggB is not None else None

        gX_f, gW_f, ggO_f = layerNorm_double_backwards_fn(
            x=x_f,
            gamma=gamma_f,
            ggX=ggX_f,
            ggG=ggW_f,
            ggB=ggB_f,
            gO=gO_f,
            normalized_shape=(D,),
            eps=eps,
        )

        if gX_f is not None:
            gX[:, f, :, :] = gX_f
        if gW_f is not None:
            gW.view(Fuse, D)[f] = gW_f
        if ggO_f is not None:
            ggO[:, f, :, :] = ggO_f

    return gX, gW, ggO


def clone_leaf(t):
    return t.detach().clone().requires_grad_(True)


def zero_if_none(g, like):
    return torch.zeros_like(like) if g is None else g


def err_stats(name, got, ref, rtol, atol):
    diff = (got - ref).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    max_rel = (diff / (ref.abs() + atol)).max().item() if diff.numel() else 0.0
    ok = torch.allclose(got, ref, rtol=rtol, atol=atol)
    print(f"{name:>10s}: ok={str(ok):5s}  max_abs={max_abs:.6e}  max_rel={max_rel:.6e}")
    if not ok:
        idx = diff.reshape(-1).argmax().item()
        print(f"    worst got={got.reshape(-1)[idx].item():.12e}, ref={ref.reshape(-1)[idx].item():.12e}")
    return ok


def run_one_case(B, Fuse, N, D, dtype, device, eps, seed, rtol, atol):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    shape = (B, Fuse, N, D)
    # Keep values moderate. Double backward can be sensitive when variance is tiny.
    x0 = torch.randn(shape, dtype=dtype, device=device) * 0.7 + 0.1
    w0 = torch.randn(Fuse * D, dtype=dtype, device=device) * 0.5 + 1.0
    b0 = torch.randn(Fuse * D, dtype=dtype, device=device) * 0.2
    gO0 = torch.randn(shape, dtype=dtype, device=device) * 0.7

    ggX = torch.randn(shape, dtype=dtype, device=device)
    ggW = torch.randn(Fuse * D, dtype=dtype, device=device)
    ggB = torch.randn(Fuse * D, dtype=dtype, device=device)

    print(f"\n=== case B={B}, Fuse={Fuse}, N={N}, D={D}, dtype={dtype}, device={device}, eps={eps}, seed={seed} ===")

    # ------------------------------------------------------------
    # 1) First backward value check: stateless backward vs autograd.
    # ------------------------------------------------------------
    x = clone_leaf(x0)
    w = clone_leaf(w0)
    b = clone_leaf(b0)
    gO = clone_leaf(gO0)

    y = grouped_layernorm_forward(x, w, b, Fuse, eps)
    loss = (y * gO).sum()  # linear loss => grad_output is exactly gO
    dX_ref, dW_ref, dB_ref = torch.autograd.grad(loss, (x, w, b), create_graph=True)

    dX_man, dW_man, dB_man = grouped_layernorm_backward(x, w, Fuse, gO, eps)

    ok1 = True
    print("\n[first backward check]")
    ok1 &= err_stats("dX", dX_man, dX_ref, rtol, atol)
    ok1 &= err_stats("dWeight", dW_man, dW_ref, rtol, atol)
    ok1 &= err_stats("dBias", dB_man, dB_ref, rtol, atol)

    # ------------------------------------------------------------
    # 2) Double backward check.
    #    Reference: autograd first grad with create_graph=True, then grandloss.backward().
    #    Manual: call grouped_layernorm_double_backward_manual.
    # ------------------------------------------------------------
    x = clone_leaf(x0)
    w = clone_leaf(w0)
    b = clone_leaf(b0)
    gO = clone_leaf(gO0)

    y = grouped_layernorm_forward(x, w, b, Fuse, eps)
    loss = (y * gO).sum()  # grad_output is exactly gO, and gO is an independent leaf
    dX_ref, dW_ref, dB_ref = torch.autograd.grad(loss, (x, w, b), create_graph=True)

    grandloss = (dX_ref * ggX).sum() + (dW_ref * ggW).sum() + (dB_ref * ggB).sum()
    grandloss.backward()

    ref_gX = zero_if_none(x.grad, x)
    ref_gW = zero_if_none(w.grad, w)
    ref_gO = zero_if_none(gO.grad, gO)
    ref_gB = zero_if_none(b.grad, b)  # should be zero: dX/dW/dB do not depend on bias here

    man_gX, man_gW, man_gO = grouped_layernorm_double_backward_manual(
        x=x.detach(),
        weight=w.detach(),
        Fuse=Fuse,
        ggX=ggX,
        ggW=ggW,
        ggB=ggB,
        gO=gO.detach(),
        eps=eps,
    )
    man_gB = torch.zeros_like(b)

    ok2 = True
    print("\n[double backward check]")
    ok2 &= err_stats("gX", man_gX, ref_gX, rtol, atol)
    ok2 &= err_stats("gWeight", man_gW, ref_gW, rtol, atol)
    ok2 &= err_stats("ggO", man_gO, ref_gO, rtol, atol)
    ok2 &= err_stats("gBias", man_gB, ref_gB, rtol, atol)

    return ok1 and ok2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    dtype = {"float64": torch.float64, "float32": torch.float32}[args.dtype]
    if args.rtol is None:
        args.rtol = 1e-7 if dtype == torch.float64 else 3e-3
    if args.atol is None:
        args.atol = 1e-9 if dtype == torch.float64 else 3e-4

    torch.set_printoptions(precision=8, sci_mode=True)

    cases = [
        (2, 1, 3, 4),
        (2, 3, 4, 5),
        (1, 4, 2, 7),
        (3, 2, 1, 8),
    ]

    all_ok = True
    for i, (B, Fuse, N, D) in enumerate(cases):
        all_ok &= run_one_case(
            B=B,
            Fuse=Fuse,
            N=N,
            D=D,
            dtype=dtype,
            device=args.device,
            eps=args.eps,
            seed=args.seed + i,
            rtol=args.rtol,
            atol=args.atol,
        )

    print("\nRESULT:", "PASS" if all_ok else "FAIL")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
