import torch
import torch.nn.functional as F


def grouped_layernorm_forward(x, weight, bias, Fuse, embed_dim, eps=1e-5):
    """
    x:      [B, Fuse, N, D]
    weight: [Fuse * D]
    bias:   [Fuse * D]
    """
    B, Fs, N, D = x.shape
    assert Fs == Fuse
    assert D == embed_dim

    z, mean, rstd = torch.ops.aten.native_layer_norm.default(
        x,
        [D],
        None,
        None,
        eps,
    )

    weight_view = weight.view(1, Fuse, 1, D)
    bias_view = bias.view(1, Fuse, 1, D)

    out = z * weight_view + bias_view

    return out, z, mean, rstd


def grouped_layernorm_backward(
    grad_output,
    x,
    z,
    weight,
    mean,
    rstd,
    Fuse,
    embed_dim,
):
    """
    grad_output: [B, Fuse, N, D]
    x:           [B, Fuse, N, D]
    z:           LayerNorm no-affine output, [B, Fuse, N, D]
    weight:      [Fuse * D]
    mean/rstd:   saved from native_layer_norm forward
    """
    B, Fs, N, D = x.shape
    assert Fs == Fuse
    assert D == embed_dim

    weight_view = weight.view(1, Fuse, 1, D)

    # out = z * weight + bias
    dweight = (grad_output * z).sum(dim=(0, 2)).reshape(Fuse * D)
    dbias = grad_output.sum(dim=(0, 2)).reshape(Fuse * D)

    # grad wrt LayerNorm output
    dz = grad_output * weight_view

    dx, _, _ = torch.ops.aten.native_layer_norm_backward.default(
        dz,
        x,
        [D],
        mean,
        rstd,
        None,
        None,
        [True, False, False],
    )

    return dx, dweight, dbias


def autograd_grouped_layernorm(x, weight, bias, Fuse, embed_dim, eps=1e-5):
    """
    Reference forward using PyTorch autograd.
    """
    z = F.layer_norm(
        x,
        normalized_shape=(embed_dim,),
        weight=None,
        bias=None,
        eps=eps,
    )

    weight_view = weight.view(1, Fuse, 1, embed_dim)
    bias_view = bias.view(1, Fuse, 1, embed_dim)

    out = z * weight_view + bias_view
    return out


def compare(name, manual, ref, atol=1e-10, rtol=1e-8):
    abs_diff = (manual - ref).abs()
    max_abs = abs_diff.max().item()

    denom = ref.abs().clamp_min(1.0)
    max_rel = (abs_diff / denom).max().item()

    ok = torch.allclose(manual, ref, atol=atol, rtol=rtol)

    print(f"{name}:")
    print(f"  allclose = {ok}")
    print(f"  max_abs  = {max_abs:.6e}")
    print(f"  max_rel  = {max_rel:.6e}")
    print()


def run_one_test(
    B=3,
    Fuse=2,
    N=5,
    D=7,
    eps=1e-5,
    dtype=torch.float64,
    device="cpu",
):
    torch.manual_seed(0)

    # -------------------------
    # Create inputs
    # -------------------------
    x_ref = torch.randn(B, Fuse, N, D, dtype=dtype, device=device, requires_grad=True)
    w_ref = torch.randn(Fuse * D, dtype=dtype, device=device, requires_grad=True)
    b_ref = torch.randn(Fuse * D, dtype=dtype, device=device, requires_grad=True)

    # same values for manual path
    x_man = x_ref.detach().clone()
    w_man = w_ref.detach().clone()
    b_man = b_ref.detach().clone()

    grad_output = torch.randn(B, Fuse, N, D, dtype=dtype, device=device)

    # -------------------------
    # Autograd reference
    # -------------------------
    out_ref = autograd_grouped_layernorm(
        x_ref,
        w_ref,
        b_ref,
        Fuse,
        D,
        eps,
    )

    loss_ref = (out_ref * grad_output).sum()
    loss_ref.backward()

    dx_ref = x_ref.grad
    dw_ref = w_ref.grad
    db_ref = b_ref.grad

    # -------------------------
    # Manual backward
    # -------------------------
    out_man, z, mean, rstd = grouped_layernorm_forward(
        x_man,
        w_man,
        b_man,
        Fuse,
        D,
        eps,
    )

    dx_man, dw_man, db_man = grouped_layernorm_backward(
        grad_output,
        x_man,
        z,
        w_man,
        mean,
        rstd,
        Fuse,
        D,
    )

    # -------------------------
    # Compare forward too
    # -------------------------
    compare("forward out", out_man, out_ref.detach())

    # -------------------------
    # Compare backward
    # -------------------------
    compare("dx", dx_man, dx_ref)
    compare("dw", dw_man, dw_ref)
    compare("db", db_man, db_ref)


if __name__ == "__main__":
    run_one_test(
        B=3,
        Fuse=2,
        N=5,
        D=7,
        eps=1e-5,
        dtype=torch.float64,
        device="cpu",
    )