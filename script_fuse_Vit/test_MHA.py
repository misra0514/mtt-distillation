from ViT_FlexFused import MultiHeadSelfAttention_Fused

import torch


def compare(name, manual, ref, atol=1e-10, rtol=1e-8):
    diff = (manual - ref).abs()
    max_abs = diff.max().item()
    max_rel = (diff / ref.abs().clamp_min(1.0)).max().item()
    ok = torch.allclose(manual, ref, atol=atol, rtol=rtol)

    print(name)
    print("  allclose:", ok)
    print("  max_abs :", f"{max_abs:.6e}")
    print("  max_rel :", f"{max_rel:.6e}")
    print()


def run_one_test(
    B=2,
    Fuse=3,
    N=5,
    C=12,
    num_heads=3,
    dtype=torch.float64,
    device="cpu",
):
    torch.manual_seed(0)

    assert C % num_heads == 0

    model = MultiHeadSelfAttention_Fused(
        embed_dim=C,
        num_heads=num_heads,
        dropout=0.0,
        Fuse=Fuse,
    ).to(dtype=dtype, device=device)

    x = torch.randn(
        B,
        Fuse,
        N,
        C,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    out, tape = model(x)

    grad_output = torch.randn_like(out)

    # --------------------------------------------------
    # Autograd reference
    # --------------------------------------------------
    loss = (out * grad_output).sum()
    loss.backward()

    dx_ref = x.grad.detach()
    dqkvw_ref = model.qkv.weight.grad.detach()
    dqkvb_ref = model.qkv.bias.grad.detach()
    doutprojw_ref = model.out_proj.weight.grad.detach()
    doutprojb_ref = model.out_proj.bias.grad.detach()

    # --------------------------------------------------
    # Manual backward
    # --------------------------------------------------
    dx_man, d_activates, d_weights, d_weights_all = (
        model.MultiHeadSelfAttention_Fused_bwd(
            tape,
            grad_output.detach(),
            Fuse=Fuse,
        )
    )

    dqkvw_man = d_weights["dqkvw"]
    dqkvb_man = d_weights["dqkvb"]
    doutprojw_man = d_weights["doutprojw"]
    doutprojb_man = d_weights["doutprojb"]

    # --------------------------------------------------
    # Compare
    # --------------------------------------------------
    print("========== MultiHeadSelfAttention_Fused backward test ==========")
    print("out shape:", tuple(out.shape))
    print("x shape  :", tuple(x.shape))
    print()

    compare("dx", dx_man, dx_ref)
    compare("dqkvw", dqkvw_man, dqkvw_ref)
    compare("dqkvb", dqkvb_man, dqkvb_ref)
    compare("doutprojw", doutprojw_man, doutprojw_ref)
    compare("doutprojb", doutprojb_man, doutprojb_ref)

    # Optional: check some intermediate shapes
    print("========== intermediate grad shapes ==========")
    for k, v in d_activates.items():
        print(f"{k:16s}: {tuple(v.shape)}")


if __name__ == "__main__":
    run_one_test(
        B=2,
        Fuse=3,
        N=5,
        C=12,
        num_heads=3,
        dtype=torch.float64,
        device="cpu",
    )