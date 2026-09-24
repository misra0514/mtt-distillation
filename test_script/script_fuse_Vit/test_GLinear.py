import torch
import torch.nn as nn


class GroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, Fuse):
        super().__init__()
        self.Fuse = Fuse
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(Fuse * out_features, in_features))
        self.bias = nn.Parameter(torch.randn(Fuse * out_features))

    def forward(self, x):
        W = self.weight.view(
            self.Fuse,
            self.out_features,
            self.in_features,
        ).transpose(-1, -2)  # [F, I, O]

        if x.ndim == 3:
            x = x.view(-1, self.Fuse, self.in_features)
            x = torch.einsum("bfi,fio->bfo", x, W)
            x = x + self.bias.view(1, self.Fuse, self.out_features)

        elif x.ndim == 4:
            B, Fs, N, I = x.shape
            x = torch.einsum("bfni,fio->bfno", x, W)
            x = x + self.bias.view(1, self.Fuse, 1, self.out_features)

        return x


def grouped_linear_bwd(x, w, grad_output, Fuse=2):
    in_features = w.shape[1]
    out_features = w.shape[0] // Fuse
    W = w.view(Fuse, out_features, in_features)  # [F, O, I]

    if x.ndim == 3:
        B, Fs, I = x.shape

        dx = torch.einsum(
            "bfo,foi->bfi",
            grad_output,
            W,
        )

        dw = torch.einsum(
            "bfo,bfi->foi",
            grad_output,
            x,
        )

        db = grad_output.sum(dim=0)  # [F, O]

        dx = dx.reshape_as(x)
        dw = dw.reshape(Fuse * out_features, in_features)
        db = db.reshape(Fuse * out_features)

        return dx, dw, db

    elif x.ndim == 4:
        B, Fs, N, I = x.shape

        dx = torch.einsum(
            "bfno,foi->bfni",
            grad_output,
            W,
        )

        dw = torch.einsum(
            "bfno,bfni->foi",
            grad_output,
            x,
        )

        db = grad_output.sum(dim=(0, 2))  # [F, O]

        dw = dw.reshape(Fuse * out_features, in_features)
        db = db.reshape(Fuse * out_features)

        return dx, dw, db

    else:
        raise ValueError(f"Unsupported x.ndim={x.ndim}")


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


def test_3d():
    torch.manual_seed(0)

    B = 4
    Fuse = 3
    In = 5
    Out = 7
    dtype = torch.float64
    device = "cpu"

    model = GroupedLinear(In, Out, Fuse).to(dtype).to(device)

    x = torch.randn(B, Fuse, In, dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn(B, Fuse, Out, dtype=dtype, device=device)

    # autograd reference
    out = model(x)
    loss = (out * grad_output).sum()
    loss.backward()

    dx_ref = x.grad.detach()
    dw_ref = model.weight.grad.detach()
    db_ref = model.bias.grad.detach()

    # manual backward
    dx_man, dw_man, db_man = grouped_linear_bwd(
        x.detach(),
        model.weight.detach(),
        grad_output.detach(),
        Fuse=Fuse,
    )

    print("========== 3D input test ==========")
    compare("dx", dx_man, dx_ref)
    compare("dw", dw_man, dw_ref)
    compare("db", db_man, db_ref)


def test_4d():
    torch.manual_seed(1)

    B = 2
    Fuse = 3
    N = 4
    In = 5
    Out = 7
    dtype = torch.float64
    device = "cpu"

    model = GroupedLinear(In, Out, Fuse).to(dtype).to(device)

    x = torch.randn(B, Fuse, N, In, dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn(B, Fuse, N, Out, dtype=dtype, device=device)

    # autograd reference
    out = model(x)
    loss = (out * grad_output).sum()
    loss.backward()

    dx_ref = x.grad.detach()
    dw_ref = model.weight.grad.detach()
    db_ref = model.bias.grad.detach()

    # manual backward
    dx_man, dw_man, db_man = grouped_linear_bwd(
        x.detach(),
        model.weight.detach(),
        grad_output.detach(),
        Fuse=Fuse,
    )

    print("========== 4D input test ==========")
    compare("dx", dx_man, dx_ref)
    compare("dw", dw_man, dw_ref)
    compare("db", db_man, db_ref)


if __name__ == "__main__":
    test_3d()
    test_4d()