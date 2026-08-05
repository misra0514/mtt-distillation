"""
Pure InstanceNorm double backward implemented in Triton.

Drop-in replacement for:

    instanceNorm_double_backwards_fn(
        x, gamma, beta, ggX, ggG, ggB, gO,
        eps=1e-5, training=True,
    )

Semantics
---------
Pure InstanceNorm double backward. There is NO ReLU mask.

InstanceNorm is applied independently to each (N, C) row over M=H*W.

Inputs:
    x:      forward input                         [N,C,H,W]
    gamma:  affine weight, normally non-None      [C]
    beta:   accepted only for API compatibility; unused
    ggX:    cotangent of first-bwd dX             [N,C,H,W] or None
    ggG:    cotangent of first-bwd dGamma         [C] or None
    ggB:    cotangent of first-bwd dBeta          [C] or None
    gO:     grad_output used by first backward    [N,C,H,W]

Returns:
    gX:     gradient wrt forward x                [N,C,H,W] or None
    gG:     gradient wrt forward gamma            [C] or None
    ggO:    cotangent wrt first-bwd grad_output   [N,C,H,W] or None

Important correction
--------------------
The old PyTorch implementation computes:

    gG = (...).sum(dim=2)

which returns [N,C]. The gradient wrt gamma must reduce across BOTH N and M
and therefore has shape [C]. This Triton implementation performs the required
cross-batch reduction using FP32 atomic_add.
"""

from __future__ import annotations

import argparse
import statistics
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _instance_norm_double_backward_pure_kernel(
    x_ptr,
    gamma_ptr,
    ggX_ptr,
    ggG_ptr,
    ggB_ptr,
    gO_ptr,
    gX_ptr,
    gG_ptr,
    ggO_ptr,
    M: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    HAS_GAMMA: tl.constexpr,
    HAS_GGX: tl.constexpr,
    HAS_GGG: tl.constexpr,
    HAS_GGB: tl.constexpr,
    WRITE_GX: tl.constexpr,
    WRITE_GG: tl.constexpr,
    WRITE_GGO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program handles one complete (n,c) row.

    For the project's ResNet/CIFAR shapes:
        M = 1024, 256, 64, or 16.

    All statistics and algebra are evaluated in FP32.
    """
    pid_nc = tl.program_id(0)
    c = pid_nc % C

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    base = pid_nc * M
    index = base + offsets

    x = tl.load(x_ptr + index, mask=mask, other=0.0).to(tl.float32)
    gO = tl.load(gO_ptr + index, mask=mask, other=0.0).to(tl.float32)

    m_f = tl.full([], M, tl.float32)
    inv_m = 1.0 / m_f

    # Statistics: match unbiased=False InstanceNorm.
    mean = tl.sum(x, axis=0) * inv_m
    x_centered = x - mean

    # Two-pass-style centered variance is more stable than E[x^2]-E[x]^2.
    var = tl.sum(x_centered * x_centered, axis=0) * inv_m
    var = tl.maximum(var, 0.0)

    inv_std = tl.rsqrt(var + EPS)
    inv_std2 = inv_std * inv_std
    inv_std3 = inv_std2 * inv_std

    sum_gO = tl.sum(gO, axis=0)
    sum_gO_xmu = tl.sum(gO * x_centered, axis=0)

    gamma = tl.full([], 1.0, tl.float32)
    if HAS_GAMMA:
        gamma = tl.load(gamma_ptr + c).to(tl.float32)

    ggX = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sum_ggX = tl.full([], 0.0, tl.float32)
    sum_ggX_xmu = tl.full([], 0.0, tl.float32)
    dot_ggX_gO = tl.full([], 0.0, tl.float32)

    if HAS_GGX:
        ggX = tl.load(
            ggX_ptr + index,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        sum_ggX = tl.sum(ggX, axis=0)
        sum_ggX_xmu = tl.sum(ggX * x_centered, axis=0)
        dot_ggX_gO = tl.sum(ggX * gO, axis=0)

    ggG = tl.full([], 0.0, tl.float32)
    if HAS_GGG:
        ggG = tl.load(ggG_ptr + c).to(tl.float32)

    # ============================================================
    # gX: gradient wrt forward input x
    # ============================================================
    gX = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    if HAS_GGX:
        A = (
            sum_ggX * sum_gO * inv_m
            - dot_ggX_gO
            + 3.0
            * inv_std2
            * sum_gO_xmu
            * sum_ggX_xmu
            * inv_m
        )

        term0 = x_centered * inv_std3 * A * inv_m
        term1 = (
            sum_ggX_xmu
            * inv_std3
            * (sum_gO * inv_m - gO)
            * inv_m
        )
        term2 = (
            sum_gO_xmu
            * inv_std3
            * (sum_ggX * inv_m - ggX)
            * inv_m
        )

        gX += gamma * (term0 + term1 + term2)

    if HAS_GGG:
        # Contribution through dGamma = sum(gO * x_hat).
        gX += ggG * (
            gO * inv_std
            - inv_std * sum_gO * inv_m
            - x_centered * inv_std3 * sum_gO_xmu * inv_m
        )

    if WRITE_GX:
        tl.store(gX_ptr + index, gX, mask=mask)

    # ============================================================
    # gG: gradient wrt forward gamma
    #
    # Each (n,c) row contributes a scalar. Accumulate across N.
    # ============================================================
    if WRITE_GG:
        first_back_gO_no_gamma = inv_std * (
            gO
            - sum_gO * inv_m
            - x_centered * inv_std2 * sum_gO_xmu * inv_m
        )

        gG_part = tl.sum(
            ggX * first_back_gO_no_gamma,
            axis=0,
        )
        tl.atomic_add(gG_ptr + c, gG_part)

    # ============================================================
    # ggO: cotangent wrt first-backward grad_output gO
    # ============================================================
    ggO = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    if HAS_GGX:
        first_back_ggX = gamma * inv_std * (
            ggX
            - sum_ggX * inv_m
            - x_centered * inv_std2 * sum_ggX_xmu * inv_m
        )
        ggO += first_back_ggX

    if HAS_GGG:
        ggO += ggG * x_centered * inv_std

    if HAS_GGB:
        ggB = tl.load(ggB_ptr + c).to(tl.float32)
        ggO += ggB

    if WRITE_GGO:
        tl.store(ggO_ptr + index, ggO, mask=mask)


def _num_warps(block_size: int) -> int:
    if block_size >= 2048:
        return 8
    if block_size >= 256:
        return 4
    return 2


def _dummy_cuda(device: torch.device) -> torch.Tensor:
    return torch.empty(1, device=device, dtype=torch.float32)


@torch.no_grad()
def instanceNorm_double_backwards_fn(
    x: torch.Tensor,
    gamma: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    ggX: Optional[torch.Tensor],
    ggG: Optional[torch.Tensor],
    ggB: Optional[torch.Tensor],
    gO: torch.Tensor,
    eps: float = 1e-5,
    training: bool = True,
):
    """
    Drop-in pure InstanceNorm double backward using Triton.

    `beta` is intentionally unused; it remains in the signature so existing
    ResNet call sites do not need to change.
    """
    del beta

    if not training:
        raise NotImplementedError(
            "This project uses per-instance training statistics. "
            "training=False is not implemented by this Triton wrapper."
        )

    if not (x.is_cuda and gO.is_cuda):
        raise ValueError("x and gO must be CUDA tensors.")
    if x.ndim != 4:
        raise ValueError(f"x must be NCHW, got {tuple(x.shape)}")
    if gO.shape != x.shape:
        raise ValueError(
            f"gO shape {tuple(gO.shape)} != x shape {tuple(x.shape)}"
        )

    n, c, h, w = x.shape
    m = h * w

    if gamma is not None:
        if not gamma.is_cuda:
            raise ValueError("gamma must be CUDA when provided.")
        if gamma.numel() != c:
            raise ValueError(
                f"gamma has {gamma.numel()} elements; expected C={c}"
            )

    if ggX is not None:
        if not ggX.is_cuda or ggX.shape != x.shape:
            raise ValueError("ggX must be CUDA and have the same shape as x.")

    if ggG is not None:
        if gamma is None:
            raise ValueError("ggG cannot be provided when gamma is None.")
        if not ggG.is_cuda or ggG.numel() != c:
            raise ValueError("ggG must be CUDA with C elements.")

    if ggB is not None:
        if not ggB.is_cuda or ggB.numel() != c:
            raise ValueError("ggB must be CUDA with C elements.")

    has_gamma = gamma is not None
    has_ggx = ggX is not None
    has_ggg = ggG is not None
    has_ggb = ggB is not None

    write_gx = has_ggx or has_ggg
    write_gg = has_gamma and has_ggx
    write_ggo = has_ggx or has_ggg or has_ggb

    x_c = x.contiguous()
    gO_c = gO.contiguous()

    gamma_c = gamma.contiguous() if gamma is not None else None
    ggX_c = ggX.contiguous() if ggX is not None else None
    ggG_c = ggG.contiguous() if ggG is not None else None
    ggB_c = ggB.contiguous() if ggB is not None else None

    dummy = _dummy_cuda(x.device)

    # Preserve practical dtype behavior while calculating internally in FP32.
    gX = torch.empty_like(x_c) if write_gx else None
    ggO = torch.empty_like(gO_c) if write_ggo else None

    # gG needs reduction across N, so accumulate atomically in FP32.
    gG_fp32 = (
        torch.zeros(c, device=x.device, dtype=torch.float32)
        if write_gg
        else None
    )

    block_size = triton.next_power_of_2(m)
    if block_size > 65536:
        raise ValueError(
            f"H*W={m} is too large for the row-wise Triton kernel "
            f"(BLOCK_SIZE={block_size})."
        )

    grid = (n * c,)

    _instance_norm_double_backward_pure_kernel[grid](
        x_c,
        gamma_c if gamma_c is not None else dummy,
        ggX_c if ggX_c is not None else dummy,
        ggG_c if ggG_c is not None else dummy,
        ggB_c if ggB_c is not None else dummy,
        gO_c,
        gX if gX is not None else dummy,
        gG_fp32 if gG_fp32 is not None else dummy,
        ggO if ggO is not None else dummy,
        M=m,
        C=c,
        EPS=eps,
        HAS_GAMMA=has_gamma,
        HAS_GGX=has_ggx,
        HAS_GGG=has_ggg,
        HAS_GGB=has_ggb,
        WRITE_GX=write_gx,
        WRITE_GG=write_gg,
        WRITE_GGO=write_ggo,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size),
        num_stages=1,
    )

    if gG_fp32 is None:
        gG = None
    elif gamma is not None and gamma.dtype != torch.float32:
        gG = gG_fp32.to(gamma.dtype)
    else:
        gG = gG_fp32

    return gX, gG, ggO


@torch.no_grad()
def instanceNorm_double_backwards_torch_reference(
    x: torch.Tensor,
    gamma: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    ggX: Optional[torch.Tensor],
    ggG: Optional[torch.Tensor],
    ggB: Optional[torch.Tensor],
    gO: torch.Tensor,
    eps: float = 1e-5,
    training: bool = True,
):
    """
    Corrected Torch reference.

    Difference from the old project function:
        gG reduces over (N,M), producing [C], not [N,C].
    """
    del beta

    if not training:
        raise NotImplementedError

    n, c, h, w = x.shape
    m = h * w

    x_flat = x.view(n, c, m)
    gO_flat = gO.view(n, c, m)

    mean = x_flat.mean(dim=2, keepdim=True)
    var = x_flat.var(dim=2, unbiased=False, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    inv_std2 = inv_std * inv_std
    inv_std3 = inv_std2 * inv_std
    x_centered = x_flat - mean

    sum_gO = gO_flat.sum(dim=2, keepdim=True)
    sum_gO_xmu = (gO_flat * x_centered).sum(
        dim=2,
        keepdim=True,
    )

    gamma_exp = (
        gamma.view(1, c, 1)
        if gamma is not None
        else 1.0
    )
    ggG_exp = (
        ggG.view(1, c, 1)
        if ggG is not None
        else None
    )

    gX = None

    if ggX is not None:
        ggX_flat = ggX.view(n, c, m)
        sum_ggX = ggX_flat.sum(dim=2, keepdim=True)
        sum_ggX_xmu = (ggX_flat * x_centered).sum(
            dim=2,
            keepdim=True,
        )
        dot_ggX_gO = (ggX_flat * gO_flat).sum(
            dim=2,
            keepdim=True,
        )

        A = (
            sum_ggX * sum_gO / m
            - dot_ggX_gO
            + 3.0
            * inv_std2
            * sum_gO_xmu
            * sum_ggX_xmu
            / m
        )

        term0 = x_centered * inv_std3 * A / m
        term1 = (
            sum_ggX_xmu
            * inv_std3
            * (sum_gO / m - gO_flat)
            / m
        )
        term2 = (
            sum_gO_xmu
            * inv_std3
            * (sum_ggX / m - ggX_flat)
            / m
        )

        gX = gamma_exp * (term0 + term1 + term2)

    if gamma is not None and ggG is not None:
        gX_from_ggG = ggG_exp * (
            gO_flat * inv_std
            - inv_std * sum_gO / m
            - x_centered * inv_std3 * sum_gO_xmu / m
        )
        gX = (
            gX_from_ggG
            if gX is None
            else gX + gX_from_ggG
        )

    gG = None
    if gamma is not None and ggX is not None:
        ggX_flat = ggX.view(n, c, m)

        first_back_gO_no_gamma = inv_std * (
            gO_flat
            - sum_gO / m
            - x_centered
            * inv_std2
            * sum_gO_xmu
            / m
        )

        # Correct gradient shape: reduce across N and M.
        gG = (
            ggX_flat * first_back_gO_no_gamma
        ).sum(dim=(0, 2))

    ggO = None

    if ggX is not None:
        ggX_flat = ggX.view(n, c, m)
        sum_ggX = ggX_flat.sum(dim=2, keepdim=True)
        sum_ggX_xmu = (ggX_flat * x_centered).sum(
            dim=2,
            keepdim=True,
        )

        ggO = gamma_exp * inv_std * (
            ggX_flat
            - sum_ggX / m
            - x_centered
            * inv_std2
            * sum_ggX_xmu
            / m
        )

    if ggG is not None:
        ggO_from_ggG = ggG_exp * x_centered * inv_std
        ggO = (
            ggO_from_ggG
            if ggO is None
            else ggO + ggO_from_ggG
        )

    if ggB is not None:
        ggO_from_ggB = ggB.view(1, c, 1)
        ggO = (
            ggO_from_ggB
            if ggO is None
            else ggO + ggO_from_ggB
        )

    return (
        gX.view_as(x) if gX is not None else None,
        gG,
        ggO.view_as(gO) if ggO is not None else None,
    )


def _relative_l2(
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    a = actual.float()
    b = reference.float()
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


def _compare(
    name: str,
    actual: Optional[torch.Tensor],
    reference: Optional[torch.Tensor],
) -> None:
    if actual is None or reference is None:
        print(f"{name:<5} actual={actual} reference={reference}")
        return

    diff = actual.float() - reference.float()
    print(
        f"{name:<5} "
        f"shape={tuple(actual.shape)!s:<20} "
        f"max_abs={diff.abs().max().item():.6e} "
        f"rel_l2={_relative_l2(actual, reference):.6e}"
    )


def _benchmark(
    fn,
    args,
    warmup: int,
    iters: int,
) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    values = []
    peak_values = []

    for _ in range(iters):
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        out = fn(*args)
        end.record()
        end.synchronize()

        values.append(start.elapsed_time(end))
        peak_values.append(
            max(
                0,
                torch.cuda.max_memory_allocated() - baseline,
            )
        )
        del out

    return (
        statistics.median(values),
        statistics.fmean(values),
        max(peak_values) / (1024.0 ** 2),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    shape = (
        args.batch,
        args.channels,
        args.height,
        args.width,
    )

    x = torch.randn(
        shape,
        device="cuda",
        dtype=torch.float32,
    )
    gamma = torch.randn(
        args.channels,
        device="cuda",
        dtype=torch.float32,
    )
    beta = torch.randn_like(gamma)
    ggX = torch.randn_like(x)
    ggG = torch.randn_like(gamma)
    ggB = torch.randn_like(gamma)
    gO = torch.randn_like(x)

    call_args = (
        x,
        gamma,
        beta,
        ggX,
        ggG,
        ggB,
        gO,
        args.eps,
        True,
    )

    reference = instanceNorm_double_backwards_torch_reference(
        *call_args
    )
    actual = instanceNorm_double_backwards_fn(*call_args)
    torch.cuda.synchronize()

    print(f"shape={shape}")
    for name, actual_i, reference_i in zip(
        ("gX", "gG", "ggO"),
        actual,
        reference,
    ):
        _compare(name, actual_i, reference_i)

    triton_result = _benchmark(
        instanceNorm_double_backwards_fn,
        call_args,
        args.warmup,
        args.iters,
    )
    torch_result = _benchmark(
        instanceNorm_double_backwards_torch_reference,
        call_args,
        args.warmup,
        args.iters,
    )

    print()
    print(
        "Triton: "
        f"median={triton_result[0]:.4f} ms, "
        f"mean={triton_result[1]:.4f} ms, "
        f"peak_delta={triton_result[2]:.2f} MiB"
    )
    print(
        "Torch:  "
        f"median={torch_result[0]:.4f} ms, "
        f"mean={torch_result[1]:.4f} ms, "
        f"peak_delta={torch_result[2]:.2f} MiB"
    )
    print(
        f"speedup={torch_result[0] / triton_result[0]:.2f}x"
    )


if __name__ == "__main__":
    main()