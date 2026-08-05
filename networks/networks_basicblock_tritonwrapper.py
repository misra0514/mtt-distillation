# 5.26 原来的fused3 改名。把所有triton code都拿出去了。但是还有很多python kernel 在里面

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import torchvision.transforms as transforms

from typing import Optional

import torch
from functools import reduce
from operator import mul


import triton
import triton.language as tl

BLOCK_M = 64

from networks.networks_basicblock_triton import _instancenorm_backward_fused_kernel,_instancenorm_backward_kernel,\
    _instancenorm_stats_kernel, instance_norm_double_backward_kernel_blocked, \
    _gelu_drop_grad_kernel, _gelu_drop_double_grad_kernel

# trion for unfused  import
from networks.networks_basicblock_triton import _instance_norm_backward_pure_kernel, _instance_norm_double_backward_pure_kernel


def instancenorm_relu_backward_plain( x, gamma, grad_output, out, eps=1e-5):
    N, C, H, W = x.shape
    M = H * W
    # use fp32 for stats/accumulation to avoid fp16/bf16 error
    x_reshaped = x.view(N, C, M).float()
    relu_mask = (out > 0).view(N, C, M)
    grad_output_reshaped = grad_output.view(N, C, M).float() * relu_mask

    mean = x_reshaped.mean(dim=2, keepdim=True)  # (N, C, 1)
    var = x_reshaped.var(dim=2, unbiased=False, keepdim=True)  # (N, C, 1)
    std = torch.sqrt(var + eps)  # (N, C, 1)
    x_hat = (x_reshaped - mean) / std  # (N, C, M)

    grad_output_hat = grad_output_reshaped * gamma.view(1, C, 1).float()  # (N, C, M)
    dx = (1. / M) / std * (
        M * grad_output_hat
        - grad_output_hat.sum(dim=2, keepdim=True)
        - x_hat * (grad_output_hat * x_hat).sum(dim=2, keepdim=True)
    )  # (N, C, M)
    grad_gamma = (grad_output_reshaped * x_hat).sum(dim=(0, 2))  # (C,)
    grad_beta = grad_output_reshaped.sum(dim=(0, 2))             # (C,)
    return dx.view(N, C, H, W).to(x.dtype), grad_gamma, grad_beta

def sum_exclude_dim1(to_sum, keepdim=True):
    to_sum = to_sum.sum(dim=0, keepdim=keepdim)
    start_point_exclusive = 1 if keepdim else 0
    for dim in range(to_sum.dim() - 1, start_point_exclusive, -1):
        to_sum = to_sum.sum(dim=dim, keepdim=keepdim)
    return to_sum

def unsqueeze_dim1(src, target):
    src_expanded = src
    while len(src_expanded.size()) < len(target.size()) - 1:
        src_expanded = src_expanded.unsqueeze(1)
    if len(src_expanded.size()) == len(target.size()) - 1:
        src_expanded = src_expanded.unsqueeze(0)
    return src_expanded

def expand_as_dim1(src, target):
    src_expanded = src
    while len(src_expanded.size()) < len(target.size()) - 1:
        src_expanded = src_expanded.unsqueeze(1)
    return src_expanded.expand_as(target)

def batchnorm_double_backwards_fn_new(input, gamma, ggI, ggG, ggB, gO, eps=1e-5,
                                  running_mean=None, running_var=None, training=True):
    N, C, H, W = input.shape
    M = N * H * W

    device = input.device
    affine = gamma is not None
    if training:
        # per-channel mean/var over (N,H,W)
        # 用 float32 算统计量更稳（尤其输入是 fp16/bf16）
        x32 = input.float()
        save_mean = x32.mean(dim=(0, 2, 3))                         # (C,)
        var = x32.var(dim=(0, 2, 3), unbiased=False)           # (C,)
        rstd = torch.rsqrt(var + eps).to(input.dtype)          # (C,)  = (var+eps)^(-1/2)

        mu = unsqueeze_dim1(save_mean.to(input.dtype), input)
        sigma2_eps_neg_1_2 = unsqueeze_dim1(rstd, input)

    if affine:
        gamma = gamma.to(device)
        gamma_expanded = expand_as_dim1(gamma, input)
        if ggG is not None:
            ggG = ggG.to(device)
            ggG_expanded = expand_as_dim1(ggG, input)
        if ggB is not None:
            ggB = ggB.to(device)
            ggB_expanded = expand_as_dim1(ggB, input)
    else:
        gamma_expanded = 1.0

    mu = unsqueeze_dim1(save_mean if training else running_mean, input)
    input_sub_mu = input - mu
    # sigma2_eps_neg_1_2 = unsqueeze_dim1(
    #     save_std if training else (running_var + eps).pow(-1. / 2),
    #     input
    # )
    sigma2_eps_neg_1_2 = unsqueeze_dim1(rstd, input)
    sigma2_eps_neg_1 = sigma2_eps_neg_1_2.pow(2)
    sigma2_eps_neg_3_2 = sigma2_eps_neg_1_2.pow(3)


    input_sub_mu = input - mu
    input_mu_sigma2_neg_3_2 = input_sub_mu * sigma2_eps_neg_3_2
    gOinmu_sum = sum_exclude_dim1(gO * input_sub_mu)
    gO_sum = sum_exclude_dim1(gO)

    gI = None
    if ggI is not None and training:
        ggI = ggI.to(device)
        ggI_sum = sum_exclude_dim1(ggI)
        ggIinmu_sum = sum_exclude_dim1(ggI * input_sub_mu)
        all_sub = ((ggI_sum * gO_sum).div_(M)).sub_(sum_exclude_dim1(gO * ggI)).add_(
            (sigma2_eps_neg_1 * gOinmu_sum * ggIinmu_sum).mul_(3.0 / M)
        )
        gI_0t = (input_mu_sigma2_neg_3_2 * all_sub).div_(M)
        gI_1t = (ggIinmu_sum * sigma2_eps_neg_3_2).div_(M) * (gO_sum.div(M) - gO)
        gI_2t = (gOinmu_sum * sigma2_eps_neg_3_2).div_(M) * (ggI_sum.div(M) - ggI)
        gI = gamma_expanded * (gI_0t + gI_1t + gI_2t)

    if affine and ggG is not None:
        if training:
            t0 = gO * sigma2_eps_neg_1_2
            t1 = (sigma2_eps_neg_1_2 * gO_sum).div_(-M)
            t2 = (input_mu_sigma2_neg_3_2 * sum_exclude_dim1(gO * input_sub_mu)).div_(-M)
            gI_G_term = ggG_expanded * (t0 + t1 + t2)
        else:
            gI_G_term = ggG_expanded * sigma2_eps_neg_1_2 * gO
        gI = gI + gI_G_term if gI is not None else gI_G_term

    def first_back_grad_input(gO, gamma):
        h0 = (gamma * sigma2_eps_neg_1_2).div(M)
        h1 = M * gO - sum_exclude_dim1(gO) - (
            input_sub_mu * sigma2_eps_neg_1 * sum_exclude_dim1(gO * input_sub_mu)
        )
        return h0 * h1

    gG = None
    if affine and ggI is not None:
        if training:
            gG = ggI * first_back_grad_input(gO, torch.ones_like(gamma_expanded))
            gG = sum_exclude_dim1(gG, keepdim=False)
        else:
            gG = sum_exclude_dim1(ggI * gO * sigma2_eps_neg_1_2, keepdim=False)

    ggO = None
    if ggI is not None:
        if training:
            ggO = first_back_grad_input(ggI, gamma_expanded)
        else:
            ggO = ggI * sigma2_eps_neg_1_2 * gamma_expanded

    if ggG is not None:
        ggO_G_term = ggG_expanded * input_sub_mu * sigma2_eps_neg_1_2
        ggO = ggO + ggO_G_term if ggO is not None else ggO_G_term

    if ggB is not None:
        ggO_B_term = ggB_expanded
        ggO = ggO + ggO_B_term if ggO is not None else ggO_B_term

    # return gI, gG, ggO
    # return gX.view(N, C, H, W) if gX is not None else None, gG, ggO.view(N, C, H, W)
    return gI,gG, ggO


def batchnorm_double_backwards_fn(input, gamma, ggI, ggG, ggB, gO, eps,
                                  save_mean, save_std, running_mean, running_var, training):
    device = input.device
    affine = gamma is not None

    if affine:
        gamma = gamma.to(device)
        gamma_expanded = expand_as_dim1(gamma, input)
        if ggG is not None:
            ggG = ggG.to(device)
            ggG_expanded = expand_as_dim1(ggG, input)
        if ggB is not None:
            ggB = ggB.to(device)
            ggB_expanded = expand_as_dim1(ggB, input)
    else:
        gamma_expanded = 1.0

    mu = unsqueeze_dim1(save_mean if training else running_mean, input)
    input_sub_mu = input - mu
    sigma2_eps_neg_1_2 = unsqueeze_dim1(
        save_std if training else (running_var + eps).pow(-1. / 2),
        input
    )
    sigma2_eps_neg_1 = sigma2_eps_neg_1_2.pow(2)
    sigma2_eps_neg_3_2 = sigma2_eps_neg_1_2.pow(3)


    input_sub_mu = input - mu
    input_mu_sigma2_neg_3_2 = input_sub_mu * sigma2_eps_neg_3_2
    gOinmu_sum = sum_exclude_dim1(gO * input_sub_mu)
    gO_sum = sum_exclude_dim1(gO)

    gI = None
    if ggI is not None and training:
        ggI = ggI.to(device)
        ggI_sum = sum_exclude_dim1(ggI)
        ggIinmu_sum = sum_exclude_dim1(ggI * input_sub_mu)
        all_sub = ((ggI_sum * gO_sum).div_(M)).sub_(sum_exclude_dim1(gO * ggI)).add_(
            (sigma2_eps_neg_1 * gOinmu_sum * ggIinmu_sum).mul_(3.0 / M)
        )
        gI_0t = (input_mu_sigma2_neg_3_2 * all_sub).div_(M)
        gI_1t = (ggIinmu_sum * sigma2_eps_neg_3_2).div_(M) * (gO_sum.div(M) - gO)
        gI_2t = (gOinmu_sum * sigma2_eps_neg_3_2).div_(M) * (ggI_sum.div(M) - ggI)
        gI = gamma_expanded * (gI_0t + gI_1t + gI_2t)

    if affine and ggG is not None:
        if training:
            t0 = gO * sigma2_eps_neg_1_2
            t1 = (sigma2_eps_neg_1_2 * gO_sum).div_(-M)
            t2 = (input_mu_sigma2_neg_3_2 * sum_exclude_dim1(gO * input_sub_mu)).div_(-M)
            gI_G_term = ggG_expanded * (t0 + t1 + t2)
        else:
            gI_G_term = ggG_expanded * sigma2_eps_neg_1_2 * gO
        gI = gI + gI_G_term if gI is not None else gI_G_term

    def first_back_grad_input(gO, gamma):
        h0 = (gamma * sigma2_eps_neg_1_2).div(M)
        h1 = M * gO - sum_exclude_dim1(gO) - (
            input_sub_mu * sigma2_eps_neg_1 * sum_exclude_dim1(gO * input_sub_mu)
        )
        return h0 * h1

    gG = None
    if affine and ggI is not None:
        if training:
            gG = ggI * first_back_grad_input(gO, torch.ones_like(gamma_expanded))
            gG = sum_exclude_dim1(gG, keepdim=False)
        else:
            gG = sum_exclude_dim1(ggI * gO * sigma2_eps_neg_1_2, keepdim=False)

    ggO = None
    if ggI is not None:
        if training:
            ggO = first_back_grad_input(ggI, gamma_expanded)
        else:
            ggO = ggI * sigma2_eps_neg_1_2 * gamma_expanded

    if ggG is not None:
        ggO_G_term = ggG_expanded * input_sub_mu * sigma2_eps_neg_1_2
        ggO = ggO + ggO_G_term if ggO is not None else ggO_G_term

    if ggB is not None:
        ggO_B_term = ggB_expanded
        ggO = ggO + ggO_B_term if ggO is not None else ggO_B_term

    # return gI, gG, ggO
    # return ggO, gI, gG
    return gI,gG, ggO


def batchNorm2d_backward(x, gamma, grad_output, eps=1e-5):
    """
    x: (N, C, H, W)
    gamma, beta: (C,)
    grad_output: same shape as x
    returns: dx, grad_gamma, grad_beta
    """
    N, C, H, W = x.shape
    M = N * H * W  # 元素总数 per channel
    # 重塑为 (C, M)
    x_flat = x.permute(1, 0, 2, 3).reshape(C, -1)  # (C, M)
    dout_flat = grad_output.permute(1, 0, 2, 3).reshape(C, -1)
    # 统计 mean / var
    mu = x_flat.mean(dim=1, keepdim=True)           # (C, 1)
    var = x_flat.var(dim=1, unbiased=False, keepdim=True)
    std = torch.sqrt(var + eps)
    x_hat = (x_flat - mu) / std                       # (C, M)
    # grad w.r.t. gamma, beta
    grad_gamma = (dout_flat * x_hat).sum(dim=1)       # (C,)
    grad_beta = dout_flat.sum(dim=1)                  # (C,)
    # grad_output_hat = grad_output * gamma
    grad_output_hat = dout_flat * gamma.view(C, 1)
    # dx calculation per channel
    inv_std = 1.0 / std
    sum_dy = grad_output_hat.sum(dim=1, keepdim=True)                 # (C,1)
    sum_dy_xhat = (grad_output_hat * x_hat).sum(dim=1, keepdim=True)  # (C,1)
    
    dx_flat = (1.0 / M) * inv_std * (
        M * grad_output_hat
        - sum_dy
        - x_hat * sum_dy_xhat
    )  # (C, M)
    # 恢复原始形状
    dx = dx_flat.view(C, N, H, W).permute(1, 0, 2, 3).contiguous()
    return dx, grad_gamma, grad_beta



    # has_relu: tl.constexpr,       # 新增常量：是否需要做 ReLU 掩码


def instancenorm_relu_backward_triton(x, gamma, grad_output,out, eps=1e-5):
    assert x.is_cuda and grad_output.is_cuda and gamma.is_cuda and out.is_cuda
    N, C, H, W = x.shape
    HW = H * W

    # (N,C,HW) view -> 底层仍是连续的 NCHW 展平
    x_flat = x.contiguous().view(N, C, HW)
    dy_flat = grad_output.contiguous().view(N, C, HW)
    out_flat = out.contiguous().view(N, C, HW)

    # 直接把 (N,C,HW) 当成 1D 指针 + stride 访问
    stride_n = C * HW
    stride_c = HW

    # mean/rstd: 展平为 (N*C,)
    mean = torch.empty((N * C,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((N * C,), device=x.device, dtype=torch.float32)

    # 输出
    dX = torch.empty_like(x_flat, dtype=torch.float32)  # 你也可以输出和 x 同 dtype，但一般 dx 用 fp32 更稳
    dgamma = torch.zeros((C,), device=x.device, dtype=torch.float32)
    dbeta = torch.zeros((C,), device=x.device, dtype=torch.float32)

    grid = (N, C)

    _instancenorm_stats_kernel[grid](
        x_flat, mean, rstd,
        stride_n, stride_c, C, HW,
        eps=eps,
        BLOCK_SIZE=256
    )

    _instancenorm_backward_fused_kernel[grid](
        dy_flat, x_flat, gamma, out_flat,
        mean, rstd,
        dX, dgamma, dbeta,
        stride_n, stride_c, C, HW,
        BLOCK_SIZE=256
    )

    dX = dX.view(N, C, H, W)
    return dX, dgamma, dbeta
    # """
    # 计算 InstanceNorm 的反向传播 (dX, dgamma, dbeta)。
    # x, grad_output: (N, C, H, W)
    # gamma: (C,)
    # """
    # N, C, H, W = x.shape
    # HW = H * W
    # stride_n = C * HW
    # stride_c = HW

    # # 展平输入以便在 Triton 内核中按 HW 维度遍历
    # x_flat = x.contiguous().view(N, C, HW)
    # out_flat = out.contiguous().view(N, C, HW)

    # grad_output_flat = grad_output.contiguous().view(N, C, HW)
    # grad_output_flat[out_flat <= 0] = 0

    # x_buf = x_flat.reshape(-1, HW)
    # dy_buf = grad_output_flat.reshape(-1, HW)

    # dX = torch.empty_like(x_buf)
    # dgamma = torch.zeros(C, device=x.device, dtype=torch.float32)
    # dbeta = torch.zeros(C, device=x.device, dtype=torch.float32)

    # # 预先计算均值和反标准差
    # mean = x_buf.mean(dim=1)
    # var = x_buf.var(dim=1, unbiased=False)
    # rstd = 1.0 / torch.sqrt(var + eps)

    # mean_ptr = mean.contiguous()
    # rstd_ptr = rstd.contiguous()

    # # 使用二维 grid 调度 (N, C)
    # grid = (N, C)
    # _instancenorm_backward_kernel[grid](
    #     dy_buf, x_buf, gamma, out_flat,
    #     mean_ptr, rstd_ptr,
    #     dX, dgamma, dbeta,
    #     stride_n, stride_c, C, HW,
    # )

    # dX = dX.view(N, C, H, W)
    # return dX, dgamma, dbeta




def _num_warps_for_block(block_size: int) -> int:
    if block_size >= 2048:
        return 8
    if block_size >= 512:
        return 4
    return 2




def instanceNorm_backward(
    x: torch.Tensor,
    gamma: torch.Tensor,
    grad_output: torch.Tensor,
    eps: float = 1e-5,
):
    """
    Drop-in pure InstanceNorm backward.

    Args:
        x:           [N, C, H, W]
        gamma:       [C]
        grad_output: [N, C, H, W]
        eps:          InstanceNorm epsilon

    Returns:
        dx:       [N, C, H, W]
        dgamma:   [C]
        dbeta:    [C]
        mean:     [N, C, 1]
        std:      [N, C, 1]

    This function does not apply a ReLU mask.
    """
    if not (x.is_cuda and gamma.is_cuda and grad_output.is_cuda):
        raise ValueError("x, gamma, and grad_output must all be CUDA tensors.")
    if x.ndim != 4:
        raise ValueError(f"x must be 4D NCHW, got shape={tuple(x.shape)}")
    if grad_output.shape != x.shape:
        raise ValueError(
            f"grad_output shape {tuple(grad_output.shape)} "
            f"does not match x shape {tuple(x.shape)}"
        )

    n, c, h, w = x.shape
    if gamma.numel() != c:
        raise ValueError(
            f"gamma has {gamma.numel()} elements, expected C={c}"
        )

    hw = h * w
    block_size = triton.next_power_of_2(hw)

    # Current ResNet/CIFAR maximum is HW=1024. Larger rows can work, but very
    # large power-of-two blocks may become register-heavy. Use a clear guard
    # rather than silently producing a poor kernel.
    if block_size > 65536:
        raise ValueError(
            f"H*W={hw} is too large for this row-wise kernel "
            f"(BLOCK_SIZE={block_size})."
        )

    x_c = x.contiguous()
    dy_c = grad_output.contiguous()
    gamma_c = gamma.contiguous()

    dx = torch.empty_like(x_c)

    # FP32 atomic accumulation avoids low-precision parameter reductions.
    dgamma_fp32 = torch.zeros(c, device=x.device, dtype=torch.float32)
    dbeta_fp32 = torch.zeros(c, device=x.device, dtype=torch.float32)
    mean_fp32 = torch.empty(n * c, device=x.device, dtype=torch.float32)
    std_fp32 = torch.empty(n * c, device=x.device, dtype=torch.float32)

    grid = (n * c,)

    _instance_norm_backward_pure_kernel[grid](
        x_c,
        dy_c,
        gamma_c,
        dx,
        dgamma_fp32,
        dbeta_fp32,
        mean_fp32,
        std_fp32,
        HW=hw,
        C=c,
        EPS=eps,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps_for_block(block_size),
        num_stages=1,
    )
    # Preserve parameter/stat dtype behavior at the Python boundary.
    dgamma = (
        dgamma_fp32
        if gamma.dtype == torch.float32
        else dgamma_fp32.to(gamma.dtype)
    )
    dbeta = (
        dbeta_fp32
        if gamma.dtype == torch.float32
        else dbeta_fp32.to(gamma.dtype)
    )
    mean = mean_fp32.view(n, c, 1)
    std = std_fp32.view(n, c, 1)
    if x.dtype != torch.float32:
        mean = mean.to(x.dtype)
        std = std.to(x.dtype)

    return dx.view_as(x), dgamma, dbeta




# TODO: 原来的instanceNorm_backward 改成了这个，不用了。目前应该没有引用。
def instanceNorm_backward_torch( x, gamma, grad_output, eps=1e-5):
    N, C, H, W = x.shape
    M = H * W
    x_reshaped = x.view(N, C, M)
    grad_output_reshaped = grad_output.view(N, C, M)
    mean = x_reshaped.mean(dim=2, keepdim=True)  # (N, C, 1)
    var = x_reshaped.var(dim=2, unbiased=False, keepdim=True)  # (N, C, 1)
    std = torch.sqrt(var + eps)  # (N, C, 1)
    x_hat = (x_reshaped - mean) / std  # (N, C, M)
    grad_output_hat = grad_output_reshaped * gamma.view(1, C, 1)  # (N, C, M)
    dx = (1. / M) / std * (
        M * grad_output_hat
        - grad_output_hat.sum(dim=2, keepdim=True)
        - x_hat * (grad_output_hat * x_hat).sum(dim=2, keepdim=True)
    )  # (N, C, M)
    grad_gamma = (grad_output_reshaped * x_hat).sum(dim=(0, 2))  # (C,)
    grad_beta = grad_output_reshaped.sum(dim=(0, 2))             # (C,)
    return dx.view(N, C, H, W), grad_gamma, grad_beta, mean, std



def instanceNorm_double_backwards_fn_new(x, gamma, ggX, ggG, ggB, gO,
                                     eps=1e-5, training=True):
    N, C, H, W = x.shape
    M = H * W

    x_flat = x.view(N, C, M)
    gO_flat = gO.view(N, C, M)
    with torch.no_grad():
        mean = x_flat.mean(dim=2, keepdim=True)
        var = x_flat.var(dim=2, unbiased=False, keepdim=True)
        std = torch.sqrt(var + eps)
        inv_std = 1.0 / std
        x_centered = x_flat - mean
        x_hat = x_centered * inv_std
        # inv_std3 = inv_std.pow(3)
        # 稳定性增强
        inv_std3 = inv_std / (var + eps)

        sum_gO = gO_flat.sum(dim=2, keepdim=True)
        sum_gO_xmu = (gO_flat * x_centered).sum(dim=2, keepdim=True)

    # Broadcast gamma safely
    if gamma is not None:
        gamma_exp = gamma.view(1, C, 1)
        ggG_exp = ggG.view(1, C, 1) if ggG is not None else None
    else:
        gamma_exp = 1.0
        ggG_exp = None
    gX = None
    if ggX is not None and training:
        ggX_flat = ggX.view(N, C, M)
        with torch.no_grad():
            sum_ggX = ggX_flat.sum(dim=2, keepdim=True)
            sum_ggX_xmu = (ggX_flat * x_centered).sum(dim=2, keepdim=True)
            dot_ggX_gO = (ggX_flat * gO_flat).sum(dim=2, keepdim=True)

            A = (sum_ggX * sum_gO) / M - dot_ggX_gO + 3 * (inv_std ** 2) * sum_gO_xmu * sum_ggX_xmu / M
            term0 = x_centered * inv_std3 * A / M
            term1 = sum_ggX_xmu * inv_std3 * (sum_gO / M - gO_flat) / M
            term2 = sum_gO_xmu * inv_std3 * (sum_ggX / M - ggX_flat) / M
        gX = gamma_exp * (term0 + term1 + term2)
    # gamma 分支贡献
    if gamma is not None and ggG is not None:
        if training:
            t0 = gO_flat * inv_std
            t1 = -(inv_std * sum_gO) / M
            t2 = -x_centered * inv_std3 * sum_gO_xmu / M
            gX_G = ggG_exp * (t0 + t1 + t2)
        else:
            gX_G = ggG_exp * inv_std * gO_flat
        gX = gX + gX_G if gX is not None else gX_G

    # gG
    gG = None
    if gamma is not None and ggX is not None:
        def first_back(g, gamma_val):
            return (gamma_val * inv_std / M) * (
                M * g - g.sum(dim=2, keepdim=True) -
                x_centered * inv_std ** 2 * (g * x_centered).sum(dim=2, keepdim=True)
            )
        if training:
            gG = (ggX_flat * first_back(gO_flat, torch.ones_like(gamma_exp))).sum(dim=2)
        else:
            gG = (ggX_flat * gO_flat * inv_std).sum(dim=2)
    # ggO
    ggO = None
    if ggX is not None:
        if training:
            ggO = first_back(ggX_flat, gamma_exp)
        else:
            ggO = ggX_flat * gamma_exp * inv_std
    if ggG is not None:
        # ggO_G = ggG_exp * x_centered
        # ggO =  ggO_G if ggO is not None else ggO_G
        ggO_G = ggG_exp * x_centered * inv_std
        ggO = ggO + ggO_G if ggO is not None else ggO_G
    if ggB is not None:
        ggB_exp = ggB.view(1, C, 1)
        ggO = ggO + ggB_exp if ggO is not None else ggB_exp
    return gX.view(N, C, H, W) if gX is not None else None, gG, ggO.view(N, C, H, W)

# TODO: instanceNorm_double_backwards_fn 改成了这个，不用了。目前应该没有引用。
def instanceNorm_double_backwards_fn_torch(x, gamma, beta, ggX, ggG, ggB, gO,
                                     eps=1e-5, training=True):
    N, C, H, W = x.shape
    M = H * W

    x_flat = x.view(N, C, M)
    gO_flat = gO.view(N, C, M)
    with torch.no_grad():
        mean = x_flat.mean(dim=2, keepdim=True)
        var = x_flat.var(dim=2, unbiased=False, keepdim=True)
        std = torch.sqrt(var + eps)
        inv_std = 1.0 / std
        x_centered = x_flat - mean
        x_hat = x_centered * inv_std
        # inv_std3 = inv_std.pow(3)
        # 稳定性增强
        inv_std3 = inv_std / (var + eps)

        sum_gO = gO_flat.sum(dim=2, keepdim=True)
        sum_gO_xmu = (gO_flat * x_centered).sum(dim=2, keepdim=True)

    # Broadcast gamma safely
    if gamma is not None:
        gamma_exp = gamma.view(1, C, 1)
        ggG_exp = ggG.view(1, C, 1) if ggG is not None else None
    else:
        gamma_exp = 1.0
        ggG_exp = None
    gX = None
    if ggX is not None and training:
        ggX_flat = ggX.view(N, C, M)
        with torch.no_grad():
            sum_ggX = ggX_flat.sum(dim=2, keepdim=True)
            sum_ggX_xmu = (ggX_flat * x_centered).sum(dim=2, keepdim=True)
            dot_ggX_gO = (ggX_flat * gO_flat).sum(dim=2, keepdim=True)

            A = (sum_ggX * sum_gO) / M - dot_ggX_gO + 3 * (inv_std ** 2) * sum_gO_xmu * sum_ggX_xmu / M
            term0 = x_centered * inv_std3 * A / M
            term1 = sum_ggX_xmu * inv_std3 * (sum_gO / M - gO_flat) / M
            term2 = sum_gO_xmu * inv_std3 * (sum_ggX / M - ggX_flat) / M
        gX = gamma_exp * (term0 + term1 + term2)
    # gamma 分支贡献
    if gamma is not None and ggG is not None:
        if training:
            t0 = gO_flat * inv_std
            t1 = -(inv_std * sum_gO) / M
            t2 = -x_centered * inv_std3 * sum_gO_xmu / M
            gX_G = ggG_exp * (t0 + t1 + t2)
        else:
            gX_G = ggG_exp * inv_std * gO_flat
        gX = gX + gX_G if gX is not None else gX_G

    # gG
    gG = None
    if gamma is not None and ggX is not None:
        def first_back(g, gamma_val):
            return (gamma_val * inv_std / M) * (
                M * g - g.sum(dim=2, keepdim=True) -
                x_centered * inv_std ** 2 * (g * x_centered).sum(dim=2, keepdim=True)
            )
        if training:
            gG = (ggX_flat * first_back(gO_flat, torch.ones_like(gamma_exp))).sum(dim=2)
        else:
            gG = (ggX_flat * gO_flat * inv_std).sum(dim=2)
    # ggO
    ggO = None
    if ggX is not None:
        if training:
            ggO = first_back(ggX_flat, gamma_exp)
        else:
            ggO = ggX_flat * gamma_exp * inv_std
    if ggG is not None:
        # ggO_G = ggG_exp * x_centered
        # ggO =  ggO_G if ggO is not None else ggO_G
        ggO_G = ggG_exp * x_centered * inv_std
        ggO = ggO + ggO_G if ggO is not None else ggO_G
    if ggB is not None:
        ggB_exp = ggB.view(1, C, 1)
        ggO = ggO + ggB_exp if ggO is not None else ggB_exp
    return gX.view(N, C, H, W) if gX is not None else None, gG, ggO.view(N, C, H, W)



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
def instanceNorm_double_backwards_plain(
    x, gamma, ggX, ggG, ggB, gO, out,
    eps=1e-5, training=True,
):
    # gO 是 ReLU 输出端的 upstream grad
    relu_mask = (out > 0)

    # 进入 InstanceNorm backward 的 gO 必须先过 ReLU mask
    gO_in = gO * relu_mask

    # 这里调用你已有的纯 InstanceNorm double-bwd plain
    gX, gG, ggO_norm = instanceNorm_double_backwards_fn_new(
        x, gamma, ggX, ggG, ggB, gO_in,
        eps=eps,
        training=training,
    )

    # 返回给 dLdy / ReLU 输出端的二阶梯度，也要过 ReLU mask
    ggO = ggO_norm * relu_mask if ggO_norm is not None else None

    # 你的 instanceNorm_double_backwards_fn_new 现在 gG 可能是 (N, C)，
    # 但 weight/gamma 的梯度应该是 (C,)
    if gG is not None and gG.dim() == 2:
        gG = gG.sum(dim=0)

    return gX, gG, ggO


# def instanceNorm_double_backwards_fn_cln(x, gamma, ggX, ggG, ggB, gO,
#                                      eps=1e-5, training=True):
#     N, C, H, W = x.shape
#     M = H * W

#     x_flat = x.view(N, C, M)
#     gO_flat = gO.view(N, C, M)
#     with torch.no_grad():
#         mean = x_flat.mean(dim=2, keepdim=True)
#         var = x_flat.var(dim=2, unbiased=False, keepdim=True)
#         std = torch.sqrt(var + eps)
#         inv_std = 1.0 / std
#         x_centered = x_flat - mean
#         x_hat = x_centered * inv_std
#         # inv_std3 = inv_std.pow(3)
#         # 稳定性增强
#         inv_std3 = inv_std / (var + eps)

#         sum_gO = gO_flat.sum(dim=2, keepdim=True)
#         sum_gO_xmu = (gO_flat * x_centered).sum(dim=2, keepdim=True)

#     # Broadcast gamma safely
#     if gamma is not None:
#         gamma_exp = gamma.view(1, C, 1)
#         ggG_exp = ggG.view(1, C, 1) if ggG is not None else None
#     else:
#         gamma_exp = 1.0
#         ggG_exp = None
#     gX = None
#     if ggX is not None:
#         ggX_flat = ggX.view(N, C, M)
#         with torch.no_grad():
#             sum_ggX = ggX_flat.sum(dim=2, keepdim=True)
#             sum_ggX_xmu = (ggX_flat * x_centered).sum(dim=2, keepdim=True)
#             dot_ggX_gO = (ggX_flat * gO_flat).sum(dim=2, keepdim=True)

#             A = (sum_ggX * sum_gO) / M - dot_ggX_gO + 3 * (inv_std ** 2) * sum_gO_xmu * sum_ggX_xmu / M
#             term0 = x_centered * inv_std3 * A / M
#             term1 = sum_ggX_xmu * inv_std3 * (sum_gO / M - gO_flat) / M
#             term2 = sum_gO_xmu * inv_std3 * (sum_ggX / M - ggX_flat) / M
#         gX = gamma_exp * (term0 + term1 + term2)
#     # gamma 分支贡献
#     if gamma is not None and ggG is not None:
#         t0 = gO_flat * inv_std
#         t1 = -(inv_std * sum_gO) / M
#         t2 = -x_centered * inv_std3 * sum_gO_xmu / M
#         gX_G = ggG_exp * (t0 + t1 + t2)
#         gX = gX + gX_G if gX is not None else gX_G

#     # gG
#     gG = None
#     if gamma is not None and ggX is not None:
#         def first_back(g, gamma_val):
#             return (gamma_val * inv_std / M) * (
#                 M * g - g.sum(dim=2, keepdim=True) -
#                 x_centered * inv_std ** 2 * (g * x_centered).sum(dim=2, keepdim=True)
#             )
#         gG = (ggX_flat * first_back(gO_flat, torch.ones_like(gamma_exp))).sum(dim=2)
#     # ggO
#     ggO = None
#     if ggX is not None:
#         ggO = first_back(ggX_flat, gamma_exp)
#     if ggG is not None:
#         ggO_G = ggG_exp * x_centered * inv_std
#         ggO = ggO + ggO_G if ggO is not None else ggO_G
#     if ggB is not None:
#         ggB_exp = ggB.view(1, C, 1)
#         ggO = ggO + ggB_exp if ggO is not None else ggB_exp
#     return gX.view(N, C, H, W) if gX is not None else None, gG, ggO.view(N, C, H, W)


class Snd_Order_NormActive(torch.autograd.Function):
    '''2 forward: relu+pool+linear.backward '''
    # TODO: 如果做ckpt，那么记得保证forward可以在算完之后全释放掉。 然后backward再重新算一遍。
    # 现在也没有做save ctx，为啥内存消耗还是1483？
    @staticmethod
    def forward(ctx, dLdy, input, weight, out):
        ctx.save_for_backward( input, weight, dLdy, out )
        # dLdy[out<=0 ] = 0
        grad_output, dw, db = instancenorm_relu_backward_triton(input, weight, dLdy, out)
        # grad_output, dw, db,mean, std = instanceNorm_backward(input, weight, dLdy)
        return grad_output, dw, db
    @staticmethod
    def backward(ctx, grad_grad_input, grad_grad_w, grad_grad_b):
        input, weight, dLdy, out = ctx.saved_tensors
        gx, gG, ggO = instanceNorm_double_backwards_triton(input, weight, grad_grad_input, grad_grad_w,grad_grad_b, dLdy, out,1e-5)
        # gx, gG, ggO = instanceNorm_double_backwards_fn_cln(input, weight, grad_grad_input, grad_grad_w,grad_grad_b, dLdy,1e-5,True   )
        ggO = ggO.view_as(dLdy)
        # ggO[out <= 0] = 0

        # return None, gx, gG, None
        return ggO,gx, gG, None


class Fst_Order_NormActive(torch.autograd.Function):
    '''forward: relu+pool+linear '''
    @staticmethod
    def forward(ctx, input, weight, bias):
        # out1 = F.instance_norm(input,weight= weight, bias = bias)
        # out = F.relu(out1)
        out = F.relu(F.instance_norm(input,weight= weight, bias = bias))
        ctx.save_for_backward( input, weight ,out)
        return  out
    @staticmethod
    def backward(ctx, dLdy):
        input, weight, out = ctx.saved_tensors
        return Snd_Order_NormActive.apply(dLdy ,input, weight, out)
        # db = gin.sum(0)
        # return gin, weight, db




# class NormActive(nn.Module):
#     # in_features 应该是1
#     def __init__(self, channel_num, affine=True):
#         super().__init__()
#         self.weight = nn.Parameter(torch.randn([channel_num]))
#         self.bias = nn.Parameter(torch.randn([channel_num]))
#     def forward(self, input):
#         out = Fst_Order_NormActive.apply(input, self.weight, self.bias)
#         return out


# 下面的gelu 部分可能需要找个时间修改一下------- 


_INV_SQRT2 = 1.0 / (2.0 ** 0.5)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)  # 0.3989422804014327


def gelu_drop_grad_triton(gY: torch.Tensor, x: torch.Tensor, m: torch.Tensor, out: torch.Tensor = None):
    """
    计算 gX = gY * m * GELU'(x)
    - x, gY, m: 同形状张量（m 为已按 1/(1-p) 缩放后的 mask）
    - 输出 dtype 默认与 gY.dtype 一致
    """
    assert x.is_cuda and gY.is_cuda and m.is_cuda, "use CUDA tensors"
    assert x.shape == gY.shape == m.shape, "shape mismatch"

    # 为了内存访问合并，这里用 1D contiguous 缓冲
    x_c  = x.contiguous()
    gy_c = gY.contiguous()
    m_c  = m.contiguous()

    if out is None:
        out = torch.empty_like(gy_c)
    else:
        assert out.is_cuda and out.dtype == gY.dtype and out.shape == gY.shape
    gx_c = out.contiguous()

    n = x_c.numel()

    # 选择 Triton 输出 dtype
    if gx_c.dtype == torch.float16:
        DTYPE = tl.float16
    elif gx_c.dtype == torch.bfloat16:
        DTYPE = tl.bfloat16
    elif gx_c.dtype == torch.float32:
        DTYPE = tl.float32
    else:
        raise TypeError(f"unsupported dtype: {gx_c.dtype}")

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)

    _gelu_drop_grad_kernel[grid](
        x_c, gy_c, m_c, gx_c,
        n_elements=n,
        DTYPE=DTYPE,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )
    return gx_c



def gelu_drop_double_grad_triton(ggX: torch.Tensor, gY: torch.Tensor, x: torch.Tensor, m: torch.Tensor,
                                 out_ggY: torch.Tensor = None, out_ggx: torch.Tensor = None):
    """
    计算 (ggY, ggx)：
      ggY = ggX * m * GELU'(x)
      ggx = ggX * gY * m * GELU''(x)
    - ggX, gY, x, m 需同形状且均在 CUDA
    - out_ggY dtype 建议与 gY.dtype 对齐；out_ggx dtype 建议与 x.dtype 对齐
    """
    assert ggX.is_cuda and gY.is_cuda and x.is_cuda and m.is_cuda
    assert ggX.shape == gY.shape == x.shape == m.shape

    ggX_c = ggX.contiguous()
    gY_c  = gY.contiguous()
    x_c   = x.contiguous()
    m_c   = m.contiguous()

    if out_ggY is None:
        out_ggY = torch.empty_like(gY_c)
    if out_ggx is None:
        out_ggx = torch.empty_like(x_c)
    ggy_c  = out_ggY.contiguous()
    ggx_c2 = out_ggx.contiguous()

    n = ggX_c.numel()

    def _to_tl_dtype(t: torch.Tensor):
        if t.dtype == torch.float16:  return tl.float16
        if t.dtype == torch.bfloat16: return tl.bfloat16
        if t.dtype == torch.float32:  return tl.float32
        raise TypeError(f"unsupported dtype: {t.dtype}")

    DTYPE_GGY = _to_tl_dtype(ggy_c)
    DTYPE_GGX = _to_tl_dtype(ggx_c2)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)

    _gelu_drop_double_grad_kernel[grid](
        ggX_c, gY_c, x_c, m_c,
        ggy_c, ggx_c2,
        n_elements=n,
        DTYPE_GGY=DTYPE_GGY,
        DTYPE_GGX=DTYPE_GGX,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )
    return ggy_c, ggx_c2


def _phi(x):
    # x 已在 GPU；常数是 Python float，会作为 kernel 的标量传入，不触发 H2D 拷贝
    return torch.exp(-0.5 * x * x) * _INV_SQRT2PI

def _gelu_prime(x):
    # GELU'(x) = 0.5(1+erf(x/√2)) + x φ(x)
    return 0.5 * (1.0 + torch.special.erf(x * _INV_SQRT2)) + x * _phi(x)

def _gelu_double_prime(x):
    # GELU''(x) = (2 - x^2) φ(x)
    return (2.0 - x * x) * _phi(x)



# 应该是relu+norm fused kernel
@torch.no_grad()
def instanceNorm_double_backwards_triton(
    x, gamma, ggX, ggG, ggB, gO, out,
    eps=1e-5, BLOCK_M=256
):
    N, C, H, W = x.shape
    M = H * W

    # flatten contiguous
    x_flat   = x.contiguous().view(N, C, M)
    out_flat = out.contiguous().view(N, C, M)
    gO_flat  = gO.contiguous().view(N, C, M)

    # (2) ReLU mask: gO entering IN must be masked
    relu_mask = (out_flat > 0)
    gO_in = (gO_flat * relu_mask).to(torch.float32)   # fp32 + masked

    # (3) stats in fp32
    x_f = x_flat.to(torch.float32)
    mean = x_f.mean(dim=2)                            # (N,C) fp32
    var  = x_f.var(dim=2, unbiased=False)             # (N,C) fp32
    inv_std = torch.rsqrt(var + eps)                  # (N,C) fp32
    xcm = x_f - mean.unsqueeze(2)                     # (N,C,M) fp32

    # sums based on masked gO_in (fp32)
    sum_gO     = gO_in.sum(dim=2)                     # (N,C)
    sum_gO_xmu = (gO_in * xcm).sum(dim=2)             # (N,C)

    if ggX is not None:
        ggX_flat = ggX.contiguous().view(N, C, M).to(torch.float32)
        sum_ggX     = ggX_flat.sum(dim=2)
        sum_ggX_xmu = (ggX_flat * xcm).sum(dim=2)
        dot_ggX_gO  = (ggX_flat * gO_in).sum(dim=2)   # IMPORTANT: use gO_in
    else:
        ggX_flat = torch.empty(1, device=x.device, dtype=torch.float32)
        sum_ggX = torch.empty(1, device=x.device, dtype=torch.float32)
        sum_ggX_xmu = torch.empty(1, device=x.device, dtype=torch.float32)
        dot_ggX_gO = torch.empty(1, device=x.device, dtype=torch.float32)

    # outputs fp32 (2nd order more stable)
    gX  = torch.empty((N, C, M), device=x.device, dtype=torch.float32)
    ggO = torch.empty((N, C, M), device=x.device, dtype=torch.float32)

    # (1) atomic_add target must be zero-init
    if (gamma is not None) and (ggX is not None):
        gG = torch.zeros((C,), device=x.device, dtype=torch.float32)
    else:
        gG = None

    gamma_f = gamma.to(torch.float32) if gamma is not None else torch.empty(1, device=x.device, dtype=torch.float32)
    ggG_f   = ggG.to(torch.float32)   if ggG   is not None else torch.empty(1, device=x.device, dtype=torch.float32)
    ggB_f   = ggB.to(torch.float32)   if ggB   is not None else torch.empty(1, device=x.device, dtype=torch.float32)

    # (4) 2D grid over (N*C, blocks_of_M)
    grid = (N * C, triton.cdiv(M, BLOCK_M))

    instance_norm_double_backward_kernel_blocked[grid](
        gO_in,
        ggX_flat,
        out_flat,
        gamma_f,
        ggG_f, ggB_f,
        var.contiguous(), inv_std.contiguous(),
        xcm.contiguous(),
        sum_gO.contiguous(), sum_gO_xmu.contiguous(),
        sum_ggX.contiguous(), sum_ggX_xmu.contiguous(),
        dot_ggX_gO.contiguous(),
        gX,
        (gG if gG is not None else torch.empty((1,), device=x.device, dtype=torch.float32)),
        ggO,
        M,
        eps=eps,
        has_gamma=(gamma is not None),
        has_ggX=(ggX is not None),
        has_ggG=(ggG is not None),
        has_ggB=(ggB is not None),
        C=C,
        BLOCK_M=BLOCK_M,
        num_warps=4,
    )

    return gX.view(N, C, H, W), gG, ggO.view(N, C, H, W)





class Snd_Order_GeluDrop(torch.autograd.Function):
    '''2 forward: relu+pool+linear.backward '''
    @staticmethod
    def forward(ctx, gY, x, m):
        # gX = gY * m * _gelu_prime(x)
        gX = gelu_drop_grad_triton(gY, x, m)
        ctx.save_for_backward(gY, x, m)
        return gX, None


    @staticmethod
    def backward(ctx, ggX, _ggNone):
        gY, x, m = ctx.saved_tensors
        # 二阶：单核同时得到 (ggY, ggx)
        ggY, ggx = gelu_drop_double_grad_triton(ggX, gY, x, m)
        # 对 y 与 m 不回传梯度
        return ggY, ggx, None


class Fst_Order_GeluDrop(torch.autograd.Function):
    '''forward: gelu+dropout '''
    @staticmethod
    def forward(ctx, x, p):
        # 生成缩放后掩码：m = mask / (1-p)
        if p <= 0.0:
            m = torch.ones_like(x)
        else:
            # 为了数值安全，防止 p 接近 1
            p_clamped = torch.clamp(torch.as_tensor(p, dtype=x.dtype, device=x.device),
                                    min=0.0, max=1.0 - 1e-6).item()
            keep_prob = 1.0 - p_clamped
            mask = (torch.rand_like(x) < keep_prob).to(x.dtype)
            m = mask / keep_prob

        y_gelu = F.gelu(x)  # 精确 erf 版本
        y = y_gelu * m

        # 保存必要信息（注意把已缩放的 m 存起来，避免二阶里还要知道 p）
        ctx.save_for_backward(x, m)
        return y

    @staticmethod
    def backward(ctx, dLdy):
        x, m = ctx.saved_tensors
        # p = ctx.p
        return Snd_Order_GeluDrop.apply(dLdy ,x, m)


class GeluDrop(nn.Module):
    def __init__(self,  p=0.1):
        self.p = p
        super().__init__()
    def forward(self, input):
        out = Fst_Order_GeluDrop.apply(input,self.p)
        return out