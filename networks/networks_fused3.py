

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import torchvision.transforms as transforms


import torch
from functools import reduce
from operator import mul


import triton
import triton.language as tl


BLOCK_M = 64

def instanceNorm_backward_plain( x, gamma, grad_output, out, eps=1e-5):
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




@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': bs}, num_warps=4, num_stages=2)
        for bs in [64, 128, 256, 512, 1024]
    ],
    key=['HW'],
)
@triton.jit
def _instancenorm_backward_kernel(
    dY, X, gamma, out_ptr, 
    mean_ptr, rstd_ptr, dX, dgamma, dbeta,
    stride_n, stride_c, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    # program id layout: (N, C)
    n = tl.program_id(0)        # batch dimension
    c = tl.program_id(1)        # channel dimension
    idx_offset = n * stride_n + c * stride_c

    dy_ptr = dY + idx_offset
    x_ptr = X + idx_offset
    dx_ptr = dX + idx_offset

    # 缓存通道对应的 gamma, mean, rstd
    g = tl.load(gamma + c).to(tl.float32)
    mean_val = tl.load(mean_ptr + (n * C + c)).to(tl.float32)
    rstd_val = tl.load(rstd_ptr + (n * C + c)).to(tl.float32)
    N = HW

    # 初始化局部累加器：用于 dgamma/dbeta 和 dX 求和
    dgam_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    dbet_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    term1_sum_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    term1_xhat_sum_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # 第一次遍历：累积 dgamma/dbeta 以及计算 dX 所需的全局求和
    for off in range(0, HW, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < HW

        x_val = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        dy_val = tl.load(dy_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        # out_val = tl.load(out_ptr + idx_offset + idx, mask=mask, other=0.0).to(tl.float32)
        # out_val = tl.where(out_val <= 0.0, 0.0, 1.0)
        # dy_val *= out_val
        

        x_hat = (x_val - mean_val) * rstd_val

        # 累积 dgamma, dbeta
        dgam_acc += dy_val * x_hat
        dbet_acc += dy_val

        # 累积计算 dX 时所需的求和：sum_j (g*dy_j) 和 sum_j (g*dy_j * x_hat_j)
        term1_val = dy_val * g
        term1_sum_acc += term1_val
        term1_xhat_sum_acc += term1_val * x_hat

    # 将向量求和 reduce 为标量
    dgam = tl.sum(dgam_acc, axis=0)
    dbet = tl.sum(dbet_acc, axis=0)
    term1_sum = tl.sum(term1_sum_acc, axis=0)
    term1_xhat_sum = tl.sum(term1_xhat_sum_acc, axis=0)

    # 写回 dgamma, dbeta（每个程序实例写入相同结果，不会产生数据竞争）
    tl.store(dgamma + c, dgam)
    tl.store(dbeta + c, dbet)

    # 计算公式中的 1/N 系数
    term2_global = term1_sum / N            # = mean(g * dy)
    term3_global = term1_xhat_sum / N       # = mean(g * dy * x_hat)

    # 第二次遍历：根据全局求和结果计算 dX
    for off in range(0, HW, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < HW

        x_val = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        dy_val = tl.load(dy_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        # out_val = tl.load(out_ptr + idx_offset + idx, mask=mask, other=0.0).to(tl.float32)
        # dy_val = tl.where(out_val <= 0.0, 0.0, dy_val)
        x_hat = (x_val - mean_val) * rstd_val
        term1_val = dy_val * g

        # 公式：dx_i = rstd * [g*dy_i - term2_global - term3_global * x_hat_i]
        dx_val = rstd_val * (term1_val - term2_global - term3_global * x_hat)
        tl.store(dx_ptr + idx, dx_val, mask=mask)


def instance_norm_backward_triton(x, gamma, grad_output,out, eps=1e-5):
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



# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE': bs}, num_warps=4, num_stages=2)
#         for bs in [64, 128, 256, 512, 1024]
#     ],
#     key=['HW'],
# )
@triton.jit
def _instancenorm_stats_kernel(
    X, mean_ptr, rstd_ptr,
    stride_n, stride_c, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # program id layout: (N, C)
    n = tl.program_id(0)
    c = tl.program_id(1)

    idx_offset = n * stride_n + c * stride_c
    x_ptr = X + idx_offset

    # 用标量累加，避免你原来那种 [BLOCK_SIZE] 累加器占用大量寄存器
    sum_x = tl.zeros([], dtype=tl.float32)
    sum_x2 = tl.zeros([], dtype=tl.float32)

    for off in range(0, HW, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < HW
        x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    invN = 1.0 / tl.full([], HW, tl.float32)
    mean = sum_x * invN
    # var = E[x^2] - (E[x])^2
    var = sum_x2 * invN - mean * mean
    rstd = tl.rsqrt(var + eps)

    # mean/rstd buffer 是按 (N*C) 展平的：index = n*C + c
    idx_nc = n * C + c
    tl.store(mean_ptr + idx_nc, mean)
    tl.store(rstd_ptr + idx_nc, rstd)


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE': bs}, num_warps=4, num_stages=2)
#         for bs in [64, 128, 256, 512, 1024]
#     ],
#     key=['HW'],
# )
@triton.jit
def _instancenorm_backward_fused_kernel(
    dY, X, gamma, out_ptr,
    mean_ptr, rstd_ptr,
    dX, dgamma, dbeta,
    stride_n, stride_c, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    # program id layout: (N, C)
    n = tl.program_id(0)
    c = tl.program_id(1)

    idx_offset = n * stride_n + c * stride_c
    dy_ptr = dY + idx_offset
    x_ptr = X + idx_offset
    dx_ptr = dX + idx_offset
    o_ptr = out_ptr + idx_offset  # out 是 ReLU 后的输出（或你也可以传 pre-activation + mask）

    g = tl.load(gamma + c).to(tl.float32)
    idx_nc = n * C + c
    mean = tl.load(mean_ptr + idx_nc).to(tl.float32)
    rstd = tl.load(rstd_ptr + idx_nc).to(tl.float32)

    # 第一遍：算 dbeta = sum(dy), dgamma = sum(dy*xhat)
    sum_dy = tl.zeros([], dtype=tl.float32)
    sum_dy_xhat = tl.zeros([], dtype=tl.float32)

    for off in range(0, HW, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < HW

        x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + idx, mask=mask, other=0.0).to(tl.float32)

        # ReLU backward gate：out<=0 的位置 dy=0
        outv = tl.load(o_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        dy = tl.where(outv > 0.0, dy, 0.0)

        xhat = (x - mean) * rstd
        sum_dy += tl.sum(dy, axis=0)
        sum_dy_xhat += tl.sum(dy * xhat, axis=0)

    # 跨 N 的正确累加（修复你原来的数据竞争）
    tl.atomic_add(dbeta + c, sum_dy)
    tl.atomic_add(dgamma + c, sum_dy_xhat)

    # 第二遍：算 dX
    invN = 1.0 / tl.full([], HW, tl.float32)
    term2 = (g * sum_dy) * invN             # mean(g*dy)
    term3 = (g * sum_dy_xhat) * invN        # mean(g*dy*xhat)

    for off in range(0, HW, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < HW

        x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + idx, mask=mask, other=0.0).to(tl.float32)

        outv = tl.load(o_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        dy = tl.where(outv > 0.0, dy, 0.0)

        xhat = (x - mean) * rstd
        dx = rstd * (g * dy - term2 - term3 * xhat)

        tl.store(dx_ptr + idx, dx, mask=mask)


def instanceNorm_backward( x, gamma, grad_output, eps=1e-5):
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




import triton
import triton.language as tl

@triton.jit
def instance_norm_double_backward_kernel_blocked(
    # (N,C,M) fp32, gO 已经按 ReLU mask 处理过
    gO_ptr, ggX_ptr, out_ptr,

    # (C,) fp32
    gamma_ptr,
    ggG_ptr, ggB_ptr,

    # (N,C) fp32
    var_ptr, inv_std_ptr,

    # (N,C,M) fp32
    xcm_ptr,

    # (N,C) fp32  —— 这些必须是“全 M 的总和”，由 host 端算好
    sum_gO_ptr, sum_gO_xmu_ptr,
    sum_ggX_ptr, sum_ggX_xmu_ptr,
    dot_ggX_gO_ptr,

    # outputs
    gX_ptr, gG_ptr, ggO_ptr,   # gG 是 (C,) fp32, atomic_add

    M,                         # runtime
    eps: tl.constexpr,

    has_gamma: tl.constexpr,
    has_ggX: tl.constexpr,
    has_ggG: tl.constexpr,
    has_ggB: tl.constexpr,

    C: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_nc = tl.program_id(0)       # 0 .. N*C-1
    pid_blk = tl.program_id(1)      # 0 .. ceil(M/BLOCK_M)-1
    c = pid_nc % C
    nc = pid_nc

    offs_m = pid_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    m = offs_m < M
    base = nc * M
    offs = base + offs_m

    # ---- load scalars (fp32) ----
    var     = tl.load(var_ptr + nc).to(tl.float32)
    inv_std = tl.load(inv_std_ptr + nc).to(tl.float32)
    inv_std2 = inv_std * inv_std
    inv_std3 = inv_std2 * inv_std

    M_f = tl.full([], M, tl.float32)
    invM = 1.0 / M_f

    sum_gO     = tl.load(sum_gO_ptr     + nc).to(tl.float32)
    sum_gO_xmu = tl.load(sum_gO_xmu_ptr + nc).to(tl.float32)

    gamma_val = tl.full([], 1.0, tl.float32)
    if has_gamma:
        gamma_val = tl.load(gamma_ptr + c).to(tl.float32)

    # ---- load block vectors (fp32) ----
    gO  = tl.load(gO_ptr  + offs, mask=m, other=0.0).to(tl.float32)
    xcm = tl.load(xcm_ptr + offs, mask=m, other=0.0).to(tl.float32)

    ggX = tl.zeros([BLOCK_M], dtype=tl.float32)
    if has_ggX:
        ggX = tl.load(ggX_ptr + offs, mask=m, other=0.0).to(tl.float32)

    # ---- precomputed ggX sums (fp32) ----
    sum_ggX     = tl.full([], 0.0, tl.float32)
    sum_ggX_xmu = tl.full([], 0.0, tl.float32)
    dot_ggX_gO  = tl.full([], 0.0, tl.float32)
    if has_ggX:
        sum_ggX     = tl.load(sum_ggX_ptr     + nc).to(tl.float32)
        sum_ggX_xmu = tl.load(sum_ggX_xmu_ptr + nc).to(tl.float32)
        dot_ggX_gO  = tl.load(dot_ggX_gO_ptr  + nc).to(tl.float32)

    # =========================
    # gX (only if has_ggX)
    # =========================
    if has_ggX:
        A = (sum_ggX * sum_gO) * invM - dot_ggX_gO + (3.0 * inv_std2 * sum_gO_xmu * sum_ggX_xmu) * invM
        term0 = xcm * inv_std3 * A * invM
        term1 = sum_ggX_xmu * inv_std3 * (sum_gO * invM - gO) * invM
        term2 = sum_gO_xmu  * inv_std3 * (sum_ggX * invM - ggX) * invM
        gX_val = term0 + term1 + term2
        if has_gamma:
            gX_val = gX_val * gamma_val
        tl.store(gX_ptr + offs, gX_val, mask=m)

    # =========================
    # gG (atomic accumulate over blocks)
    # only if has_gamma & has_ggX
    # =========================
    if has_gamma and has_ggX:
        fb = (inv_std * invM) * (M_f * gO - sum_gO - xcm * inv_std2 * sum_gO_xmu)
        gG_part = tl.sum(ggX * fb, axis=0)
        tl.atomic_add(gG_ptr + c, gG_part)

    # =========================
    # ggO  (blocked)
    # =========================
    ggO_val = tl.zeros([BLOCK_M], dtype=tl.float32)

    if has_ggX:
        fb2 = (gamma_val * inv_std * invM) * (M_f * ggX - sum_ggX - xcm * inv_std2 * sum_ggX_xmu)
        ggO_val += fb2

    if has_ggG:
        ggG_val = tl.load(ggG_ptr + c).to(tl.float32)
        ggO_val += ggG_val * xcm * inv_std

    if has_ggB:
        ggB_val = tl.load(ggB_ptr + c).to(tl.float32)
        ggO_val += ggB_val

    # ReLU gate on ggO: multiply by mask(out>0)
    outv = tl.load(out_ptr + offs, mask=m, other=0.0).to(tl.float32)
    ggO_val = tl.where(outv > 0.0, ggO_val, 0.0)

    tl.store(ggO_ptr + offs, ggO_val, mask=m)
import torch
import triton

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


def instanceNorm_double_backwards_fn(x, gamma, beta, ggX, ggG, ggB, gO,
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


class Snd_Order_MyLinearFunction(torch.autograd.Function):
    '''2 forward: relu+pool+linear.backward '''
    # TODO: 如果做ckpt，那么记得保证forward可以在算完之后全释放掉。 然后backward再重新算一遍。
    # 现在也没有做save ctx，为啥内存消耗还是1483？
    @staticmethod
    def forward(ctx, dLdy, input, weight, out):
        ctx.save_for_backward( input, weight, dLdy, out )
        # dLdy[out<=0 ] = 0

        grad_output, dw, db = instance_norm_backward_triton(input, weight, dLdy, out)
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


class MyLinearFunction(torch.autograd.Function):
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
        return Snd_Order_MyLinearFunction.apply(dLdy ,input, weight, out)
        # db = gin.sum(0)
        # return gin, weight, db




class NormActive(nn.Module):
    # in_features 应该是1
    def __init__(self, channel_num, affine=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn([channel_num]))
        self.bias = nn.Parameter(torch.randn([channel_num]))
    def forward(self, input):
        out = MyLinearFunction.apply(input, self.weight, self.bias)
        return out
