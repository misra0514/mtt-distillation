# 理论上fused 2值不对
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

    return gI, gG, ggO

def batchNorm2d_backward(x, gamma, beta, grad_output, eps=1e-5):
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

@triton.jit
def _instancenorm_backward_kernel(
    dY, X, gamma, out_ptr, mean_ptr, rstd_ptr,
    dX, dgamma, dbeta,
    stride_n, stride_c, stride_hw,
    C, HW,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = pid // C
    c = pid % C
    start = batch_id * stride_n + c * stride_c
    dy  = dY + start
    x   = X  + start
    dx  = dX + start
    out = out_ptr + start
    mean = tl.load(mean_ptr + pid).to(tl.float32)
    rstd = tl.load(rstd_ptr + pid).to(tl.float32)
    # accumulate dgamma, dbeta
    _dgam = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _dbet = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, HW, BLOCK_SIZE):
        idx  = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < HW
        x_val  = tl.load(x  + idx, mask=mask, other=0.).to(tl.float32)
        dy_val = tl.load(dy + idx, mask=mask, other=0.).to(tl.float32)
        # 如果启用了 ReLU，则根据 out_val 决定 dy_val 是否为 0
        # if has_relu:
        # out_val = tl.load(out + idx, mask=mask, other=0.).to(tl.float32)
        # dy_val = tl.where(out_val <= 0.0, 0.0, dy_val)
        x_hat = (x_val - mean) * rstd
        _dgam += dy_val * x_hat
        _dbet += dy_val
    # reduce 到通道维度
    dgam = tl.sum(_dgam, axis=0)
    dbet = tl.sum(_dbet, axis=0)
    tl.store(dgamma + c, dgam)
    tl.store(dbeta + c, dbet)

    # compute dX per-element
    for off in range(0, HW, BLOCK_SIZE):
        idx  = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < HW
        x_val  = tl.load(x  + idx, mask=mask, other=0.).to(tl.float32)
        dy_val = tl.load(dy + idx, mask=mask, other=0.).to(tl.float32)

        # if has_relu:                                   # ★ 再做一次 ReLU 屏蔽
        # out_val = tl.load(out + idx,
        #                 mask=mask, other=0.).to(tl.float32)
        # df  = tl.where(out_val > 0,  1.0, 0.0)
        # dy_val = df *dy_val
        
        x_hat = (x_val - mean) * rstd
        N = HW
        g = tl.load(gamma + c).to(tl.float32)
        term1 = dy_val * g
        term2 = tl.sum(term1, axis=0) / N
        term3 = tl.sum(term1 * x_hat, axis=0) * x_hat / N
        dx_hat = rstd * (term1 - term2 - term3)
        tl.store(dx + idx, dx_hat, mask=mask)

def instancenorm_relu_backward_triton(x, gamma, grad_output, out, eps=1e-5):
    # x, grad_output: (N, C, H, W)
    N, C, H, W = x.shape
    HW = H * W
    stride_n = C * HW
    stride_c = HW
    BLOCK_SIZE = 1024  # 可以调优
    grad_output[out<=0 ] = 0
    # relu_grad = (out > 0).float()
    # grad_output *= relu_grad

    x_flat = x.contiguous().view(N, C, HW)
    out = out.contiguous().view(N, C, HW)
    grad_output_flat = grad_output.contiguous().view(N, C, HW)
    x_buf = x_flat.reshape(-1, HW)
    dy_buf = grad_output_flat.reshape(-1, HW)
    dX = torch.empty_like(x_buf)
    dgamma = torch.zeros(C, device=x.device, dtype=torch.float32)
    dbeta = torch.zeros(C, device=x.device, dtype=torch.float32)
    # 使用 forward 计算 mean 和 rstd（可选提前缓存）
    mean = x_buf.mean(dim=1, keepdim=False)
    var = x_buf.var(dim=1, unbiased=False)
    rstd = 1.0 / torch.sqrt(var + eps)
    mean_ptr = mean.contiguous()
    rstd_ptr = rstd.contiguous()
    _instancenorm_backward_kernel[(N * C,)](
        dy_buf, x_buf, gamma, out,
        mean_ptr, rstd_ptr,
        dX, dgamma, dbeta,
        stride_n, stride_c, HW, C, HW,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    dX = dX.view(N, C, H, W)
    return dX, dgamma, dbeta

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



@triton.jit
def instance_norm_double_backward_kernel(
    x_ptr, gO_ptr, ggX_ptr, out_ptr,
    gamma_ptr,     # gamma_ptr 用于 γ，beta_ptr 保留但不使用
    ggG_ptr, ggB_ptr,        # ggG_ptr 用于 ggG，ggB_ptr 用于 ggB
    mean_ptr, var_ptr, inv_std_ptr, xcm_ptr,
    sum_gO_ptr, sum_gO_xmu_ptr, sum_ggX_ptr, sum_ggX_xmu_ptr, dot_ggX_gO_ptr,
    gX_ptr, gG_ptr, ggO_ptr,
    N, C, M,
    eps: tl.constexpr,
    has_gamma: tl.constexpr, 
    has_ggX: tl.constexpr, has_ggG: tl.constexpr, has_ggB: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % C
    n = pid // C
    offs_m = tl.arange(0, BLOCK_M)
    mask = offs_m < M
    offs = (n * C + c) * M + offs_m
    x   = tl.load(x_ptr   + offs, mask=mask, other=0.0)
    gO  = tl.load(gO_ptr  + offs, mask=mask, other=0.0)
    xcm = tl.load(xcm_ptr + offs, mask=mask, other=0.0)
    # mean    = tl.load(mean_ptr    + n * C + c)
    var     = tl.load(var_ptr     + n * C + c)
    inv_std = tl.load(inv_std_ptr + n * C + c)
    inv_std2 = inv_std * inv_std
    inv_std3 = inv_std / (var + eps)
    sum_gO     = tl.load(sum_gO_ptr     + n * C + c)
    sum_gO_xmu = tl.load(sum_gO_xmu_ptr + n * C + c)
    # -------- gX 部分：利用 ggX 计算，再乘上 gamma --------
    if has_ggX:
        ggX          = tl.load(ggX_ptr + offs, mask=mask, other=0.0)
        sum_ggX      = tl.load(sum_ggX_ptr      + n * C + c)
        sum_ggX_xmu  = tl.load(sum_ggX_xmu_ptr  + n * C + c)
        dot_ggX_gO   = tl.load(dot_ggX_gO_ptr   + n * C + c)
        A = (sum_ggX * sum_gO) / M - dot_ggX_gO + 3 * inv_std2 * sum_gO_xmu * sum_ggX_xmu / M
        term0 = xcm * inv_std3 * A / M
        term1 = sum_ggX_xmu * inv_std3 * (sum_gO / M - gO) / M
        term2 = sum_gO_xmu  * inv_std3 * (sum_ggX / M - ggX) / M
        gX_val = term0 + term1 + term2
        if has_gamma:
            # 这里应该从 gamma_ptr 读取 γ，而不是从 ggG_ptr 读取
            gamma_val = tl.load(gamma_ptr + c)
            gX_val *= gamma_val
        tl.store(gX_ptr + offs, gX_val, mask=mask)
    # -------- gG 部分：仅当 gamma 和 ggX 均存在时计算 --------
    if has_gamma and has_ggX:
        # 这里只涉及 gO 和 ggX，与 gamma 或 ggG 无关
        gO_masked     = gO * mask
        gO_xcm_masked = gO * xcm * mask
        fb = (inv_std / M) * (
            M * gO_masked - tl.sum(gO_masked, axis=0) -
            xcm * inv_std2 * tl.sum(gO_xcm_masked, axis=0)
        )
        ggX_masked = ggX * mask
        gG_val = tl.sum(ggX_masked * fb, axis=0)
        tl.atomic_add(gG_ptr + c, gG_val)
    # -------- ggO 第一部分：first_back(ggX, gamma) --------
    if has_ggX:
        ggX_masked     = ggX * mask
        ggX_xcm_masked = ggX * xcm * mask
        # 若有 gamma，用 gamma_ptr 读取；否则默认为 1
        if has_gamma:
            gamma_val = tl.load(gamma_ptr + c)
        else:
            gamma_val = 1.0
        fb2 = (gamma_val * inv_std / M) * (
            M * ggX_masked - tl.sum(ggX_masked, axis=0) -
            xcm * inv_std2 * tl.sum(ggX_xcm_masked, axis=0)
        )
        ggO_val = fb2
    else:
        ggO_val = tl.zeros([BLOCK_M], dtype=tl.float32)
    # -------- ggO 第二部分：ggG * x_centered * inv_std --------
    if has_ggG:
        ggG_val = tl.load(ggG_ptr + c)
        ggO_val += ggG_val * xcm * inv_std * mask
    # -------- ggO 第三部分：ggB --------
    if has_ggB:
        ggB_val = tl.load(ggB_ptr + c)
        # ggB 是一维的 (C,) 张量，需要在 M 维上广播，所以直接加上 ggB_val * mask
        ggO_val += ggB_val * mask
    
    out_val = tl.load(out_ptr + offs, mask=mask, other=0.).to(tl.float32)
    ggO_val = tl.where(out_val <= 0.0, 0.0, ggO_val)
    tl.store(ggO_ptr + offs, ggO_val, mask=mask)
    
@torch.no_grad()

def instanceNorm_double_backwards_triton(x, gamma, ggX, ggG, ggB, gO, out, eps=1e-5):

    N, C, H, W = x.shape
    M = H * W
    x_flat = x.view(N, C, M)
    gO_flat = gO.view(N, C, M)
    mean = x_flat.mean(dim=2)
    var = x_flat.var(dim=2, unbiased=False)
    std = torch.sqrt(var + eps)
    inv_std = 1.0 / std
    xcm = x_flat - mean.unsqueeze(2)
    sum_gO = gO_flat.sum(dim=2)
    sum_gO_xmu = (gO_flat * xcm).sum(dim=2)
    if ggX is not None:
        ggX_flat = ggX.view(N, C, M)
        sum_ggX = ggX_flat.sum(dim=2)
        sum_ggX_xmu = (ggX_flat * xcm).sum(dim=2)
        dot_ggX_gO = (ggX_flat * gO_flat).sum(dim=2)
    else:
        ggX_flat = torch.empty(1, device=x.device)
        sum_ggX = sum_ggX_xmu = dot_ggX_gO = torch.empty(1, device=x.device)
    gX = torch.empty_like(x_flat)
    gG = torch.empty_like(gamma) if gamma is not None and ggX is not None else None
    ggO = torch.empty_like(x_flat)
    grid = (N * C,)
    instance_norm_double_backward_kernel[grid](
        x_flat, gO_flat, ggX_flat, out,
        gamma if gamma is not None else torch.empty(1, device=x.device),
        ggG   if ggG   is not None else torch.empty(1, device=x.device),
        ggB   if ggB   is not None else torch.empty(1, device=x.device),
        mean, var, inv_std, xcm,
        sum_gO, sum_gO_xmu,
        sum_ggX, sum_ggX_xmu, dot_ggX_gO,
        gX,
        gG   if gG   is not None else torch.empty_like(gamma),
        ggO,
        N, C, M,
        eps,
        gamma is not None, 
        ggX is not None, ggG is not None, ggB is not None,
        BLOCK_M=M,
    )
    # ggO = ggO.view(N, C, H, W)
    # ggO[out <= 0] = 0
    # print("CKPT--Norm",ggO.sum().item())

    return gX.view(N, C, H, W), gG, ggO.view(N, C, H, W)



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

def instanceNorm_double_backwards_fn_cln(x, gamma, ggX, ggG, ggB, gO,
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
    if ggX is not None:
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
        t0 = gO_flat * inv_std
        t1 = -(inv_std * sum_gO) / M
        t2 = -x_centered * inv_std3 * sum_gO_xmu / M
        gX_G = ggG_exp * (t0 + t1 + t2)
        gX = gX + gX_G if gX is not None else gX_G

    # gG
    gG = None
    if gamma is not None and ggX is not None:
        def first_back(g, gamma_val):
            return (gamma_val * inv_std / M) * (
                M * g - g.sum(dim=2, keepdim=True) -
                x_centered * inv_std ** 2 * (g * x_centered).sum(dim=2, keepdim=True)
            )
        gG = (ggX_flat * first_back(gO_flat, torch.ones_like(gamma_exp))).sum(dim=2)
    # ggO
    ggO = None
    if ggX is not None:
        ggO = first_back(ggX_flat, gamma_exp)
    if ggG is not None:
        ggO_G = ggG_exp * x_centered * inv_std
        ggO = ggO + ggO_G if ggO is not None else ggO_G
    if ggB is not None:
        ggB_exp = ggB.view(1, C, 1)
        ggO = ggO + ggB_exp if ggO is not None else ggB_exp
    return gX.view(N, C, H, W) if gX is not None else None, gG, ggO.view(N, C, H, W)


class Snd_Order_MyLinearFunction(torch.autograd.Function):
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


class MyLinearFunction(torch.autograd.Function):
    '''forward: relu+pool+linear '''
    @staticmethod
    def forward(ctx, input, weight, bias):
        out1 = F.instance_norm(input,weight= weight, bias = bias)
        out = F.relu(out1)
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
    def __init__(self, channel_num):
        super().__init__()
        self.weight = nn.Parameter(torch.randn([channel_num]))
        self.bias = nn.Parameter(torch.randn([channel_num]))
    def forward(self, input):
        out = MyLinearFunction.apply(input, self.weight, self.bias)
        return out




_INV_SQRT2 = 1.0 / (2.0 ** 0.5)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)  # 0.3989422804014327

# ------ Triton kernel: gX = gY * m * gelu'(x) ------
@triton.jit
def _gelu_drop_grad_kernel(
    x_ptr, gy_ptr, m_ptr, gx_ptr,
    n_elements: tl.constexpr,
    DTYPE: tl.constexpr,           # tl.float32 / tl.float16 / tl.bfloat16
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # load -> fp32 计算，提升数值稳定性
    x  = tl.load(x_ptr  + offs, mask=mask, other=0).to(tl.float32)
    gy = tl.load(gy_ptr + offs, mask=mask, other=0).to(tl.float32)
    mm = tl.load(m_ptr  + offs, mask=mask, other=0).to(tl.float32)

    inv_sqrt2   = 0.7071067811865476  # _INV_SQRT2
    inv_sqrt2pi = 0.3989422804014327  # _INV_SQRT2PI

    # φ(x) = exp(-x^2/2) / sqrt(2π)
    phi = tl.exp(-0.5 * x * x) * inv_sqrt2pi
    # gelu'(x) = 0.5*(1+erf(x/√2)) + x*φ(x)
    gp = 0.5 * (1.0 + tl.erf(x * inv_sqrt2)) + x * phi

    gx = gy * mm * gp
    tl.store(gx_ptr + offs, gx.to(DTYPE), mask=mask)


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


# -------- 二阶：同时算 ggY 与 ggx ------------
@triton.jit
def _gelu_drop_double_grad_kernel(
    ggx_in_ptr, gy_ptr, x_ptr, m_ptr,     # inputs: ggX, gY, x, m
    ggy_out_ptr, ggx_out_ptr,             # outputs: ggY, ggx
    n_elements: tl.constexpr,
    DTYPE_GGY: tl.constexpr,              # 输出 ggY 的 dtype（通常跟 gY 一致）
    DTYPE_GGX: tl.constexpr,              # 输出 ggx 的 dtype（通常跟 x 一致）
    BLOCK_SIZE: tl.constexpr
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    ggX = tl.load(ggx_in_ptr + offs, mask=mask, other=0).to(tl.float32)
    gY  = tl.load(gy_ptr     + offs, mask=mask, other=0).to(tl.float32)
    x   = tl.load(x_ptr      + offs, mask=mask, other=0).to(tl.float32)
    mm  = tl.load(m_ptr      + offs, mask=mask, other=0).to(tl.float32)

    inv_sqrt2   = 0.7071067811865476
    inv_sqrt2pi = 0.3989422804014327

    # φ(x), g'(x), g''(x)
    phi = tl.exp(-0.5 * x * x) * inv_sqrt2pi
    gp  = 0.5 * (1.0 + tl.erf(x * inv_sqrt2)) + x * phi
    gpp = (2.0 - x * x) * phi

    # ggY = ggX * m * g'(x)
    ggY = ggX * mm * gp
    # ggx = ggX * gY * m * g''(x)
    ggx = ggX * gY * mm * gpp

    tl.store(ggy_out_ptr + offs, ggY.to(DTYPE_GGY), mask=mask)
    tl.store(ggx_out_ptr + offs, ggx.to(DTYPE_GGX), mask=mask)


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
        # out = F.gelu(input)
        # out = F.dropout(out, p=self.p)
        out = Fst_Order_GeluDrop.apply(input,self.p)
        return out