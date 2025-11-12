# 10.25
# 最基本的并行方法，F+Fcg并行，要求backward之后上半部分已经完全释放，下面还保留着计算图。用来做下一个B。

# 思路1：用with nograd，再手动写二阶展开，不能调用cg=t了，而且这样合并不了（必须把with内的op单独拿出来）
# 思路2：正常forward，但是想办法让一半的图loss.backward(create graph)（这个好像不太靠谱，因为就不能并行了） 
# 思路3：全面重写kernel。forward用Batchgemm。一阶导数正常算，但是只save下半部分，二阶导数不变。（好像也不行，因为只要cg=t中间节点就保留，也不可能没有中间节点？）

# 难点在于，同一个op，似乎很难让其一半保存在计算图中，一半不保存；op的输出tensor节点，我也不能让他一半保存一半不保存。

# 手动展开一阶导数。然后每个op的输出手动分成两个量，然后上半部分detach，这样可能可以。


# 尝试用无状态函数来解决这个问题： 正常做两个不存cg的forward，然后把第二半的grad 拿到已经做好的stateless model里面去求。
# 即便是无状态的模型，也是得做一遍forward，然后才能backword吧？毕竟里面有那种多临时变量？
# 所以stateless的办法看起来可行，但是实际实现的时候，为了做backward，不只需要一个loss，还需要所有的中间变量

# 新的问题： 不能做到只根据loss信息就做backward。做backward必须在某一个已经建立好的计算图上，而且还需要各种中间变量。这些中间变量也很难自定义修改。


# 使用了test 10的base code。
# 目前策略： 使用一个stateless model来处理二阶导数。

#---------------------------------------------------#
# 本testcode 主要对比cg+autograd 实现的backward 和手写的stateless backwards的精度（速度）差距
#---------------------------------------------------#



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
import time 
import random
import numpy as np
import os

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭自动优化，确保计算确定性
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 保证 CUDA 计算稳定（仅对 PyTorch 1.8+ 有效）


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
    """
    计算 InstanceNorm 的反向传播 (dX, dgamma, dbeta)。
    x, grad_output: (N, C, H, W)
    gamma: (C,)
    """
    N, C, H, W = x.shape
    HW = H * W
    stride_n = C * HW
    stride_c = HW

    # 展平输入以便在 Triton 内核中按 HW 维度遍历
    x_flat = x.contiguous().view(N, C, HW)
    out_flat = out.contiguous().view(N, C, HW)

    grad_output_flat = grad_output.contiguous().view(N, C, HW)
    grad_output_flat[out_flat <= 0] = 0

    x_buf = x_flat.reshape(-1, HW)
    dy_buf = grad_output_flat.reshape(-1, HW)

    dX = torch.empty_like(x_buf)
    dgamma = torch.zeros(C, device=x.device, dtype=torch.float32)
    dbeta = torch.zeros(C, device=x.device, dtype=torch.float32)

    # 预先计算均值和反标准差
    mean = x_buf.mean(dim=1)
    var = x_buf.var(dim=1, unbiased=False)
    rstd = 1.0 / torch.sqrt(var + eps)

    mean_ptr = mean.contiguous()
    rstd_ptr = rstd.contiguous()

    # 使用二维 grid 调度 (N, C)
    grid = (N, C)
    _instancenorm_backward_kernel[grid](
        dy_buf, x_buf, gamma, out_flat,
        mean_ptr, rstd_ptr,
        dX, dgamma, dbeta,
        stride_n, stride_c, C, HW,
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

# TODO: einsum之后是不连续的？？这里需要解决。

class LinearStacked_2(nn.Module):
    # batch* fusion * channel * WH。 
    def __init__(self ,in_features, out_features, Fuse):
        super(LinearStacked_2, self).__init__()
        self.Fuse = Fuse
        self.in_features = in_features
        self.out_features = out_features
        # TODO: 在nn实现中，这里是一个转制，也就是说应该是Fuse, out_features, in_features
        self.weight = torch.nn.Parameter(torch.randn(Fuse* out_features,in_features))
        self.bias = torch.nn.Parameter(torch.randn(self.Fuse* out_features))

    def forward(self, x):
        """
        x目前仅支持二维输入： B* STK * In。 B和stk可以view 在一起。 weight  STK*IN*OUT 
        """
        # self.weight = self.weight.view(self.Fuse,self.out_features,self.in_features)
        # self.bias = self.bias.view(self.Fuse, self.out_features)

        x = x.view(-1,self.Fuse, self.in_features)
        x = torch.einsum("abc,bcd->abd",x,self.weight.view(self.Fuse,self.out_features,self.in_features).transpose(-1, -2)) 
        x = x+self.bias.view(self.Fuse, self.out_features)

        x = x.contiguous()
        return x


# TODO: 3 实现一个stateless 的snd order backward 包含2nd order + 1st order bwd；__init__不定义任何变量。
# double conv  bwd 可以直接import/

def group_conv_double_backward_fn():
    # const std::optional<Tensor>& ggI_opt, const std::optional<Tensor>& ggW_r_opt, const std::optional<Tensor>& ggb_opt,
    # const Tensor& gO_r, const Tensor& weight_r, const Tensor& input,
    # IntArrayRef stride_, IntArrayRef padding_, IntArrayRef dilation_,
    # bool transposed_, IntArrayRef output_padding_, int64_t groups_,
    # std::array<bool, 3> output_mask
    torch.ops.aten._convolution_double_backward(ggI_opt, )

# class double_bwd_conv():
#     pass

class Conv_original(nn.Module):
    def __init__(self, net_width, Fuse):
        super(Conv_original, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=net_width, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(net_width, affine=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.classifier = nn.Linear(net_width * 16 * 16, 10)
    def forward(self, x):
        # x = x.view(-1, self.num_feat)        # 10, 256, 4,4   -> 20, 2048
        x = self.conv1(x)          # N x net_width x 32 x 32
        x = self.norm1(x)          # N x net_width x 32 x 32
        x = F.relu(x)             # N x net_width x 32 x 32
        x = self.pool1(x)          # N x net_width x 16 x 16
        x = x.view(x.size(0), -1) # Flatten to N x (net_width*16*16)
        x = self.classifier(x)    # N x 10
        return x
    
class ConvNet(nn.Module):
    def __init__(self, net_width, Fuse):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        self.norm1 = nn.BatchNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.classifier = LinearStacked_2(net_width * 16 * 16, 10,2 )
    def forward(self, x):
        x = x.view(-1,6 ,32,32)        # 10, 256, 4,4   -> 20, 2048
        x = self.conv1(x)          # N x net_width x 32 x 32
        x = self.norm1(x)          # N x net_width x 32 x 32
        x = F.relu(x)             # N x net_width x 32 x 32
        x = self.pool1(x)          # N x net_width x 16 x 16
        x = x.view(x.size(0), -1) # Flatten to N x (net_width*16*16)
        x = self.classifier(x)    # N x 10
        return x
    
class Myconv(nn.Module):
    def __init__(self, net_width, Fuse):
        super(Myconv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        self.norm1 = nn.BatchNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.classifier = LinearStacked_2(net_width * 16 * 16, 10,2 )
    def forward(self, x, criterion, target):
        x = x.view(-1,6 ,32,32)        # 10, 256, 4,4   -> 20, 2048
        x_norm = self.conv1(x)          # N x net_width x 32 x 32
        x_relu = self.norm1(x_norm)          # N x net_width x 32 x 32
        x_pool = F.relu(x_relu)             # N x net_width x 32 x 32
        x_pool = self.pool1(x_pool)          # N x net_width x 16 x 16
        x_lin = x_pool.view(x_pool.size(0), -1) # Flatten to N x (net_width*16*16)
        out = self.classifier(x_lin)    # N x 10
        # TODO: 2 需要手动实现一阶的loss backward。并且同时把需要的ctx输出。
        # 2.1 criterion
        out = out.view(-1,10)
        loss = criterion(out, target) 

        # 2.2 求dw，以及所有中间要存ctx的量。 这个地方不知道能不能用autograd
        # dw = torch.torch.autograd.grad(loss, list(model.parameters()))
        dw=[]
        dout = torch.torch.autograd.grad(loss, out) # criterion bwd
        grad_output, dlinw, dlinb = torch.torch.autograd.grad(out, [x_lin, self.classifier.weight, self.classifier.bias], grad_outputs= dout)  # linear bwd
        # relu bwd， pooling bwd
        grad_output = grad_output.view(x_pool.shape)
        grad_output = F.interpolate(grad_output, scale_factor=2, mode='nearest') /4
        relu_grad = (x_relu > 0).float()
        grad_output = grad_output * relu_grad
        grad_output, d_gamma, d_beta = torch.torch.autograd.grad(x_relu, [x_norm,self.norm1.weight, self.norm1.bias], grad_outputs=grad_output ) # norm bwd
        grad_output, dconvw, dconvb = torch.torch.autograd.grad(x_norm, [x,self.conv1.weight, self.conv1.bias], grad_outputs=grad_output ) # conv bwd
        dw = [ dconvw, dconvb, d_gamma, d_beta, dlinw, dlinb]

        # 2.3 求grand_loss, 这一步根据具体情况调整。
        weight = [d.sum() for d in dw]
        grad_loss = sum(weight)
        return grad_loss, dw

def load_state_dict_by_position(model, pretrained_state_dict):
    model_state_dict = model.state_dict()
    new_state_dict = {}
    # 取出当前模型的参数名字和值（有顺序）
    model_items = list(model_state_dict.items())
    pretrained_items = list(pretrained_state_dict.items())
    assert len(model_items) == len(pretrained_items), \
        f"参数数量不一致：当前模型有 {len(model_items)} 个参数，预训练模型有 {len(pretrained_items)} 个参数"

    for (model_key, _), (_, pretrained_val) in zip(model_items, pretrained_items):
        new_state_dict[model_key] = pretrained_val

    model.load_state_dict(new_state_dict)


if __name__ == "__main__":
    flag = 'myconv'
    Fuse = 2
    batch_size = 1024
    set_random_seed() # 仅仅在ACC test的时候使用。会严重影响性能。 

    model1 = ConvNet(32, Fuse).to("cuda")
    model2 = Myconv(32, Fuse).to("cuda")
    if flag =='conv':
        model = model1
    else:
        model = model2

    transform = transforms.ToTensor()
    cifar10 = torchvision.datasets.CIFAR10(root='/scratch/yguo25/files/mtt-distillation/data', train=True, download=True, transform=transform)
    img, label = cifar10[0]
    x = img.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # shape: [batch_size, 3, 32, 32]
    x = x.clone().detach().to("cuda").requires_grad_(True)
    target = torch.tensor([label] * batch_size, device="cuda")  # shape: [batch_size]


    # TODO: 1 输入和target、weight先变成两倍

    # torch.save(model.state_dict(), 'model_test7.pt')
    # exit()
    # # model.load_state_dict(torch.load('model_test5.pt'), strict = False)

    pretrained_dict = torch.load("model_test7.pt")
    for i,j in pretrained_dict.items():
        if j.ndim  != 0 :
            pretrained_dict[i] = torch.cat([j, j], dim=0)

    # 如何复制，取决于原始weight里面是怎么排布的， linear放在最外面，但是weight不知道。（应该也是最外面吧）
    # pretrained_dict = [(torch.cat([x, x], dim=0)) for x in pretrained_dict.items()] 
    load_state_dict_by_position(model, pretrained_dict)

    # x = x.repeat_interleave(Fuse,dim =1)
    x = x.repeat(1, Fuse, 1, 1).detach().clone().requires_grad_()
    target = target.repeat(Fuse)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([x], lr=1e-1)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start = time.time()

    for step in range(1):
        optimizer.zero_grad()

        if flag =='conv':
            output = model(x)  # forward
            # print(output.stride())           # 查看各维度的步长  
            # print(output.is_contiguous())    # 如果返回 True，说明是 contiguous；False 则不是。 :contentReference[oaicite:0]{index=0}
            output = output.view(-1,10)
            # print(output.shape)
            loss = criterion(output, target)  # compute loss
            # print(loss.item())
            dw = torch.torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
            # weight = list(model.parameters()) 
            # weight = [(1- p + g).sum() for p, g in zip(weight, dw)]
            weight = [d.sum() for d in dw]
            grad_loss = sum(weight)
            print("----GRANDLOSS-----", grad_loss.item())
            grad_loss.backward()  
        else: 
            output, dw = model(x,criterion, target)  # forward+1stbwd+weight op
            print("----GRANDLOSS-----", output.item())
            


        print("----GRAD-----")
        # print(x.grad.sum().item())
        optimizer.step()  # update x

    end = time.time()

    print("当前显存使用:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("峰值显存使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    print("时间占用：", end-start)