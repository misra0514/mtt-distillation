
from __future__ import annotations


import triton
import triton.language as tl
import torch
import math
import triton

BLOCK_M = 64

_INV_SQRT2 = 1.0 / (2.0 ** 0.5)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)  # 0.3989422804014327



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




@triton.jit
def _instance_norm_backward_pure_kernel(
    x_ptr,
    dy_ptr,
    gamma_ptr,
    dx_ptr,
    dgamma_ptr,
    dbeta_ptr,
    mean_ptr,
    std_ptr,
    HW: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One Triton program handles one (n, c) row of length HW.

    dgamma/dbeta must be zero-initialized by the host because each row
    contributes one scalar and rows are accumulated across batch with atomics.
    """
    pid_nc = tl.program_id(0)
    c = pid_nc % C

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW
    base = pid_nc * HW

    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c).to(tl.float32)

    inv_hw = 1.0 / tl.full([], HW, tl.float32)

    # More stable than E[x^2] - E[x]^2.
    mean = tl.sum(x, axis=0) * inv_hw
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) * inv_hw
    var = tl.maximum(var, 0.0)

    std = tl.sqrt(var + EPS)
    rstd = 1.0 / std
    x_hat = x_centered * rstd

    sum_dy = tl.sum(dy, axis=0)
    sum_dy_xhat = tl.sum(dy * x_hat, axis=0)

    # dx = gamma * rstd *
    #      (dy - mean(dy) - x_hat * mean(dy*x_hat))
    dx = gamma * rstd * (
        dy
        - sum_dy * inv_hw
        - x_hat * sum_dy_xhat * inv_hw
    )

    tl.store(dx_ptr + base + offsets, dx, mask=mask)

    # Parameter gradients reduce over both batch N and spatial HW.
    tl.atomic_add(dgamma_ptr + c, sum_dy_xhat)
    tl.atomic_add(dbeta_ptr + c, sum_dy)

    # Preserve the old wrapper's mean/std outputs: [N, C, 1].
    tl.store(mean_ptr + pid_nc, mean)
    tl.store(std_ptr + pid_nc, std)



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





# -------- 下面是一些旧的代码。目前没有引用。 -----------



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

    ggG_val = tl.full([], 0.0, tl.float32)
    if has_ggG:
        ggG_val = tl.load(ggG_ptr + c).to(tl.float32)

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
    # gX
    #   contribution 1: ggX -> gX
    #   contribution 2: ggG -> gX
    # =========================
    gX_val = tl.zeros([BLOCK_M], dtype=tl.float32)

    if has_ggX:
        A = (sum_ggX * sum_gO) * invM - dot_ggX_gO + (3.0 * inv_std2 * sum_gO_xmu * sum_ggX_xmu) * invM
        term0 = xcm * inv_std3 * A * invM
        term1 = sum_ggX_xmu * inv_std3 * (sum_gO * invM - gO) * invM
        term2 = sum_gO_xmu  * inv_std3 * (sum_ggX * invM - ggX) * invM
        gX_from_ggX = term0 + term1 + term2
        if has_gamma:
            gX_from_ggX = gX_from_ggX * gamma_val
        gX_val += gX_from_ggX

    if has_ggG:
        # Python plain equivalent:
        # t0 = gO * inv_std
        # t1 = -(inv_std * sum_gO) / M
        # t2 = -x_centered * inv_std^3 * sum_gO_xmu / M
        # gX_G = ggG * (t0 + t1 + t2)
        t0 = gO * inv_std
        t1 = -(inv_std * sum_gO) * invM
        t2 = -(xcm * inv_std3 * sum_gO_xmu) * invM
        gX_val += ggG_val * (t0 + t1 + t2)

    if has_ggX or has_ggG:
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
        ggO_val += ggG_val * xcm * inv_std

    if has_ggB:
        ggB_val = tl.load(ggB_ptr + c).to(tl.float32)
        ggO_val += ggB_val

    # ReLU gate on ggO: multiply by mask(out>0)
    outv = tl.load(out_ptr + offs, mask=m, other=0.0).to(tl.float32)
    ggO_val = tl.where(outv > 0.0, ggO_val, 0.0)

    tl.store(ggO_ptr + offs, ggO_val, mask=m)

