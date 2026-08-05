# 5.3 尝试写一个Forward 可以fuse 的code。 目前先对照原版和fused（autograd）
# 5.8 加入manuel bwd


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

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.nn.attention import sdpa_kernel, SDPBackend
from testing_utils import clear_tensorlists, set_random_seed,load_state_dict_by_position, repeat_params_for_fuse

from networks.networks_Fuse import LinearStacked_2,GroupedLinear,GroupedLayerNorm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.networks_stacked import LinearStacked_2 # NOTE 这里和flex fuse 不太一样。
from networks.networks_basicblock_fused3 import batchNorm2d_backward, batchnorm_double_backwards_fn, batchnorm_double_backwards_fn_new
from networks.networks_basicblock_fused3 import instanceNorm_backward ,instanceNorm_double_backwards_fn, instancenorm_relu_backward_triton,instanceNorm_double_backwards_triton
from networks.networks_stateless_basicblock import linear_bwd, conv_bwd, insNormNRelu_bwd, \
linear_double_bwd, conv_double_bwd, insNormNRelu_double_bwd, avgPool_bwd, crossEntropy_bwd, \
    avgPool_double_bwd,bmm_bwd, linerFused_bwd, linearFused_double_bwd,crossEntropy_double_bwd
    

def grouped_layernorm_backward(
    x,             # original input, [B, Fuse, N, D]
    weight,        # [Fuse * D]
    Fuse,
    grad_output,   # [B, Fuse, N, D]
    eps=1e-5,
    # mean,          # from native_layer_norm forward
    # rstd,          # from native_layer_norm forward
):
    B, Fs, N, D = x.shape
    assert Fs == Fuse
    normalized_shape = [D]
    z, mean, rstd = torch.ops.aten.native_layer_norm.default( x, [D], None, None, eps)
    weight_view = weight.view(1, Fuse, 1, D)
    dweight = (grad_output * z).sum(dim=(0, 2))      # [Fuse, D]
    dbias = grad_output.sum(dim=(0, 2))              # [Fuse, D]
    dweight = dweight.reshape(Fuse * D)
    dbias = dbias.reshape(Fuse * D)
    dz = grad_output * weight_view                   # [B, Fuse, N, D]
    #  LN weight/ bias was None,# only need dx
    dx, _, _ = torch.ops.aten.native_layer_norm_backward.default( dz, x, normalized_shape, mean, rstd, None, None, [True, False, False] )
    # dx, dweight, dbias = torch.ops.aten.native_layer_norm_backward.default( grad_output, x, normalized_shape, mean, rstd, weight, None, [True, True, True] )
    return dx, dweight, dbias


def grouped_linear_bwd(x, w, grad_output, Fuse =2 ):
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
    elif x.ndim == 4:
        B, Fs, N, I = x.shape
        dx = torch.einsum( "bfno,foi->bfni", grad_output, W, )
        dw = torch.einsum(   "bfno,bfni->foi", grad_output, x, )
        db = grad_output.sum(dim=(0, 2))  # [F, O]
        dw = dw.reshape(Fuse * out_features, in_features)
        db = db.reshape(Fuse * out_features)
    return dx, dw, db

def gelu_bwd(x, grad_output):
    """
    GELU exact backward.
    forward: gelu(x) = x * Phi(x)
    derivative: Phi(x) + x * phi(x)
    """
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(x * inv_sqrt2))
    pdf = torch.exp(-0.5 * x * x) * inv_sqrt2pi
    return grad_output * (cdf + x * pdf)


def sdpa_no_mask_no_dropout_bwd(q, k, v, grad_output):
    """
    q, k, v:     [B, Fuse, H, N, Dh]
    grad_output: [B, Fuse, H, N, Dh]

    return:
      dq, dk, dv: [B, Fuse, H, N, Dh]
    """
    Dh = q.shape[-1]
    scale = Dh ** -0.5
    # 这里需要重新算一下BHNN的attention score，否则没法求导。
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    prob = torch.softmax(scores, dim=-1)
    dv = torch.matmul(prob.transpose(-2, -1), grad_output)

    dprob = torch.matmul(grad_output, v.transpose(-2, -1))
    dscores = prob * (dprob - (dprob * prob).sum(dim=-1, keepdim=True))
    # scores = q @ k^T * scale
    dq = torch.matmul(dscores, k) * scale
    dk = torch.matmul(dscores.transpose(-2, -1), q) * scale
    return dq, dk, dv, dprob, dscores

####----------------------------------------------------------------

def layerNorm_double_bwd_fn(
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
    输入:
        x:      原始 input, shape = outer_shape + normalized_shape
        gamma:  LayerNorm weight, shape = normalized_shape, 可以是 None
        ggX:    grad of dX, shape 同 x, 可以是 None
        ggG:    grad of dGamma, shape 同 gamma, 可以是 None
        ggB:    grad of dBeta, shape 同 gamma, 可以是 None
        gO:     grad_output of forward, shape 同 x
        normalized_shape: LayerNorm 的 normalized_shape
    返回:
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

        # inv_std ** 3, 写成这个形式更稳一点
        inv_std3 = inv_std / (var + eps)

    if gamma is not None:
        gamma_flat = gamma.reshape(1, M)
    else:
        gamma_flat = None

    def first_back_no_weight(g):
        """
        rP(g) = inv_std * (g - mean(g) - x_hat * mean(g * x_hat))

        这是 LayerNorm backward 中去掉 gamma 之后的线性部分。
        """
        return inv_std * (
            g
            - g.mean(dim=1, keepdim=True)
            - x_hat * (g * x_hat).mean(dim=1, keepdim=True)
        )

    # ------------------------------------------------------------
    # gX: contribution from ggX
    # ------------------------------------------------------------
    gX_flat = None

    if ggX is not None:
        ggX_flat = ggX.reshape(K, M)

        with torch.no_grad():
            # 对 LayerNorm 来说，一阶 dX 里真正进入 norm backward 的是:
            #     b = gO * gamma
            # 如果 gamma is None，则等价于 gamma = 1
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

    # ------------------------------------------------------------
    # gX: contribution from ggG
    # dGamma = sum_outer(gO * x_hat)
    # 所以 ggG 对 x 的贡献是:
    #     rP(gO * ggG)
    # 注意这里不能写成 ggG * rP(gO)，因为 LayerNorm 的 ggG 是逐元素的。
    # ------------------------------------------------------------
    if ggG is not None:
        ggG_flat = ggG.reshape(1, M)

        with torch.no_grad():
            gX_G = first_back_no_weight(gO_flat * ggG_flat)

        gX_flat = gX_G if gX_flat is None else gX_flat + gX_G

    # ------------------------------------------------------------
    # gG: grad wrt gamma
    # 只有 ggX 分支会对 gamma 产生梯度
    #
    # dX = rP(gO * gamma)
    # d/dgamma <ggX, dX> = gO * rP(ggX)
    # 然后对 outer dims 求和。
    # ------------------------------------------------------------
    gG = None

    if gamma is not None and ggX is not None:
        ggX_flat = ggX.reshape(K, M)

        with torch.no_grad():
            rP_ggX = first_back_no_weight(ggX_flat)
            gG_flat = (gO_flat * rP_ggX).sum(dim=0)

        gG = gG_flat.reshape(normalized_shape)

    # ------------------------------------------------------------
    # ggO: grad wrt gO
    # ------------------------------------------------------------
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



def grouped_layernorm_double_bwd_fn(
    x,          # [B, Fuse, N, D]
    weight,     # [Fuse * D]
    ggX,        # grad of dx,      [B, Fuse, N, D] or None
    ggW,        # grad of dweight, [Fuse * D]       or None
    ggB,        # grad of dbias,   [Fuse * D]       or None
    gO,         # original grad_output, [B, Fuse, N, D]
    Fuse,
    eps=1e-5,
):
    B, Fs, N, D = x.shape
    assert Fs == Fuse
    assert weight is not None
    assert weight.numel() == Fuse * D
    assert gO.shape == x.shape
    M = D
    K = B * Fuse * N
    x_flat = x.reshape(K, D)
    gO_flat = gO.reshape(K, D)
    # 每个 [B, Fuse, N] row 对应一个 Fuse group 的 weight
    weight_view = weight.reshape(1, Fuse, 1, D)
    weight_flat = weight_view.expand(B, Fuse, N, D).reshape(K, D)
    mean = x_flat.mean(dim=1, keepdim=True)
    var = x_flat.var(dim=1, unbiased=False, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    x_centered = x_flat - mean
    x_hat = x_centered * inv_std
    # inv_std ** 3
    inv_std3 = inv_std / (var + eps)
    def first_back_no_weight(g):
        # no-affine LayerNorm backward 的线性部分： rP(g) = inv_std * (g - mean(g) - x_hat * mean(g * x_hat))  g: [K, D]
        return inv_std * (
            g
            - g.mean(dim=1, keepdim=True)
            - x_hat * (g * x_hat).mean(dim=1, keepdim=True)
        )
    gX_flat = None
    gWeight = None
    ggO_flat = None

    # 1. ggX 分支 
    if ggX is not None:
        assert ggX.shape == x.shape
        ggX_flat = ggX.reshape(K, D)
        # with torch.no_grad():
        # b 是传进 no-affine LN backward 的 grad_out
        b = gO_flat * weight_flat
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
        # wrt weight:
        #   <ggX, LN_backward(gO * weight)> 对 weight 求导
        # = sum_BN(gO * LN_backward(ggX))
        rP_ggX = first_back_no_weight(ggX_flat)
        gWeight_full = (
            gO_flat * rP_ggX
        ).reshape(B, Fuse, N, D).sum(dim=(0, 2))  # [Fuse, D]
        gWeight = gWeight_full.reshape(Fuse * D)
        # wrt gO
        ggO_X = rP_ggX * weight_flat
        ggO_flat = ggO_X
    # 2. ggW 分支 dweight = sum_BN(gO * x_hat)
    #   gX  += LN_backward(gO * ggW)
    #   ggO += ggW * x_hat
    if ggW is not None:
        assert ggW.numel() == Fuse * D
        ggW_view = ggW.reshape(1, Fuse, 1, D)
        ggW_flat = ggW_view.expand(B, Fuse, N, D).reshape(K, D)
        with torch.no_grad():
            gX_W = first_back_no_weight(gO_flat * ggW_flat)
            ggO_W = ggW_flat * x_hat
        gX_flat = gX_W if gX_flat is None else gX_flat + gX_W
        ggO_flat = ggO_W if ggO_flat is None else ggO_flat + ggO_W
    # 3. ggB 分支: ggO += ggB
    if ggB is not None:
        assert ggB.numel() == Fuse * D
        ggB_view = ggB.reshape(1, Fuse, 1, D)
        ggB_flat = ggB_view.expand(B, Fuse, N, D).reshape(K, D)
        with torch.no_grad():
            ggO_B = ggB_flat
        ggO_flat = ggO_B if ggO_flat is None else ggO_flat + ggO_B
    gX = gX_flat.reshape_as(x) if gX_flat is not None else None
    ggO = ggO_flat.reshape_as(gO) if ggO_flat is not None else None
    return gX, gWeight, ggO



def grouped_linear_double_bwd(
    x,
    w,
    grad_output,
    gg_grad_input=None,   # same shape as dx from grouped_linear_bwd
    gg_grad_w=None,       # same shape as dw: [Fuse * O, I]
    gg_grad_b=None,       # same shape as db: [Fuse * O]
    Fuse=2,
):
    in_features = w.shape[1]
    out_features = w.shape[0] // Fuse
    W = w.view(Fuse, out_features, in_features)  # [F, O, I]
    dgrad_output = torch.zeros_like(grad_output)
    dx = torch.zeros_like(x)
    dW = torch.zeros_like(W)

    if x.ndim == 3:
        B, Fs, I = x.shape
        assert Fs == Fuse
        assert I == in_features
        G = grad_output                    # [B, F, O]
        X = x                              # [B, F, I]
        assert G.shape == (B, Fuse, out_features)

        if gg_grad_input is not None:
            H = gg_grad_input
            assert H.shape == X.shape
            dgrad_output = dgrad_output + torch.einsum( "bfi,foi->bfo", H, W, )
            dW = dW + torch.einsum( "bfo,bfi->foi", G, H, )
        # if gg_grad_w is not None:
        V = gg_grad_w.view(Fuse, out_features, in_features)  # [F, O, I]
        dgrad_output = dgrad_output + torch.einsum( "foi,bfi->bfo", V, X, )
        dx = dx + torch.einsum( "bfo,foi->bfi", G, V, )
        if gg_grad_b is not None:
            ggb = gg_grad_b.view(Fuse, out_features)  # [F, O]
            dgrad_output = dgrad_output + ggb.view(1, Fuse, out_features)

    elif x.ndim == 4:
        #       grad_input = einsum("bfno,foi->bfni", G, W)
        #       dG += einsum("bfni,foi->bfno", H, W)
        #       dW += einsum("bfno,bfni->foi", G, H)
        B, Fs, N, I = x.shape
        G = grad_output                    # [B, F, N, O]
        X = x                              # [B, F, N, I]
        if gg_grad_input is not None:
            H = gg_grad_input
            assert H.shape == X.shape
            dgrad_output = dgrad_output + torch.einsum( "bfni,foi->bfno", H, W )
            dW = dW + torch.einsum( "bfno,bfni->foi", G, H, )
        if gg_grad_w is not None:
            V = gg_grad_w.view(Fuse, out_features, in_features)  # [F, O, I]

            dgrad_output = dgrad_output + torch.einsum( "foi,bfni->bfno", V, X )
            dx = dx + torch.einsum(  "bfno,foi->bfni",  G,  V, )
        if gg_grad_b is not None:
            ggb = gg_grad_b.view(Fuse, out_features)  # [F, O]
            dgrad_output = dgrad_output + ggb.view(1, Fuse, 1, out_features)
    else:
        raise ValueError(f"Unsupported x.ndim={x.ndim}, expected 3 or 4")
    dw = dW.reshape_as(w)
    return dgrad_output, dx, dw



def gelu_double_bwd(
    x,
    grad_output,
    gg_grad_input=None,
):
    if gg_grad_input is None:
        return None, None
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(x * inv_sqrt2))
    pdf = torch.exp(-0.5 * x * x) * inv_sqrt2pi
    # gelu'(x)
    gelu_grad = cdf + x * pdf
    # gelu''(x)
    gelu_double_grad = pdf * (2.0 - x * x)
    # grad wrt x
    dx = gg_grad_input * grad_output * gelu_double_grad
    # grad wrt grad_output
    dgrad_output = gg_grad_input * gelu_grad
    return dx, dgrad_output


def sdpa_no_mask_no_dropout_double_bwd(
    q,
    k,
    v,
    grad_output,
    ggQ=None,
    ggK=None,
    ggV=None,
    ggDprob=None,
    ggDscores=None,
):
    """
    Double backward for:

        scores = q @ k.T * scale
        prob = softmax(scores)
        dprob = grad_output @ v.T
        dv = prob.T @ grad_output
        dscores = prob * (dprob - sum(dprob * prob))
        dq = dscores @ k * scale
        dk = dscores.T @ q * scale

    q, k, v, grad_output:
        [B, Fuse, H, N, Dh]

    ggQ, ggK, ggV:
        upstream grads of dq, dk, dv.

    Optional:
        ggDprob:   upstream grad of returned dprob, if you expose dprob as output.
        ggDscores: upstream grad of returned dscores, if you expose dscores as output.

    Return:
        gQ, gK, gV, ggO
    where:
        gQ  = grad wrt q
        gK  = grad wrt k
        gV  = grad wrt v
        ggO = grad wrt grad_output
    """

    Dh = q.shape[-1]
    scale = Dh ** -0.5
    # Recompute forward pieces used by backward
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    prob = torch.softmax(scores, dim=-1)
    dprob = torch.matmul(grad_output, v.transpose(-2, -1))
    alpha = (dprob * prob).sum(dim=-1, keepdim=True)
    dscores = prob * (dprob - alpha)
    # Initialize cotangents
    gQ = torch.zeros_like(q)
    gK = torch.zeros_like(k)
    gV = torch.zeros_like(v)
    ggO = torch.zeros_like(grad_output)
    bar_dscores = torch.zeros_like(dscores)
    # dq = dscores @ k * scale
    if ggQ is not None:
        bar_dscores = bar_dscores + torch.matmul( ggQ, k.transpose(-2, -1) ) * scale
        gK = gK + torch.matmul( dscores.transpose(-2, -1), ggQ ) * scale
    # dk = dscores.T @ q * scale
    if ggK is not None:
        bar_dscores = bar_dscores + torch.matmul( q, ggK.transpose(-2, -1) ) * scale
        gQ = gQ + torch.matmul( dscores, ggK  ) * scale
    # If dscores itself is exposed as an output
    if ggDscores is not None:
        bar_dscores = bar_dscores + ggDscores
    # dv = prob.T @ grad_output
    bar_prob = torch.zeros_like(prob)
    if ggV is not None:
        bar_prob = bar_prob + torch.matmul( grad_output, ggV.transpose(-2, -1) )
        ggO = ggO + torch.matmul(prob, ggV)
    # dscores = prob * (dprob - sum(dprob * prob))
    #
    # Given R = bar_dscores:
    #
    # beta = sum(R * prob)
    # bar_dprob = prob * (R - beta)
    # bar_prob += R * (dprob - alpha) - beta * dprob
    beta = (bar_dscores * prob).sum(dim=-1, keepdim=True)
    bar_dprob = prob * (bar_dscores - beta)
    bar_prob = bar_prob + (   bar_dscores * (dprob - alpha) - beta * dprob)
    # If dprob itself is exposed as an output
    if ggDprob is not None:
        bar_dprob = bar_dprob + ggDprob
    # dprob = grad_output @ v.T
    ggO = ggO + torch.matmul(bar_dprob, v)
    gV = gV + torch.matmul(  bar_dprob.transpose(-2, -1), grad_output )
    # prob = softmax(scores)
    tau = (bar_prob * prob).sum(dim=-1, keepdim=True)
    bar_scores = prob * (bar_prob - tau)
    # scores = q @ k.T * scale
    gQ = gQ + torch.matmul(bar_scores, k) * scale
    gK = gK + torch.matmul(
        bar_scores.transpose(-2, -1), q
    ) * scale
    return gQ, gK, gV, ggO



class MultiHeadSelfAttention_Fused(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, Fuse=1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.Fuse = Fuse
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.qkv = GroupedLinear(embed_dim, 3 * embed_dim, Fuse)
        self.out_proj = GroupedLinear(embed_dim, embed_dim, Fuse)

    def forward(self, x):
        """
        x: [B, Fuse, N, C]
        """
        B, Fs, N, C = x.shape
        H = self.num_heads
        Dh = self.head_dim
        tape = {}
        qkv_in = x
        qkv = self.qkv(qkv_in)        # [B, Fuse, N, 3C]
        qkv_view = qkv.view(B, Fs, N, 3, H, Dh)        # [B, Fuse, N, 3, H, Dh]
        qkv_perm = qkv_view.permute(3, 0, 1, 4, 2, 5).contiguous()        # [3, B, Fuse, H, N, Dh]
        q, k, v = qkv_perm[0], qkv_perm[1], qkv_perm[2]        # each: [B, Fuse, H, N, Dh]
        with sdpa_kernel(SDPBackend.MATH):
            attn_out = F.scaled_dot_product_attention(q, k, v)        # [B, Fuse, H, N, Dh]
        attn_out_perm = attn_out.permute(0, 1, 3, 2, 4).contiguous()        # [B, Fuse, N, H, Dh]
        attn_out = attn_out_perm.view(B, Fs, N, C)    
        out = self.out_proj(attn_out)   # [B, Fuse, N, C]
        tape = {
            "x": qkv_in,
            "q": q,
            "k": k,
            "v": v,
            "attn_out": attn_out,
        }
        return out, tape
    
    def run_first_bwd(module, tape, grad_output, Fuse=None):
        """
        module: MultiHeadSelfAttention_Fused
        tape: forward 里面返回的 tape
        grad_output: dL/dout, shape [B, Fuse, N, C]
        return:
        dx: [B, Fuse, N, C]
        d_activates: 中间梯度，后面 double bwd 可能用
        d_weights: dict
        d_weights_all: list
        """
        if Fuse is None:
            Fuse = module.Fuse
        x = tape["x"]
        q = tape["q"]
        k = tape["k"]
        v = tape["v"]
        attn_out = tape["attn_out"]
        B, Fs, N, C = x.shape
        _, _, H, N_q, Dh = q.shape
        assert Fs == Fuse
        assert C == H * Dh
        assert grad_output.shape == (B, Fs, N, C)
        dout_merge, doutprojw, doutprojb = grouped_linear_bwd( attn_out, module.out_proj.weight,
            grad_output=grad_output, Fuse=Fuse,)
        # dout_merge: [B, Fuse, N, C]
        # 2. reverse head merge
        dattn_out_perm = dout_merge.view(B, Fs, N, H, Dh)
        # [B, Fuse, N, H, Dh]
        dattn_out = dattn_out_perm.permute(0, 1, 3, 2, 4).contiguous()
        del dout_merge, dattn_out_perm
        # [B, Fuse, H, N, Dh]
        # 3. SDPA bwd。 因为目前double bwd是重新算了attention score， 所以dprob和dscore 没有用上。
        #   attn_out = softmax(q @ k^T / sqrt(Dh)) @ v
        dq, dk, dv, dprob, dscores = sdpa_no_mask_no_dropout_bwd( q=q, k=k, v=v, grad_output=dattn_out )
        # each dq/dk/dv: [B, Fuse, H, N, Dh]
        # 4. reverse qkv split + permute + view
        dqkv_perm = torch.stack((dq, dk, dv), dim=0)
        # [3, B, Fuse, H, N, Dh]
        dqkv_view = dqkv_perm.permute(1, 2, 4, 0, 3, 5).contiguous()
        # [B, Fuse, N, 3, H, Dh]
        dqkv_linear = dqkv_view.reshape(B, Fs, N, 3 * C)
        # [B, Fuse, N, 3C]
        # 5. qkv linear bwd
        dx, dqkvw, dqkvb = grouped_linear_bwd( x, module.qkv.weight, grad_output=dqkv_linear, Fuse=Fuse )
        # dx: [B, Fuse, N, C]
        d_activates = {
            "grad_output": grad_output,
            "dattn_out": dattn_out,
            "dqkv_linear": dqkv_linear,
        }
        d_weights = {
            "dqkvw": dqkvw,
            "dqkvb": dqkvb,
            "doutprojw": doutprojw,
            "doutprojb": doutprojb,
        }
        d_weights_all = [dqkvw]
        if dqkvb is not None:
            d_weights_all.append(dqkvb)
        d_weights_all.append(doutprojw)
        if doutprojb is not None:
            d_weights_all.append(doutprojb)
        return dx, d_activates, d_weights, d_weights_all

    def run_double_bwd(
        self,
        tape,
        d_activates,
        dd_weights=None,
        ddgrad_in=None,
        Fuse=None,
    ):
        if Fuse is None:
            Fuse = self.Fuse
        if dd_weights is None:
            dd_weights = {}
        x = tape["x"]
        q = tape["q"]
        k = tape["k"]
        v = tape["v"]
        attn_out = tape["attn_out"]
        B, Fs, N, C = x.shape
        _, _, H, N_q, Dh = q.shape
        assert Fs == Fuse
        assert N_q == N
        assert C == H * Dh
        grad_output = d_activates.pop("grad_output")
        dqkv_linear = d_activates.pop("dqkv_linear")
        dattn_out = d_activates.pop("dattn_out")
        if ddgrad_in is None:
            ddgrad_in = torch.zeros_like(x)
        def get_dd(*names):
            for name in names:
                if name in dd_weights:
                    return dd_weights[name]
            return None
        ddqkvw = get_dd("ddqkvw", "dqkvw", "qkvw")
        ddqkvb = get_dd("ddqkvb", "dqkvb", "qkvb")
        ddoutprojw = get_dd("ddoutprojw", "doutprojw", "outprojw")
        ddoutprojb = get_dd("ddoutprojb", "doutprojb", "outprojb")
        # 1. qkv linear double-bwd
        dd_dqkv_linear, dx_d2, _ = grouped_linear_double_bwd( x=x, w=self.qkv.weight, grad_output=dqkv_linear,\
              gg_grad_input=ddgrad_in,  gg_grad_w=ddqkvw,  gg_grad_b=ddqkvb,  Fuse=Fuse, )
        # dqkv_linear 用完，可以显式删局部引用
        del dqkv_linear
        # 2. unpack dd_dqkv_linear -> ggQ, ggK, ggV
        dd_dqkv_view = dd_dqkv_linear.reshape(B, Fs, N, 3, H, Dh)
        dd_dqkv_perm = dd_dqkv_view.permute(3, 0, 1, 4, 2, 5).contiguous()
        ggQ = dd_dqkv_perm[0]
        ggK = dd_dqkv_perm[1]
        ggV = dd_dqkv_perm[2]
        del dd_dqkv_linear, dd_dqkv_view, dd_dqkv_perm
        # 3. SDPA double-bwd
        gQ, gK, gV, dd_dattn_out = sdpa_no_mask_no_dropout_double_bwd( q=q, k=k, v=v, grad_output=dattn_out,
            ggQ=ggQ, ggK=ggK, ggV=ggV, ggDprob=None, ggDscores=None, )

        del dattn_out, ggQ, ggK, ggV
        # pack gQ/gK/gV -> dqkv_linear_d2
        dqkv_d2_perm = torch.stack((gQ, gK, gV), dim=0)
        dqkv_d2_view = dqkv_d2_perm.permute(1, 2, 4, 0, 3, 5).contiguous()
        dqkv_linear_d2 = dqkv_d2_view.reshape(B, Fs, N, 3 * C)
        del gQ, gK, gV, dqkv_d2_perm, dqkv_d2_view
        # 4. reverse dattn_out reshape -> dd_dout_merge
        dd_dattn_out_perm = dd_dattn_out.permute(0, 1, 3, 2, 4).contiguous()
        dd_dout_merge = dd_dattn_out_perm.reshape(B, Fs, N, C)
        del dd_dattn_out, dd_dattn_out_perm
        # 5. out_proj linear double-bwd
        dd_grad_output, dout_merge_d2, _ = grouped_linear_double_bwd(
            x=attn_out,
            w=self.out_proj.weight,
            grad_output=grad_output,
            gg_grad_input=dd_dout_merge,
            gg_grad_w=ddoutprojw,
            gg_grad_b=ddoutprojb,
            Fuse=Fuse,
        )
        del grad_output, dd_dout_merge
        # 到这里，first-bwd 的激活都已经 pop 掉了。
        # 清空后只留下 bwd2_1 真正需要的三个。
        d_activates.clear()
        d_activates["dx_d2"] = dx_d2
        d_activates["dqkv_linear_d2"] = dqkv_linear_d2
        d_activates["dout_merge_d2"] = dout_merge_d2
        return dd_grad_output, d_activates

    def run_bwd2_1(
        self,
        tape,
        d_activates,
        grad_output,
        Fuse=None,
    ):
        """
        Re-run first backward of MultiHeadSelfAttention_Fused,
        while injecting activation-level second-order contributions
        generated by run_double_bwd.

        forward:
            x
            -> qkv linear
            -> reshape / split q,k,v
            -> SDPA
            -> merge heads
            -> out_proj
            -> out

        first-bwd:
            grad_output
            -> out_proj bwd gives dout_merge
            -> reshape gives dattn_out
            -> SDPA bwd gives dq, dk, dv
            -> pack gives dqkv_linear
            -> qkv linear bwd gives dx
        bwd2_1 injections:
            dout_merge += dout_merge_d2
            dqkv_linear += dqkv_linear_d2
            dx += dx_d2
        Inputs:
            tape:
                forward tape from attention forward.
            d_activates:
                first-bwd activation dict, already updated by run_double_bwd.
                Expected optional keys:
                    "dout_merge_d2"
                    "dqkv_linear_d2"
                    "dx_d2"
            grad_output:
                current corrected upstream grad wrt attention output,
                shape [B, Fuse, N, C].
        Return:
            dx:
                corrected grad wrt attention input x,
                shape [B, Fuse, N, C].
        """
        if Fuse is None:
            Fuse = self.Fuse
        x = tape["x"]
        q = tape["q"]
        k = tape["k"]
        v = tape["v"]
        attn_out = tape["attn_out"]
        B, Fs, N, C = x.shape
        _, _, H, N_q, Dh = q.shape
        assert Fs == Fuse
        assert C == H * Dh
        assert grad_output.shape == (B, Fs, N, C)
        # 1. out_proj bwd
        # forward:
        #   out = out_proj(attn_out)
        # first-bwd:
        #   dout_merge = dL/dout_merge
        dout_merge, _, _ = grouped_linear_bwd(
            attn_out,
            self.out_proj.weight,
            grad_output=grad_output,
            Fuse=Fuse, )
        # [B, Fuse, N, C]
        # Inject d2 contribution wrt forward attn_out.
        # This comes from double-bwd of out_proj bwd.
        dout_merge_d2 = d_activates.pop("dout_merge_d2")
        dqkv_linear_d2 = d_activates.pop("dqkv_linear_d2")
        dx_d2 = d_activates.pop("dx_d2")
        assert dout_merge_d2.shape == dout_merge.shape
        dout_merge = dout_merge + dout_merge_d2
        # 2. reverse merge-head reshape
        # forward:
        #   attn_out:      [B, Fuse, H, N, Dh]
        #   attn_out_perm: [B, Fuse, N, H, Dh]
        #   attn_out:     [B, Fuse, N, C]
        #
        # backward:
        #   dout_merge -> dattn_out
        dattn_out_perm = dout_merge.view(B, Fs, N, H, Dh)
        # [B, Fuse, N, H, Dh]
        dattn_out = dattn_out_perm.permute(0, 1, 3, 2, 4).contiguous()
        # [B, Fuse, H, N, Dh]
        # 3. SDPA bwd
        # forward:
        #   attn_out = softmax(q @ k^T / sqrt(Dh)) @ v
        # first-bwd:
        #   dq, dk, dv
        dq, dk, dv, _, _ = sdpa_no_mask_no_dropout_bwd( q=q, k=k, v=v, grad_output=dattn_out  )
        # 4. pack dq, dk, dv back to qkv-linear grad
        # forward:
        #   qkv:      [B, Fuse, N, 3C]
        #   qkv_view: [B, Fuse, N, 3, H, Dh]
        #   qkv_perm: [3, B, Fuse, H, N, Dh]
        # backward:
        #   dq,dk,dv -> dqkv_linear [B, Fuse, N, 3C]
        dqkv_perm = torch.stack((dq, dk, dv), dim=0)         # [3, B, Fuse, H, N, Dh]
        dqkv_view = dqkv_perm.permute(1, 2, 4, 0, 3, 5).contiguous()        # [B, Fuse, N, 3, H, Dh]
        dqkv_linear = dqkv_view.reshape(B, Fs, N, 3 * C)
        # [B, Fuse, N, 3C]
        del dq, dk, dv, dqkv_perm, dqkv_view
        # Inject d2 contribution wrt forward qkv output. This comes from double-bwd of SDPA bwd.
        dqkv_linear = dqkv_linear + dqkv_linear_d2
        # 5. qkv linear bwd
        # forward:
        #   qkv = qkv_linear(x)
        # first-bwd:
        #   dx = dL/dx
        dx, _, _ = grouped_linear_bwd(x,self.qkv.weight,grad_output=dqkv_linear,Fuse=Fuse )
        if dx_d2 is not None:
            assert dx_d2.shape == dx.shape
            dx = dx + dx_d2

        return dx



class TransformerBlock_Fused(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0, Fuse=1):
        super().__init__()
        self.Fuse = Fuse
        self.embed_dim = embed_dim

        self.norm1 = GroupedLayerNorm(embed_dim, Fuse)
        self.attn = MultiHeadSelfAttention_Fused(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            Fuse=Fuse,
        )

        self.norm2 = GroupedLayerNorm(embed_dim, Fuse)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = GroupedLinear(embed_dim, hidden_dim, Fuse)
        self.act = nn.GELU()
        self.fc2 = GroupedLinear(hidden_dim, embed_dim, Fuse)

    def forward(self, x):
        tape = {}
        # residual branch 1:  x_res1 = x + attn(norm1(x))
        x_in = x
        x_norm1 = self.norm1(x_in)
        x_attn, attn_tape = self.attn(x_norm1)
        x_res1 = x_in + x_attn
        # MLP branch:
        #   y = fc2(gelu(fc1(norm2(x_res1))))
        #   x_out = x_res1 + y
        x_norm2 = self.norm2(x_res1)
        x_fc1 = self.fc1(x_norm2)
        x_gelu = self.act(x_fc1)
        x_fc2 = self.fc2(x_gelu)
        x_out = x_res1 + x_fc2
        tape = {
            "x_in": x_in,
            "attn": attn_tape,
            "x_res1": x_res1,
            "x_norm2": x_norm2,
            "x_fc1": x_fc1,
            "x_gelu": x_gelu,
        }
        return x_out, tape
    
    def run_first_bwd(module, tape, grad_output, Fuse=None):
        """
        module: TransformerBlock_Fused
        tape: block forward 返回的 tape
        grad_output: dL/dx_out, shape [B, Fuse, N, C]
        return:
        dx_in: [B, Fuse, N, C]
        d_activates: dict
        d_weights: dict
        d_weights_all: list
        """
        if Fuse is None:
            Fuse = module.Fuse
        x_in = tape["x_in"]
        attn_tape = tape["attn"]
        x_res1 = tape["x_res1"]
        x_norm2 = tape["x_norm2"]
        x_fc1 = tape["x_fc1"]
        x_gelu = tape["x_gelu"]

        assert grad_output.shape == x_in.shape
        # backward of:
        #   x_out = x_res1 + x_fc2
        dx_res1 = grad_output
        dx_fc2 = grad_output
        # backward of:
        #   x_fc2 = fc2(x_gelu)
        dx_gelu, dfc2w, dfc2b = grouped_linear_bwd( x_gelu, module.fc2.weight, grad_output=dx_fc2, Fuse=Fuse )
        # del dx_res1_from_norm2, dx_res1_total, dx_attn, dx_in_from_norm1
        # backward of:
        #   x_gelu = gelu(x_fc1)
        dx_fc1 = gelu_bwd(x_fc1, dx_gelu)
        # backward of:
        #   x_fc1 = fc1(x_norm2)
        dx_norm2, dfc1w, dfc1b = grouped_linear_bwd(
            x_norm2,
            module.fc1.weight,
            grad_output=dx_fc1,
            Fuse=Fuse,
        )
        # backward of:
        #   x_norm2 = norm2(x_res1)
        dx_res1_from_norm2, dnorm2w, dnorm2b = grouped_layernorm_backward(
            x_res1,
            module.norm2.weight,
            Fuse=Fuse,
            grad_output=dx_norm2,
        )
        # x_res1 has two outgoing paths:
        #   1. x_out = x_res1 + x_fc2
        #   2. x_norm2 = norm2(x_res1)
        dx_res1_total = dx_res1 + dx_res1_from_norm2
        # backward of:
        #   x_res1 = x_in + x_attn
        dx_in_from_skip = dx_res1_total
        dx_attn = dx_res1_total
        # backward of:
        #   x_attn = attn(x_norm1)
        dx_norm1, d_attn_activates, d_attn_weights, d_attn_weights_all = module.attn.run_first_bwd(
            attn_tape,
            grad_output=dx_attn,
            Fuse=Fuse,
        )
        # backward of:
        #   x_norm1 = norm1(x_in)
        dx_in_from_norm1, dnorm1w, dnorm1b = grouped_layernorm_backward(
            x_in,
            module.norm1.weight,
            Fuse=Fuse,
            grad_output=dx_norm1,
        )
        # x_in has two outgoing paths:
        #   1. residual skip into x_res1
        #   2. norm1 -> attn branch
        dx_in = dx_in_from_skip + dx_in_from_norm1
        # collect
        d_activates = {
            "dx_fc2": dx_fc2,
            "dx_gelu": dx_gelu,
            "dx_fc1": dx_fc1,
            "dx_norm2": dx_norm2,
            "dx_norm1": dx_norm1,
            "attn": d_attn_activates,
        }
        d_weights = {
            "dnorm1w": dnorm1w,
            "dnorm1b": dnorm1b,
            "attn": d_attn_weights,
            "dnorm2w": dnorm2w,
            "dnorm2b": dnorm2b,
            "dfc1w": dfc1w,
            "dfc1b": dfc1b,
            "dfc2w": dfc2w,
            "dfc2b": dfc2b,
        }
        # 顺序要对应 module.parameters():
        # norm1.weight, norm1.bias,
        # attn.qkv.weight, attn.qkv.bias, attn.out_proj.weight, attn.out_proj.bias,
        # norm2.weight, norm2.bias,
        # fc1.weight, fc1.bias,
        # fc2.weight, fc2.bias
        d_weights_all = []
        d_weights_all.append(dnorm1w)
        d_weights_all.append(dnorm1b)
        for g in d_attn_weights_all:
            if g is not None:
                d_weights_all.append(g)
        d_weights_all.append(dnorm2w)
        d_weights_all.append(dnorm2b)
        d_weights_all.append(dfc1w)
        d_weights_all.append(dfc1b)
        d_weights_all.append(dfc2w)
        d_weights_all.append(dfc2b)
        return dx_in, d_activates, d_weights, d_weights_all

    def run_double_bwd(
        self,
        tape,
        d_activates,
        dd_weights,
        ddgrad_in=None,
        Fuse=None,
    ):
        """
        Double backward for TransformerBlock_Fused.

        first backward 是:
            dx_res1 = grad_output
            dx_fc2  = grad_output
            dx_gelu = linear_bwd(fc2)(x_gelu, dx_fc2)
            dx_fc1  = gelu_bwd(x_fc1, dx_gelu)
            dx_norm2 = linear_bwd(fc1)(x_norm2, dx_fc1)
            dx_res1_from_norm2 = layernorm_bwd(norm2)(x_res1, dx_norm2)
            dx_res1_total = dx_res1 + dx_res1_from_norm2
            dx_in_from_skip = dx_res1_total
            dx_attn = dx_res1_total

            dx_norm1 = attn_bwd(x_norm1, dx_attn)
            dx_in_from_norm1 = layernorm_bwd(norm1)(x_in, dx_norm1)

            dx_in = dx_in_from_skip + dx_in_from_norm1

        输入:
            ddgrad_in:
                cotangent wrt dx_in, shape [B, Fuse, N, C].
                如果这个 block 是 double-bwd 链的起点，可以传 zeros_like(x_in)。

            dd_weights:
                cotangent wrt first-bwd 产生的参数梯度。
                如果你是在算:
                    grad_loss = sum(d.sum() for d in d_weights_all)
                那这些一般就是 ones_like(parameter_grad)。

        返回:
            dd_grad_output:
                cotangent wrt original first-bwd grad_output,
                shape [B, Fuse, N, C].

            d_activates:
                会被补充若干 *_d2，用于之后 bwd2_1。
        """
        if Fuse is None:
            Fuse = self.Fuse

        x_in = tape["x_in"]
        attn_tape = tape["attn"]
        x_res1 = tape["x_res1"]
        x_norm2 = tape["x_norm2"]
        x_fc1 = tape["x_fc1"]
        x_gelu = tape["x_gelu"]
        dx_norm1 = d_activates.pop("dx_norm1")
        dx_norm2 = d_activates.pop("dx_norm2")
        dx_fc1 = d_activates.pop("dx_fc1")
        dx_gelu = d_activates.pop("dx_gelu")
        dx_fc2 = d_activates.pop("dx_fc2")
        d_attn_activates = d_activates.pop("attn")
        d_activates.clear()
        if ddgrad_in is None:
            ddgrad_in = torch.zeros_like(x_in)
        # First-bwd final:
        #   dx_in = dx_in_from_skip + dx_in_from_norm1
        dd_dx_in_from_skip = ddgrad_in
        dd_dx_in_from_norm1 = ddgrad_in
        # ==================================================
        # Double of:
        #   dx_in_from_norm1, dnorm1w, dnorm1b
        #       = LayerNorm_bwd(x_in, norm1.weight, grad_output=dx_norm1)
        # ==================================================
        dx_in_d2_from_norm1, dnorm1w_d2, dd_dx_norm1 = grouped_layernorm_double_bwd_fn(
            x=x_in,
            weight=self.norm1.weight,
            ggX=dd_dx_in_from_norm1,
            ggW=dd_weights.get("ddnorm1w", None),
            ggB=dd_weights.get("ddnorm1b", None),
            gO=dx_norm1,
            Fuse=Fuse,
        )
        # 这个是 wrt forward x_in 的二阶贡献，后面 bwd2_1 要加到 block input gradient 上。
        d_activates["dx_in_d2"] = dx_in_d2_from_norm1
        # ==================================================
        # Double of:
        #   dx_norm1, dattn_weights = attn_bwd(x_norm1, grad_output=dx_attn)
        #
        # 注意:
        #   self.attn.run_double_bwd 应该返回的是 wrt dx_attn 的 cotangent。
        #   也就是 dd_dx_attn。
        # ==================================================
        dd_attn_weights = dd_weights.get("attn", None)
        dd_dx_attn, d_attn_activates = self.attn.run_double_bwd( tape=attn_tape,
            d_activates=d_attn_activates, dd_weights=dd_attn_weights, ddgrad_in=dd_dx_norm1, Fuse=Fuse )

        d_activates["attn"] = d_attn_activates
        # ==================================================
        # first-bwd:
        #   dx_in_from_skip = dx_res1_total
        #   dx_attn         = dx_res1_total
        # 所以 cotangent wrt dx_res1_total 要累加两条路径:
        #   1. from dx_in_from_skip
        #   2. from dx_attn
        # ==================================================
        dd_dx_res1_total = dd_dx_in_from_skip + dd_dx_attn
        # ==================================================
        # first-bwd:
        #   dx_res1_total = dx_res1 + dx_res1_from_norm2
        # ==================================================
        dd_dx_res1 = dd_dx_res1_total
        dd_dx_res1_from_norm2 = dd_dx_res1_total
        # ==================================================
        # Double of:
        #   dx_res1_from_norm2, dnorm2w, dnorm2b
        #       = LayerNorm_bwd(x_res1, norm2.weight, grad_output=dx_norm2)
        # ==================================================
        dx_res1_d2_from_norm2, dnorm2w_d2, dd_dx_norm2 = grouped_layernorm_double_bwd_fn(
            x=x_res1,
            weight=self.norm2.weight,
            ggX=dd_dx_res1_from_norm2,
            ggW=dd_weights.get("ddnorm2w", None),
            ggB=dd_weights.get("ddnorm2b", None),
            gO=dx_norm2,
            Fuse=Fuse,
        )
        # 这是 wrt forward x_res1 的二阶贡献。
        d_activates["dx_res1_d2"] = dx_res1_d2_from_norm2
        # ==================================================
        # Double of:
        #   dx_norm2, dfc1w, dfc1b
        #       = grouped_linear_bwd(x_norm2, fc1.weight, grad_output=dx_fc1)
        # ==================================================
        dd_dx_fc1, dx_norm2_d2, dfc1w_d2 = grouped_linear_double_bwd(
            x=x_norm2,
            w=self.fc1.weight,
            grad_output=dx_fc1,
            gg_grad_input=dd_dx_norm2,
            gg_grad_w=dd_weights.get("ddfc1w", None),
            gg_grad_b=dd_weights.get("ddfc1b", None),
            Fuse=Fuse,
        )
        # wrt forward x_norm2 的二阶贡献。
        d_activates["dx_norm2_d2"] = dx_norm2_d2
        # ==================================================
        # Double of:
        #   dx_fc1 = gelu_bwd(x_fc1, grad_output=dx_gelu)
        # ==================================================
        dx_fc1_d2, dd_dx_gelu = gelu_double_bwd( x=x_fc1, grad_output=dx_gelu, gg_grad_input=dd_dx_fc1 )
        # wrt forward x_fc1 的二阶贡献。
        d_activates["dx_fc1_d2"] = dx_fc1_d2
        # ==================================================
        # Double of:
        #   dx_gelu, dfc2w, dfc2b
        #       = grouped_linear_bwd(x_gelu, fc2.weight, grad_output=dx_fc2)
        # ==================================================
        dd_dx_fc2, dx_gelu_d2, dfc2w_d2 = grouped_linear_double_bwd(
            x=x_gelu,
            w=self.fc2.weight,
            grad_output=dx_fc2,
            gg_grad_input=dd_dx_gelu,
            gg_grad_w=dd_weights.get("ddfc2w", None),
            gg_grad_b=dd_weights.get("ddfc2b", None),
            Fuse=Fuse,
        )
        # wrt forward x_gelu 的二阶贡献。
        d_activates["dx_gelu_d2"] = dx_gelu_d2
        # ==================================================
        # first-bwd:
        #   dx_res1 = grad_output
        #   dx_fc2  = grad_output
        #
        # 所以 dd wrt original grad_output 是两条路径相加。
        # ==================================================
        dd_grad_output = dd_dx_res1 + dd_dx_fc2
        # 可选：如果你之后想 debug，可以把这些也存下来。
        # d_activates["dd_dx_norm1"] = dd_dx_norm1
        # d_activates["dd_dx_attn"] = dd_dx_attn
        # d_activates["dd_dx_res1_total"] = dd_dx_res1_total
        # d_activates["dd_dx_norm2"] = dd_dx_norm2
        # d_activates["dd_dx_fc1"] = dd_dx_fc1
        # d_activates["dd_dx_gelu"] = dd_dx_gelu
        # d_activates["dd_dx_fc2"] = dd_dx_fc2
        return dd_grad_output, d_activates
    
    def run_bwd2_1(
        self,
        tape,
        d_activates,
        grad_output,
        Fuse=None,
    ):
        """
        Re-run first backward of TransformerBlock_Fused,
        while injecting activation-level second-order contributions
        generated by run_double_bwd.

        forward:
            x_in
            -> norm1 -> attn
            -> residual: x_res1 = x_in + x_attn
            -> norm2 -> fc1 -> gelu -> fc2
            -> residual: x_out = x_res1 + x_fc2

        first-bwd:
            grad_output
            -> fc2 bwd
            -> gelu bwd
            -> fc1 bwd
            -> norm2 bwd
            -> residual split to skip + attn
            -> attn bwd
            -> norm1 bwd
            -> dx_in

        bwd2_1 injection positions:
            dx_gelu      += dx_gelu_d2       after fc2 bwd
            dx_fc1       += dx_fc1_d2        after gelu bwd
            dx_norm2     += dx_norm2_d2      after fc1 bwd
            dx_res1_total += dx_res1_d2      after norm2 bwd + final skip
            dx_in        += dx_in_d2         after norm1 bwd + residual skip

        Inputs:
            tape:
                forward tape from TransformerBlock_Fused.forward

            d_activates:
                first-bwd activation dict, already updated by run_double_bwd

            grad_output:
                corrected upstream gradient wrt block output x_out,
                shape [B, Fuse, N, C]

        Return:
            dx_in:
                corrected gradient wrt block input x_in,
                shape [B, Fuse, N, C]
        """
        if Fuse is None:
            Fuse = self.Fuse

        # --------------------------------------------------
        # unpack forward activations
        # --------------------------------------------------
        x_in = tape["x_in"]
        attn_tape = tape["attn"]
        x_res1 = tape["x_res1"]
        x_norm2 = tape["x_norm2"]
        x_fc1 = tape["x_fc1"]
        x_gelu = tape["x_gelu"]

        assert grad_output.shape == x_in.shape

        # ==================================================
        # 1. backward of:
        #       x_out = x_res1 + x_fc2
        # ==================================================
        dx_res1 = grad_output
        dx_fc2 = grad_output
        # ==================================================
        # 2. backward of:
        #       x_fc2 = fc2(x_gelu)
        # ==================================================
        dx_gelu, _, _ = grouped_linear_bwd(
            x_gelu,
            self.fc2.weight,
            grad_output=dx_fc2,
            Fuse=Fuse,
        )
        # Inject d2 contribution wrt forward x_gelu.
        dx_gelu_d2 = d_activates.pop("dx_gelu_d2")
        assert dx_gelu_d2.shape == dx_gelu.shape
        dx_gelu = dx_gelu + dx_gelu_d2
        # ==================================================
        # 3. backward of:
        #       x_gelu = GELU(x_fc1)
        # ==================================================
        dx_fc1 = gelu_bwd(
            x_fc1,
            dx_gelu,
        )
        # Inject d2 contribution wrt forward x_fc1.
        dx_fc1_d2 = d_activates.pop("dx_fc1_d2")
        if dx_fc1_d2 is not None:
            assert dx_fc1_d2.shape == dx_fc1.shape
            dx_fc1 = dx_fc1 + dx_fc1_d2

        # ==================================================
        # 4. backward of:
        #       x_fc1 = fc1(x_norm2)
        # ==================================================
        dx_norm2, _, _ = grouped_linear_bwd( x_norm2, self.fc1.weight, grad_output=dx_fc1, Fuse=Fuse)
        # Inject d2 contribution wrt forward x_norm2.
        dx_norm2_d2 = d_activates.pop("dx_norm2_d2")

        if dx_norm2_d2 is not None:
            assert dx_norm2_d2.shape == dx_norm2.shape
            dx_norm2 = dx_norm2 + dx_norm2_d2
        # ==================================================
        # 5. backward of:
        #       x_norm2 = norm2(x_res1)
        # ==================================================
        dx_res1_from_norm2, _, _ = grouped_layernorm_backward(
            x_res1,
            self.norm2.weight,
            Fuse=Fuse,
            grad_output=dx_norm2,
        )

        # x_res1 has two outgoing paths in forward:
        #   1. x_out = x_res1 + x_fc2
        #   2. norm2(x_res1)
        dx_res1_total = dx_res1 + dx_res1_from_norm2

        # Inject d2 contribution wrt forward x_res1.
        #
        # Important:
        #   This must be added BEFORE the residual split:
        #       x_res1 = x_in + x_attn
        #
        #   so it flows into both:
        #       dx_in_from_skip
        #       dx_attn
        dx_res1_d2 = d_activates.pop("dx_res1_d2")
        assert dx_res1_d2.shape == dx_res1_total.shape
        dx_res1_total = dx_res1_total + dx_res1_d2

        # ==================================================
        # 6. backward of:
        #       x_res1 = x_in + x_attn
        # ==================================================
        dx_in_from_skip = dx_res1_total
        dx_attn = dx_res1_total

        # ==================================================
        # 7. backward of:
        #       x_attn = attn(x_norm1)
        #
        # This attention bwd2_1 should already inject:
        #       dout_merge_d2
        #       dqkv_linear_d2
        #       dx_d2
        # ==================================================
        d_attn_activates = d_activates["attn"]
        dx_norm1 = self.attn.run_bwd2_1(
            tape=attn_tape,
            d_activates=d_attn_activates,
            grad_output=dx_attn,
            Fuse=Fuse,
        )
        # 8. backward of:
        #       x_norm1 = norm1(x_in)
        dx_in_from_norm1, _, _ = grouped_layernorm_backward(
            x_in,
            self.norm1.weight,
            Fuse=Fuse,
            grad_output=dx_norm1,
        )
        # x_in has two outgoing paths in forward:
        #   1. residual skip into x_res1
        #   2. norm1 -> attention branch
        dx_in = dx_in_from_skip + dx_in_from_norm1
        # Inject d2 contribution wrt forward x_in from norm1 double-bwd.
        # This is added at the very end because it is directly wrt the
        # block input x_in.
        dx_in_d2 = d_activates.pop("dx_in_d2")
        d_activates.clear()
        assert dx_in_d2.shape == dx_in.shape
        dx_in = dx_in + dx_in_d2

        return dx_in

    def init_dd_weights(self):
        dd_weights = {
            "ddnorm1w": torch.ones_like(self.norm1.weight),
            "ddnorm1b": torch.ones_like(self.norm1.bias),

            "attn": {
                "ddqkvw": torch.ones_like(self.attn.qkv.weight),
                "ddqkvb": torch.ones_like(self.attn.qkv.bias),
                "ddoutprojw": torch.ones_like(self.attn.out_proj.weight),
                "ddoutprojb": torch.ones_like(self.attn.out_proj.bias),
            },

            "ddnorm2w": torch.ones_like(self.norm2.weight),
            "ddnorm2b": torch.ones_like(self.norm2.bias),

            "ddfc1w": torch.ones_like(self.fc1.weight),
            "ddfc1b": torch.ones_like(self.fc1.bias),

            "ddfc2w": torch.ones_like(self.fc2.weight),
            "ddfc2b": torch.ones_like(self.fc2.bias),
        }
        return dd_weights



class ViT_Fused(nn.Module):
    def __init__( self, image_size=32, patch_size=4, in_channels=3, num_classes=10, embed_dim=128, num_heads=4, mlp_ratio=4.0, dropout=0.0, Fuse=1,
    ):
        super().__init__()
        self.Fuse = Fuse
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d( in_channels=in_channels * Fuse, out_channels=embed_dim * Fuse,  kernel_size=patch_size, stride=patch_size, groups=Fuse, )
        self.cls_token = nn.Parameter(torch.zeros(1, Fuse, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, Fuse, self.num_patches + 1, embed_dim))
        self.block = TransformerBlock_Fused( embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout, Fuse= Fuse )
        # 注意：如果 forward 里面的 shape 是 [B, Fuse, N, D]，
        # 那么 LayerNorm 应该是 embed_dim，而不是 embed_dim * Fuse。
        self.norm = GroupedLayerNorm(embed_dim,Fuse)
        # self.head = LinearStacked_2(embed_dim, num_classes, Fuse)
        self.head = GroupedLinear(embed_dim, num_classes, Fuse)
        self._init_weights()

    def forward(self, x):
        Fuse = self.Fuse
        B = x.shape[0]
        D = self.embed_dim
        tape = {  "patch": {},  "block": None,  "head": {},}
        x_in = x
        # [B, C*Fuse, H, W]
        x_patch = self.patch_embed(x_in)
        # [B, D*Fuse, H/P, W/P]
        x_flat = x_patch.flatten(2).transpose(1, 2)
        # [B, N, D*Fuse]
        x_reshape = x_flat.reshape(B, self.num_patches, Fuse, D)
        # [B, N, Fuse, D]
        x_tokens = x_reshape.permute(0, 2, 1, 3).contiguous()
        # [B, Fuse, N, D]
        tape["patch"] = {
            "x_in": x_in,
            "x_patch": x_patch,
        }
        cls_token = self.cls_token.expand(B, -1, -1, -1)
        # [B, Fuse, 1, D]
        x_cat = torch.cat((cls_token, x_tokens), dim=2)
        # [B, Fuse, N+1, D]
        x_pos = x_cat + self.pos_embed
        # [B, Fuse, N+1, D]
        x_block, block_tape = self.block(x_pos)
        tape["block"] = block_tape
        # [B, Fuse, N+1, D]
        x_norm_in = x_block
        x_norm = self.norm(x_norm_in)
        # [B, Fuse, N+1, D]
        x_cls = x_norm[:, :, 0]
        # [B, Fuse, D]
        x_head = self.head(x_cls)
        # [B, Fuse, num_classes]
        # 重要：保留你的原始约定。
        # 无 permute，直接 reshape。所以 target 要用 repeat_interleave(Fuse)。
        x_out = x_head.reshape(B * Fuse, self.num_classes)
        # [B*Fuse, num_classes]
        # 顺序是 b0f0, b0f1, b1f0, b1f1, ...
        tape["head"] = {
            "x_norm_in": x_norm_in,
            "x_cls": x_cls,
            "x_out": x_out,
        }
        return x_out, tape
    
    def run_first_bwd(self, tape, target, Fuse=None):
        if Fuse is None:
            Fuse = self.Fuse
        patch = tape["patch"]
        head = tape["head"]
        block_tape = tape["block"]
        x_cls = head["x_cls"]
        x_out = head["x_out"]
        x_patch = patch["x_patch"]
        B, Fs, D = x_cls.shape
        C = x_out.shape[-1]
        N = x_patch.shape[-2] * x_patch.shape[-1]
        x_in = patch["x_in"]
        x_patch = patch["x_patch"]
        x_norm_in = head["x_norm_in"]
        x_cls = head["x_cls"]
        x_out = head["x_out"]

        # 1. CE bwd
        dx_out = crossEntropy_bwd(x_out, target, Fuse = Fuse)
        # forward: x_out = x_head.reshape(B*Fuse, C)
        dx_head = dx_out.reshape(B, Fuse, C)
        dx_cls, dheadw, dheadb = grouped_linear_bwd(
            x_cls,
            self.head.weight,
            grad_output=dx_head,
            Fuse=Fuse,
        )
        # dx_cls: [B, Fuse, D]
        # forward:  x_cls = x_norm[:, :, 0]
        # backward:  only cls position receives dx_cls
        # dx_norm = torch.zeros_like(x_norm)
        dx_norm = torch.zeros_like(x_norm_in)
        dx_norm[:, :, 0, :] = dx_cls # [B, Fuse, N+1, D]
        dx_block, dnormw, dnormb = grouped_layernorm_backward( x_norm_in, self.norm.weight, grad_output=dx_norm, Fuse=Fuse )
        # dx_block: [B, Fuse, N+1, D]
        dx_pos, d_block_activates, d_block_weights, d_block_weights_all = self.block.run_first_bwd( block_tape, grad_output=dx_block, Fuse=Fuse )
        # dx_pos: [B, Fuse, N+1, D]
        # forward: x_pos = x_cat + self.pos_embed
        # self.pos_embed: [1, Fuse, N+1, D]
        dpos_embed = dx_pos.sum(dim=0, keepdim=True)
        dx_cat = dx_pos
        # dx_cat: [B, Fuse, N+1, D]
        # forward: x_cat = cat(cls_token_expand, x_tokens, dim=2)
        dcls_expand = dx_cat[:, :, 0:1, :]
        dx_tokens = dx_cat[:, :, 1:, :]
        # dcls_expand: [B, Fuse, 1, D]
        # dx_tokens:   [B, Fuse, N, D]
        # cls_token 是 expand 出来的，所以 batch 维度求和
        dcls_token = dcls_expand.sum(dim=0, keepdim=True)
        # [1, Fuse, 1, D]
        dx_reshape = dx_tokens.permute(0, 2, 1, 3).contiguous()
        # [B, N, Fuse, D]
        dx_flat = dx_reshape.reshape(B, N, Fuse * D)
        # [B, N, Fuse*D]
        dx_patch = dx_flat.transpose(1, 2).contiguous().view_as(x_patch)
        # [B, Fuse*D, Hp, Wp]
        dx_in, dpatchw, _ = conv_bwd( x_in, self.patch_embed.weight, grad_output=dx_patch,
            stride=self.patch_embed.stride[0],  padding=self.patch_embed.padding[0], groups=Fuse )
        if self.patch_embed.bias is not None:
            dpatchb = dx_patch.sum(dim=(0, 2, 3))
        else:
            dpatchb = None
        d_activates = {
            "dx_head": dx_head,
            "dx_norm": dx_norm,
            "dx_patch": dx_patch,
            "block": d_block_activates,
        }
        d_weights = {
            "dpatchw": dpatchw,
            "dpatchb": dpatchb,
            "dcls_token": dcls_token,
            "dpos_embed": dpos_embed,
            "block": d_block_weights,
            "dnormw": dnormw,
            "dnormb": dnormb,
            "dheadw": dheadw,
            "dheadb": dheadb,
        }
        # 这个 list 用来做你后面的  weight_sum = [d.sum() for d in d_weights_list_all]
        d_weights_list_all = []
        # 注意顺序最好和 model.parameters() 尽量一致： patch_embed, cls_token, pos_embed, block, norm, head
        d_weights_list_all.append(dpatchw)
        if dpatchb is not None:
            d_weights_list_all.append(dpatchb)
        d_weights_list_all.append(dcls_token)
        d_weights_list_all.append(dpos_embed)
        for g in d_block_weights_all:
            if g is not None:
                d_weights_list_all.append(g)
        d_weights_list_all.append(dnormw)
        d_weights_list_all.append(dnormb)
        d_weights_list_all.append(dheadw)
        if dheadb is not None:
            d_weights_list_all.append(dheadb)
        return dx_in, d_activates, d_weights, d_weights_list_all

    def init_dd_weights(self):
        """
        用于:
            grad_loss = sum(d.sum() for d in d_weights_list_all)

        所以每个一阶参数梯度的 cotangent 都是 ones_like。
        """
        return {
            "ddpatchw": torch.ones_like(self.patch_embed.weight),
            "ddpatchb": torch.ones_like(self.patch_embed.bias) if self.patch_embed.bias is not None else None,
            "ddcls_token": torch.ones_like(self.cls_token),
            "ddpos_embed": torch.ones_like(self.pos_embed),
            # 这里要求 TransformerBlock_Fused 里面也有 init_dd_weights()
            "block": self.block.init_dd_weights(),
            "ddnormw": torch.ones_like(self.norm.weight),
            "ddnormb": torch.ones_like(self.norm.bias),
            "ddheadw": torch.ones_like(self.head.weight),
            "ddheadb": torch.ones_like(self.head.bias) if self.head.bias is not None else None,
        }

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, GroupedLayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def run_double_bwd(
        self,
        tape,
        d_activates,
        dd_weights,
        ddgrad_in=None,
        Fuse=None,
    ):
        """
        ViT_Fused double backward stage.
`       沿着 first-bwd graph 反向传播 cotangent.并把 activation-level d2 contribution 写入 d_activates。
        输入:
            ddgrad_in:   cotangent wrt first-bwd output dx_in。
                如果 grand_loss 只来自参数梯度 sum，通常传 zeros_like(dx_in)。
            dd_weights:  cotangent wrt first-bwd 产生的参数梯度。
                对于 grad_loss = sum(d.sum() for d in d_weights_list_all)，
                通常就是 init_dd_weights() 里面的 ones_like。
        返回:
            dd_dx_out:  cotangent wrt CE first-bwd output dx_out。 
                 后面 run_bwd2_1 里会传给 crossEntropy_double_bwd。
            d_activates 会被补充:  "x_in_d2"     "x_norm_in_d2"     "x_cls_d2"     block 内部的 *_d2
        """
        if Fuse is None:
            Fuse = self.Fuse
        patch = tape["patch"]
        head = tape["head"]
        block_tape = tape["block"]
        D = self.embed_dim
        C = self.num_classes
        N = self.num_patches
        x_cls = head["x_cls"]
        x_out = head["x_out"]
        x_patch = patch["x_patch"]
        B= x_cls.shape[0]
        C = x_out.shape[-1]
        N = x_patch.shape[-2] * x_patch.shape[-1]
        x_in = patch["x_in"]
        x_patch = patch["x_patch"]
        x_pos = block_tape["x_in"]
        x_norm_in = head["x_norm_in"]
        x_cls = head["x_cls"]
        x_out = head["x_out"]
        dx_head = d_activates.pop("dx_head")
        dx_norm = d_activates.pop("dx_norm")
        dx_patch = d_activates.pop("dx_patch")
        d_block_activates = d_activates.pop("block")

        if ddgrad_in is None:
            ddgrad_in = torch.zeros_like(x_in)
        assert ddgrad_in.shape == x_in.shape
        assert dx_patch.shape == x_patch.shape
        assert dx_head.shape == (B, Fuse, C)
        def get_dd(*keys):
            for k in keys:
                if k in dd_weights:
                    return dd_weights[k]
            return None
        # ==================================================
        # 1. Double of patch_embed conv bwd
        # first-bwd:
        #   dx_in, dpatchw = conv_bwd(x_in, patch_weight, grad_output=dx_patch)
        #   dpatchb = dx_patch.sum(...)
        # double-bwd returns:
        #   dd_dx_patch: cotangent wrt dx_patch
        #   x_in_d2:     contribution wrt forward x_in
        # ==================================================
        ddpatchw = get_dd("ddpatchw", "dpatchw")
        ddpatchb = get_dd("ddpatchb", "dpatchb")
        dd_dx_patch, x_in_d2, _ = conv_double_bwd( ddgrad_in, ddpatchw, ddpatchb, dx_patch, self.patch_embed.weight,
            x_in, stride_=list(self.patch_embed.stride), padding_=list(self.patch_embed.padding), groups_=Fuse,  )
        d_activates["x_in_d2"] = x_in_d2
        # ==================================================
        # 2. Reverse patch reshape path
        # first-bwd path:
        #   dx_tokens -> dx_patch
        # reverse cotangent:
        #   dd_dx_patch -> dd_dx_tokens
        # ==================================================
        dd_dx_flat = dd_dx_patch.view(B, Fuse * D, N).transpose(1, 2).contiguous()
        # [B, N, Fuse*D]
        dd_dx_reshape = dd_dx_flat.reshape(B, N, Fuse, D)
        # [B, N, Fuse, D]
        dd_dx_tokens = dd_dx_reshape.permute(0, 2, 1, 3).contiguous()
        # [B, Fuse, N, D]
        # ==================================================
        # 3. Reverse cat + pos gradients
        #
        # first-bwd:
        #   dpos_embed = dx_pos.sum(dim=0)
        #   dcls_token = dx_pos[:, :, 0:1, :].sum(dim=0)
        #   dx_tokens  = dx_pos[:, :, 1:, :]
        #
        # So dd wrt dx_pos receives:
        #   dd_dx_tokens at patch-token positions
        #   ddcls_token broadcast to cls position
        #   ddpos_embed broadcast to all positions
        # ==================================================
        dd_dx_pos = torch.zeros_like(x_pos)
        dd_dx_pos[:, :, 1:, :] = dd_dx_pos[:, :, 1:, :] + dd_dx_tokens
        ddcls_token = get_dd("ddcls_token", "dcls_token")
        if ddcls_token is not None:
            assert ddcls_token.shape == self.cls_token.shape
            dd_dx_pos[:, :, 0:1, :] = dd_dx_pos[:, :, 0:1, :] + ddcls_token.expand(B, -1, -1, -1)

        ddpos_embed = get_dd("ddpos_embed", "dpos_embed")
        if ddpos_embed is not None:
            assert ddpos_embed.shape == self.pos_embed.shape
            dd_dx_pos = dd_dx_pos + ddpos_embed.expand(B, -1, -1, -1)

        # ==================================================
        # 4. Double of TransformerBlock bwd
        #
        # first-bwd:
        #   dx_pos, d_block_weights = self.block.run_first_bwd(...)
        #
        # double-bwd:
        #   dd_dx_pos -> dd_dx_block
        # ==================================================
        dd_dx_block, d_block_activates = self.block.run_double_bwd(
            tape=block_tape,
            d_activates=d_block_activates,
            dd_weights=dd_weights["block"],
            ddgrad_in=dd_dx_pos,
            Fuse=Fuse,
        )
        d_activates["block"] = d_block_activates
        # ==================================================
        # 5. Double of final LayerNorm bwd
        #
        # first-bwd:
        #   dx_block, dnormw, dnormb =
        #       grouped_layernorm_backward(x_norm_in, norm.weight, grad_output=dx_norm)
        #
        # double-bwd:
        #   dd_dx_block -> dd_dx_norm
        #   also produces x_norm_in_d2 wrt forward block output
        # ==================================================
        x_norm_in_d2, _, dd_dx_norm = grouped_layernorm_double_bwd_fn(
            x=x_norm_in,
            weight=self.norm.weight,
            ggX=dd_dx_block,
            ggW=get_dd("ddnormw", "dnormw"),
            ggB=get_dd("ddnormb", "dnormb"),
            gO=dx_norm,
            Fuse=Fuse,
        )
        d_activates["x_norm_in_d2"] = x_norm_in_d2
        # ==================================================
        # 6. Reverse cls slice
        #
        # first-bwd:
        #   dx_norm[:, :, 0, :] = dx_cls
        #
        # only cls position flows back to dx_cls.
        # non-cls positions in dd_dx_norm are discarded here.
        # ==================================================
        dd_dx_cls = dd_dx_norm[:, :, 0, :].contiguous()
        # [B, Fuse, D]

        # ==================================================
        # 7. Double of head grouped linear bwd
        #
        # first-bwd:
        #   dx_cls, dheadw, dheadb =
        #       grouped_linear_bwd(x_cls, head.weight, grad_output=dx_head)
        #
        # double-bwd:
        #   dd_dx_cls -> dd_dx_head
        #   also produces x_cls_d2 wrt forward x_cls
        # ==================================================
        dd_dx_head, x_cls_d2, _ = grouped_linear_double_bwd(
            x=x_cls,
            w=self.head.weight,
            grad_output=dx_head,
            gg_grad_input=dd_dx_cls,
            gg_grad_w=get_dd("ddheadw", "dheadw"),
            gg_grad_b=get_dd("ddheadb", "dheadb"),
            Fuse=Fuse,
        )
        d_activates["x_cls_d2"] = x_cls_d2
        # ==================================================
        # 8. Reverse head reshape
        #
        # first-bwd:
        #   dx_head = dx_out.reshape(B, Fuse, C)
        #
        # reverse:
        #   dd_dx_head -> dd_dx_out
        # ==================================================
        dd_dx_out = dd_dx_head.reshape(B * Fuse, C).contiguous()
        assert dd_dx_out.shape == x_out.shape
        d_activates["dd_dx_out"] = dd_dx_out
        return dd_dx_out, d_activates

    def run_bwd2_1(
        self,
        tape,
        d_activates,
        dd_dx_out,
        Fuse=None,
    ):
        """
        ViT_Fused bwd2_1 stage.
        这个函数重新跑一遍 ViT first-bwd，
        但是在正确位置加回 run_double_bwd 产生的 activation-level d2。
        输入: dd_dx_out: run_double_bwd 返回的 cotangent wrt CE first-bwd output dx_out。
        返回:  dx_in: d grand_loss / d input。
        """
        if Fuse is None:
            Fuse = self.Fuse

        patch = tape["patch"]
        head = tape["head"]
        block_tape = tape["block"]
        D = self.embed_dim
        C = self.num_classes
        N = self.num_patches
        x_in = patch["x_in"]
        x_patch = patch["x_patch"]
        x_norm_in = head["x_norm_in"]
        x_cls = head["x_cls"]
        B= x_cls.shape[0]
        x_out = head["x_out"]
        assert dd_dx_out.shape == x_out.shape

        # ==================================================
        # 1. CE double-bwd
        #
        # first-bwd:
        #   dx_out = crossEntropy_bwd(x_out, target, Fuse=Fuse)
        #
        # double-bwd gives corrected grad wrt logits x_out.
        # CE Hessian 不依赖 target，所以你的 crossEntropy_double_bwd
        # 和 ResNet 里一样只需要 x_out, dd_dx_out, Fuse。
        # ==================================================
        dx_out_d1 = crossEntropy_double_bwd(
            x_out,
            dd_dx_out,
            Fuse,
        )
        # [B*Fuse, C]
        # forward:
        #   x_out = x_head.reshape(B*Fuse, C)
        dx_head = dx_out_d1.reshape(B, Fuse, C)
        # [B, Fuse, C]
        # ==================================================
        # 2. head bwd
        # ==================================================
        dx_cls, _, _ = grouped_linear_bwd(
            x_cls,
            self.head.weight,
            grad_output=dx_head,
            Fuse=Fuse,
        )
        # [B, Fuse, D]
        # Inject d2 wrt forward x_cls from head double-bwd.
        x_cls_d2 = d_activates.pop("x_cls_d2")
        assert x_cls_d2.shape == dx_cls.shape
        dx_cls = dx_cls + x_cls_d2
        # ==================================================
        # 3. cls slice bwd
        #
        # forward:
        #   x_cls = x_norm[:, :, 0]
        # ==================================================
        dx_norm = torch.zeros_like(x_norm_in)
        dx_norm[:, :, 0, :] = dx_cls
        # ==================================================
        # 4. final LayerNorm bwd
        # ==================================================
        dx_block, _, _ = grouped_layernorm_backward(
            x_norm_in,
            self.norm.weight,
            Fuse=Fuse,
            grad_output=dx_norm,
        )
        # Inject d2 wrt forward x_norm_in, i.e. block output.
        # x_norm_in_d2 = d_activates.get("x_norm_in_d2", None)
        x_norm_in_d2 = d_activates.pop("x_norm_in_d2")
        d_block_activates = d_activates.pop("block")
        assert x_norm_in_d2.shape == dx_block.shape
        dx_block = dx_block + x_norm_in_d2
        # ==================================================
        # 5. TransformerBlock bwd2_1
        # ==================================================
        dx_pos = self.block.run_bwd2_1(
            tape=block_tape,
            d_activates=d_block_activates,
            grad_output=dx_block,
            Fuse=Fuse,
        )
        # ==================================================
        # 6. pos add + cat bwd
        #
        # forward:
        #   x_pos = cat(cls_token_expand, x_tokens) + pos_embed
        #
        # For bwd2_1, no direct d2 injection here because add/cat/expand
        # are linear. Their parameter cotangents were already propagated
        # inside run_double_bwd through dd_dx_pos.
        # ==================================================
        dx_cat = dx_pos
        dx_tokens = dx_cat[:, :, 1:, :]
        # [B, Fuse, N, D]
        # ==================================================
        # 7. reverse patch token reshape
        #
        # forward:
        #   x_patch:   [B, Fuse*D, Hp, Wp]
        #   x_flat:    [B, N, Fuse*D]
        #   x_reshape: [B, N, Fuse, D]
        #   x_tokens:  [B, Fuse, N, D]
        # ==================================================
        dx_reshape = dx_tokens.permute(0, 2, 1, 3).contiguous()
        # [B, N, Fuse, D]
        dx_flat = dx_reshape.reshape(B, N, Fuse * D)
        # [B, N, Fuse*D]
        dx_patch = dx_flat.transpose(1, 2).contiguous().view_as(x_patch)
        # [B, Fuse*D, Hp, Wp]
        # ==================================================
        # 8. patch_embed conv bwd
        # ==================================================
        dx_in, _, _ = conv_bwd( x_in, self.patch_embed.weight, grad_output=dx_patch,
            stride=self.patch_embed. stride[0], padding=self.patch_embed.padding[0],groups=Fuse )
        # Inject d2 wrt forward input x_in from patch conv double-bwd.
        x_in_d2 = d_activates.pop("x_in_d2")
        assert x_in_d2.shape == dx_in.shape
        dx_in = dx_in + x_in_d2
        return dx_in


###################################
###################################
flag = 'ManuelBwd'          
flag = 'VFuse'
Fuse = 1
batch_size = 128
num_class=10
out_channel = 128 # in shape是写死了64， 所以out是128的话就是short cut
stride = 2
test_iter = 1
set_random_seed() # 仅仅在ACC test的时候使≈≈≈≈≈≈≈≈≈
###################################



if __name__ == "__main__":
    print("flag = " + flag)
    model1 = ViT_Fused( image_size=32, patch_size=4, in_channels=3, num_classes=10, embed_dim=128, num_heads=4, Fuse=Fuse, ).to("cuda")
    model = model1
    transform = transforms.ToTensor()
    cifar10 = torchvision.datasets.CIFAR10(root='/scratch/yguo25/files/mtt-distillation/data', train=True, download=True, transform=transform)
    # 取一个 batch
    indices = torch.arange(batch_size)   # 或 torch.randperm(len(cifar10))[:batch_size]
    imgs, labels = zip(*(cifar10[i] for i in indices))
    x = torch.stack(imgs).to("cuda").requires_grad_(True)
    target = torch.tensor(labels, device="cuda")

    # torch.save(model.state_dict(), 'model_test_Vit.pt')
    # exit()
    pretrained_dict = torch.load("model_test_Vit.pt")
    x = x.repeat(1, Fuse, 1, 1).detach().clone().requires_grad_()
    # target = target.repeat(Fuse) # 这个只能对应Linear stacked 的输出。否则BFC还是FBC顺序不同。
    target = target.repeat_interleave(Fuse) # 这步也很关键。因为之前是直接把batch维度重复了，所以target也要对应地重复。repeat_interleave 可以把每个元素重复Fuse次。

    if(Fuse != 1):
        pretrained_dict = repeat_params_for_fuse(pretrained_dict,Fuse)
    load_state_dict_by_position(model, pretrained_dict)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([x], lr=1e-1)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    # print(model.block.bn1.track_running_stats, model.block.bn2.track_running_stats)
    start = time.time()

    for step in range(test_iter):
        optimizer.zero_grad()
        if flag =='VFuse':
            x_out,_ = model(x)
            # x_out = x_out.reshape(-1, x_out.shape[-1])
            loss = criterion(x_out, target)  # compute loss
            loss *= Fuse  # 应该有一个这个。在VFuse 乘倍是有意义的会影响后面的值。
            print("----CELOSS-----", loss.item())
            dw = torch.torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
            weight_sum = [d.sum() for d in dw]
            grad_loss = sum(weight_sum)
            print("----GRANDLOSS-----", grad_loss.item())
            grad_loss.backward()  
            print("----GRAD-----", x.grad.sum().item())
        elif flag =='ManuelBwd':
            with torch.no_grad():
                print('Fuse = ',Fuse)
                x_out, tape = model(x)
                loss = criterion(x_out, target)
                loss *= Fuse  
                print("----CELOSS-----", loss.item())
                dx_in, d_activates, d_weights, d_weights_list_all = model.run_first_bwd(
                    tape=tape,
                    target=target,
                    Fuse=Fuse,
                )
                weight_sum = [d.sum() for d in d_weights_list_all if d is not None]
                grad_loss = sum(weight_sum)
                print("----GRANDLOSS-----", grad_loss.item())

                dd_weights = model.init_dd_weights()
                # grand_loss 只来自参数梯度 sum，不包含 dx_in.sum()
                # 所以 wrt first-bwd output dx_in 的 cotangent 是 0。
                ddgrad_in = torch.zeros_like(dx_in)
                dd_dx_out, d_activates = model.run_double_bwd(
                    tape=tape,
                    d_activates=d_activates,
                    dd_weights=dd_weights,
                    ddgrad_in=ddgrad_in,
                    Fuse=Fuse,
                )
                dx_grand = model.run_bwd2_1(
                    tape=tape,
                    d_activates=d_activates,
                    dd_dx_out=dd_dx_out,
                    Fuse=Fuse,
                )
                print("----GRAD-----", dx_grand.sum().item())
    end = time.time()
    print("当前显存使用:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("峰值显存使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    print("时间占用：", end-start)