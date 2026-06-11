
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import torchvision.transforms as transforms
from typing import Optional, Sequence, Tuple

from functools import reduce
from operator import mul


import triton
import triton.language as tl
import time 
import random
import numpy as np
import os
from typing import Optional, Sequence, Tuple
from networks.networks_basicblock_tritonwrapper import instance_norm_backward_triton, instanceNorm_backward_plain
# from networks_fused2 import instance_norm_backward_triton
from networks.networks_basicblock_tritonwrapper import instanceNorm_double_backwards_triton
from networks.networks_basicblock_tritonwrapper import instanceNorm_backward ,instanceNorm_double_backwards_fn,\
     instance_norm_backward_triton,instanceNorm_double_backwards_plain, gelu_drop_double_grad_triton,gelu_drop_grad_triton


#########################
# 一阶导数：
#########################


def insNormNRelu_bwd( x, weight, output, grad_output, v_fuse = True):
    if(v_fuse):
        grad_x, grad_weight, grad_bias = instance_norm_backward_triton(x, weight, grad_output, output)
    else:
        grad_x, grad_weight, grad_bias =  instanceNorm_backward_plain(x, weight, grad_output, output)
    return grad_x, grad_weight, grad_bias


def crossEntropy_bwd(logits, target, Fuse = 2):
    """
    logits: [N, C]
    target: [N]
    grad_output: same shape as logits, or broadcastable
                 represents dL/d(out) from next layer
                 default is 1 (scalar loss)
    """
    N = target.shape[0]
    softmax = F.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes=logits.size(1)).type_as(logits)
    grad_logits = (softmax - one_hot) * Fuse / N   # dCE/d(logits)
    return grad_logits




def avgPool_bwd(
    x,
    grad_output,
    kernel_size=[2, 2],
    stride=[2, 2],
    padding=[0, 0],
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    # TODO:  精度不太一样找不到原因。目前只能认为是累加误差。
    return torch.ops.aten.avg_pool2d_backward(grad_output,x,kernel_size,stride, padding,ceil_mode,count_include_pad, divisor_override)

def linear_bwd( x, weight, grad_output):
    grad_x = grad_output @ weight                     # (N, in_features)
    grad_weight = grad_output.t() @ x                # (out_features, in_features)
    grad_bias = grad_output.sum(0)                      # (out_features,)

    return grad_x, grad_weight, grad_bias

def conv_bwd( x, weight,grad_output, stride=1, padding=1, dilation=1, groups=1):
    input_shape = x.shape
    weight_size = weight.shape
    grad_x = torch.nn.grad.conv2d_input(input_shape, weight, grad_output, stride, padding, dilation, groups)
    grad_weight = torch.nn.grad.conv2d_weight(x, weight_size, grad_output, stride, padding, dilation, groups)
    # grad_bias = grad_output.sum(0)
    grad_bias = grad_output.sum(dim=(0, 2, 3))

    return grad_x, grad_weight, grad_bias

def bmm_bwd(mat1, mat2, grad_output):
    # 因为classifier（Linear_stacked） 本身就做了很多transpose + reshape 。这里很难调整。
    # x和w都是在linear_stack里面先view 再transpose。这里拿到的是原始tensor（因为view 在forward），又做一遍后两维，相当于抵消了。
    # print(mat1.shape)
    # print(mat2.shape)
    # print(grad_output.shape) # [2048, 10]
    # grad_input = grad_output.bmm(mat2.transpose(1, 2))
    grad_output = grad_output.view(2,1024,10)
    mat2 = mat2.view(2, 10, 8192)
    mat1= mat1.view(1024,2,8192).transpose(0,1).transpose(1, 2)
    
    grad_input = torch.bmm(grad_output, mat2)
    # grad_mat2  = mat1.transpose(1, 2).bmm(grad_output)
    grad_mat2 = torch.bmm(mat1, grad_output)
    return grad_input, grad_mat2

def linerFused_bwd(x, w, grad_output, Fuse =2 ):
    # 参数形状约定与 LinearStacked_2.forward 保持一致：
    #   x: (..., Fuse * I) 经过 view(B, Fuse, I) 使用
    #   w: (Fuse * O, I)   其中每个 fuse 切片是 (O, I)
    #   grad_output: (B * Fuse, O) 或 (Fuse, B, O) 展平的上游梯度
    F = Fuse
    O = w.shape[0] // F
    I = w.shape[1]

    # 根据 x 反推 batch 数，避免 grad_output 形状不一致带来的误差
    B = x.numel() // (F * I)

    # 将梯度、权重、输入都还原到与 forward 相同的内部布局
    G = grad_output.reshape(F, B, O)       # (F, B, O)
    W = w.reshape(F, O, I)                 # (F, O, I)
    X = x.reshape(B, F, I).transpose(0, 1) # (F, B, I)

    grad_input_int = torch.bmm(G, W)                   # (F, B, I)
    grad_w_int     = torch.bmm(G.transpose(1, 2), X)   # (F, O, I)
    grad_b_int     = G.sum(dim=1)                      # (F, O)

    # 回到原始 x 的布局（batch 在前，Fuse 在内层）
    grad_input = grad_input_int.transpose(0, 1).reshape_as(x)
    grad_w     = grad_w_int.reshape(F * O, I)          # (F*O, I)
    grad_b     = grad_b_int.reshape(F * O)             # (F*O,)

    return grad_input, grad_w, grad_b


#########################
# 二阶导数：
#########################


def crossEntropy_double_bwd(x_out, v, Fuse = 2,  reduction='mean'):
    """
    手动实现:
        torch.autograd.grad(dx_out, x_out, grad_outputs=v)[0]
    其中 dx_out = d CE / d x_out
    x_out: (N, C) logits
    v:     (N, C)  对应 ddx_out
    """
    # softmax 概率
    p = torch.softmax(x_out, dim=1)  # (N, C)
    # 对每个样本 n，算 s_n = sum_c p_{n,c} * v_{n,c}
    s = torch.sum(p * v, dim=1, keepdim=True)  # (N, 1)
    # Hv = p ⊙ v - s * p   （样本内的 H·v）
    Hv = p * v - s * p     # (N, C)
    if reduction == 'mean':
        # PyTorch CrossEntropyLoss 默认是 mean，会多一个 1/N
        N = x_out.size(0)
        Hv = Hv / N 
        Hv*=Fuse # 不用grad_output，只要这里加上就可以了。
    return Hv

def linearFused_double_bwd(
    x, w, grad_output,
    gg_grad_input=None,   # same shape as grad_input: (B*F, I)
    gg_grad_w=None,       # same shape as grad_w:   (F*O, I)
    gg_grad_b=None,       # same shape as grad_b:   (F*O,)
    Fuse=2,
):
    batch_full, C = grad_output.shape[-2], grad_output.shape[-1]
    F = Fuse
    B = batch_full // F
    # 内部布局：与一阶 backward 完全一致
    G = grad_output.view(F, B, C)    # (F, B, O)
    W = w.view(F, C, -1)             # (F, O, I)
    I = W.shape[-1]
    X = x.view(B, F, I).transpose(0, 1)   # (F, B, I)
    dG = torch.zeros_like(G)
    dW = torch.zeros_like(W)
    dX = torch.zeros_like(X)
    # ---- 1) 来自 grad_input_int = G @ W ----
    if gg_grad_input is not None:
        # 外部 grad_input: (B*F, I)
        # 一阶里是 grad_input_int.transpose(0,1).reshape(B*F, I)
        # 反向映射回内部 GI 的梯度：
        H_ext = gg_grad_input.view(B, F, I)    # (B, F, I)
        H     = H_ext.transpose(0, 1)          # (F, B, I)  对应 GI
        # GI = G @ W
        # ∂L/∂G += H @ Wᵀ
        dG = dG + torch.bmm(H, W.transpose(1, 2))       # (F, B, C)
        # ∂L/∂W += Gᵀ @ H
        dW = dW + torch.bmm(G.transpose(1, 2), H)       # (F, C, I)

    # ---- 2) 来自 grad_w_int = Gᵀ @ X ----
    if gg_grad_w is not None:
        # 外部 grad_w: (F*O, I) → (F, O, I)
        ggW = gg_grad_w.view(F, C, I)     # (F, O, I)

        # 对 Y = Gᵀ X 的二阶：
        # dX += G @ ggW
        dX = dX + torch.bmm(G, ggW)                     # (F, B, I)
        # dG += X @ ggWᵀ
        dG = dG + torch.bmm(X, ggW.transpose(1, 2))     # (F, B, C)

    # ---- 3) 来自 grad_b_int = G.sum(dim=1) ----
    if gg_grad_b is not None:
        ggb = gg_grad_b.view(F, C)                     # (F, O)
        # 每个 batch 位置都加上同一 ggb
        dG = dG + ggb.unsqueeze(1).expand(F, B, C)     # (F, B, O)

    # ---- 内部 dX, dW, dG → 外部 dx, dw, dgrad_output ----
    # X = x.view(B, F, I).transpose(0,1)
    # 反向：先 transpose 回去，再 reshape
    dx = dX.transpose(0, 1).reshape_as(x)              # (B*F, I)
    # W = w.view(F, O, I)
    dw = dW.reshape_as(w)                              # (F*O, I)
    # G = grad_output.view(F, B, C)
    dgrad_output = dG.view(batch_full, C).reshape_as(grad_output)  # (B*F, C)
    # 顺序：∂L/∂grad_output, ∂L/∂x, ∂L/∂w
    return dgrad_output, dx, dw



def avgPool_double_bwd(
    grad_grad_input,
    kernel_size=[2, 2],
    stride=[2, 2],
    padding=[0, 0],
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    """
    grad_grad_input: d^2 L / d (input) d(...)
    返回: grad_grad_output（对应 avg_pool 的输出位置），形状 [N,C,OH,OW]
    只要 grad_grad_input 在 CUDA 上，这里全程走 GPU。
    """
    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    else:
        kh, kw = kernel_size

    if isinstance(stride, int):
        sh = sw = stride
    else:
        sh, sw = stride

    if isinstance(padding, int):
        ph = pw = padding
    else:
        ph, pw = padding

    N, C, H, W = grad_grad_input.shape
    device = grad_grad_input.device
    dtype = grad_grad_input.dtype

    # 1) 先用一个 dummy 输入算出 OH, OW（这样 ceil_mode 等细节由 avg_pool2d 自己处理）
    with torch.no_grad():
        dummy = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
        y_dummy = F.avg_pool2d(
            dummy,
            kernel_size=(kh, kw),
            stride=(sh, sw),
            padding=(ph, pw),
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
        OH, OW = y_dummy.shape[2], y_dummy.shape[3]

    # 2) 构造 divisor map（和 backward 的一样）
    if divisor_override is not None:
        div = torch.full((1, 1, OH, OW), float(divisor_override), device=device, dtype=dtype)
    else:
        if count_include_pad:
            div = torch.full((1, 1, OH, OW), float(kh * kw), device=device, dtype=dtype)
        else:
            oh_idx = torch.arange(OH, device=device)
            ow_idx = torch.arange(OW, device=device)
            hstart = oh_idx * sh - ph
            hend   = torch.clamp(hstart + kh, max=H)
            hstart_clamped = torch.clamp(hstart, min=0)
            eff_h = (hend - hstart_clamped).clamp(min=0)

            wstart = ow_idx * sw - pw
            wend   = torch.clamp(wstart + kw, max=W)
            wstart_clamped = torch.clamp(wstart, min=0)
            eff_w = (wend - wstart_clamped).clamp(min=0)

            div_hw = eff_h[:, None] * eff_w[None, :]
            div_hw = div_hw.clamp(min=1)
            div = div_hw.unsqueeze(0).unsqueeze(0).to(dtype=dtype)

    # 3) 计算 conv_transpose2d 的完整输出尺寸，用 dummy 走一遍
    weight = torch.ones((C, 1, kh, kw), device=device, dtype=dtype)
    with torch.no_grad():
        dummy_grad_scaled = torch.zeros(N, C, OH, OW, device=device, dtype=dtype)
        full_out = F.conv_transpose2d(
            dummy_grad_scaled,
            weight,
            bias=None,
            stride=(sh, sw),
            padding=(ph, pw),
            groups=C,
        )
        full_H, full_W = full_out.shape[2], full_out.shape[3]

    # 4) 反裁剪：把 grad_grad_input 填回到“大” tensor 里
    grad_grad_input_full = torch.zeros(
        (N, C, full_H, full_W), device=device, dtype=dtype
    )
    grad_grad_input_full[:, :, :H, :W] = grad_grad_input

    # 5) conv_transpose2d 对输入的梯度是 conv2d
    #    对 weight=ones 的情况，就是用同一个 ones kernel 做 conv2d
    grad_S = F.conv2d(
        grad_grad_input_full,
        weight,
        bias=None,
        stride=(sh, sw),
        padding=(ph, pw),
        groups=C,
    )  # 形状 [N,C,OH,OW]

    # 6) 最后一步：对应 forward 里的 `/ div`，对 G 的梯度 = grad_S / div
    grad_grad_output = grad_S / div  # broadcast 到 [N,C,OH,OW]

    return grad_grad_output



def insNormNRelu_double_bwd(ggI, ggw, ggb, grad_output, output ,weight, x, v_fuse=True):
    if v_fuse:
        gx, gw, ggO = instanceNorm_double_backwards_triton(x=x, gamma=weight, ggX=ggI, ggG=ggw, ggB=ggb, gO= grad_output,out= output,eps= 1e-5)
        ggO = ggO.view_as(grad_output)
    else:
        # gx, gw, ggO = instanceNorm_double_backwards_triton(x=x, gamma=weight, ggX=ggI, ggG=ggw, ggB=ggb, gO= grad_output,out= output,eps= 1e-5)
        gx, gw, ggO = instanceNorm_double_backwards_plain(x=x, gamma=weight, ggX=ggI, ggG=ggw, ggB=ggb, gO= grad_output,out= output,eps= 1e-5)
        ggO = ggO.view_as(grad_output)
    # gG是d_gamma , 也就是weight。
    return ggO, gx, gw


def linear_double_bwd(
    # grads: Sequence[Optional[torch.Tensor]],  # [grad_grad_input, grad_grad_weight, grad_grad_bias]
    x: torch.Tensor,                         # forward 的 input, shape [..., in_features]
    weight: torch.Tensor,                    # forward 的 weight, shape [out_features, in_features]
    grad_output: torch.Tensor,               # 一阶 backward 里的 grad_output, shape [..., out_features]
    ggi: torch.Tensor,
    ggw: torch.Tensor,
    ggb: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    # 这里直接用 weight 推出 in/out 维度，避免靠 dim 猜
    out_features, in_features = weight.shape
    # -------- 把所有有关的张量 flatten 到 [B, *] --------
    # B = 所有 batch 维的乘积，方便统一用 matmul
    x_flat = x.reshape(-1, in_features)                    # [B, in]
    go_flat = grad_output.reshape(-1, out_features)        # [B, out]

    ggi_flat = None
    if ggi is not None:
        ggi_flat = ggi.reshape(-1, in_features)            # [B, in]
    ggw_2d = None
    if ggw is not None:
        # 线性层的 grad_grad_weight 应该就是 [out, in]，这里强行 reshape 一下更保险
        ggw_2d = ggw.reshape(out_features, in_features)    # [out, in]
    ggb_flat = None
    if ggb is not None:
        # bias 方向的高阶梯度，最后一维必须是 out_features
        ggb_ = ggb
        if ggb_.dim() == 1:
            ggb_ = ggb_.unsqueeze(0)                       # [1, out]
        last_dim = ggb_.shape[-1]
        assert last_dim == out_features, \
            f"ggb last dim ({last_dim}) != out_features ({out_features})"
        ggb_flat = ggb_.reshape(-1, out_features)          # [B_ggb, out]
        # 如果只给了一份，就 broadcast 到所有 batch
        if ggb_flat.shape[0] == 1:
            ggb_flat = ggb_flat.expand_as(go_flat)         # [B, out]
        else:
            assert ggb_flat.shape[0] == go_flat.shape[0], \
                f"ggb batch dim {ggb_flat.shape[0]} != grad_output batch dim {go_flat.shape[0]}"

    # -------- 计算 dx2, dw2, dgradout2（全部在 flat 空间）--------
    dx2_flat: Optional[torch.Tensor] = None
    dw2:     Optional[torch.Tensor] = None
    dgo2_flat: Optional[torch.Tensor] = None
    # 1) dx2_flat = grad_output @ ggw
    if ggw_2d is not None:  # [B, out] @ [out, in] = [B, in]
        dx2_flat = go_flat.matmul(ggw_2d)
    # 2) dw2 = grad_output^T @ ggi
    if ggi_flat is not None:  # [B, out]^T @ [B, in] = [out, in]
        dw2 = go_flat.transpose(0, 1).matmul(ggi_flat)
    # 3) dgo2_flat = ggi @ W^T + x @ ggw^T + ggb
    need = (ggi_flat is not None) or (ggw_2d is not None) or (ggb_flat is not None)
    if need:
        dgo2_flat = torch.zeros_like(go_flat)              # [B, out]
        if ggi_flat is not None:
            # [B, in] @ [in, out] = [B, out]
            dgo2_flat = dgo2_flat + ggi_flat.matmul(weight.transpose(0, 1))
        if ggw_2d is not None:
            # [B, in] @ [in, out] = [B, out]
            dgo2_flat = dgo2_flat + x_flat.matmul(ggw_2d.transpose(0, 1))
        if ggb_flat is not None:
            dgo2_flat = dgo2_flat + ggb_flat
    # -------- 把 flat 结果 reshape 回原状 --------
    dx2 = dx2_flat.reshape_as(x) if dx2_flat is not None else None
    dgrad_out = dgo2_flat.reshape_as(grad_output) if dgo2_flat is not None else None

    # 返回 gg0 gI gw
    return dgrad_out,dx2, dw2,



def conv_double_bwd(ggI_opt, ggW_r_opt, ggb_opt, gO_r, weight_r, input,
    stride_ = [1,1],
    padding_        = [1, 1],
    dilation_       = [1, 1],
    transposed_     = False,
    output_padding_ = [0, 0],
    groups_         = 1,
    output_mask     = [True, True, True],   # 返回 ggO, gI, gW
    ):
    # ggI_opt输入梯度的梯度； ggW_r_opt： w的二阶梯度？； ggb_opt：b的二阶梯度？； gO_r一阶导数的输入gradoutput； input 是conv的input？
    # 返回：输出tensor（grad_output)的二阶梯度。 gI：x的梯度累计； gw weight的梯度累计； bias 应该没有梯度。

    ggO, gI, gW = torch.ops.aten._convolution_double_backward(ggI_opt,ggW_r_opt, ggb_opt, gO_r, weight_r, input,     stride_,         # [1, 1]
    padding_,        # [1, 1]
    dilation_,       # [1, 1]
    transposed_,     # False
    output_padding_, # [0, 0]
    groups_,         # 1
    output_mask      # [True, True, True]
    )
    return ggO, gI, gW


def adaptivepooling_bwd(x, grad_output):
    out = torch.ops.aten._adaptive_avg_pool2d_backward(grad_output, x)
    return out

def adaptivepooling_double_bwd(x,shape = (1, 1)):
    return F.adaptive_avg_pool2d(x, shape)


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






def _to_float_p(p):
    # p 需要是 Python float，native_dropout 不适合直接传 Tensor
    if isinstance(p, torch.Tensor):
        p_float = float(p.detach().item())
    else:
        p_float = float(p)

    # 数值安全：避免 p >= 1 导致除 0
    p_clamped = min(max(p_float, 0.0), 1.0 - 1e-6)
    return p_clamped


def dropout_fwd(x, p,training =True):
    p_clamped = _to_float_p(p)
    keep_prob = 1.0 - p_clamped

    #TODO:  当p=0的时候，按理说是不生成mask的。 这也是p=0内存还上涨了的原因。
    # 可以在p=0的时候m = None， 但是不知道是否会对后面的bwd产生影响。
    if p_clamped <= 0.0:
        out = x
        m = torch.ones_like(x)
    else:
        # native_dropout returns:
        # out  = x * mask / (1 - p)
        # mask = bool mask, not scaled
        out, mask = torch.ops.aten.native_dropout(x, p_clamped, True)
        m = mask.to(dtype=x.dtype) / keep_prob
    return out, m

def dropout_bwd(dout, m):
    dx = dout * m
    return dx

def dropout_double_bwd(ddx, m):
    ddout = ddx * m
    return ddout


def geluDropout_fwd(x, p, training=True):
    p_clamped = _to_float_p(p)
    keep_prob = 1.0 - p_clamped
    if p_clamped <= 0.0:
        out = F.gelu(x)
        mask = torch.ones_like(x, dtype=torch.bool)
    else:
        gelu_x = F.gelu(x)
        out, mask = torch.ops.aten.native_dropout(gelu_x, p_clamped, True)
    return out, mask

def geluDropout_bwd( x, mask, dout, out=None,v_fuse=False,):
    if v_fuse:
        dx = gelu_drop_grad_triton( gY=dout, x=x, m=mask, out=out,)
    else:
        # y = dropout(gelu(x))
        # dgelu_out = dout * mask
        dgelu_out = dropout_bwd( dout, mask )
        # dx = dgelu_out * gelu'(x)
        dx = gelu_bwd( x, dgelu_out, )
    return dx

def geluDropout_double_bwd(
    x,
    mask,
    dout,
    ddx,
    out_dd_dout=None,
    out_x_d2=None,
    v_fuse=False,
):
    """
    Double backward for fused:
        y = dropout(gelu(x)) = gelu(x) * mask

    First backward:
        dx = dout * mask * gelu'(x)

    Given:
        ddx = cotangent wrt dx

    Return:
        x_d2:
            contribution wrt forward input x:
                ddx * dout * mask * gelu''(x)

        dd_dout:
            cotangent wrt first-bwd input dout:
                ddx * mask * gelu'(x)

    These names match usage in TransformerBlock:
        x_d2    -> add to dx_fc1 in bwd2_1
        dd_dout -> pass backward to fc2 double-bwd as cotangent wrt dx_dp1
    """
    if ddx is None:
        return None, None

    if v_fuse:
        # Your Triton function returns:
        #   ggY  = ddx * mask * gelu'(x)        wrt dout
        #   ggx  = ddx * dout * mask * gelu''   wrt x
        dd_dout, x_d2 = gelu_drop_double_grad_triton(
            ggX=ddx,
            gY=dout,
            x=x,
            m=mask,
            out_ggY=out_dd_dout,
            out_ggx=out_x_d2,
        )
    else:
        dz = dropout_bwd(
            dout,
            mask,
        )

        # Double of GELU backward:
        #   x_d2  = ddx * dz * gelu''(x)
        #   dd_dz = ddx * gelu'(x)
        x_d2, dd_dz = gelu_double_bwd(
            x=x,
            grad_output=dz,
            gg_grad_input=ddx,
        )

        # Double of dropout backward:
        #   dz = dout * mask
        #   dd_dout = dd_dz * mask
        dd_dout = dropout_double_bwd(
            dd_dz,
            mask,
        )

    return x_d2, dd_dout