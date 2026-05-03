
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
from networks.networks_basicblock_fused3 import instance_norm_backward_triton, instanceNorm_backward_plain
# from networks_fused2 import instance_norm_backward_triton
from networks.networks_basicblock_fused3 import instanceNorm_double_backwards_triton
from networks.networks_basicblock_fused3 import instanceNorm_backward ,instanceNorm_double_backwards_fn, instance_norm_backward_triton,instanceNorm_double_backwards_triton


#########################
# 一阶导数：
#########################


def insNormNRelu_bwd( x, weight, output, grad_output):
    grad_x, grad_weight, grad_bias = instance_norm_backward_triton(x, weight, grad_output, output)
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



def insNormNRelu_double_bwd(ggI, ggw, ggb, grad_output, output ,weight, x):
    gx, gw, ggO = instanceNorm_double_backwards_triton(x=x, gamma=weight, ggX=ggI, ggG=ggw, ggB=ggb, gO= grad_output,out= output,eps= 1e-5)
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
