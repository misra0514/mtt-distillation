from ViT_FlexFused import MultiHeadSelfAttention_Fused
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
from networks.networks_basicblock_fused3 import instanceNorm_backward ,instanceNorm_double_backwards_fn, instance_norm_backward_triton,instanceNorm_double_backwards_triton
from networks.networks_stateless_basicblock import linear_bwd, conv_bwd, insNormNRelu_bwd, \
linear_double_bwd, conv_double_bwd, insNormNRelu_double_bwd, avgPool_bwd, crossEntropy_bwd, \
    avgPool_double_bwd,bmm_bwd, linerFused_bwd, linearFused_double_bwd,crossEntropy_double_bwd
    

###################################
###################################
flag = "Both"
Fuse = 2
batch_size = 16
embed_dim = 128
num_heads = 4
num_tokens = 65       # 类似 ViT: 64 patches + 1 cls token
test_iter = 1
loss_scale = Fuse     # 如果你想模仿 VFuse 里面 loss *= Fuse，就保留这个
set_random_seed()
###################################
###################################


def make_attn_dd_weights(model):
    """
    dd_weights 对应:
        grad_loss = sum(d.sum() for d in d_weights_all)

    所以每个一阶参数梯度的 cotangent 都是 ones_like。
    """
    return {
        "ddqkvw": torch.ones_like(model.qkv.weight),
        "ddqkvb": torch.ones_like(model.qkv.bias) if model.qkv.bias is not None else None,
        "ddoutprojw": torch.ones_like(model.out_proj.weight),
        "ddoutprojb": torch.ones_like(model.out_proj.bias) if model.out_proj.bias is not None else None,
    }


def max_abs_rel_err(a, b, name):
    abs_err = (a - b).abs().max().item()
    denom = b.abs().max().item() + 1e-12
    rel_err = abs_err / denom
    print(f"{name}: abs_err={abs_err:.6e}, rel_err={rel_err:.6e}")
    return abs_err, rel_err


if __name__ == "__main__":
    print("Testing MultiHeadSelfAttention_Fused")
    print("Fuse =", Fuse)

    device = "cuda"

    model = MultiHeadSelfAttention_Fused(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        Fuse=Fuse,
    ).to(device)

    model.eval()

    # --------------------------------------------------
    # fake attention input:
    #   [B, Fuse, N, C]
    # --------------------------------------------------
    x0 = torch.randn(
        batch_size,
        Fuse,
        num_tokens,
        embed_dim,
        device=device,
    )

    # ==================================================
    # 1. Autograd baseline
    # ==================================================
    x_auto = x0.detach().clone().requires_grad_(True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    out_auto, _ = model(x_auto)
    # out_auto: [B, Fuse, N, C]

    # standalone test loss:
    #   loss = sum(out) * loss_scale
    #
    # 这个 loss 的 Hessian wrt out 是 0。
    # 所以后面 manual bwd2_1 的 corrected grad_output 要用 zeros_like(out)，不是 ones_like(out)。
    loss_auto = out_auto.sum() * loss_scale
    print("----AUTO LOSS-----", loss_auto.item())

    params = list(model.parameters())

    grads_auto = torch.autograd.grad(
        loss_auto,
        params + [x_auto],
        create_graph=True,
        retain_graph=True,
    )

    dw_auto = grads_auto[:-1]
    dx_loss_auto = grads_auto[-1]

    grad_loss_auto = sum([g.sum() for g in dw_auto])
    print("----AUTO GRANDLOSS-----", grad_loss_auto.item())

    dx_grand_auto = torch.autograd.grad(
        grad_loss_auto,
        x_auto,
        retain_graph=False,
        create_graph=False,
    )[0]

    print("----AUTO FIRST DX SUM-----", dx_loss_auto.sum().item())
    print("----AUTO GRAND DX SUM-----", dx_grand_auto.sum().item())

    # ==================================================
    # 2. Manual first-bwd + double-bwd + bwd2_1
    # ==================================================
    with torch.no_grad():
        x_manual = x0.detach().clone()

        out_manual, tape = model(x_manual)
        loss_manual = out_manual.sum() * loss_scale
        print("----MANUAL LOSS-----", loss_manual.item())

        # 因为 loss = out.sum() * loss_scale
        # 所以 dloss/dout = ones_like(out) * loss_scale
        grad_output = torch.ones_like(out_manual) * loss_scale

        # --------------------------------------------------
        # manual first backward
        # --------------------------------------------------
        dx_first_manual, d_activates, d_weights, d_weights_all = model.run_first_bwd(
            tape=tape,
            grad_output=grad_output,
            Fuse=Fuse,
        )

        grad_loss_manual = sum([d.sum() for d in d_weights_all if d is not None])
        print("----MANUAL GRANDLOSS-----", grad_loss_manual.item())

        # --------------------------------------------------
        # manual double backward
        #
        # grad_loss 只包含参数梯度的 sum，不包含 dx_first_manual。
        # 所以 ddgrad_in = zeros_like(dx_first_manual)
        # --------------------------------------------------
        dd_weights = make_attn_dd_weights(model)
        ddgrad_in = torch.zeros_like(dx_first_manual)

        dd_grad_output, d_activates = model.run_double_bwd(
            tape=tape,
            d_activates=d_activates,
            dd_weights=dd_weights,
            ddgrad_in=ddgrad_in,
            Fuse=Fuse,
        )

        # --------------------------------------------------
        # bwd2_1
        #
        # 注意：
        #   这里不能传原来的 grad_output = ones。
        #
        # 因为 loss = out.sum() * loss_scale，
        # 它的 Hessian wrt out 是 0。
        #
        # 所以 corrected_grad_output = 0。
        #
        # 如果以后换成 CE/MSE，这里应该传 loss double-bwd 的结果。
        # --------------------------------------------------
        corrected_grad_output = torch.zeros_like(out_manual)

        dx_grand_manual = model.run_bwd2_1(
            tape=tape,
            d_activates=d_activates,
            grad_output=corrected_grad_output,
            Fuse=Fuse,
        )

        print("----MANUAL FIRST DX SUM-----", dx_first_manual.sum().item())
        print("----MANUAL GRAND DX SUM-----", dx_grand_manual.sum().item())

    # ==================================================
    # 3. Compare
    # ==================================================
    print("\n========== CHECK ==========")
    max_abs_rel_err(out_manual, out_auto.detach(), "forward out")
    max_abs_rel_err(dx_first_manual, dx_loss_auto.detach(), "first-bwd dx")
    max_abs_rel_err(dx_grand_manual, dx_grand_auto.detach(), "double-bwd+bwd2_1 dx")

    print("\nscalar check:")
    print("loss diff:", abs(loss_manual.item() - loss_auto.item()))
    print("grand_loss diff:", abs(grad_loss_manual.item() - grad_loss_auto.item()))

    print("\n当前显存使用:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("峰值显存使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")