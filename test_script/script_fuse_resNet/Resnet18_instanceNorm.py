# 3.26
# 终于迟迟的调完了正确性。目前虽然只能保证一个instance norm的。但是感觉可以开始搭建resnet18做测试了。
# basic code 复制与basic block
# 
# 
# # 4.24  现在终于完成了封装和bug修复。 再试一下res 18 
# 4.26 存了一个back up 方法。现在在保证内存不变的情况下把操作都封装起来。
# 4.27 加入fused 接口。

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
from networks.networks_stacked import LinearStacked_2 # NOTE 这里和flex fuse 不太一样。
from networks.networks_basicblock_fused3 import batchNorm2d_backward, batchnorm_double_backwards_fn, batchnorm_double_backwards_fn_new
from networks.networks_basicblock_fused3 import instanceNorm_backward ,instanceNorm_double_backwards_fn, instancenorm_relu_backward_triton,instanceNorm_double_backwards_triton
from networks.networks_stateless_basicblock import linear_bwd, conv_bwd, insNormNRelu_bwd, \
linear_double_bwd, conv_double_bwd, insNormNRelu_double_bwd, avgPool_bwd, crossEntropy_bwd, \
    avgPool_double_bwd,bmm_bwd, linerFused_bwd, linearFused_double_bwd,crossEntropy_double_bwd
    
# from torch.aps.aten import adaptive_avg_pool2d_backward_cuda
# from networks.networks_basicblock_fused3 import NormActive
# from networks_flexFuse import Conv_Flexfused, ConvBlock_double_bwd,ConvBlock_bwd2_1

def clear_tensorlists(*dicts):
    for d in dicts:
        d.clear()

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭自动优化，确保计算确定性
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 保证 CUDA 计算稳定（仅对 PyTorch 1.8+ 有效）
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



def adaptivepooling_bwd(x, grad_output):
    out = torch.ops.aten._adaptive_avg_pool2d_backward(grad_output, x)
    return out

def adaptivepooling_double_bwd(x,shape = (1, 1)):
    return F.adaptive_avg_pool2d(x, shape)

def BasicBlock_bwd( activates, weights, grad_output,SCstride=1, Fuse=1 ):
    # -------- unpack activates --------
    x_conv1 = activates["x_conv1"]
    x_bn1   = activates["x_bn1"]
    x_conv2 = activates["x_conv2"]
    x_bn2   = activates["x_bn2"]
    out     = activates["x_out"]
    x_bnsc  = activates.get("x_bnsc", None)
    # -------- unpack weights --------
    conv1w  = weights["conv1w"]
    bn1w    = weights["bn1w"]
    bn1b    = weights["bn1b"]
    conv2w  = weights["conv2w"]
    bn2w    = weights["bn2w"]
    bn2b    = weights["bn2b"]
    convscw = weights.get("convscw", None)
    bnscw   = weights.get("bnscw", None)
    has_downsample = (convscw is not None)
    grad_output[out <= 0] = 0

    # ---------------- main branch ----------------
    dx_bn2, dbn2w, dbn2b, _, _ = instanceNorm_backward(x_bn2, bn2w, grad_output=grad_output)
    dx_conv2, dconv2w, _ = conv_bwd(x_conv2, conv2w, grad_output=dx_bn2, groups=Fuse)
    dbno1 = dx_conv2
    dbno1[x_conv2 <= 0] = 0
    dx_bn1, dbn1w, dbn1b, _, _ = instanceNorm_backward(x_bn1, bn1w, grad_output=dbno1)
    dx_main, dconv1w, _ = conv_bwd(x_conv1, conv1w, grad_output=dx_bn1, stride=SCstride, groups=Fuse)
    dbno2 = grad_output
    # ---------------- shortcut branch ----------------
    if has_downsample:
        dbnosc = grad_output
        dx_bnsc, dbnscw, dbnscb, _, _ = instanceNorm_backward( x_bnsc, bnscw, grad_output=dbnosc)
        dx_short, dconvscw, _ = conv_bwd( x_conv1, convscw, grad_output=dx_bnsc, stride=SCstride, padding=0, groups=Fuse )
        dx_in = dx_main + dx_short
    else:
        dbnosc = None
        dx_bnsc = None
        dbnscw = None
        dbnscb = None
        dconvscw = None
        dx_in = dx_main + grad_output
    d_activates = {
        # "dx_conv1": dx_in,
        "dbno2": dbno2,
        "dbno1": dbno1,
        "dx_bn1": dx_bn1,
        "dx_bn2": dx_bn2,
        "dx_bnsc": dx_bnsc,
    }
    d_weights = {
        "dconv1w": dconv1w,
        "dbn1w": dbn1w,
        "dbn1b": dbn1b,
        "dconv2w": dconv2w,
        "dbn2w": dbn2w,
        "dbn2b": dbn2b,
        "dconvscw": dconvscw,
        "dbnscw": dbnscw,
        "dbnscb": dbnscb,
    }
    return dx_in, d_activates, d_weights

def BasicBlock_bwd2_1(
    activates, weights, d2_activates,
    grad_output, SCstride=1, Fuse = 1
):
    x_conv1 = activates["x_conv1"]
    x_bn1   = activates["x_bn1"]
    x_conv2 = activates["x_conv2"]
    x_bn2   = activates["x_bn2"]
    out     = activates["x_out"]
    x_bnsc  = activates.get("x_bnsc", None)
    conv1w  = weights["conv1w"]
    bn1w    = weights["bn1w"]
    conv2w  = weights["conv2w"]
    bn2w    = weights["bn2w"]
    convscw = weights.get("convscw", None)
    bnscw   = weights.get("bnscw", None)
    dx_conv1_d2   = d2_activates["dx_in_d2"]
    dx_bn1_d2   = d2_activates["dx_bn1_d2"]
    dx_conv2_d2  = d2_activates["dx_conv2_d2"]
    dx_bn2_d2  = d2_activates["dx_bn2_d2"]
    dx_bnsc_d2 = d2_activates.get("dx_bnsc_d2", None)

    has_downsample = (convscw is not None)
    # g_out = grad_output.clone()
    grad_output[out <= 0] = 0
    dx_bn2, dbn2w, dbn2b, _, _ = instanceNorm_backward( x_bn2, bn2w, grad_output=grad_output )
    dx_bn2 += dx_bn2_d2
    dx_conv2, dconv2w, _ = conv_bwd( x_conv2, conv2w, grad_output=dx_bn2, groups=Fuse)
    # dx_conv2 = dx_conv2.clone()
    dx_conv2 += dx_conv2_d2 # TODO: 应该需要先加法再relu。 每一个bwd2_1结束后需要立刻合并dbwd梯度。
    dx_conv2[x_conv2 <= 0] = 0
    dx_bn1, dbn1w, dbn1b, _, _ = instanceNorm_backward( x_bn1, bn1w, grad_output=dx_conv2 )
    dx_bn1 += dx_bn1_d2
    del dx_bn1_d2
    dx_conv1_main, dconv1w, _ = conv_bwd( x_conv1, conv1w, grad_output=dx_bn1, stride=SCstride,groups= Fuse )
    if has_downsample:
        dx_bnsc, dbnscw, dbnscb, _, _ = instanceNorm_backward( x_bnsc, bnscw, grad_output=grad_output )
        dx_bnsc += dx_bnsc_d2
        dx_conv1_sc, dconvscw, _ = conv_bwd( x_conv1, convscw, grad_output=dx_bnsc, stride=SCstride, padding=0, groups= Fuse)
        dx_in = dx_conv1_main + dx_conv1_sc + dx_conv1_d2
    else:
        dx_bnsc = None
        dbnscw = None
        dbnscb = None
        dconvscw = None
        dx_in = dx_conv1_main + grad_output + dx_conv1_d2
    # weights["dconv1w"] = dconv1w
    # weights["dbn1w"] = dbn1w
    # weights["dbn1b"] = dbn1b
    # weights["dconv2w"] = dconv2w
    # weights["dbn2w"] = dbn2w
    # weights["dbn2b"] = dbn2b
    # weights["dconvscw"] = dconvscw
    # weights["dbnscw"] = dbnscw
    # weights["dbnscb"] = dbnscb
    return dx_in



def BasicBlock_double_bwd(
    activates, d_activates, weights, dd_weights,
    ddgrad_in, SCstride=1, Fuse=1
):
    ddx_conv1 = ddgrad_in
    # 原地修改activates 里的值为d2。（x2_1）返回值里不再体现。
    conv1w  = weights["conv1w"]
    bn1w    = weights["bn1w"]
    conv2w  = weights["conv2w"]
    bn2w    = weights["bn2w"]
    convscw = weights.get("convscw", None)
    bnscw   = weights.get("bnscw", None)
    ddconv1w  = dd_weights["ddconv1w"]
    ddbn1w    = dd_weights["ddbn1w"]
    ddbn1b = dd_weights["ddbn1b"]
    ddconv2w  = dd_weights["ddconv2w"]
    ddbn2w    = dd_weights["ddbn2w"]
    ddbn2b = dd_weights["ddbn2b"]
    ddconvscw = dd_weights.get("ddconvscw", None)
    ddbnscw   = dd_weights.get("ddbnscw", None)
    ddbnscb   = dd_weights.get("ddbnscb", None)
    x_conv1 = activates["x_conv1"]
    x_bn1   = activates["x_bn1"]
    x_conv2 = activates["x_conv2"]
    x_bn2   = activates["x_bn2"]
    out     = activates["x_out"]
    x_bnsc  = activates.get("x_bnsc", None)

    #  branch 下还有一些点要再商议一下：
    #  因为最后一个op是先加和再relu；所以dbnosc = dbno2
    has_downsample = (convscw is not None)
    # dbnosc = dbno2
    # ---------------- main branch ----------------
    ddx_bn1, dx_in_d2, dconv1w_d2 = conv_double_bwd( ddx_conv1, ddconv1w, None,
        d_activates["dx_bn1"], conv1w, x_conv1, stride_=[SCstride,SCstride], groups_= Fuse )
    d_activates.pop("dx_bn1")
    # del dx_bn1
    d_activates["dx_in_d2"] = dx_in_d2
    dx_bn1_d2, dbn1w_d2, ddx_conv2 = instanceNorm_double_backwards_fn(
        x_bn1, bn1w, None, ddx_bn1, ddbn1w, ddbn1b, d_activates["dbno1"], 1e-5 )
    d_activates.pop("dbno1")
    # del dbno1
    d_activates["dx_bn1_d2"] = dx_bn1_d2
    ddx_conv2[x_conv2 <= 0] = 0
    ddx_bn2, dx_conv2_d2, dconv2w_d2 = conv_double_bwd( ddx_conv2, ddconv2w, None, d_activates["dx_bn2"], conv2w, x_conv2, groups_= Fuse )
    d_activates.pop("dx_bn2")
    # del dx_bn2
    d_activates["dx_conv2_d2"] = dx_conv2_d2
    dx_bn2_d2, dbn2w_d2, ddO_main = instanceNorm_double_backwards_fn(
        x_bn2, bn2w, None, ddx_bn2, ddbn2w, ddbn2b, gO=d_activates["dbno2"] )
    d_activates["dx_bn2_d2"] = dx_bn2_d2
    # ---------------- shortcut branch ----------------
    if has_downsample:
        ddx_scbn, dx_conv1_d2_sc, dconvscw_d2 = conv_double_bwd(
            ddx_conv1, ddconvscw, None,  d_activates["dx_bnsc"], convscw, x_conv1,
            stride_=[SCstride,SCstride],padding_=[0,0], groups_= Fuse)
        d_activates.pop("dx_bnsc")
        dx_bnsc_d2, dbnscw_d2, ddO_sc = instanceNorm_double_backwards_fn(
            x_bnsc, bnscw, None, ddx_scbn, ddbnscw, ddbnscb, gO=d_activates["dbno2"] )
        d_activates.pop("dbno2")
        d_activates["dx_bnsc_d2"] = dx_bnsc_d2
        d_activates["dx_in_d2"]+= dx_conv1_d2_sc
        ddO_main += ddO_sc
    else:
        d_activates.pop("dbno2")
        dx_bnsc_d2 = None
        ddO_main += ddx_conv1
    # 没有少返回，因为被累加到dx_conv1_d2_total 里面了，确实也合理，这个是x_conv在两个conv 下面产生的一阶梯度和。
    # dx_conv1_d2_total = dx_conv1_d2_sc(shortcut) + dx_conv1_d2_main(原来的dxconv2d2)
    ddO_main[out <= 0] = 0
    return ddO_main, d_activates


def conv_norm_relu_bwd(x, x_bn,x_block,convw, bnw, grad_output, Fuse = 1  ):
    dx_block = grad_output
    dx_block[x_block <= 0] = 0
    dx_bn, dbnw, dbnb,_,_ = instanceNorm_backward(x_bn, bnw, grad_output=dx_block)
    _, dconvw ,_ = conv_bwd(x, convw, grad_output=dx_bn, groups= Fuse)
    return dx_bn, dbnw, dbnb, dconvw

def conv_norm_relu_double_bwd(x, x_bn,x_block,dx_bn,dx_block,ddx_conv,  convw ,bnw,ddconvw,ddbnw,ddbnb):
    ddx_bn, dx_conv_d2, dconvw_d2 = conv_double_bwd(ddx_conv,ddconvw,None,dx_bn, convw, x )
    del dx_bn, ddx_conv
    dx_bn_d2, dbnw_d2, ddx_block = instanceNorm_double_backwards_fn(x_bn,bnw, None ,ddx_bn,ddbnw,ddbnb,dx_block)
    del dx_block, ddx_bn # 可以试试用覆盖的话，这里就不用单独del了更加工整内存也更好。
    ddx_block[x_block<=0] = 0
    return  ddx_block, dx_bn_d2, dx_conv_d2


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, Fuse = 1):
        # 注意一下这个basic block已经把 fuse隔离在外面了。内部init的时候再扩充in out channel
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels*Fuse, out_channels*Fuse, 3, stride, 1, bias=False, groups=Fuse)
        self.bn1 = nn.InstanceNorm2d(out_channels*Fuse, affine=True)
        self.conv2 = nn.Conv2d(out_channels*Fuse, out_channels*Fuse, 3, 1, 1, bias=False, groups=Fuse)
        self.bn2 = nn.InstanceNorm2d(out_channels*Fuse, affine=True)
        if stride != 1 or in_channels != out_channels:
            self.convsc = nn.Conv2d(in_channels*Fuse, out_channels*Fuse, 1, stride, bias=False, groups=Fuse )
            self.bnsc = nn.InstanceNorm2d(out_channels*Fuse, affine=True)
        else:
            self.convsc = None
            self.bnsc = None
    def forward(self, x):
        identity = x
        x_bnsc = None
        if self.convsc is not None:
            x_bnsc = self.convsc(x)
            identity = self.bnsc(x_bnsc)
        x_bn1 = self.conv1(x)
        x_conv2 = F.relu(self.bn1(x_bn1), inplace=True)
        x_bn2 = self.conv2(x_conv2)
        out = self.bn2(x_bn2)
        out = F.relu(out + identity, inplace=True)
        activates = {
            "x_conv1": x,
            "x_bn1": x_bn1,
            "x_conv2": x_conv2,
            "x_bn2": x_bn2,
            "x_bnsc": x_bnsc,
            "x_out": out,
        }
        return out, activates


class ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, Fuse=1):
        super().__init__()
        self.Fuse = Fuse
        blk_in_ch = 64   # 这里仍然是逻辑通道数
        cfg = [ (blk_in_ch,  2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2), ]
        self.conv = nn.Conv2d(in_channels * Fuse, blk_in_ch * Fuse, kernel_size=3, stride=1, padding=1, bias=False, groups=Fuse )
        self.bn = nn.InstanceNorm2d(blk_in_ch * Fuse, affine=True)
        self.stages = nn.ModuleList()
        for stage_id, (out_ch, num_blocks, first_stride) in enumerate(cfg):
            stage = nn.ModuleList()
            for block_id in range(num_blocks):
                stride = first_stride if block_id == 0 else 1
                stage.append(BasicBlock(blk_in_ch, out_ch, stride=stride, Fuse=Fuse))
                blk_in_ch = out_ch
            self.stages.append(stage)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = LinearStacked_2(512, num_classes, Fuse)

    def forward(self, x_conv):
        tape = { "stem": {}, "blocks": [], "head": {} }
        x_bn = self.conv(x_conv)
        x_block = self.bn(x_bn)
        x_block = F.relu(x_block, inplace=True)
        tape["stem"] = {  "x_conv": x_conv, "x_bn": x_bn,  "x_block": x_block}
        h = x_block
        for stage_id, stage in enumerate(self.stages):
            for block_id, blk in enumerate(stage):
                h, activates = blk(h)
                activates["stage_id"] = stage_id
                activates["block_id"] = block_id
                tape["blocks"].append(activates)
        x_pool = h
        x_fc = self.pool(x_pool)
        x_fc = torch.flatten(x_fc, 1)
        x_out = self.fc(x_fc)
        x_out = x_out.view(-1,num_class)
        tape["head"] = { "x_pool": x_pool, "x_fc": x_fc, "x_out": x_out, }
        return x_out, tape
    
    def get_flat_blocks(self):
        return [blk for stage in self.stages for blk in stage]
    
    def get_block_weights(self, blk):
        bnscw = blk.bnsc.weight if blk.bnsc is not None else None
        bnscb = blk.bnsc.bias if blk.bnsc is not None else None
        convscw = blk.convsc.weight if blk.convsc is not None else None
        weights = {
            "conv1w": blk.conv1.weight,
            "bn1w": blk.bn1.weight,
            "bn1b": blk.bn1.bias,
            "conv2w": blk.conv2.weight,
            "bn2w": blk.bn2.weight,
            "bn2b": blk.bn2.bias,
            "convscw": convscw,
            "bnscw": bnscw,
            "bnscb": bnscb,
        }
        return weights

    def init_dd_block_weights(self, blk):
        convscw = blk.convsc.weight if blk.convsc is not None else None
        bnscw = blk.bnsc.weight if blk.bnsc is not None else None
        bnscb = blk.bnsc.bias if blk.bnsc is not None else None
        dd_weights = {
            "ddconv1w": torch.ones_like(blk.conv1.weight),
            "ddbn1w": torch.ones_like(blk.bn1.weight),
            "ddbn1b": torch.ones_like(blk.bn1.bias),
            "ddconv2w": torch.ones_like(blk.conv2.weight),
            "ddbn2w": torch.ones_like(blk.bn2.weight),
            "ddbn2b": torch.ones_like(blk.bn2.bias),
            "ddconvscw": torch.ones_like(convscw) if convscw is not None else None,
            "ddbnscw": torch.ones_like(bnscw) if bnscw is not None else None,
            "ddbnscb": torch.ones_like(bnscb) if bnscb is not None else None,
        }
        return dd_weights
        
    def run_double_bwd(
        self,
        tape,
        d_activates_list,
        dd_weights_list,
        d_stem_tensors,
        dd_stem_tensors,
        Fuse = 1,
    ):
        flat_blocks = self.get_flat_blocks()
        stem = tape["stem"]
        head = tape["head"]
        x = stem["x_conv"]
        x_bn = stem["x_bn"]
        x_block = stem["x_block"]
        x_pool = head.pop("x_pool")
        x_fc = head.pop("x_fc")
        x_out = head.pop("x_out")

        clear_tensorlists(stem, head)
        tape["stem"] = None
        tape["head"] = None
        dx_bn = d_stem_tensors.pop("dx_bn")
        dx_block = d_stem_tensors.pop("dx_block")
        dx_out = d_stem_tensors.pop("dx_out")
        ddx_conv = dd_stem_tensors.pop("ddx_conv")
        ddconvw = dd_stem_tensors.pop("ddconvw")
        ddbnw = dd_stem_tensors.pop("ddbnw")
        ddbnb = dd_stem_tensors.pop("ddbnb")
        ddfcw = dd_stem_tensors.pop("ddfcw")
        ddfcb = dd_stem_tensors.pop("ddfcb")
        ddx_bn, dx_conv_d2, _ = conv_double_bwd( ddx_conv, ddconvw, None,dx_bn, self.conv.weight, x, groups_=Fuse)
        del dx_bn, ddx_conv, ddconvw
        # TODO: instance norm好像根本就不需要考虑fuse的问题。bn再说吧。
        dx_bn_d2, _, dd_cur = instanceNorm_double_backwards_fn( x_bn, self.bn.weight, None, ddx_bn, ddbnw, ddbnb, dx_block)
        del dx_block, ddx_bn,ddbnw, ddbnb
        dd_cur[x_block <= 0] = 0
        for i in range(len(flat_blocks)):
            blk = flat_blocks[i]
            activates_i = tape["blocks"][i]
            d_activates_i = d_activates_list[i]
            weights_i = self.get_block_weights(blk)
            dd_weights_i = dd_weights_list[i]
            dd_cur, _ = BasicBlock_double_bwd(
                activates_i, d_activates_i, weights_i, dd_weights_i, ddgrad_in=dd_cur,
                SCstride=blk.conv1.stride[0], Fuse= Fuse )
            clear_tensorlists(dd_weights_i)
            dd_weights_list[i] = None
            del activates_i, d_activates_i, weights_i, dd_weights_i

        ddx_pool = dd_cur
        del dd_cur
        # head double backward
        ddx_lin = adaptivepooling_double_bwd(ddx_pool)
        del ddx_pool
        ddx_out, dx_lin_d2, _ = linearFused_double_bwd( x_fc, self.fc.weight, dx_out, ddx_lin, ddfcw, ddfcb, Fuse )
        del dx_out, ddx_lin,ddfcw,ddfcb
        dx_out_d1 = crossEntropy_double_bwd(x_out, ddx_out, Fuse)
        del ddx_out,x_out
        dx_lin_d1, _, _ = linerFused_bwd( x_fc, self.fc.weight, grad_output=dx_out_d1, Fuse=Fuse)
        del dx_out_d1,x_fc
        dx_lin_d1 += dx_lin_d2
        del dx_lin_d2
        dx_lin_d1 = dx_lin_d1.view(x_pool.size(0), x_pool.size(1), 1, 1)
        g = adaptivepooling_bwd(x_pool, grad_output=dx_lin_d1)
        del dx_lin_d1, x_pool

        # blocks: bwd2_1, backward direction
        for i in reversed(range(len(flat_blocks))):
            blk = flat_blocks[i]
            activates_i = tape["blocks"][i]
            weights_i = self.get_block_weights(blk)
            d2_activates_i = d_activates_list[i]
            if weights_i["convscw"] is None:
                d2_activates_i.setdefault("dx_bnsc_d2", None)
            g = BasicBlock_bwd2_1( activates_i, weights_i, d2_activates_i,grad_output=g, SCstride=blk.conv1.stride[0], Fuse = Fuse)
            clear_tensorlists(activates_i, d2_activates_i, weights_i)
            tape["blocks"][i] = None
            d_activates_list[i] = None
            del activates_i, d2_activates_i, weights_i
        dx_block_d1 = g
        del g
        dx_block_d1[x_block <= 0] = 0
        del x_block
        dx_bn_d1, _, _, _, _ = instanceNorm_backward( x_bn, self.bn.weight, grad_output=dx_block_d1)
        del dx_block_d1
        dx_bn_d1 += dx_bn_d2
        del dx_bn_d2
        dx_conv, _, _ = conv_bwd( x, self.conv.weight, grad_output=dx_bn_d1,groups=Fuse)
        del dx_bn_d1
        dx_conv += dx_conv_d2

        return dx_conv


    def run_first_bwd(self, tape, target, Fuse = 1):
        flat_blocks = self.get_flat_blocks()
        # 这里只读，不 pop。double-bwd 还要继续用 tape
        stem = tape["stem"]
        head = tape["head"]
        x_conv  = stem["x_conv"]
        x_bn    = stem["x_bn"]
        x_block = stem["x_block"]
        x_pool = head["x_pool"]
        x_fc   = head["x_fc"]
        x_out  = head["x_out"]
        d_activates_list = [None] * len(flat_blocks)
        d_weights_list   = [None] * len(flat_blocks)

        dx_out = crossEntropy_bwd(x_out, target, Fuse=Fuse) 
        # dx_fc, dfcw, dfcb = linear_bwd( x_fc, self.fc.weight, grad_output=dx_out)
        dx_fc, dfcw, dfcb = linerFused_bwd( x_fc, self.fc.weight, grad_output=dx_out, Fuse= Fuse)
        dx_fc = dx_fc.view(x_pool.size(0), x_pool.size(1), 1, 1)
        g = adaptivepooling_bwd(x_pool, grad_output=dx_fc)
        del dx_fc
        for i in reversed(range(len(flat_blocks))):
            blk = flat_blocks[i]
            activates_i = tape["blocks"][i]
            weights_i = self.get_block_weights(blk)
            g, d_activates_i, d_weights_i = BasicBlock_bwd( activates_i, weights_i, grad_output=g, SCstride=blk.conv1.stride[0],Fuse=Fuse )
            d_activates_list[i] = d_activates_i
            d_weights_list[i]   = d_weights_i
            # 这里只删局部引用，不动 tape 里的本体
            del activates_i, weights_i
        dx_block = g
        del g
        dx_bn, dbnw, dbnb, dconvw = conv_norm_relu_bwd( x_conv, x_bn, x_block, self.conv.weight, self.bn.weight, grad_output=dx_block, Fuse=Fuse )
        d_stem_tensors = { "dx_bn": dx_bn, "dx_block": dx_block, "dx_out": dx_out, }
        d_weights_list_all = [dconvw, dbnw, dbnb, dfcw, dfcb]
        for d_weights_i in d_weights_list:
            if d_weights_i is None:
                continue
            for v in d_weights_i.values():
                if v is not None:
                    d_weights_list_all.append(v)
        return  d_stem_tensors, d_activates_list, d_weights_list, d_weights_list_all

    # def blocks_double_bwd(self, tape, d_activates_list, dd_weights_list, ddx_block):
    #     flat_blocks = self.get_flat_blocks()
    #     dd_cur = ddx_block
    #     d2_activates_list = [None] * len(flat_blocks)
    #     for i in range(len(flat_blocks)):
    #         blk = flat_blocks[i]
    #         activates_i = tape["blocks"][i]
    #         d_activates_i = d_activates_list[i]
    #         weights_i = self.get_block_weights(blk)
    #         dd_weights_i = dd_weights_list[i]
    #         stride = blk.conv1.stride[0]

    #         dd_cur, d2_activates_i = BasicBlock_double_bwd(
    #             activates_i,
    #             d_activates_i,
    #             weights_i,
    #             dd_weights_i,
    #             ddgrad_in=dd_cur,
    #             SCstride=stride
    #         )
    #         # d2_activates_i 和 d_activates_i 是同一个 dict
    #         d2_activates_list[i] = d2_activates_i
    #         # 这里只移除旧引用，不能 clear dict 本体
    #         d_activates_list[i] = None
    #         # dd_weights 用完了，可以清
    #         clear_tensorlists(dd_weights_i)
    #         dd_weights_list[i] = None
    #         del activates_i, d_activates_i, weights_i, dd_weights_i
    #     ddx_pool = dd_cur
    #     return ddx_pool, d2_activates_list
    

###################################
###################################
# flag = 'flex'        
flag = 'original'
flag = 'manuel'         # 只是一个写开bwd的版本。fuse 永远是1
flag = 'fused'          # Fuse 大小可以控制。
Fuse = 2
batch_size = 128
num_class=10
out_channel = 128 # in shape是写死了64， 所以out是128的话就是short cut
stride = 2
test_iter = 1
set_random_seed() # 仅仅在ACC test的时候使用。会严重影响性能。 
###################################
###################################



if __name__ == "__main__":
    print("flag = " + flag)
    if flag == 'manuel' :
        Fuse =1
    model1 = ResNet18(Fuse = Fuse).to("cuda")
    model = model1

    transform = transforms.ToTensor()
    cifar10 = torchvision.datasets.CIFAR10(root='/scratch/yguo25/files/mtt-distillation/data', train=True, download=True, transform=transform)
    # 取一个 batch
    indices = torch.arange(batch_size)   # 或 torch.randperm(len(cifar10))[:batch_size]
    imgs, labels = zip(*(cifar10[i] for i in indices))
    x = torch.stack(imgs).to("cuda").requires_grad_(True)
    target = torch.tensor(labels, device="cuda")


    # torch.save(model.state_dict(), 'model_test_resnet18_instanceNorm.pt')
    # exit()
    pretrained_dict = torch.load("model_test_resnet18_instanceNorm.pt")
    if flag =='fused':
        x = x.repeat(1, Fuse, 1, 1).detach().clone().requires_grad_()
        target = target.repeat(Fuse)
        for i,j in pretrained_dict.items():
            if j.ndim  != 0 :
                pretrained_dict[i] = j.repeat((Fuse,) + (1,) * (j.ndim - 1))         
    load_state_dict_by_position(model, pretrained_dict)



    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([x], lr=1e-1)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    # print(model.block.bn1.track_running_stats, model.block.bn2.track_running_stats)
    start = time.time()

    for step in range(test_iter):
        optimizer.zero_grad()

        if flag =='original':
            x_out,_ = model(x)
            # x_out, x_fc,x_bnsc, x_pool, x_bn2, x_conv2, x_bn1, x_block, x_bn= model(x)  # forward
            loss = criterion(x_out, target)  # compute loss
            print("----CELOSS-----", loss.item())
            dw = torch.torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
            weight_sum = [d.sum() for d in dw]
            grad_loss = sum(weight_sum)
            print("----GRANDLOSS-----", grad_loss.item())
            grad_loss.backward()  
            print("----GRAD-----", x.grad.sum().item())

        elif flag =='manuel':
            with torch.no_grad():
                x_out, tape = model(x)  # forward
                loss = criterion(x_out, target)  
                print("----CELOSS-----", loss.item())
                del x_out

                d_stem_activates, d_activates_list, d_weights_list, d_weights_list_all = model.run_first_bwd( tape=tape, target=target )

                weight_sum = [d.sum() for d in d_weights_list_all]
                grad_loss = sum(weight_sum)
                print("----GRANDLOSS-----", grad_loss.item())

                ddconvw = torch.ones_like(model.conv.weight).cuda()
                ddbnw = torch.ones_like(model.bn.weight).cuda()
                ddbnb = torch.ones_like(model.bn.bias).cuda()
                ddfcw = torch.ones_like(model.fc.weight).cuda()
                ddfcb = torch.ones_like(model.fc.bias).cuda()
                ddx_conv = torch.zeros_like(x).cuda()
                flat_blocks = model.get_flat_blocks()
                dd_weights_list = [model.init_dd_block_weights(blk) for blk in flat_blocks]
                dd_stem_tensors = {
                    "ddconvw": ddconvw,
                    "ddbnw": ddbnw,
                    "ddbnb": ddbnb,
                    "ddfcw": ddfcw,
                    "ddfcb": ddfcb,
                    "ddx_conv": ddx_conv
                }
                del ddconvw,ddbnw,ddbnb,ddfcw,ddfcb,ddx_conv

                dx_conv= model.run_double_bwd(
                    tape=tape,
                    d_activates_list=d_activates_list,
                    dd_weights_list=dd_weights_list,
                    d_stem_tensors=d_stem_activates,
                    dd_stem_tensors=dd_stem_tensors,
                    Fuse=1)
                print("----GRAD-----", dx_conv.sum().item())

        # 区别就是fuse 写死了是1，这里写成了fuse
        # 然后fuse>1 之后，有一些view不一样。
        elif flag =='fused':
            with torch.no_grad():
                x_out, tape = model(x)  # forward
                # x_out = x_out.view(-1,num_class)
                # tape["head"]['x_out'] = x_out #TODO:  因为tape里的没有做view。 所以这里也要做一下。后面看考虑直接放在fwd里面吧
                loss = criterion(x_out, target)  
                loss*=Fuse 
                print("----CELOSS-----", loss.item())
                del x_out
                d_stem_activates, d_activates_list, d_weights_list, d_weights_list_all = model.run_first_bwd( tape=tape, target=target ,Fuse=Fuse)
                weight_sum = [d.sum() for d in d_weights_list_all]
                grad_loss = sum(weight_sum)
                print("----GRANDLOSS-----", grad_loss.item())
                ddconvw = torch.ones_like(model.conv.weight).cuda()
                ddbnw = torch.ones_like(model.bn.weight).cuda()
                ddbnb = torch.ones_like(model.bn.bias).cuda()
                ddfcw = torch.ones_like(model.fc.weight).cuda()
                ddfcb = torch.ones_like(model.fc.bias).cuda()
                ddx_conv = torch.zeros_like(x).cuda()
                flat_blocks = model.get_flat_blocks()
                dd_weights_list = [model.init_dd_block_weights(blk) for blk in flat_blocks]
                dd_stem_tensors = {
                    "ddconvw": ddconvw,
                    "ddbnw": ddbnw,
                    "ddbnb": ddbnb,
                    "ddfcw": ddfcw,
                    "ddfcb": ddfcb,
                    "ddx_conv": ddx_conv
                }
                del ddconvw,ddbnw,ddbnb,ddfcw,ddfcb,ddx_conv
                dx_conv= model.run_double_bwd( tape=tape, d_activates_list=d_activates_list, dd_weights_list=dd_weights_list,
                    d_stem_tensors=d_stem_activates, dd_stem_tensors=dd_stem_tensors, Fuse=Fuse)
                print("----GRAD-----", dx_conv.sum().item())
            
        optimizer.step()  # update x

    end = time.time()

    print("当前显存使用:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("峰值显存使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    print("时间占用：", end-start)