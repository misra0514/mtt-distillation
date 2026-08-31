# 11.24
# conv block double bwd， conv model bwd, basic block bwd & double bwd （一些稍微大一点的组成元素）

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

from networks.networks_stateless_basicblock import linear_bwd, conv_bwd, insNormNRelu_bwd, \
linear_double_bwd, conv_double_bwd, insNormNRelu_double_bwd, avgPool_bwd, \
    avgPool_double_bwd,bmm_bwd, linerFused_bwd, linearFused_double_bwd, crossEntropy_bwd,crossEntropy_double_bwd,\
    insNorm_bwd ,instanceNorm_double_backwards_fn
from script_fuseParallel.base import instanceNorm_backward


def ConvBlock_bwd1_2(x_conv, x_norm, x_pool, conv_w, norm_w, dx_lin_d1, Fuse =2, v_fuse = True):
    # double bwd 的bwd阶段。区别与1-2的主要特点是有dx_norm_d2？ 然后dxnorm 和dxpool 也需要
    dx_pool_d1 = avgPool_bwd( x_pool, grad_output= dx_lin_d1 )
    dx_norm_d1, d_norm_weight, d_norm_bias = insNormNRelu_bwd(x_norm, norm_w, x_pool, grad_output=dx_pool_d1, v_fuse=v_fuse)
    # del dx_pool_d1
    dx_conv_d1, d_conv_weight_d1, d_conv_bias_d1 = conv_bwd(x_conv, conv_w, grad_output=dx_norm_d1, groups=Fuse)
    # del dx_norm_d1
    
    return dx_conv_d1, dx_norm_d1, dx_pool_d1, d_conv_weight_d1, d_conv_bias_d1 , d_norm_weight, d_norm_bias


def ConvBlock_bwd2_1(x_conv, x_norm, x_pool, conv_w, norm_w, dx_lin_d1, dx_norm_d2=torch.zeros([1]).cuda() ,dxconv_d2 =torch.zeros([1]).cuda(), Fuse =2, v_fuse=True):
    # double bwd 的bwd阶段。区别与1-2的主要特点是有dx_norm_d2？
    dx_pool_d1 = avgPool_bwd( x_pool, grad_output= dx_lin_d1 )
    dx_norm_d1, d_norm_weight, d_norm_bias = insNormNRelu_bwd(x_norm, norm_w, x_pool, grad_output=dx_pool_d1, v_fuse=v_fuse)
    del dx_pool_d1
    dx_norm_d1 += dx_norm_d2
    dx_conv_d1, d_conv_weight_d1, d_conv_bias_d1 = conv_bwd(x_conv, conv_w, grad_output=dx_norm_d1, groups=Fuse)
    del dx_norm_d1
    dx_conv_d1 += dxconv_d2
    
    return dx_conv_d1, d_conv_weight_d1, d_conv_bias_d1 , d_norm_weight, d_norm_bias

def ConvBlock_bwd_full(x_conv, x_norm, x_pool, conv_w, norm_w, dx_lin_d1, dx_norm_d2=torch.zeros([1]).cuda() ,dxconv_d2 =torch.zeros([1]).cuda(), Fuse =2, v_fuse=True):
    # 2-1 和 1-2 的结合体。 也就是把dx_norm_d2 和 dxconv_d2 都加上了。 
    # dx_norm_d2 和 dxconv_d2 应该提前配置好shape？
    dx_pool_d1 = avgPool_bwd( x_pool, grad_output= dx_lin_d1 )
    dx_norm_d1, d_norm_weight, d_norm_bias = insNormNRelu_bwd(x_norm, norm_w, x_pool, grad_output=dx_pool_d1, v_fuse=v_fuse)
    del dx_pool_d1
    dx_norm_d1 += dx_norm_d2
    dx_conv_d1, d_conv_weight_d1, d_conv_bias_d1 = conv_bwd(x_conv, conv_w, grad_output=dx_norm_d1, groups=Fuse)
    del dx_norm_d1
    dx_conv_d1 += dxconv_d2
    
    return dx_conv_d1, d_conv_weight_d1, d_conv_bias_d1 , d_norm_weight, d_norm_bias
    


    

def ConvBlock_double_bwd(x_conv, x_norm, x_pool, dx_norm, dx_pool, ddx_conv, conv_w, norm_w,ddcon_w, ddconv_b, ddnorm_w, ddnorm_b, Fuse=2, v_fuse=True):
    ddx_norm, dxconv_d2, dconvw_d2 = conv_double_bwd(ddx_conv, ddcon_w, ddconv_b, dx_norm, conv_w, x_conv, groups_=Fuse )
    ddx_pool, dx_norm_d2, dnormw_d2 = insNormNRelu_double_bwd(ddx_norm, ddnorm_w, ddnorm_b, dx_pool, x_pool, norm_w, x_norm, v_fuse=v_fuse)
    del ddx_norm
    ddx_lin = avgPool_double_bwd(ddx_pool)

    return ddx_lin,dxconv_d2, dconvw_d2,dx_norm_d2,dnormw_d2


def conv3_bwd(
    x_lin,
    dx_out_d1,
    dx_lin_d2,
    x_conv3, x_norm3, x_pool3,
    dx_norm3_d2, dxconv3_d2,
    x_conv2, x_norm2, x_pool2,
    dx_norm2_d2, dxconv2_d2,
    x_conv1, x_norm1, x_pool1,
    dx_norm1_d2, dxconv1_d2,
    model,
    Fuse
):
    # 使用之后从2737 继续上涨至3141.177734375。需要注意！！
    dx_lin_d1, _, _ = linerFused_bwd(x_lin, model.linear.weight, grad_output=dx_out_d1, Fuse=Fuse)
    del x_lin,dx_out_d1
    dx_lin_d1 = dx_lin_d1.reshape(-1, 32 * Fuse, 4,4) 
    dx_lin_d1 += dx_lin_d2 
    dx_conv3_d1 , _ ,_ ,_ ,_ = ConvBlock_bwd2_1(x_conv3, x_norm3, x_pool3, model.conv3.weight, model.norm3.weight, dx_lin_d1,dx_norm3_d2,dxconv3_d2 ,Fuse=Fuse)
    del dx_norm3_d2,dxconv3_d2, x_conv3, x_norm3,x_pool3,dx_lin_d1

    dx_conv2_d1 , _ ,_ ,_ ,_ = ConvBlock_bwd2_1(x_conv2, x_norm2, x_pool2, model.conv2.weight, model.norm2.weight, dx_conv3_d1,dx_norm2_d2,dxconv2_d2 ,Fuse=Fuse)
    del dx_norm2_d2,dxconv2_d2, x_conv2, x_norm2,x_pool2,dx_conv3_d1

    dx_conv1_d1 , _ ,_ ,_ ,_ = ConvBlock_bwd2_1(x_conv1, x_norm1, x_pool1, model.conv1.weight, model.norm1.weight, dx_conv2_d1,dx_norm1_d2,dxconv1_d2 ,Fuse=Fuse)
    del dx_norm1_d2,dxconv1_d2, x_conv1, x_norm1,x_pool1, dx_conv2_d1
    return dx_conv1_d1


# 使用这个函数的话，mem 从2717 --> 2737.099609375（有上涨。）
def conv3_double_bwd(x_conv1,x_norm1, x_pool1,x_conv2,x_norm2, x_pool2,x_conv3,x_norm3, x_pool3, x_lin,x_out ,dx_norm1, dx_norm2,dx_norm3,dx_pool1,dx_pool2,dx_pool3,
                     dconv1_w,dconv1_b,dnorm1_w,dnorm1_b,dconv2_w,dconv2_b,dnorm2_w,dnorm2_b,dconv3_w,dconv3_b,dnorm3_w,dnorm3_b,dlin_w,ddlin_b,ddx_conv,dx_out,
                      model, Fuse=2 ,v_fuse=True):
    ddx_conv2, dxconv1_d2, _,dx_norm1_d2,_ = ConvBlock_double_bwd(x_conv1, x_norm1, x_pool1, dx_norm1, dx_pool1, \
                                                                    ddx_conv, model.conv1.weight, model.norm1.weight,dconv1_w*2, dconv1_b*2, dnorm1_w*2, dnorm1_b*2,Fuse  )
    del ddx_conv,dx_norm1,dx_pool1,dconv1_w,dconv1_b,dnorm1_w,dnorm1_b

    ddx_conv3, dxconv2_d2, _,dx_norm2_d2,_ = ConvBlock_double_bwd(x_conv2, x_norm2, x_pool2, dx_norm2, dx_pool2, \
                                                                    ddx_conv2, model.conv2.weight, model.norm2.weight,dconv2_w*2, dconv2_b*2, dnorm2_w*2, dnorm2_b*2,Fuse  )
    del ddx_conv2,dx_norm2,dx_pool2

    ddx_lin, dxconv3_d2, _,dx_norm3_d2,_ = ConvBlock_double_bwd(x_conv3, x_norm3, x_pool3, dx_norm3, dx_pool3, \
                                                                ddx_conv3, model.conv3.weight, model.norm3.weight,dconv3_w*2, dconv3_b*2, dnorm3_w*2, dnorm3_b*2,Fuse  )
    del ddx_conv3,dx_norm3,dx_pool3

    ddx_out, dx_lin_d2, _ = linearFused_double_bwd(x_lin,model.linear.weight, dx_out, ddx_lin, dlin_w*2 ,ddlin_b*2 ,Fuse)
    del dx_out, ddx_lin
    dx_out_d1 = crossEntropy_double_bwd(x_out, ddx_out, Fuse)
    del ddx_out,x_out

    return dx_out_d1, dx_norm1_d2, dx_norm2_d2, dx_norm3_d2, dxconv1_d2, dxconv2_d2, dxconv3_d2,dx_lin_d2




def BasicBlock_bwd( activates, weights, grad_output,SCstride=1, Fuse=1, v_fuse=True):
    # v_fuse=False
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
    dx_bn2, dbn2w, dbn2b, _, _ = insNorm_bwd(x_bn2, bn2w, grad_output=grad_output, v_fuse=v_fuse)
    dx_conv2, dconv2w, _ = conv_bwd(x_conv2, conv2w, grad_output=dx_bn2, groups=Fuse)
    # dbno1 = dx_conv2
    # dbno1[x_conv2 <= 0] = 0
    # dx_bn1, dbn1w, dbn1b, _, _ = instanceNorm_backward(x_bn1, bn1w, grad_output=dbno1)
    dx_bn1, dbn1w, dbn1b = insNormNRelu_bwd(x_bn1, bn1w, x_conv2, grad_output=dx_conv2, v_fuse=v_fuse)
    
    
    dx_main, dconv1w, _ = conv_bwd(x_conv1, conv1w, grad_output=dx_bn1, stride=SCstride, groups=Fuse)
    dbno2 = grad_output
    # ---------------- shortcut branch ----------------
    if has_downsample:
        dbnosc = grad_output
        dx_bnsc, dbnscw, dbnscb, _, _ = insNorm_bwd( x_bnsc, bnscw, grad_output=dbnosc, v_fuse=v_fuse)
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
        "dbno1": dx_conv2,
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
    grad_output, SCstride=1, Fuse = 1, v_fuse=True
):
    # v_fuse=False
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
    dx_bn2, dbn2w, dbn2b, _, _ = insNorm_bwd( x_bn2, bn2w, grad_output=grad_output, v_fuse=v_fuse )
    dx_bn2 += dx_bn2_d2
    dx_conv2, dconv2w, _ = conv_bwd( x_conv2, conv2w, grad_output=dx_bn2, groups=Fuse)
    dx_conv2 = dx_conv2.clone()
    dx_conv2 += dx_conv2_d2 # TODO: 应该需要先加法再relu。 每一个bwd2_1结束后需要立刻合并dbwd梯度。
    # dx_conv2[x_conv2 <= 0] = 0
    # dx_bn1, dbn1w, dbn1b, _, _ = instanceNorm_backward( x_bn1, bn1w, grad_output=dx_conv2 )
    dx_bn1, dbn1w, dbn1b = insNormNRelu_bwd(x_bn1, bn1w, x_conv2, grad_output=dx_conv2, v_fuse=v_fuse)

    dx_bn1 += dx_bn1_d2
    del dx_bn1_d2
    dx_conv1_main, dconv1w, _ = conv_bwd( x_conv1, conv1w, grad_output=dx_bn1, stride=SCstride,groups= Fuse )
    if has_downsample:
        dx_bnsc, dbnscw, dbnscb, _, _ = insNorm_bwd( x_bnsc, bnscw, grad_output=grad_output, v_fuse=v_fuse )
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
    ddgrad_in, SCstride=1, Fuse=1, v_fuse=True
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
    # dx_bn1_d2, dbn1w_d2, ddx_conv2 = instanceNorm_double_backwards_fn(
    #     x_bn1, bn1w, None, ddx_bn1, ddbn1w, ddbn1b, d_activates["dbno1"], 1e-5 )
    # # del dbno1
    # d_activates.pop("dbno1")
    # d_activates["dx_bn1_d2"] = dx_bn1_d2
    # ddx_conv2[x_conv2 <= 0] = 0
    ddx_conv2, dx_bn1_d2, dbn1w_d2 = insNormNRelu_double_bwd(ddx_bn1, ddbn1w, ddbn1b, d_activates["dbno1"], x_conv2\
    , bn1w, x_bn1, v_fuse=v_fuse)
    d_activates.pop("dbno1")
    d_activates["dx_bn1_d2"] = dx_bn1_d2



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

def conv_norm_relu_bwd(x, x_bn,x_block,convw, bnw, grad_output, Fuse = 1  , v_fuse=True):
    dx_bn, dbnw, dbnb = insNormNRelu_bwd(x_bn, bnw,output=x_block, grad_output=grad_output, v_fuse=v_fuse)
    _, dconvw ,_ = conv_bwd(x, convw, grad_output=dx_bn, groups= Fuse)
    return dx_bn, dbnw, dbnb, dconvw

def conv_norm_relu_double_bwd(x, x_bn,x_block,dx_bn,dx_block,ddx_conv,  convw ,bnw,ddconvw,ddbnw,ddbnb):
    ddx_bn, dx_conv_d2, dconvw_d2 = conv_double_bwd(ddx_conv,ddconvw,None,dx_bn, convw, x )
    del dx_bn, ddx_conv
    dx_bn_d2, dbnw_d2, ddx_block = instanceNorm_double_backwards_fn(x_bn,bnw, None ,ddx_bn,ddbnw,ddbnb,dx_block)
    del dx_block, ddx_bn # 可以试试用覆盖的话，这里就不用单独del了更加工整内存也更好。
    ddx_block[x_block<=0] = 0
    return  ddx_block, dx_bn_d2, dx_conv_d2


