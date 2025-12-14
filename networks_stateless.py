# 11.24
# conv block double bwd， conv model bwd （一些稍微大一点的组成元素）

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
from networks_stacked_basicblock import LinearStacked_2_flexFuse as linear_flex
from networks_stacked_basicblock import LinearStacked_2
from networks_fused3 import NormActive
from networks_stateless_basicblock import linear_bwd, conv_bwd, insNormNRelu_bwd, \
linear_double_bwd, conv_double_bwd, insNormNRelu_double_bwd, avgPool_bwd, \
    avgPool_double_bwd,bmm_bwd, linerFused_bwd, linearFused_double_bwd, crossEntropy_bwd,crossEntropy_double_bwd


def ConvBlock_bwd1_2(x_conv, x_norm, x_pool, conv_w, norm_w, dx_lin_d1, Fuse =2):
    # double bwd 的bwd阶段。区别与1-2的主要特点是有dx_norm_d2？ 然后dxnorm 和dxpool 也需要
    dx_pool_d1 = avgPool_bwd( x_pool, grad_output= dx_lin_d1 )
    # dx_lin_d1.copy_(dx_lin_d1[:, :, ...].contiguous())
    dx_norm_d1, d_norm_weight, d_norm_bias = insNormNRelu_bwd(x_norm, norm_w, x_pool, grad_output=dx_pool_d1)
    # del dx_pool_d1
    dx_conv_d1, d_conv_weight_d1, d_conv_bias_d1 = conv_bwd(x_conv, conv_w, grad_output=dx_norm_d1, groups=Fuse)
    # del dx_norm_d1
    
    return dx_conv_d1, dx_norm_d1, dx_pool_d1, d_conv_weight_d1, d_conv_bias_d1 , d_norm_weight, d_norm_bias


def ConvBlock_bwd2_1(x_conv, x_norm, x_pool, conv_w, norm_w, dx_lin_d1, dx_norm_d2=torch.zeros([1]).cuda() ,dxconv_d2 =torch.zeros([1]).cuda(), Fuse =2):
    # double bwd 的bwd阶段。区别与1-2的主要特点是有dx_norm_d2？
    dx_pool_d1 = avgPool_bwd( x_pool, grad_output= dx_lin_d1 )
    dx_norm_d1, d_norm_weight, d_norm_bias = insNormNRelu_bwd(x_norm, norm_w, x_pool, grad_output=dx_pool_d1)
    del dx_pool_d1
    dx_norm_d1 += dx_norm_d2
    dx_conv_d1, d_conv_weight_d1, d_conv_bias_d1 = conv_bwd(x_conv, conv_w, grad_output=dx_norm_d1, groups=Fuse)
    del dx_norm_d1
    dx_conv_d1 += dxconv_d2
    
    return dx_conv_d1, d_conv_weight_d1, d_conv_bias_d1 , d_norm_weight, d_norm_bias
    

def ConvBlock_double_bwd(x_conv, x_norm, x_pool, dx_norm, dx_pool, ddx_conv, conv_w, norm_w,ddcon_w, ddconv_b, ddnorm_w, ddnorm_b, Fuse=2):
    ddx_norm, dxconv_d2, dconvw_d2 = conv_double_bwd(ddx_conv, ddcon_w, ddconv_b, dx_norm, conv_w, x_conv, groups_=Fuse )
    ddx_pool, dx_norm_d2, dnormw_d2 = insNormNRelu_double_bwd(ddx_norm, ddnorm_w, ddnorm_b, dx_pool, x_pool, norm_w, x_norm)
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
                      model, Fuse=2 ):
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



