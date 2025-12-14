# 11/16 : 
# CONV_NET
# 完成了flex 之后的forward + backward(statteless) + doublebwd(stateless)
# 从base_flexModel 复制而来。
# LinearStacked_2 做了一些变动。因为不想影响已经调试好的结果。所以这里新import 一个。

# 最早测试用的文件。后面可能逐渐弃用。

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
from networks_stateless_basicblock import linear_bwd, conv_bwd, insNormNRelu_bwd, \
linear_double_bwd, conv_double_bwd, insNormNRelu_double_bwd, avgPool_bwd, \
    avgPool_double_bwd,bmm_bwd, linerFused_bwd, linearFused_double_bwd, crossEntropy_bwd

from networks_fused3 import NormActive
# from networks_fused2 import NormActive



# class Conv_Flexfuse(nn.Module):
#     def __init__(self, channel=3, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='maxpooling', im_size = (32,32), Fuse=2):
#         super(Conv_Flexfuse, self).__init__()
#         self.Fuse = Fuse
#         self.conv1 = nn.Conv2d(in_channels=channel*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
#         self.norm1 = NormActive(net_width*Fuse) #BN在channel上单独计算，所以目前不用管。
#         self.pool1 = nn.AvgPool2d(kernel_size=2)
#         self.conv2 = nn.Conv2d(in_channels=net_width*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
#         self.norm2 = NormActive(net_width*Fuse) #BN在channel上单独计算，所以目前不用管。
#         self.pool2 = nn.AvgPool2d(kernel_size=2)
#         self.conv3 = nn.Conv2d(in_channels=net_width*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
#         self.norm3 = NormActive(net_width*Fuse) #BN在channel上单独计算，所以目前不用管。
#         self.pool3 = nn.AvgPool2d(kernel_size=2)
#         self.linear = LinearStacked_2(net_width * 4 * 4, num_classes,Fuse )
#         self.net_width= net_width

#     def forward(self, x_conv1):
#         x_conv1 = x_conv1.view(-1,self.Fuse*3 ,32,32)        # 10, 256, 4,4   -> 20, 2048
#         x_norm1 = self.conv1(x_conv1)          
#         x_pool1 = self.norm1(x_norm1)    
#         x_conv2 = self.pool1(x_pool1)
#         # x_conv2 = x_conv1.view(-1,self.Fuse*3 ,32,32)        # 10, 256, 4,4   -> 20, 2048
#         x_norm2 = self.conv2(x_conv2)          
#         x_pool2 = self.norm2(x_norm2)    
#         x_conv3 = self.pool2(x_pool2)
#         # x_conv3 = x_conv1.view(-1,self.Fuse*3 ,32,32)        # 10, 256, 4,4   -> 20, 2048
#         x_norm3 = self.conv3(x_conv3)          
#         x_pool3 = self.norm3(x_norm3)    
#         x_lin  = self.pool3(x_pool3)
#         x_out = self.linear(x_lin)    # N x 10
#         x_out = x_out.view(-1,10)
#         return  x_conv1,x_norm1, x_pool1,x_conv2,x_norm2, x_pool2,x_conv3,x_norm3, x_pool3, x_lin,x_out


class Conv_Flexfuse(nn.Module):
    def __init__(self, channel=3, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='maxpooling', im_size = (32,32), Fuse=2):
        super(Conv_Flexfuse, self).__init__()
        self.Fuse = Fuse
        self.conv1 = nn.Conv2d(in_channels=channel*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        self.norm1 = nn.InstanceNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        # self.norm1 = nn.GroupNorm(net_width*Fuse,net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=net_width*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        self.norm2 = nn.InstanceNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        # self.norm2 = nn.GroupNorm(net_width*Fuse,net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=net_width*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        self.norm3 = nn.InstanceNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        # self.norm3 = nn.GroupNorm(net_width*Fuse,net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool3 = nn.AvgPool2d(kernel_size=2)
        self.linear = LinearStacked_2(net_width * 4 * 4, num_classes,Fuse )
        self.net_width= net_width

    def forward(self, x_conv1):
        x_conv1 = x_conv1.view(-1,self.Fuse*3 ,32,32)        # 10, 256, 4,4   -> 20, 2048
        x_norm1 = self.conv1(x_conv1)          
        x_pool1 = F.relu(self.norm1(x_norm1)    )
        x_conv2 = self.pool1(x_pool1)
        x_norm2 = self.conv2(x_conv2)          
        x_pool2 = F.relu(self.norm2(x_norm2) )   
        x_conv3 = self.pool2(x_pool2)
        x_norm3 = self.conv3(x_conv3)          
        x_pool3 = F.relu(self.norm3(x_norm3))    
        x_lin  = self.pool3(x_pool3)
        x_out = self.linear(x_lin)    # N x 10
        x_out = x_out.view(-1,10)
        return  x_conv1,x_norm1, x_pool1,x_conv2,x_norm2, x_pool2,x_conv3,x_norm3, x_pool3, x_lin,x_out


# def ConvBlock_bwd2_1(x_conv, x_norm, x_pool, conv_w, norm_w, dx_lin_d1, dx_norm_d2=torch.zeros([1]).cuda() ,dxconv_d2 =torch.zeros([1]).cuda(), Fuse =2):
#     # 一个conv block的bwd conv+norm+relu+pool 。 如果是2_1 的话，中间有个累加项
#     # print(x_pool.shape)
#     # print(dx_lin_d1.shape)
#     dx_pool_d1 = avgPool_bwd( x_pool, grad_output= dx_lin_d1 )
#     dx_norm_d1, d_norm_weight, d_norm_bias = insNormNRelu_bwd(x_norm, norm_w, x_pool, grad_output=dx_pool_d1)
#     dx_norm_d2 += dx_norm_d1
#     del dx_pool_d1,dx_norm_d1
#     dxconv_d1, d_conv_weight_d1, d_conv_bias_d1 = conv_bwd(x_conv, conv_w, grad_output=dx_norm_d2, groups=Fuse)
#     del dx_norm_d2
#     dxconv_d2 += dxconv_d1
    
#     return dxconv_d2, d_conv_weight_d1, d_conv_bias_d1 , d_norm_weight, d_norm_bias
    


def ConvBlock_double_bwd(x_conv, x_norm, x_pool, dx_norm, dx_pool, ddx_conv, conv_w, norm_w,ddcon_w, ddconv_b, ddnorm_w, ddnorm_b, Fuse=2):
    ddx_norm, dxconv_d2, dconvw_d2 = conv_double_bwd(ddx_conv, ddcon_w, ddconv_b, dx_norm, conv_w, x_conv, groups_=Fuse )
    ddx_pool, dx_norm_d2, dnormw_d2 = insNormNRelu_double_bwd(ddx_norm, ddnorm_w, ddnorm_b, dx_pool, x_pool, norm_w, x_norm)
    del ddx_norm
    ddx_lin = avgPool_double_bwd(ddx_pool)

    return ddx_lin,dxconv_d2, dconvw_d2,dx_norm_d2,dnormw_d2

def Conv3_double_bwd():
    pass