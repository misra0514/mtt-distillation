# 11.1 conv 模型已经更新成了更高的fuse 版本。现在rundouble bwd 这个字段只做bwd。不确定所有模型都更新了

# 4.27
# 现在的model 应该都可以把Fuse 写好了。唯一的区别就是 flex fuse 有没有做以及有没有封装到model 里面。所以现在准备重新整理一下这些网络。这里希望可以包含Conv3 、 Resnet18（resnet18test file 复制）
# 这里是完整的大模型, 来源主要有两个： stateless & stateless_basic block 。两个分别是一个封装后的模块，一个是原始的函数接口。

import operator
from torch.nn.modules.utils import _pair

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import torchvision.transforms as transforms
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

# from networks.networks_stacked import GroupedLinear # NOTE: 这里取消注释了。因为Fuse->stacked->stacked_basicblock 太长了。不好。
from networks.networks_basicblock_tritonwrapper import batchNorm2d_backward, batchnorm_double_backwards_fn, batchnorm_double_backwards_fn_new
from networks.networks_basicblock_tritonwrapper import instanceNorm_backward ,instanceNorm_double_backwards_fn
from networks.utils import clear_tensorlists
from networks.networks_stateless import BasicBlock_double_bwd,BasicBlock_bwd,BasicBlock_bwd2_1, conv_norm_relu_bwd
from networks.networks_basicblock_tritonwrapper import Fst_Order_NormActive # fuse+基本优化
from networks.networks_stateless_basicblock import insNorm_bwd, linear_bwd, conv_bwd, insNormNRelu_bwd, \
linear_double_bwd, conv_double_bwd, insNormNRelu_double_bwd, avgPool_bwd, crossEntropy_bwd, \
    avgPool_double_bwd,bmm_bwd, linerFused_bwd, linearFused_double_bwd,crossEntropy_double_bwd, \
    adaptivepooling_bwd, adaptivepooling_double_bwd,\
    grouped_layernorm_double_bwd_fn, grouped_linear_double_bwd,gelu_double_bwd,sdpa_no_mask_no_dropout_double_bwd,\
    grouped_layernorm_backward, grouped_linear_bwd, gelu_bwd,sdpa_no_mask_no_dropout_bwd


from networks.networks_stateless import  ConvBlock_double_bwd,ConvBlock_bwd2_1,ConvBlock_bwd1_2,conv3_double_bwd,conv3_bwd
from networks.networks_stateless_basicblock import linear_bwd, conv_bwd, insNormNRelu_bwd, \
linear_double_bwd, conv_double_bwd, insNormNRelu_double_bwd, avgPool_bwd, crossEntropy_bwd, \
    avgPool_double_bwd,bmm_bwd, linerFused_bwd, linearFused_double_bwd,crossEntropy_double_bwd,\
        dropout_fwd,dropout_bwd, dropout_double_bwd,geluDropout_bwd,geluDropout_double_bwd

from utils_flex import build_global_group_mask, fuse_params_with_mask,split_half_snd_dim,recover_params,set_random_seed


from networks.grouped_kernels.grouped_conv_double_bwd import grouped_conv2d
from networks.networks_stateless_basicblock import grouped_instancenorm_fwd as groupd_instance_norm



def _ff_to_bf(x, fuse):
    """[F*B,C,...] -> [B,F*C,...]."""
    if x.shape[0] % fuse:
        raise ValueError(f"FF leading dimension {x.shape[0]} not divisible by Fuse={fuse}")
    b = x.shape[0] // fuse
    return x.reshape(fuse, b, *x.shape[1:]).permute(1, 0, *range(2, x.ndim + 1)).reshape(b, fuse * x.shape[1], *x.shape[2:])


def _bf_to_ff(x, fuse):
    """[B,F*C,...] -> [F*B,C,...]."""
    if x.shape[1] % fuse:
        raise ValueError(f"BF channel dimension {x.shape[1]} not divisible by Fuse={fuse}")
    b, c = x.shape[:2]
    return x.reshape(b, fuse, c // fuse, *x.shape[2:]).permute(1, 0, *range(2, x.ndim + 1)).reshape(fuse * b, c // fuse, *x.shape[2:])


def _layout_conv(module, x, layout, fuse):
    if layout == "f_f":
        return _bf_to_ff(module(_ff_to_bf(x, fuse)), fuse)
    return module(x)


class GroupedLayerNorm(nn.Module):
    def __init__(self, embed_dim, Fuse=1, eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.Fuse = Fuse
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(Fuse * embed_dim))
        self.bias = nn.Parameter(torch.zeros(Fuse * embed_dim))

    def forward(self, x):
        # x: [B, Fuse, N, D]
        B, Fs, N, D = x.shape
        assert Fs == self.Fuse
        assert D == self.embed_dim
        # 只在最后一维 D 上做 LayerNorm
        x = F.layer_norm( x, normalized_shape=(D,), weight=None, bias=None, eps=self.eps, )
        weight = self.weight.view(1, self.Fuse, 1, self.embed_dim)
        bias = self.bias.view(1, self.Fuse, 1, self.embed_dim)
        x = x * weight + bias
        return x

# class LinearStacked_2(nn.Module):
#     # batch* fusion * In。 --> fusion,Batch ,Out
#     def __init__(self ,in_features, out_features, Fuse):
#         super(LinearStacked_2, self).__init__()
#         self.Fuse = Fuse
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = torch.nn.Parameter(torch.randn(Fuse* out_features,in_features))
#         self.bias = torch.nn.Parameter(torch.randn(self.Fuse* out_features))

#     def forward(self, x):
#         x = x.view(-1,self.Fuse, self.in_features)
#         # TODO: 这里输入如果是BAD/（而不是ABD）的话，可以得到is_contiguous 的结果。那就很简单只要调整target即可
#         # 现在是BAD，意味着Fusion，batch的排序，B到了第一位
#         # 因为显然有bug，所以后续想慢慢换掉这个接口。加了一个dim==3。遇到问题再说吧
#         if x.ndim == 3:
#             x = torch.einsum("abc,bcd->bad",x,self.weight.view(self.Fuse,self.out_features,self.in_features).transpose(-1, -2)) 
#             x = x+self.bias.view(self.Fuse,1 ,self.out_features)
#             x = x.squeeze(0) #加一个squeeze，为了适配在Fuse=1的时候的模型。
#             return x



# class GroupedLinear(nn.Module):
#     def __init__(self, in_features, out_features, Fuse):
#         super().__init__()
#         self.Fuse = Fuse
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(
#             torch.randn(Fuse * out_features, in_features)
#         )
#         self.bias = nn.Parameter(
#             torch.randn(Fuse * out_features)
#         )

#     def forward(self, x, setup='vit'):
#         W = self.weight.view(
#             self.Fuse, self.out_features, self.in_features
#         ).transpose(1, 2)  # [G,I,O]

#         if setup == 'conv':
#             gb = x.shape[0]
#             X = x.reshape(gb, -1)
#             if gb % self.Fuse != 0:
#                 raise ValueError('Input batch must be divisible by Fuse')
#             if X.shape[1] != self.in_features:
#                 raise ValueError(
#                     f'Expected {self.in_features} features, got {X.shape[1]}'
#                 )
#             X = X.reshape(self.Fuse, gb // self.Fuse, self.in_features)
#             out = torch.bmm(X, W)
#             out = out + self.bias.view(self.Fuse, 1, self.out_features)
#             return out.reshape(gb, self.out_features)

#         elif setup == 'vit':
#             # if x.ndim == 4:
#             # [B,G,N,I] -> [B,G,N,O]
#             out = torch.einsum("bfni,fio->bfno", x, W)
#             return out + self.bias.view( 1, self.Fuse, 1, self.out_features)
#             # elif x.ndim == 3:
#             #     # 保留原来的三维输入兼容
#             #     X = x.view(-1, self.Fuse, self.in_features)
#             #     out = torch.einsum("bfi,fio->bfo", X, W)
#             #     return out + self.bias.view(
#             #         1, self.Fuse, self.out_features
#             #     )
#             raise ValueError('vit input must be [B,G,I] or [B,G,N,I]')

#         raise ValueError(f'Unsupported setup: {setup}')


# 新的grouped linear包含两种layout+vit。
# bwd可能还是各自搞各自的
class GroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, Fuse,
                 layout='b_f'):
        super().__init__()

        self.Fuse = Fuse
        self.in_features = in_features
        self.out_features = out_features
        self.layout = layout

        self.weight = nn.Parameter(
            torch.randn(Fuse * out_features, in_features)
        )
        self.bias = nn.Parameter(
            torch.randn(Fuse * out_features)
        )

    def forward(self, x, setup='vit', layout=None):
        layout = self.layout if layout is None else layout
        F = self.Fuse
        I = self.in_features
        O = self.out_features
        W = self.weight.view(F, O, I).transpose(1, 2)
        # W: [F, I, O]
        bias = self.bias.view(F, O)
        if setup == 'vit':
            # [B, F, N, I] -> [B, F, N, O]

            if x.ndim == 3:
                return torch.einsum("bfi,fio->bfo", x, W) + bias.view(1, F, O)
            if x.ndim == 4:
                return torch.einsum("bfni,fio->bfno", x, W) + bias.view(1, F, 1, O)
            raise ValueError("vit input must be 3D or 4D")
        elif setup == 'conv':
            if layout == 'b_f':
                # Original Conv / ResNet layout
                # [B, F*I] -> [B, F, I]
                if math.prod(x.shape[1:]) != F * I:
                    raise ValueError(f"Expected {F * I} features per BF sample, got {math.prod(x.shape[1:])}")
                X = x.reshape(-1, F, I)
                out = torch.einsum(
                    "bfi,fio->fbo", X, W
                )
                out = out + bias.view(F, 1, O)
                # Preserve GroupedLinear behavior
                return out.squeeze(0)
            elif layout == 'f_f':
                # Group-first Conv / ResNet layout
                # [F*B, I] -> [F, B, I]
                gb = x.shape[0]
                X = x.reshape(gb, -1)
                if gb % F != 0:
                    raise ValueError(
                        "Input batch must be divisible by Fuse"
                    )
                if X.shape[1] != I:
                    raise ValueError(
                        f"Expected {I} features, got {X.shape[1]}"
                    )
                X = X.reshape(F, gb // F, I)
                out = torch.bmm(X, W)
                out = out + bias.view(F, 1, O)

                return out.reshape(gb, O)
            else:
                raise ValueError(
                    f"Unsupported conv layout: {layout}"
                )

        else:
            raise ValueError(
                f"Unsupported setup: {setup}"
            )


class NormActive(nn.Module):
    # TODO: 这里的代码也比较旧了，最好都更新掉，后续fwd 都换成stateless 比较好。
    # Instasnce Norm + ReLU 的组合，使用了Fst_Order_NormActive 这个triton kernel
    # 但是也只有bwd+ double bwd 有fuse 。其实fwd也没有fuse
    def __init__(self, channel_num, affine=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn([channel_num]))
        self.bias = nn.Parameter(torch.randn([channel_num]))
    def forward(self, input, layout = 'b_f'):
        if layout == 'b_f':
            out = Fst_Order_NormActive.apply(input, self.weight, self.bias)
            return out
        else:
        # input: [G*B,C,H,W]
            return groupd_instance_norm(input, self.weight, self.bias, eps=1e-5, v_fuse=True)

class Conv_Flexfuse_backup(nn.Module):
    # 区别就是加上了v_fuse的开关。v_fuse=True的时候，使用NormActive；否则使用InstanceNorm2d。
    def __init__(self, channel=3, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='maxpooling', im_size = (32,32), Fuse=2, v_fuse=True):
        super(Conv_Flexfuse_backup, self).__init__()
        self.Fuse = Fuse
        self.v_fuse = v_fuse
        self.num_classes = num_classes
        self.channel = channel
        self.im_size = im_size
        self.conv1 = nn.Conv2d(in_channels=channel*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        if v_fuse:
            self.norm1 = NormActive(net_width*Fuse, affine=True) 
        else:
            self.norm1 = nn.InstanceNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
            # self.norm1 = nn.GroupNorm(net_width*Fuse,net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=net_width*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        if v_fuse:
            self.norm2 = NormActive(net_width*Fuse,affine=True) #BN在channel上单独计算，所以目前不用管。
        else:
            self.norm2 = nn.InstanceNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
            # self.norm2 = nn.GroupNorm(net_width*Fuse,net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=net_width*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        if v_fuse:
            self.norm3 = NormActive(net_width*Fuse,affine=True) #BN在channel上单独计算，所以目前不用管。
        else:
            self.norm3 = nn.InstanceNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
            # self.norm3 = nn.GroupNorm(net_width*Fuse,net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool3 = nn.AvgPool2d(kernel_size=2)
        # self.linear = GroupedLinear(net_width * 4 * 4, num_classes,Fuse )
        feat_h = im_size[0] // 8
        feat_w = im_size[1] // 8
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.linear = GroupedLinear(
            net_width * feat_h * feat_w,
            num_classes,
            Fuse,
        )

        self.net_width= net_width

    def forward(self, x_conv1):
        if self.v_fuse:
            # x_conv1 = x_conv1.view(-1,self.Fuse*3 ,32,32)        # 10, 256, 4,4   -> 20, 2048
            # x_conv1 = x_conv1.view( -1, self.Fuse * self.channel, self.im_size[0], self.im_size[1] )
            x_norm1 = self.conv1(x_conv1)          
            x_pool1 = self.norm1(x_norm1)    
            x_conv2 = self.pool1(x_pool1)
            x_norm2 = self.conv2(x_conv2)          
            x_pool2 = self.norm2(x_norm2)
            x_conv3 = self.pool2(x_pool2)
            x_norm3 = self.conv3(x_conv3)          
            x_pool3 = self.norm3(x_norm3)  
            x_lin  = self.pool3(x_pool3)
            x_out = self.linear(x_lin, setup = 'conv')    # N x 10
            x_out = x_out.view(-1,self.num_classes)
        else:
            # x_conv1 = x_conv1.view(-1,self.Fuse*3 ,32,32)        # 10, 256, 4,4   -> 20, 2048
            x_norm1 = self.conv1(x_conv1)          
            x_pool1 = F.relu(self.norm1(x_norm1)    )
            x_conv2 = self.pool1(x_pool1)
            x_norm2 = self.conv2(x_conv2)          
            x_pool2 = F.relu(self.norm2(x_norm2) )   
            x_conv3 = self.pool2(x_pool2)
            x_norm3 = self.conv3(x_conv3)          
            x_pool3 = F.relu(self.norm3(x_norm3))    
            x_lin  = self.pool3(x_pool3)
            x_out = self.linear(x_lin, setup = 'conv')    # N x 10
            x_out = x_out.view(-1,self.num_classes)
        return  x_conv1,x_norm1, x_pool1,x_conv2,x_norm2, x_pool2,x_conv3,x_norm3, x_pool3, x_lin,x_out

class Conv_Flexfuse(nn.Module):
    # 
    def __init__(self, channel=3, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='maxpooling', im_size = (32,32), Fuse=2, v_fuse=True, layout="b_f"):
        super(Conv_Flexfuse, self).__init__()
        self.Fuse = Fuse
        self.v_fuse = v_fuse
        self.layout = layout
        if layout not in ("b_f", "f_f"):
            raise ValueError("layout must be b_f or f_f")
        self.channel = channel
        self.im_size = im_size
        Conv = nn.Conv2d if layout == "b_f" else GroupedConv2d
        self.num_classes = num_classes
        self.conv1 = Conv(in_channels=channel*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        self.norm1 = NormActive(net_width*Fuse, affine=True) if self.v_fuse or layout == "f_f" else nn.InstanceNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        # self.norm1 = nn.GroupNorm(net_width*Fuse,net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = Conv(in_channels=net_width*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        self.norm2 = NormActive(net_width*Fuse, affine=True) if self.v_fuse or layout == "f_f" else nn.InstanceNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        # self.norm2 = nn.GroupNorm(net_width*Fuse,net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = Conv(in_channels=net_width*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        self.norm3 = NormActive(net_width*Fuse, affine=True) if self.v_fuse or layout == "f_f" else nn.InstanceNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        # self.norm3 = nn.GroupNorm(net_width*Fuse,net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool3 = nn.AvgPool2d(kernel_size=2)
        self.linear = GroupedLinear(net_width * (im_size[0] // 8) * (im_size[1] // 8), num_classes, Fuse, layout=self.layout)
        if layout == "f_f":
            for conv, norm in ((self.conv1, self.norm1), (self.conv2, self.norm2), (self.conv3, self.norm3)):
                if conv.weight.ndim != 4:
                    raise ValueError("Use the flat-weight GroupedConv2d from ConvNet")
                # Match the BF non-fused model initialization.
                if not self.v_fuse:
                    with torch.no_grad():
                        norm.weight.fill_(1)
                        norm.bias.zero_()
        self.net_width= net_width
    def forward(self, x_conv1):
        channels = self.channel * self.Fuse if self.layout == "b_f" else self.channel
        x_conv1 = x_conv1.view(-1, channels, *self.im_size)
        x_norm1 = self.conv1(x_conv1)
        x_pool1 = F.relu(self.norm1(x_norm1, layout=self.layout)) if self.v_fuse or self.layout == "f_f" else F.relu(self.norm1(x_norm1))
        x_conv2 = self.pool1(x_pool1)
        x_norm2 = self.conv2(x_conv2)
        x_pool2 = F.relu(self.norm2(x_norm2, layout=self.layout)) if self.v_fuse or self.layout == "f_f" else F.relu(self.norm2(x_norm2))
        x_conv3 = self.pool2(x_pool2)
        x_norm3 = self.conv3(x_conv3)
        x_pool3 = F.relu(self.norm3(x_norm3, layout=self.layout)) if self.v_fuse or self.layout == "f_f" else F.relu(self.norm3(x_norm3))
        x_lin = self.pool3(x_pool3)
        x_out = self.linear(x_lin, setup="conv").view(-1, self.num_classes)
        tape = {
            "blocks": [
                {"x_conv": x_conv1, "x_norm": x_norm1, "x_pool": x_pool1},
                {"x_conv": x_conv2, "x_norm": x_norm2, "x_pool": x_pool2},
                {"x_conv": x_conv3, "x_norm": x_norm3, "x_pool": x_pool3},
            ],
            "head": {
                "x_lin": x_lin,
                "x_out": x_out,
            },
        }
        return x_out, tape
    def get_weight_pack(self):
        return {
            "blocks": [
                {
                    "convw": self.conv1.weight,
                    "convb": self.conv1.bias,
                    "normw": self.norm1.weight,
                    "normb": self.norm1.bias,
                },
                {
                    "convw": self.conv2.weight,
                    "convb": self.conv2.bias,
                    "normw": self.norm2.weight,
                    "normb": self.norm2.bias,
                },
                {
                    "convw": self.conv3.weight,
                    "convb": self.conv3.bias,
                    "normw": self.norm3.weight,
                    "normb": self.norm3.bias,
                },
            ],
            "head": {
                "linw": self.linear.weight,
                "linb": getattr(self.linear, "bias", None),
            },
        }
    def pack_recovered_weights(self, tensors_all):
        """
        用于 ReparamModule 场景。
        tensors_all 的顺序必须和 recover_params / grad 顺序一致：
        conv1_w, conv1_b, norm1_w, norm1_b,
        conv2_w, conv2_b, norm2_w, norm2_b,
        conv3_w, conv3_b, norm3_w, norm3_b,
        lin_w, lin_b
        """
        (
            conv1_w, conv1_b, norm1_w, norm1_b,
            conv2_w, conv2_b, norm2_w, norm2_b,
            conv3_w, conv3_b, norm3_w, norm3_b,
            lin_w, lin_b,
        ) = tensors_all
        return {
            "blocks": [
                {"convw": conv1_w, "convb": conv1_b, "normw": norm1_w, "normb": norm1_b},
                {"convw": conv2_w, "convb": conv2_b, "normw": norm2_w, "normb": norm2_b},
                {"convw": conv3_w, "convb": conv3_b, "normw": norm3_w, "normb": norm3_b},
            ],
            "head": {
                "linw": lin_w,
                "linb": lin_b,
            },
        }
    def collect_ordered_grads(self, d_weights_list, dlin_w, dlin_b):
        grads = []
        for dw in d_weights_list:
            grads.extend([
                dw["dconvw"], dw["dconvb"],
                dw["dnormw"], dw["dnormb"],
            ])
        grads.extend([dlin_w, dlin_b])
        return grads
    def _active_fuse_range(self, fuse_mask_list):
        if fuse_mask_list is None:
            return None
        Fuse = len(fuse_mask_list)
        bwd_Fuse = sum(fuse_mask_list)
        if bwd_Fuse == Fuse:
            return None
        start = fuse_mask_list.index(1)
        end = start + bwd_Fuse
        assert fuse_mask_list[start:end] == [1] * bwd_Fuse, "fuse_mask_list must keep active branches contiguous"
        return start, end, Fuse
    def _split_tensor_by_fuse(self, p, active_range, cache=None, batch=False):
        if p is None or active_range is None:
            return p
        start, end, Fuse = active_range
        key = (id(p), "batch" if batch else "channel")
        if cache is not None and key in cache:
            src_weak, cached = cache[key]
            if src_weak() is p:
                return cached
        dim = 0 if batch or self.layout == "f_f" else 1
        block = p.shape[dim] // Fuse
        view = p.narrow(dim, start * block, (end - start) * block)
        # Force a new compact storage. contiguous() alone can return a view
        # that still keeps the full-Fuse allocation alive.
        out = view.clone(memory_format=torch.contiguous_format)
        if cache is not None:
            import weakref
            cache[key] = (weakref.ref(p), out)
        return out
    def _split_saved_for_double_bwd(self, tape, d_activates_list, d_head_tensors, fuse_mask_list):
        """
        first-bwd 必须用完整 Fuse 算 dW；
        但是 double-bwd 如果只跑 bwd_Fuse，就要把 tape / d_activates / dx_out 切到 bwd_Fuse。
        """
        active_range = self._active_fuse_range(fuse_mask_list)
        if active_range is None:
            return
        cache = {}
        for b in tape["blocks"]:
            for key in ("x_conv", "x_norm", "x_pool"):
                b[key] = self._split_tensor_by_fuse(b[key], active_range, cache)
        for da in d_activates_list:
            for key in ("dx_norm", "dx_pool"):
                da[key] = self._split_tensor_by_fuse(da[key], active_range, cache)
        head = tape["head"]
        head["x_lin"] = self._split_tensor_by_fuse(head["x_lin"], active_range, cache)
        head["x_out"] = self._split_tensor_by_fuse(head["x_out"], active_range, cache, batch=True)
        d_head_tensors["dx_out"] = self._split_tensor_by_fuse(
            d_head_tensors["dx_out"], active_range, cache, batch=True)
    def _convblock_first_bwd(self, block, weights, grad_output, Fuse):
        dx_conv, dx_norm, dx_pool, dconv_w, dconv_b, dnorm_w, dnorm_b = ConvBlock_bwd1_2(
            block["x_conv"],
            block["x_norm"],
            block["x_pool"],
            weights["convw"],
            weights["normw"],
            grad_output,
            Fuse=Fuse, v_fuse=self.v_fuse, layout=self.layout,
        )
        d_acts = {
            "dx_norm": dx_norm,
            "dx_pool": dx_pool,
        }
        d_w = {
            "dconvw": dconv_w,
            "dconvb": dconv_b,
            "dnormw": dnorm_w,
            "dnormb": dnorm_b,
        }
        return dx_conv, d_acts, d_w
    def run_first_bwd(self, tape, target, Fuse=1, weights=None, fuse_mask_list=None):
        """
        返回：
        d_head_tensors: 主要存 dx_out
        d_activates_list: 每层存 dx_norm / dx_pool
        d_weights_list: 每层参数梯度 dict
        d_weights_list_all: 按 recover_params 顺序排列的 grad list
        """
        if weights is None:
            weights = self.get_weight_pack()
        blocks = tape["blocks"]
        head = tape["head"]
        x_lin = head["x_lin"]
        x_out = head["x_out"]
        d_activates_list = [None, None, None]
        d_weights_list = [None, None, None]
        active_range = self._active_fuse_range(fuse_mask_list)
        split_cache = {} if active_range is not None else None
        dx_out = crossEntropy_bwd(x_out, target, Fuse)
        dx_lin, dlin_w, dlin_b = linerFused_bwd(
            x_lin,
            weights["head"]["linw"],
            grad_output=dx_out,
            Fuse=Fuse, layout=self.layout,
        )
        dx_lin = dx_lin.view_as(x_lin)
        # 与原始 Distill 相同：Linear Backward 后就切 head，而不是等整段 First Backward 结束。
        if active_range is not None:
            head["x_lin"] = self._split_tensor_by_fuse(x_lin, active_range, split_cache)
            head["x_out"] = self._split_tensor_by_fuse(x_out, active_range, split_cache, batch=True)
            dx_out_saved = self._split_tensor_by_fuse(dx_out, active_range, split_cache, batch=True)
        else:
            dx_out_saved = dx_out
        del x_lin, x_out, dx_out
        # block3
        g, d_activates_list[2], d_weights_list[2] = self._convblock_first_bwd(
            blocks[2],
            weights["blocks"][2],
            dx_lin,
            Fuse,
        )
        del dx_lin
        if active_range is not None:
            b = blocks[2]
            da = d_activates_list[2]
            for key in ("x_conv", "x_norm", "x_pool"):
                b[key] = self._split_tensor_by_fuse(b[key], active_range, split_cache)
            for key in ("dx_norm", "dx_pool"):
                da[key] = self._split_tensor_by_fuse(da[key], active_range, split_cache)
        # block2
        g, d_activates_list[1], d_weights_list[1] = self._convblock_first_bwd(
            blocks[1],
            weights["blocks"][1],
            g,
            Fuse,
        )
        if active_range is not None:
            b = blocks[1]
            da = d_activates_list[1]
            for key in ("x_conv", "x_norm", "x_pool"):
                b[key] = self._split_tensor_by_fuse(b[key], active_range, split_cache)
            for key in ("dx_norm", "dx_pool"):
                da[key] = self._split_tensor_by_fuse(da[key], active_range, split_cache)
        # block1：这里保持你原来的特殊写法，而不是强行用 ConvBlock_bwd1_2
        b1 = blocks[0]
        w1 = weights["blocks"][0]
        dx_pool1 = avgPool_bwd(b1["x_pool"], grad_output=g)
        del g
        dx_norm1, dnorm1_w, dnorm1_b = insNormNRelu_bwd(
            b1["x_norm"],
            w1["normw"],
            b1["x_pool"],
            grad_output=dx_pool1, v_fuse=self.v_fuse, layout=self.layout,
        )
        if active_range is not None:
            b1["x_norm"] = self._split_tensor_by_fuse(b1["x_norm"], active_range, split_cache)
            b1["x_pool"] = self._split_tensor_by_fuse(b1["x_pool"], active_range, split_cache)
            dx_pool1 = self._split_tensor_by_fuse(dx_pool1, active_range, split_cache)
        _, dconv1_w, dconv1_b = conv_bwd(
            b1["x_conv"],
            w1["convw"],
            grad_output=dx_norm1,
            groups=Fuse, layout=self.layout,
        )
        del _
        if active_range is not None:
            b1["x_conv"] = self._split_tensor_by_fuse(b1["x_conv"], active_range, split_cache)
            dx_norm1 = self._split_tensor_by_fuse(dx_norm1, active_range, split_cache)
        d_activates_list[0] = {
            "dx_norm": dx_norm1,
            "dx_pool": dx_pool1,
        }
        d_weights_list[0] = {
            "dconvw": dconv1_w,
            "dconvb": dconv1_b,
            "dnormw": dnorm1_w,
            "dnormb": dnorm1_b,
        }
        d_head_tensors = {
            "dx_out": dx_out_saved,
        }
        # tape 与 d_activates 已经按原始 Distill 的顺序逐层切分，无需再切一次。
        d_weights_list_all = self.collect_ordered_grads(
            d_weights_list,
            dlin_w,
            dlin_b,
        )
        return d_head_tensors, d_activates_list, d_weights_list, d_weights_list_all
    def pack_recovered_dd(self, dd_tensors_all, tape=None, x=None):
        """
        dd_tensors_all 来自：
            recover_params(ddw, shape_list, bwd_Fuse)
        顺序必须和 collect_ordered_grads 一致。
        """
        (
            ddconv1_w, ddconv1_b, ddnorm1_w, ddnorm1_b,
            ddconv2_w, ddconv2_b, ddnorm2_w, ddnorm2_b,
            ddconv3_w, ddconv3_b, ddnorm3_w, ddnorm3_b,
            ddlin_w, ddlin_b,
        ) = dd_tensors_all
        dd_weights_list = [
            {
                "ddconvw": ddconv1_w,
                "ddconvb": ddconv1_b,
                "ddnormw": ddnorm1_w,
                "ddnormb": ddnorm1_b,
            },
            {
                "ddconvw": ddconv2_w,
                "ddconvb": ddconv2_b,
                "ddnormw": ddnorm2_w,
                "ddnormb": ddnorm2_b,
            },
            {
                "ddconvw": ddconv3_w,
                "ddconvb": ddconv3_b,
                "ddnormw": ddnorm3_w,
                "ddnormb": ddnorm3_b,
            },
        ]
        if x is None and tape is not None:
            x = tape["blocks"][0]["x_conv"]
        dd_head_tensors = {
            "ddlinw": ddlin_w,
            "ddlinb": ddlin_b,
        }
        if x is not None:
            dd_head_tensors["ddx_conv"] = torch.zeros_like(x)
        return dd_head_tensors, dd_weights_list
    @torch.no_grad()
    def run_double_bwd(
        self,
        tape,
        d_activates_list,
        dd_weights_list=None,
        d_head_tensors=None,
        dd_head_tensors=None,
        Fuse=1,
        weights=None,
        dd_tensors_all=None,
        d_bwd_state=None,
        weight_tensors_all=None,
    ):
        """Run the forward-order double backward and prepare the bwd2-1 inputs.

        Consumes the first-bwd activation state and packed dd gradients.
        The forward tape and weights are still needed by bwd2-1. The
        derivatives are computed explicitly, so this entry needs no autograd tape.
        """
        return self._run_double_bwd_impl(
            tape, d_activates_list, dd_weights_list, d_head_tensors,
            dd_head_tensors, Fuse, weights, dd_tensors_all, d_bwd_state,
            weight_tensors_all,
        )

    def _run_double_bwd_impl(
        self, tape, d_activates_list, dd_weights_list, d_head_tensors,
        dd_head_tensors, Fuse, weights, dd_tensors_all, d_bwd_state,
        weight_tensors_all=None,
    ):
        if d_bwd_state is not None:
            if d_head_tensors is not None:
                raise ValueError("Pass only one of d_bwd_state and d_head_tensors")
            d_head_tensors = d_bwd_state
        if dd_tensors_all is not None:
            if dd_weights_list is not None or dd_head_tensors is not None:
                raise ValueError("Pass either dd_tensors_all or packed dd tensors, not both")
            dd_head_tensors, dd_weights_list = self.pack_recovered_dd(
                dd_tensors_all, tape=tape)
        if weight_tensors_all is not None:
            if weights is not None:
                raise ValueError("Pass either weight_tensors_all or weights, not both")
            weights = self.pack_recovered_weights(weight_tensors_all)
        if d_head_tensors is None or dd_head_tensors is None or dd_weights_list is None:
            raise ValueError("run_double_bwd requires first-bwd state and dd parameter tensors")
        if weights is None:
            weights = self.get_weight_pack()
        blocks = tape["blocks"]
        head = tape["head"]
        x_lin = head["x_lin"]
        x_out = head["x_out"]
        dx_out = d_head_tensors.pop("dx_out")
        ddx_conv = dd_head_tensors.pop("ddx_conv")
        ddlin_w = dd_head_tensors.pop("ddlinw")
        ddlin_b = dd_head_tensors.pop("ddlinb")
        # ------------------------------------------------------------
        # 1. forward-order double-bwd:
        #    conv1 -> block2 -> block3 -> linear/head
        # ------------------------------------------------------------
        b1 = blocks[0]
        w1 = weights["blocks"][0]
        da1 = d_activates_list[0]
        dd1 = dd_weights_list[0]
        ddx_norm1, dxconv1_d2, _ = conv_double_bwd(
            ddx_conv,
            dd1["ddconvw"],
            dd1["ddconvb"],
            da1["dx_norm"],
            w1["convw"],
            b1["x_conv"],
            groups_=Fuse, layout=self.layout, output_mask=(True, True, False),
        )
        del ddx_conv, _
        ddx_pool1, dx_norm1_d2, _ = insNormNRelu_double_bwd(
            ddx_norm1,
            dd1["ddnormw"],
            dd1["ddnormb"],
            da1["dx_pool"],
            b1["x_pool"],
            w1["normw"],
            b1["x_norm"], v_fuse=self.v_fuse, layout=self.layout,
        )
        del ddx_norm1, _
        dd_cur = avgPool_double_bwd(ddx_pool1)
        del ddx_pool1
        d2_cache = [None, None, None]
        d2_cache[0] = {
            "dxconv_d2": dxconv1_d2,
            "dx_norm_d2": dx_norm1_d2,
        }
        d_activates_list[0] = None
        dd_weights_list[0] = None
        del da1, dd1, b1, w1, dxconv1_d2, dx_norm1_d2
        # block2, block3
        for idx in (1, 2):
            b = blocks[idx]
            w = weights["blocks"][idx]
            da = d_activates_list[idx]
            dd = dd_weights_list[idx]
            dd_cur, dxconv_d2, _, dx_norm_d2, _ = ConvBlock_double_bwd(
                b["x_conv"],
                b["x_norm"],
                b["x_pool"],
                da["dx_norm"],
                da["dx_pool"],
                dd_cur,
                w["convw"],
                w["normw"],
                dd["ddconvw"],
                dd["ddconvb"],
                dd["ddnormw"],
                dd["ddnormb"],
                Fuse, v_fuse=self.v_fuse, layout=self.layout, output_mask=(True, True, False),
            )
            d2_cache[idx] = {
                "dxconv_d2": dxconv_d2,
                "dx_norm_d2": dx_norm_d2,
            }
            d_activates_list[idx] = None
            dd_weights_list[idx] = None
            del b, w, da, dd, dxconv_d2, dx_norm_d2, _
        ddx_lin = dd_cur
        del dd_cur
        # ------------------------------------------------------------
        # 2. head double-bwd
        # -----------------------------------------------------------
        ddx_out, dx_lin_d2, _ = linearFused_double_bwd(
            x_lin,
            weights["head"]["linw"],
            dx_out,
            ddx_lin,
            ddlin_w,
            ddlin_b,
            Fuse, layout=self.layout,
        )
        del dx_out, ddx_lin, ddlin_w, ddlin_b, _
        dx_out_d1 = crossEntropy_double_bwd(x_out, ddx_out, Fuse)
        return {
            "dx_out_d1": dx_out_d1,
            "dx_lin_d2": dx_lin_d2,
            "blocks": d2_cache,
        }

    def run_double_bwd_full(
        self,
        tape,
        d_activates_list,
        dd_weights_list=None,
        d_head_tensors=None,
        dd_head_tensors=None,
        Fuse=1,
        weights=None,
        dd_tensors_all=None,
        d_bwd_state=None,
        weight_tensors_all=None,
    ):
        """
        返回 synthetic image/input 方向的最终梯度。

        新的统一入口只需要传 d_bwd_state + dd_tensors_all；
        旧的 dd_weights_list/d_head_tensors/dd_head_tensors 接口仍然保留。
        """
        if weight_tensors_all is not None:
            if weights is not None:
                raise ValueError("Pass either weight_tensors_all or weights, not both")
            weights = self.pack_recovered_weights(weight_tensors_all)
        if weights is None:
            weights = self.get_weight_pack()
        bwd2_1_activations = self._run_double_bwd_impl(
            tape, d_activates_list, dd_weights_list, d_head_tensors,
            dd_head_tensors, Fuse, weights, dd_tensors_all, d_bwd_state,
        )
        blocks = tape["blocks"]
        head = tape["head"]
        x_lin = head["x_lin"]
        dx_out_d1 = bwd2_1_activations.pop("dx_out_d1")
        dx_lin_d2 = bwd2_1_activations.pop("dx_lin_d2")
        d2_cache = bwd2_1_activations.pop("blocks")
        dx_lin_d1, _, _ = linerFused_bwd(
            x_lin,
            weights["head"]["linw"],
            grad_output=dx_out_d1,
            Fuse=Fuse, layout=self.layout,
        )
        del _
        dx_lin_d1 = dx_lin_d1.reshape(x_lin.shape)
        dx_lin_d1 += dx_lin_d2.reshape(x_lin.shape)  # 与原始 Distill 的原地相加一致
        head.clear()
        del x_lin, dx_out_d1, dx_lin_d2
        # ------------------------------------------------------------
        # 3. backward-order bwd2_1:
        #    block3 -> block2 -> block1
        # -----------------------------------------------------------
        g = dx_lin_d1
        del dx_lin_d1
        for idx in (2, 1, 0):
            b = blocks[idx]
            w = weights["blocks"][idx]
            c = d2_cache[idx]
            g, _, _, _, _ = ConvBlock_bwd2_1(
                b["x_conv"],
                b["x_norm"],
                b["x_pool"],
                w["convw"],
                w["normw"],
                g,
                c["dx_norm_d2"],
                c["dxconv_d2"],
                Fuse=Fuse, v_fuse=self.v_fuse, layout=self.layout,
            )
            blocks[idx] = None
            d2_cache[idx] = None
            del b, w, c, _
        return g

    def _add_bwd2_1_correction(self, grad, correction, active_range):
        start, end, Fuse = active_range
        dim = 0 if self.layout == "f_f" else 1
        width = grad.shape[dim] // Fuse
        left = start * width
        right = (Fuse - end) * width
        if left:
            grad.narrow(dim, 0, left).add_(correction.narrow(dim, 0, left))
        if right:
            grad.narrow(dim, end * width, right).add_(correction.narrow(dim, left, right))

    @torch.no_grad()
    def run_fused_bwd(self, tape, target, bwd2_1_activations, fuse_mask_list, weights=None):
        """Run first-bwd and bwd2-1 in one grouped backward traversal.

        The caller packs forward activations and weights in mask order: 1 uses
        the current first-bwd job and 0 uses the earlier bwd2-1 job. Pack the
        tape along channels (axis 1) for b_f or batch (axis 0) for f_f;
        logits always use axis 0. Pack
        each parameter along axis 0. Jobs must share batch size, per-lane
        shapes, and v_fuse. Only the contiguous 1 lanes have targets.

        Pass the state returned by run_double_bwd for the 0 lanes in
        their mask order. This call consumes that state and compacts tape to
        the 1 lanes for explicit double backward. Caller-held references to
        the source tensors must be dropped separately. Returns the four
        first-bwd results for the 1 lanes and the input gradient for the 0 lanes.
        """
        mask = list(fuse_mask_list)
        if any(lane not in (0, 1) for lane in mask) or 0 not in mask or 1 not in mask:
            raise ValueError("run_fused_bwd requires both first-bwd (1) and bwd2-1 (0) lanes")
        active_range = self._active_fuse_range(mask)
        start, end, Fuse = active_range
        if weights is None:
            weights = self.get_weight_pack()
        blocks = tape["blocks"]
        head = tape["head"]
        split_cache = {}
        d_activates_list = [None] * len(blocks)
        d_weights_list = [None] * len(blocks)

        x_lin = head["x_lin"]
        head["x_out"] = self._split_tensor_by_fuse(
            head["x_out"], active_range, split_cache, batch=True)
        dx_out = crossEntropy_bwd(head["x_out"], target, end - start)
        old_dx_out = bwd2_1_activations.pop("dx_out_d1")
        rows_per_lane = dx_out.shape[0] // (end - start)
        left = start * rows_per_lane
        fused_dx_out = torch.cat((old_dx_out[:left], dx_out, old_dx_out[left:]), dim=0)
        del old_dx_out
        g, dlin_w, dlin_b = linerFused_bwd(
            x_lin, weights["head"]["linw"], grad_output=fused_dx_out, Fuse=Fuse, layout=self.layout)
        del fused_dx_out
        g = g.reshape_as(x_lin)
        old_dx_lin = bwd2_1_activations.pop("dx_lin_d2")
        old_shape = list(x_lin.shape)
        dim = 0 if self.layout == "f_f" else 1
        old_shape[dim] = x_lin.shape[dim] // Fuse * (Fuse - (end - start))
        self._add_bwd2_1_correction(g, old_dx_lin.reshape(old_shape), active_range)
        del old_dx_lin
        head["x_lin"] = self._split_tensor_by_fuse(x_lin, active_range, split_cache)
        del x_lin
        dlin_w = self._split_tensor_by_fuse(dlin_w, active_range, batch=True)
        dlin_b = self._split_tensor_by_fuse(dlin_b, active_range, batch=True)

        old_blocks = bwd2_1_activations.pop("blocks")
        for idx in reversed(range(len(blocks))):
            block = blocks[idx]
            w = weights["blocks"][idx]
            old_block = old_blocks[idx]
            dx_pool = avgPool_bwd(block["x_pool"], grad_output=g)
            del g
            dx_norm, dnorm_w, dnorm_b = insNormNRelu_bwd(
                block["x_norm"], w["normw"], block["x_pool"],
                grad_output=dx_pool, v_fuse=self.v_fuse, layout=self.layout)
            self._add_bwd2_1_correction(dx_norm, old_block.pop("dx_norm_d2"), active_range)
            for key in ("x_norm", "x_pool"):
                block[key] = self._split_tensor_by_fuse(block[key], active_range, split_cache)
            dx_pool = self._split_tensor_by_fuse(dx_pool, active_range, split_cache)

            g, dconv_w, dconv_b = conv_bwd(
                block["x_conv"], w["convw"], grad_output=dx_norm, groups=Fuse, layout=self.layout)
            self._add_bwd2_1_correction(g, old_block.pop("dxconv_d2"), active_range)
            old_blocks[idx] = None
            del old_block
            dx_norm = self._split_tensor_by_fuse(dx_norm, active_range, split_cache)
            block["x_conv"] = self._split_tensor_by_fuse(
                block["x_conv"], active_range, split_cache)
            d_activates_list[idx] = {
                "dx_norm": dx_norm,
                "dx_pool": dx_pool,
            }
            d_weights_list[idx] = {
                "dconvw": self._split_tensor_by_fuse(dconv_w, active_range, batch=True),
                "dconvb": self._split_tensor_by_fuse(dconv_b, active_range, batch=True),
                "dnormw": self._split_tensor_by_fuse(dnorm_w, active_range, batch=True),
                "dnormb": self._split_tensor_by_fuse(dnorm_b, active_range, batch=True),
            }
            del dx_norm, dconv_w, dconv_b, dnorm_w, dnorm_b

        # The old lanes are the complement of the contiguous first-bwd range.
        width = g.shape[dim] // Fuse
        old_dx_input = torch.cat((g.narrow(dim, 0, start * width),
                                  g.narrow(dim, end * width, (Fuse - end) * width)), dim=dim)
        d_weights_list_all = self.collect_ordered_grads(d_weights_list, dlin_w, dlin_b)
        return {"dx_out": dx_out}, d_activates_list, d_weights_list, d_weights_list_all, old_dx_input



class BasicBlock_Flexfuse(nn.Module):
    def __init__(self, in_channels, out_channels,   net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='maxpooling', im_size = (32,32), stride=1, Fuse = 1,v_fuse=True, layout="b_f"):
        # 注意一下这个basic block已经把 fuse隔离在外面了。内部init的时候再扩充in out channel
        super().__init__()
        self.v_fuse = v_fuse
        if layout not in ("b_f", "f_f"):
            raise ValueError("layout must be 'b_f' or 'f_f'")
        self.layout = layout
        # The FF convolution class must retain the nn.Conv2d parameter contract.
        Conv = nn.Conv2d if layout == "b_f" else GroupedConv2d
        # Native IN preserves legacy affine initialization (ones/zeros) in BF.
        # FF uses the existing pure, layout-aware InstanceNorm from ConvNet.
        Norm = nn.InstanceNorm2d if layout == "b_f" else NormActive
        self.conv1 = Conv(in_channels*Fuse, out_channels*Fuse, 3, stride, 1, bias=False, groups=Fuse)
        self.bn1 = Norm(out_channels*Fuse, affine=True) 
        self.conv2 = Conv(out_channels*Fuse, out_channels*Fuse, 3, 1, 1, bias=False, groups=Fuse)
        self.bn2 = Norm(out_channels*Fuse, affine=True) 
        if stride != 1 or in_channels != out_channels:
            self.convsc = Conv(in_channels*Fuse, out_channels*Fuse, 1, stride, bias=False, groups=Fuse )
            self.bnsc = Norm(out_channels*Fuse, affine=True) 
        else:
            self.convsc = None
            self.bnsc = None
        if self.layout == "f_f":
            for conv in (self.conv1, self.conv2, self.convsc):
                if conv is not None and conv.weight.ndim != 4:
                    raise ValueError("Use the flat-weight GroupedConv2d from ConvNet, not the older 5D-weight class")
            for norm in (self.bn1, self.bn2, self.bnsc):
                if norm is not None:
                    with torch.no_grad():
                        norm.weight.fill_(1)
                        norm.bias.zero_()

    def forward(self, x):
        norm_kw = {"layout": self.layout} if self.layout == "f_f" else {}
        identity = x
        x_bnsc = None
        if self.convsc is not None:
            x_bnsc = self.convsc(x)
            identity = self.bnsc(x_bnsc, **norm_kw)
        x_bn1 = self.conv1(x)
        x_conv2 = F.relu(self.bn1(x_bn1, **norm_kw), inplace=True)
        x_bn2 = self.conv2(x_conv2)
        out = self.bn2(x_bn2, **norm_kw)
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

class ResNet18_FlexFuse(nn.Module):
    def __init__(self, channel=3, num_classes=10, Fuse=1,v_fuse=True, layout="b_f"):
        super().__init__()
        self.Fuse = Fuse
        self.v_fuse = v_fuse
        if layout not in ("b_f", "f_f"):
            raise ValueError("layout must be 'b_f' or 'f_f'")
        self.layout = layout
        # The FF convolution class must retain the nn.Conv2d parameter contract.
        Conv = nn.Conv2d if layout == "b_f" else GroupedConv2d
        # Native IN preserves legacy affine initialization (ones/zeros) in BF.
        # FF uses the existing pure, layout-aware InstanceNorm from ConvNet.
        Norm = nn.InstanceNorm2d if layout == "b_f" else NormActive
        self.num_classes = num_classes
        blk_in_ch = 64   # 这里仍然是逻辑通道数
        cfg = [ (blk_in_ch,  2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2), ]
        self.conv = Conv(channel * Fuse, blk_in_ch * Fuse, kernel_size=3, stride=1, padding=1, bias=False, groups=Fuse )
        self.bn = Norm(blk_in_ch * Fuse, affine=True) 
        if self.layout == "f_f":
            if self.conv.weight.ndim != 4:
                raise ValueError("Use the flat-weight GroupedConv2d from ConvNet, not the older 5D-weight class")
            with torch.no_grad():
                self.bn.weight.fill_(1)
                self.bn.bias.zero_()
        self.stages = nn.ModuleList()
        for stage_id, (out_ch, num_blocks, first_stride) in enumerate(cfg):
            stage = nn.ModuleList()
            for block_id in range(num_blocks):
                stride = first_stride if block_id == 0 else 1
                stage.append(BasicBlock_Flexfuse(blk_in_ch, out_ch, stride=stride, Fuse=Fuse,v_fuse=v_fuse,layout=self.layout))
                blk_in_ch = out_ch
            self.stages.append(stage)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = GroupedLinear(512, num_classes, Fuse, layout=self.layout)
        # self.fc = nn.Linear(512, num_classes)

    def forward(self, x_conv):
        norm_kw = {"layout": self.layout} if self.layout == "f_f" else {}
        tape = { "stem": {}, "blocks": [], "head": {} }
        x_bn = self.conv(x_conv)
        x_block = self.bn(x_bn, **norm_kw)
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
        x_out = self.fc(x_fc, setup="conv")
        x_out = x_out.view(-1,self.num_classes)
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
    
    def collect_ordered_grads(self, dconvw, dbnw, dbnb, d_weights_list, dfcw, dfcb):
        grads_all = [dconvw, dbnw, dbnb]

        block_keys_main = ["dconv1w", "dbn1w", "dbn1b", "dconv2w", "dbn2w", "dbn2b"]
        block_keys_sc = ["dconvscw", "dbnscw", "dbnscb"]

        for d_weights_i in d_weights_list:
            if d_weights_i is None:
                continue

            for k in block_keys_main:
                grads_all.append(d_weights_i[k])

            for k in block_keys_sc:
                v = d_weights_i.get(k, None)
                if v is not None:
                    grads_all.append(v)

        grads_all.extend([dfcw, dfcb])
        return grads_all


    def _active_fuse_range(self, fuse_mask_list):
        if fuse_mask_list is None:
            return None
        Fuse = len(fuse_mask_list)
        bwd_Fuse = sum(fuse_mask_list)
        if bwd_Fuse == Fuse:
            return None
        start = fuse_mask_list.index(1)
        end = start + bwd_Fuse
        assert fuse_mask_list[start:end] == [1] * bwd_Fuse, "fuse_mask_list must keep active branches contiguous"
        return start, end, Fuse

    def _split_tensor_by_fuse(self, p, fuse_mask_list, layout="channel", cache=None):
        if p is None or (not torch.is_tensor(p)):
            return p
        active_range = self._active_fuse_range(fuse_mask_list)
        if active_range is None:
            return p
        start, end, Fuse = active_range
        key = (id(p), layout)
        if cache is not None and key in cache:
            src_weak, cached = cache[key]
            if src_weak() is p:
                return cached

        from utils_flex import split_half_snd_dim

        # Logits are Fuse-first in both layouts. BF x_fc is the only 2D
        # channel-packed activation: expose it as NCHW to the shared utility.
        split_layout = "f_f" if layout == "batch" or p.ndim == 1 else self.layout
        if split_layout == "b_f" and p.ndim == 2:
            out = split_half_snd_dim([p[:, :, None, None]], fuse_mask_list,
                                     layout=split_layout)[0][:, :, 0, 0]
        else:
            out = split_half_snd_dim([p], fuse_mask_list, layout=split_layout)[0]
        # A contiguous view can still retain the full-Fuse allocation.
        # Keep the original compact-storage guarantee without copying twice.
        if (out.untyped_storage().data_ptr() == p.untyped_storage().data_ptr()
                or not out.is_contiguous()):
            out = out.clone(memory_format=torch.contiguous_format)
        if cache is not None:
            import weakref
            cache[key] = (weakref.ref(p), out)
        return out

    def _split_tensor_dict_by_fuse(self, tensor_dict, fuse_mask_list, layout="channel", cache=None):
        if tensor_dict is None:
            return None
        if self._active_fuse_range(fuse_mask_list) is None:
            return tensor_dict
        for k in list(tensor_dict.keys()):
            v = tensor_dict[k]
            if torch.is_tensor(v):
                tensor_dict[k] = self._split_tensor_by_fuse(
                    v, fuse_mask_list, layout=layout, cache=cache)
                del v
        return tensor_dict

    def _split_head_saved_for_double_bwd(self, head, d_stem_tensors, fuse_mask_list, cache=None):
        if self._active_fuse_range(fuse_mask_list) is None:
            return
        # BF: x_pool/x_fc pack Fuse in the channel axis (dim=1).
        # FF: x_pool/x_fc pack Fuse in the leading axis (dim=0).
        # Logits and their gradients are [Fuse*B, num_classes] in BOTH layouts.
        head["x_pool"] = self._split_tensor_by_fuse(head["x_pool"], fuse_mask_list, layout="channel", cache=cache)
        head["x_fc"] = self._split_tensor_by_fuse(head["x_fc"], fuse_mask_list, layout="channel", cache=cache)
        head["x_out"] = self._split_tensor_by_fuse(head["x_out"], fuse_mask_list, layout="batch", cache=cache)
        d_stem_tensors["dx_out"] = self._split_tensor_by_fuse(
            d_stem_tensors["dx_out"], fuse_mask_list, layout="batch", cache=cache)

    def _split_block_saved_for_double_bwd(self, activates_i, d_activates_i, fuse_mask_list, cache=None):
        if self._active_fuse_range(fuse_mask_list) is None:
            return
        self._split_tensor_dict_by_fuse(activates_i, fuse_mask_list, layout="channel", cache=cache)
        self._split_tensor_dict_by_fuse(d_activates_i, fuse_mask_list, layout="channel", cache=cache)

    def _split_stem_saved_for_double_bwd(self, stem, d_stem_tensors, fuse_mask_list, cache=None):
        if self._active_fuse_range(fuse_mask_list) is None:
            return
        self._split_tensor_dict_by_fuse(stem, fuse_mask_list, layout="channel", cache=cache)
        d_stem_tensors["dx_bn"] = self._split_tensor_by_fuse(
            d_stem_tensors["dx_bn"], fuse_mask_list, layout="channel", cache=cache)
        d_stem_tensors["dx_block"] = self._split_tensor_by_fuse(
            d_stem_tensors["dx_block"], fuse_mask_list, layout="channel", cache=cache)

    def pack_recovered_weights(self, tensors_all):
        ptr = 0
        # ---- stem ----
        convw = tensors_all[ptr]; ptr += 1
        bnw   = tensors_all[ptr]; ptr += 1
        bnb   = tensors_all[ptr]; ptr += 1

        block_weights_list = []
        flat_blocks = self.get_flat_blocks()
        for blk in flat_blocks:
            weights_i = {
                "conv1w": tensors_all[ptr],
                "bn1w":   tensors_all[ptr + 1],
                "bn1b":   tensors_all[ptr + 2],
                "conv2w": tensors_all[ptr + 3],
                "bn2w":   tensors_all[ptr + 4],
                "bn2b":   tensors_all[ptr + 5],
                "convscw": None,
                "bnscw":   None,
                "bnscb":   None,
            }
            ptr += 6
            if blk.convsc is not None:
                weights_i["convscw"] = tensors_all[ptr]; ptr += 1
                weights_i["bnscw"]   = tensors_all[ptr]; ptr += 1
                weights_i["bnscb"]   = tensors_all[ptr]; ptr += 1
            block_weights_list.append(weights_i)

        fcw = tensors_all[ptr]; ptr += 1
        fcb = tensors_all[ptr]; ptr += 1

        assert ptr == len(tensors_all), (
            f"weight tensor parse mismatch: used {ptr}, total {len(tensors_all)}"
        )
        return {
            "stem": {
                "convw": convw,
                "bnw": bnw,
                "bnb": bnb,
            },
            "blocks": block_weights_list,
            "head": {
                "fcw": fcw,
                "fcb": fcb,
            },
        }

    def run_first_bwd(self, tape, target, Fuse = 1, fuse_mask_list=None):
        # Full-Fuse first backward; saved tape is compacted here, not by the caller.
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
        split_cache = {} if self._active_fuse_range(fuse_mask_list) is not None else None

        dx_out = crossEntropy_bwd(x_out, target, Fuse=Fuse) 
        d_stem_tensors = { "dx_out": dx_out, }
        # dx_fc, dfcw, dfcb = linear_bwd( x_fc, self.fc.weight, grad_output=dx_out)
        dx_fc, dfcw, dfcb = linerFused_bwd( x_fc, self.fc.weight, grad_output=dx_out, Fuse= Fuse, layout=self.layout)
        dx_fc = dx_fc.view(x_pool.size(0), x_pool.size(1), 1, 1)
        g = adaptivepooling_bwd(x_pool, grad_output=dx_fc)
        self._split_head_saved_for_double_bwd(
            head, d_stem_tensors, fuse_mask_list, cache=split_cache)
        del dx_fc, dx_out, x_pool, x_fc, x_out
        for i in reversed(range(len(flat_blocks))):
            blk = flat_blocks[i]
            activates_i = tape["blocks"][i]
            weights_i = self.get_block_weights(blk)
            g, d_activates_i, d_weights_i = BasicBlock_bwd( activates_i, weights_i, grad_output=g, SCstride=blk.conv1.stride[0],Fuse=Fuse,v_fuse=self.v_fuse, layout=self.layout)
            d_activates_list[i] = d_activates_i
            d_weights_list[i]   = d_weights_i
            self._split_block_saved_for_double_bwd(
                activates_i, d_activates_i, fuse_mask_list, cache=split_cache)
            # 这里只删局部引用，不动 tape 里的本体
            del activates_i, weights_i
        dx_block = g
        del g
        dx_bn, dbnw, dbnb, dconvw = conv_norm_relu_bwd( x_conv, x_bn, x_block, self.conv.weight, self.bn.weight, grad_output=dx_block, Fuse=Fuse, v_fuse=self.v_fuse, layout=self.layout )
        d_stem_tensors["dx_bn"] = dx_bn
        d_stem_tensors["dx_block"] = dx_block
        self._split_stem_saved_for_double_bwd(
            stem, d_stem_tensors, fuse_mask_list, cache=split_cache)
        del dx_bn, dx_block, x_conv, x_bn, x_block

        d_weights_list_all = self.collect_ordered_grads( dconvw, dbnw, dbnb, d_weights_list, dfcw, dfcb)
        return  d_stem_tensors, d_activates_list, d_weights_list, d_weights_list_all
        
    def run_double_bwd_full(
        self,
        tape,
        d_activates_list,
        dd_weights_list=None,
        d_stem_tensors=None,
        dd_stem_tensors=None,
        Fuse = 1,
        weights=None,
        dd_tensors_all=None,
        d_bwd_state=None,
        weight_tensors_all=None,
    ):
        if d_bwd_state is not None:
            if d_stem_tensors is not None:
                raise ValueError("Pass only one of d_bwd_state and d_stem_tensors")
            d_stem_tensors = d_bwd_state
        if dd_tensors_all is not None:
            if dd_weights_list is not None or dd_stem_tensors is not None:
                raise ValueError("Pass either dd_tensors_all or packed dd tensors, not both")
            dd_stem_tensors, dd_weights_list = self.pack_recovered_dd(
                dd_tensors_all, tape=tape)
        if weight_tensors_all is not None:
            if weights is not None:
                raise ValueError("Pass either weight_tensors_all or weights, not both")
            weights = self.pack_recovered_weights(weight_tensors_all)
        if d_stem_tensors is None or dd_stem_tensors is None or dd_weights_list is None:
            raise ValueError("run_double_bwd requires first-bwd state and dd parameter tensors")
        flat_blocks = self.get_flat_blocks()
        if weights is None:
            weights = {
                "stem": {
                    "convw": self.conv.weight,
                    "bnw": self.bn.weight,
                    "bnb": self.bn.bias,
                },
                "blocks": [self.get_block_weights(blk) for blk in flat_blocks],
                "head": {
                    "fcw": self.fc.weight,
                    "fcb": self.fc.bias,
                },
            }
        stem_weights = weights["stem"]
        block_weights_list = weights["blocks"]
        head_weights = weights["head"]
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
        ddx_bn, dx_conv_d2, _ = conv_double_bwd( ddx_conv, ddconvw, None,dx_bn, stem_weights["convw"], x, groups_=Fuse, layout=self.layout, output_mask=(True, True, False))
        del dx_bn, ddx_conv, ddconvw, _
        # InstanceNorm statistics are per sample/channel; affine mapping is layout-dependent.
        # TODO:这里不知道为什么分开做了bwd。现在改成统一的接口了。
        # dx_bn_d2, _, dd_cur = instanceNorm_double_backwards_fn( x_bn, stem_weights["bnw"], None, ddx_bn, ddbnw, ddbnb, dx_block, layout=self.layout)
        # del dx_block, ddx_bn,ddbnw, ddbnb
        # dd_cur[x_block <= 0] = 0
        dd_cur, dx_bn_d2, _ = insNormNRelu_double_bwd(
            ddx_bn, ddbnw, ddbnb, dx_block,
            x_block, stem_weights["bnw"], x_bn,
            v_fuse=self.v_fuse, layout=self.layout
        )
        del dx_block, ddx_bn, ddbnw, ddbnb, _
        for i in range(len(flat_blocks)):
            blk = flat_blocks[i]
            activates_i = tape["blocks"][i]
            d_activates_i = d_activates_list[i]
            weights_i = block_weights_list[i]
            dd_weights_i = dd_weights_list[i]
            dd_cur, _ = BasicBlock_double_bwd(
                activates_i, d_activates_i, weights_i, dd_weights_i, ddgrad_in=dd_cur,
                SCstride=blk.conv1.stride[0], Fuse= Fuse,  v_fuse=self.v_fuse, layout=self.layout )
            clear_tensorlists(dd_weights_i)
            dd_weights_list[i] = None
            del activates_i, d_activates_i, weights_i, dd_weights_i

        ddx_pool = dd_cur
        del dd_cur
        # head double backward
        ddx_lin = adaptivepooling_double_bwd(ddx_pool)
        del ddx_pool
        ddx_out, dx_lin_d2, _ = linearFused_double_bwd( x_fc, head_weights["fcw"], dx_out, ddx_lin, ddfcw, ddfcb, Fuse, layout=self.layout )
        del dx_out, ddx_lin,ddfcw,ddfcb, _
        dx_out_d1 = crossEntropy_double_bwd(x_out, ddx_out, Fuse)
        del ddx_out,x_out
        dx_lin_d1, _, _ = linerFused_bwd( x_fc, head_weights["fcw"], grad_output=dx_out_d1, Fuse=Fuse, layout=self.layout)
        del dx_out_d1,x_fc, _
        dx_lin_d1 += dx_lin_d2
        del dx_lin_d2
        dx_lin_d1 = dx_lin_d1.view(x_pool.size(0), x_pool.size(1), 1, 1)
        g = adaptivepooling_bwd(x_pool, grad_output=dx_lin_d1)
        del dx_lin_d1, x_pool

        # blocks: bwd2_1, backward direction
        for i in reversed(range(len(flat_blocks))):
            blk = flat_blocks[i]
            activates_i = tape["blocks"][i]
            weights_i = block_weights_list[i]
            d2_activates_i = d_activates_list[i]
            if weights_i["convscw"] is None:
                d2_activates_i.setdefault("dx_bnsc_d2", None)
            g = BasicBlock_bwd2_1( activates_i, weights_i, d2_activates_i,grad_output=g, SCstride=blk.conv1.stride[0], Fuse = Fuse, v_fuse=self.v_fuse, layout=self.layout)
            clear_tensorlists(activates_i, d2_activates_i, weights_i)
            tape["blocks"][i] = None
            d_activates_list[i] = None
            del activates_i, d2_activates_i, weights_i
        dx_block_d1 = g
        del g

        dx_bn_d1, _, _ = insNormNRelu_bwd(
            x_bn, stem_weights["bnw"], x_block,
            grad_output=dx_block_d1,
            v_fuse=self.v_fuse,
            layout=self.layout
        )
        del x_block, x_bn, _

        del dx_block_d1
        dx_bn_d1 += dx_bn_d2
        del dx_bn_d2
        dx_conv, _, _ = conv_bwd( x, stem_weights["convw"], grad_output=dx_bn_d1,groups=Fuse, layout=self.layout)
        del dx_bn_d1, _
        dx_conv += dx_conv_d2
        return dx_conv

    def pack_recovered_dd(self, dd_tensors_all, x=None, tape=None, Fuse=None):
        ptr = 0
        # ---- stem ----
        ddconvw = dd_tensors_all[ptr]; ptr += 1
        ddbnw   = dd_tensors_all[ptr]; ptr += 1
        ddbnb   = dd_tensors_all[ptr]; ptr += 1
        # ---- blocks ----
        dd_weights_list = []
        flat_blocks = self.get_flat_blocks()
        for blk in flat_blocks:
            dd_weights_i = {
                "ddconv1w": dd_tensors_all[ptr],
                "ddbn1w":   dd_tensors_all[ptr + 1],
                "ddbn1b":   dd_tensors_all[ptr + 2],
                "ddconv2w": dd_tensors_all[ptr + 3],
                "ddbn2w":   dd_tensors_all[ptr + 4],
                "ddbn2b":   dd_tensors_all[ptr + 5],
                "ddconvscw": None,
                "ddbnscw":   None,
                "ddbnscb":   None,
            }
            ptr += 6
            if blk.convsc is not None:
                dd_weights_i["ddconvscw"] = dd_tensors_all[ptr]; ptr += 1
                dd_weights_i["ddbnscw"]   = dd_tensors_all[ptr]; ptr += 1
                dd_weights_i["ddbnscb"]   = dd_tensors_all[ptr]; ptr += 1

            dd_weights_list.append(dd_weights_i)
        # ---- head ----
        ddfcw = dd_tensors_all[ptr]; ptr += 1
        ddfcb = dd_tensors_all[ptr]; ptr += 1

        assert ptr == len(dd_tensors_all), (
            f"dd tensor parse mismatch: used {ptr}, total {len(dd_tensors_all)}"
        )
        if x is None and tape is not None:
            x = tape["stem"]["x_conv"]
        dd_work = {
            "ddconvw": ddconvw,
            "ddbnw": ddbnw,
            "ddbnb": ddbnb,
            "ddfcw": ddfcw,
            "ddfcb": ddfcb,
        }
        if x is not None:
            dd_work["ddx_conv"] = torch.zeros_like(x)
        return dd_work, dd_weights_list










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
        weights=None,
    ):
        if Fuse is None:
            Fuse = self.Fuse
        if dd_weights is None:
            dd_weights = {}
        if weights is None:
            qkv_w = self.qkv.weight
            out_proj_w = self.out_proj.weight
        else:
            qkv_w = weights["qkvw"]
            out_proj_w = weights["outprojw"]
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
        dd_dqkv_linear, dx_d2, _ = grouped_linear_double_bwd( x=x, w=qkv_w, grad_output=dqkv_linear,\
              gg_grad_input=ddgrad_in,  gg_grad_w=ddqkvw,  gg_grad_b=ddqkvb,  Fuse=Fuse, )
        # dqkv_linear 用完，可以显式删局部引用
        del dqkv_linear, _
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
            w=out_proj_w,
            grad_output=grad_output,
            gg_grad_input=dd_dout_merge,
            gg_grad_w=ddoutprojw,
            gg_grad_b=ddoutprojb,
            Fuse=Fuse,
        )
        del grad_output, dd_dout_merge, _
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
        weights=None,
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
        if weights is None:
            qkv_w = self.qkv.weight
            out_proj_w = self.out_proj.weight
        else:
            qkv_w = weights["qkvw"]
            out_proj_w = weights["outprojw"]
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
            out_proj_w,
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
        dx, _, _ = grouped_linear_bwd(x,qkv_w,grad_output=dqkv_linear,Fuse=Fuse )
        if dx_d2 is not None:
            assert dx_d2.shape == dx.shape
            dx = dx + dx_d2

        return dx

class TransformerBlock_Fused(nn.Module):
    def __init__(
        self,
        emb_size=128,
        heads=4,
        mlp_dim=256,
        dropout=0.0,
        Fuse=1,
        v_fuse=False,
    ):
        super().__init__()
        self.Fuse = Fuse
        self.v_fuse = v_fuse
        self.embed_dim = emb_size
        self.emb_size = emb_size
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.norm1 = GroupedLayerNorm(emb_size, Fuse)
        self.attn = MultiHeadSelfAttention_Fused(
            embed_dim=emb_size,
            num_heads=heads,
            dropout=dropout,
            Fuse=Fuse,
        )
        self.norm2 = GroupedLayerNorm(emb_size, Fuse)
        self.fc1 = GroupedLinear(emb_size, mlp_dim, Fuse)
        self.act = nn.GELU()
        self.dp1 = nn.Dropout(dropout)
        self.fc2 = GroupedLinear(mlp_dim, emb_size, Fuse)
        self.dp2 = nn.Dropout(dropout)

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

        # dp1: after GELU
        x_dp1, dp1_mask = dropout_fwd(
            x_gelu,
            p=self.dropout,
            training=self.training,
        )
        x_fc2 = self.fc2(x_dp1)
        # dp2: after fc2
        x_dp2, dp2_mask = dropout_fwd(
            x_fc2,
            p=self.dropout,
            training=self.training,
        )
        x_out = x_res1 + x_dp2
        tape = {
            "x_in": x_in,
            "attn": attn_tape,
            "x_res1": x_res1,
            "x_norm2": x_norm2,
            "x_fc1": x_fc1,
            "x_dp1": x_dp1,
            "dp1_mask": dp1_mask,
            "dp2_mask": dp2_mask,
        }
        return x_out, tape

    def run_first_bwd(self, tape, grad_output, Fuse=None):
        """
        TransformerBlock_Fused first backward with dp1/dp2.

        forward MLP path:
            x_res1
            -> norm2
            -> fc1
            -> gelu
            -> dp1
            -> fc2
            -> dp2
            -> residual add

        dropout has no parameters, but its masks must come from tape.
        """
        if Fuse is None:
            Fuse = self.Fuse

        x_in = tape["x_in"]
        attn_tape = tape["attn"]
        x_res1 = tape["x_res1"]
        x_norm2 = tape["x_norm2"]
        x_fc1 = tape["x_fc1"]
        x_dp1 = tape["x_dp1"]
        dp1_mask = tape["dp1_mask"]
        dp2_mask = tape["dp2_mask"]

        assert grad_output.shape == x_in.shape

        # ==================================================
        # 1. x_out = x_res1 + x_dp2
        # ==================================================
        dx_res1 = grad_output
        dx_dp2 = grad_output

        # ==================================================
        # 2. x_dp2 = dropout(x_fc2)
        # ==================================================
        dx_fc2 = dropout_bwd(
            dx_dp2,
            dp2_mask,
        )

        # ==================================================
        # 3. x_fc2 = fc2(x_dp1)
        # ==================================================
        dx_dp1, dfc2w, dfc2b = grouped_linear_bwd(
            x_dp1,
            self.fc2.weight,
            grad_output=dx_fc2,
            Fuse=Fuse,
        )

        # ==================================================
        # 4. x_dp1 = dropout(x_gelu) + gelu
        # ==================================================
        dx_fc1 = geluDropout_bwd(
            x=x_fc1,
            mask=dp1_mask,
            dout=dx_dp1,
            v_fuse=self.v_fuse,
        )

        # ==================================================
        # 6. x_fc1 = fc1(x_norm2)
        # ==================================================
        dx_norm2, dfc1w, dfc1b = grouped_linear_bwd(
            x_norm2,
            self.fc1.weight,
            grad_output=dx_fc1,
            Fuse=Fuse,
        )

        # ==================================================
        # 7. x_norm2 = norm2(x_res1)
        # ==================================================
        dx_res1_from_norm2, dnorm2w, dnorm2b = grouped_layernorm_backward(
            x_res1,
            self.norm2.weight,
            Fuse=Fuse,
            grad_output=dx_norm2,
        )

        # x_res1 has two outgoing paths:
        #   1. residual skip to output
        #   2. norm2 -> MLP branch
        dx_res1_total = dx_res1 + dx_res1_from_norm2

        # ==================================================
        # 8. x_res1 = x_in + x_attn
        # ==================================================
        dx_in_from_skip = dx_res1_total
        dx_attn = dx_res1_total

        # ==================================================
        # 9. x_attn = attn(norm1(x_in))
        # ==================================================
        dx_norm1, d_attn_activates, d_attn_weights, d_attn_weights_all = self.attn.run_first_bwd(
            attn_tape,
            grad_output=dx_attn,
            Fuse=Fuse,
        )

        # ==================================================
        # 10. x_norm1 = norm1(x_in)
        # ==================================================
        dx_in_from_norm1, dnorm1w, dnorm1b = grouped_layernorm_backward(
            x_in,
            self.norm1.weight,
            Fuse=Fuse,
            grad_output=dx_norm1,
        )

        dx_in = dx_in_from_skip + dx_in_from_norm1

        # Only keep first-bwd activations that double-bwd needs.
        d_activates = {
            "dx_fc2": dx_fc2,
            "dx_dp1": dx_dp1,
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

        # Parameter order unchanged:
        # norm1, attn, norm2, fc1, fc2.
        # Dropout has no parameters.
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
        weights=None,
    ):
        """
        TransformerBlock_Fused double backward with dp1/dp2.

        first-bwd MLP path:
            grad_output
            -> dp2_bwd
            -> fc2_bwd
            -> dp1_bwd
            -> gelu_bwd
            -> fc1_bwd
            -> norm2_bwd
            -> residual / attn / norm1

        Dropout mask is fixed from forward.
        Therefore dropout double-bwd only propagates cotangent to its first-bwd
        grad_output. It does not create forward-activation d2.
        """
        if Fuse is None:
            Fuse = self.Fuse
        if weights is None:
            norm1_w = self.norm1.weight
            norm2_w = self.norm2.weight
            fc1_w = self.fc1.weight
            fc2_w = self.fc2.weight
            attn_weights = None
        else:
            norm1_w = weights["norm1w"]
            norm2_w = weights["norm2w"]
            fc1_w = weights["fc1w"]
            fc2_w = weights["fc2w"]
            attn_weights = weights.get("attn", None)

        x_in = tape["x_in"]
        attn_tape = tape["attn"]
        x_res1 = tape["x_res1"]
        x_norm2 = tape["x_norm2"]
        x_fc1 = tape["x_fc1"]
        x_dp1 = tape["x_dp1"]
        dp1_mask = tape["dp1_mask"]
        dp2_mask = tape["dp2_mask"]

        dx_norm1 = d_activates.pop("dx_norm1")
        dx_norm2 = d_activates.pop("dx_norm2")
        dx_fc1 = d_activates.pop("dx_fc1")
        dx_dp1 = d_activates.pop("dx_dp1")
        dx_fc2 = d_activates.pop("dx_fc2")
        d_attn_activates = d_activates.pop("attn")
        d_activates.clear()

        if ddgrad_in is None:
            ddgrad_in = torch.zeros_like(x_in)

        # ==================================================
        # dx_in = dx_in_from_skip + dx_in_from_norm1
        # ==================================================
        dd_dx_in_from_skip = ddgrad_in
        dd_dx_in_from_norm1 = ddgrad_in

        # ==================================================
        # norm1 double-bwd:
        #   dx_in_from_norm1 = LN_bwd(x_in, norm1.weight, dx_norm1)
        # ==================================================
        dx_in_d2_from_norm1, _, dd_dx_norm1 = grouped_layernorm_double_bwd_fn(
            x=x_in,
            weight=norm1_w,
            ggX=dd_dx_in_from_norm1,
            ggW=dd_weights.get("ddnorm1w", None),
            ggB=dd_weights.get("ddnorm1b", None),
            gO=dx_norm1,
            Fuse=Fuse,
        )

        del dx_norm1, dd_dx_in_from_norm1, _
        d_activates["dx_in_d2"] = dx_in_d2_from_norm1

        # ==================================================
        # attention double-bwd:
        #   dx_norm1 = attn_bwd(..., dx_attn)
        # returns cotangent wrt dx_attn
        # ==================================================
        dd_attn_weights = dd_weights.get("attn", None)

        dd_dx_attn, d_attn_activates = self.attn.run_double_bwd(
            tape=attn_tape,
            d_activates=d_attn_activates,
            dd_weights=dd_attn_weights,
            ddgrad_in=dd_dx_norm1,
            Fuse=Fuse,
            weights=attn_weights,
        )

        del dd_dx_norm1
        d_activates["attn"] = d_attn_activates

        # ==================================================
        # dx_in_from_skip = dx_res1_total
        # dx_attn         = dx_res1_total
        # ==================================================
        dd_dx_res1_total = dd_dx_in_from_skip + dd_dx_attn
        del dd_dx_attn, dd_dx_in_from_skip, ddgrad_in

        # ==================================================
        # dx_res1_total = dx_res1 + dx_res1_from_norm2
        # ==================================================
        dd_dx_res1 = dd_dx_res1_total
        dd_dx_res1_from_norm2 = dd_dx_res1_total

        # ==================================================
        # norm2 double-bwd:
        #   dx_res1_from_norm2 = LN_bwd(x_res1, norm2.weight, dx_norm2)
        # ==================================================
        dx_res1_d2_from_norm2, _, dd_dx_norm2 = grouped_layernorm_double_bwd_fn(
            x=x_res1,
            weight=norm2_w,
            ggX=dd_dx_res1_from_norm2,
            ggW=dd_weights.get("ddnorm2w", None),
            ggB=dd_weights.get("ddnorm2b", None),
            gO=dx_norm2,
            Fuse=Fuse,
        )

        del dx_norm2, dd_dx_res1_from_norm2, dd_dx_res1_total, _
        d_activates["dx_res1_d2"] = dx_res1_d2_from_norm2

        # ==================================================
        # fc1 double-bwd:
        #   dx_norm2, dfc1w, dfc1b = linear_bwd(x_norm2, fc1.weight, dx_fc1)
        # ==================================================
        dd_dx_fc1, dx_norm2_d2, _ = grouped_linear_double_bwd(
            x=x_norm2,
            w=fc1_w,
            grad_output=dx_fc1,
            gg_grad_input=dd_dx_norm2,
            gg_grad_w=dd_weights.get("ddfc1w", None),
            gg_grad_b=dd_weights.get("ddfc1b", None),
            Fuse=Fuse,
        )

        del dx_fc1, dd_dx_norm2, _
        d_activates["dx_norm2_d2"] = dx_norm2_d2

        # ==================================================
        # GELU double-bwd:
        #   dx_fc1 = gelu_bwd(x_fc1, dx_gelu)
        # ==================================================
        dx_fc1_d2, dd_dx_dp1 = geluDropout_double_bwd(
            x=x_fc1,
            mask=dp1_mask,
            dout=dx_dp1,
            ddx=dd_dx_fc1,
            v_fuse=self.v_fuse,
        )

        del dx_dp1, dd_dx_fc1
        d_activates["dx_fc1_d2"] = dx_fc1_d2
        # ==================================================
        # fc2 double-bwd:
        #   dx_dp1, dfc2w, dfc2b = linear_bwd(x_dp1, fc2.weight, dx_fc2)
        #
        # Important:
        #   forward input of fc2 is x_dp1, not x_gelu.
        #   So the d2 contribution is dx_dp1_d2.
        # ==================================================
        dd_dx_fc2, dx_dp1_d2, _ = grouped_linear_double_bwd(
            x=x_dp1,
            w=fc2_w,
            grad_output=dx_fc2,
            gg_grad_input=dd_dx_dp1,
            gg_grad_w=dd_weights.get("ddfc2w", None),
            gg_grad_b=dd_weights.get("ddfc2b", None),
            Fuse=Fuse,
        )

        del dx_fc2, dd_dx_dp1, _
        d_activates["dx_dp1_d2"] = dx_dp1_d2
        # ==================================================
        # dp2 double-bwd:
        #   dx_fc2 = dropout_bwd(dx_dp2, dp2_mask)
        #
        # No forward-activation d2 is produced by dropout.
        # It only propagates cotangent wrt dx_fc2 back to dx_dp2.
        # ==================================================
        dd_dx_dp2 = dropout_double_bwd(
            dd_dx_fc2,
            dp2_mask,
        )

        # ==================================================
        # first-bwd:
        #   dx_res1 = grad_output
        #   dx_dp2  = grad_output
        #
        # Therefore cotangent wrt original grad_output is summed.
        # ==================================================
        del dd_dx_fc2
        dd_grad_output = dd_dx_res1 + dd_dx_dp2

        return dd_grad_output, d_activates

    def run_bwd2_1(
        self,
        tape,
        d_activates,
        grad_output,
        Fuse=None,
        weights=None,
    ):
        """
        Re-run corrected first backward with dp1/dp2 injections.

        bwd2_1 MLP path:
            grad_output
            -> dp2_bwd
            -> fc2_bwd
            -> inject dx_dp1_d2
            -> dp1_bwd
            -> gelu_bwd
            -> inject dx_fc1_d2
            -> fc1_bwd
            -> inject dx_norm2_d2
            -> norm2_bwd
            -> inject dx_res1_d2 before residual split
            -> attn_bwd2_1
            -> norm1_bwd
            -> inject dx_in_d2
        """
        if Fuse is None:
            Fuse = self.Fuse
        if weights is None:
            norm1_w = self.norm1.weight
            norm2_w = self.norm2.weight
            fc1_w = self.fc1.weight
            fc2_w = self.fc2.weight
            attn_weights = None
        else:
            norm1_w = weights["norm1w"]
            norm2_w = weights["norm2w"]
            fc1_w = weights["fc1w"]
            fc2_w = weights["fc2w"]
            attn_weights = weights.get("attn", None)

        x_in = tape["x_in"]
        attn_tape = tape["attn"]
        x_res1 = tape["x_res1"]
        x_norm2 = tape["x_norm2"]
        x_fc1 = tape["x_fc1"]
        x_dp1 = tape["x_dp1"]
        dp1_mask = tape["dp1_mask"]
        dp2_mask = tape["dp2_mask"]

        assert grad_output.shape == x_in.shape

        # ==================================================
        # 1. x_out = x_res1 + x_dp2
        # ==================================================
        dx_res1 = grad_output
        dx_dp2 = grad_output

        # ==================================================
        # 2. x_dp2 = dropout(x_fc2)
        # ==================================================
        dx_fc2 = dropout_bwd(
            dx_dp2,
            dp2_mask,
        )

        # ==================================================
        # 3. x_fc2 = fc2(x_dp1)
        # ==================================================
        dx_dp1, _, _ = grouped_linear_bwd(
            x_dp1,
            fc2_w,
            grad_output=dx_fc2,
            Fuse=Fuse,
        )

        # Inject d2 wrt forward x_dp1.
        # This must happen after fc2 bwd and before dp1 bwd.
        dx_dp1_d2 = d_activates.pop("dx_dp1_d2")
        assert dx_dp1_d2.shape == dx_dp1.shape
        dx_dp1 = dx_dp1 + dx_dp1_d2

        # ==================================================
        # 4. x_dp1 = dropout(x_gelu)
        # ==================================================
        dx_fc1 = geluDropout_bwd(
            x=x_fc1,
            mask=dp1_mask,
            dout=dx_dp1,
            v_fuse=self.v_fuse,
        )
        
        # Inject d2 wrt forward x_fc1.
        dx_fc1_d2 = d_activates.pop("dx_fc1_d2")
        assert dx_fc1_d2.shape == dx_fc1.shape
        dx_fc1 = dx_fc1 + dx_fc1_d2

        # ==================================================
        # 6. x_fc1 = fc1(x_norm2)
        # ==================================================
        dx_norm2, _, _ = grouped_linear_bwd(
            x_norm2,
            fc1_w,
            grad_output=dx_fc1,
            Fuse=Fuse,
        )

        # Inject d2 wrt forward x_norm2.
        dx_norm2_d2 = d_activates.pop("dx_norm2_d2")
        assert dx_norm2_d2.shape == dx_norm2.shape
        dx_norm2 = dx_norm2 + dx_norm2_d2

        # ==================================================
        # 7. x_norm2 = norm2(x_res1)
        # ==================================================
        dx_res1_from_norm2, _, _ = grouped_layernorm_backward(
            x_res1,
            norm2_w,
            Fuse=Fuse,
            grad_output=dx_norm2,
        )

        # x_res1 has two outgoing paths:
        #   1. output residual
        #   2. norm2 -> MLP
        dx_res1_total = dx_res1 + dx_res1_from_norm2

        # Inject d2 wrt forward x_res1.
        # Important: before residual split into skip and attention.
        dx_res1_d2 = d_activates.pop("dx_res1_d2")
        assert dx_res1_d2.shape == dx_res1_total.shape
        dx_res1_total = dx_res1_total + dx_res1_d2

        # ==================================================
        # 8. x_res1 = x_in + x_attn
        # ==================================================
        dx_in_from_skip = dx_res1_total
        dx_attn = dx_res1_total

        # ==================================================
        # 9. x_attn = attn(norm1(x_in))
        # ==================================================
        d_attn_activates = d_activates.pop("attn")

        dx_norm1 = self.attn.run_bwd2_1(
            tape=attn_tape,
            d_activates=d_attn_activates,
            grad_output=dx_attn,
            Fuse=Fuse,
            weights=attn_weights,
        )

        # ==================================================
        # 10. x_norm1 = norm1(x_in)
        # ==================================================
        dx_in_from_norm1, _, _ = grouped_layernorm_backward(
            x_in,
            norm1_w,
            Fuse=Fuse,
            grad_output=dx_norm1,
        )

        dx_in = dx_in_from_skip + dx_in_from_norm1

        # Inject d2 wrt forward x_in.
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


class ViT_FlexFuse(nn.Module):
    # 请注意一下，ViT的fuse 规则和传统的不太一样。cls token不是在第一维做fuse
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        emb_size=128,
        depth=6,
        heads=4,
        mlp_dim=256,
        dropout=0.0,
        Fuse=1,
        v_fuse=False,
    ):
        super().__init__()
        self.Fuse = Fuse
        self.in_channels = in_channels
        self.num_classes = num_classes
        # 保留旧名字，避免你后面的 bwd 代码要大改
        self.embed_dim = emb_size
        self.emb_size = emb_size
        assert img_size % patch_size == 0
        self.image_size = img_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels * Fuse,
            out_channels=emb_size * Fuse,
            kernel_size=patch_size,
            stride=patch_size,
            groups=Fuse,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, Fuse, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, Fuse, self.num_patches + 1, emb_size))
        self.blocks = nn.ModuleList([
            TransformerBlock_Fused(
                emb_size=emb_size,
                heads=heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                Fuse=Fuse,
                v_fuse=v_fuse
            )
            for _ in range(depth)
        ])

        self.norm = GroupedLayerNorm(emb_size, Fuse)
        self.head = GroupedLinear(emb_size, num_classes, Fuse)
        self._init_weights()

    def get_flat_blocks(self):
        return list(self.blocks)
    def forward(self, x):
        Fuse = self.Fuse
        B = x.shape[0]
        D = self.embed_dim
        tape = {  "patch": {},  "blocks": [],  "head": {},}
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
        # x_block, block_tape = self.block(x_pos)
        # tape["block"] = block_tape
        h = x_pos
        for blk in self.blocks:
            h, block_tape = blk(h)
            tape["blocks"].append(block_tape)
        x_norm_in = h
        # [B, Fuse, N+1, D]
        # x_norm_in = x_block
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
    

    def _active_fuse_range(self, fuse_mask_list):
        if fuse_mask_list is None:
            return None
        Fuse = len(fuse_mask_list)
        bwd_Fuse = sum(fuse_mask_list)
        if bwd_Fuse == Fuse:
            return None
        active_fuse_ids = [i for i, m in enumerate(fuse_mask_list) if m == 1]
        assert len(active_fuse_ids) == bwd_Fuse and bwd_Fuse > 0
        start = active_fuse_ids[0]
        end = start + bwd_Fuse
        assert active_fuse_ids == list(range(start, end)), \
            "fuse_mask_list must keep active branches contiguous"
        return start, end, Fuse

    def _split_tensor_by_fuse(self, p, fuse_mask_list, layout="bf", cache=None):
        if p is None or (not torch.is_tensor(p)):
            return p
        active_range = self._active_fuse_range(fuse_mask_list)
        if active_range is None:
            return p
        start, end, Fuse = active_range
        key = (id(p), layout)
        if cache is not None and key in cache:
            src_weak, cached = cache[key]
            if src_weak() is p:
                return cached

        if layout == "bf":
            # ViT normal activations: [B, Fuse, ...]
            assert p.shape[1] % Fuse == 0 or p.shape[1] == Fuse
            view = p[:, start:end, ...]
            result = view.clone(memory_format=torch.contiguous_format)
        elif layout == "channel":
            # Patch conv input/output: [B, C*Fuse, H, W]
            c = p.shape[1]
            assert c % Fuse == 0, f"Cannot split channel-fused tensor shape={tuple(p.shape)} by Fuse={Fuse}"
            block = c // Fuse
            view = p[:, start * block:end * block, ...]
            result = view.clone(memory_format=torch.contiguous_format)
        elif layout == "logits":
            # ViT logits are reshape(B*Fuse, C), ordered b0f0,b0f1,b1f0,b1f1,...
            assert p.shape[0] % Fuse == 0, f"Cannot split logits shape={tuple(p.shape)} by Fuse={Fuse}"
            B = p.shape[0] // Fuse
            rest = p.shape[1:]
            view = p.view(B, Fuse, *rest)[:, start:end, ...]
            result = view.clone(memory_format=torch.contiguous_format).view(B * (end - start), *rest)
        elif layout == "param_token":
            # cls_token / pos_embed: [1, Fuse, ..., D]
            assert p.shape[0] == 1 and p.shape[1] == Fuse, \
                f"Expected token parameter layout [1,Fuse,...], got shape={tuple(p.shape)}"
            view = p[:, start:end, ...]
            result = view.clone(memory_format=torch.contiguous_format)
        else:
            raise ValueError(f"Unknown fuse split layout: {layout}")

        if cache is not None:
            import weakref
            cache[key] = (weakref.ref(p), result)
        return result

    def _split_nested_bf_by_fuse(self, obj, fuse_mask_list, cache=None):
        """
        Split TransformerBlock / attention tapes and d_activates.
        Inside blocks, every saved tensor stores Fuse at dim=1:
            block activations: [B,Fuse,N,D]
            attention q/k/v:   [B,Fuse,H,N,Dh]
            dropout masks:     [B,Fuse,N,D]
        """
        if self._active_fuse_range(fuse_mask_list) is None:
            return obj
        if torch.is_tensor(obj):
            return self._split_tensor_by_fuse(obj, fuse_mask_list, layout="bf", cache=cache)
        if isinstance(obj, dict):
            for k in list(obj.keys()):
                obj[k] = self._split_nested_bf_by_fuse(obj[k], fuse_mask_list, cache=cache)
            return obj
        if isinstance(obj, list):
            for i in range(len(obj)):
                obj[i] = self._split_nested_bf_by_fuse(obj[i], fuse_mask_list, cache=cache)
            return obj
        return obj

    def _split_saved_for_double_bwd(self, tape, d_activates, fuse_mask_list):
        """
        first-bwd 用完整 Fuse 计算 dW；随后把 double-bwd / bwd2_1 会继续使用的
        tape 和 first-bwd 激活切到 active bwd_Fuse。

        这里不能照搬 ResNet：
        - ViT block 内部激活是 [B, Fuse, N, D]，Fuse 在 dim=1。
        - patch conv 的 x_in/x_patch/dx_patch 是 [B, C*Fuse, H, W]，Fuse 在 channel 内。
        - x_out/dx_out 对应 CE 是 [B*Fuse, C]，但顺序是 b0f0,b0f1,...，不能直接按 batch 连续切。
        - cls_token/pos_embed 参数本身是 [1, Fuse, ..., D]，后面 pack weight 时也要按 dim=1 处理。
        """
        if self._active_fuse_range(fuse_mask_list) is None:
            return

        split_cache = {}
        patch = tape["patch"]
        patch["x_in"] = self._split_tensor_by_fuse(patch["x_in"], fuse_mask_list, layout="channel", cache=split_cache)
        patch["x_patch"] = self._split_tensor_by_fuse(patch["x_patch"], fuse_mask_list, layout="channel", cache=split_cache)

        head = tape["head"]
        head["x_norm_in"] = self._split_tensor_by_fuse(head["x_norm_in"], fuse_mask_list, layout="bf", cache=split_cache)
        head["x_cls"] = self._split_tensor_by_fuse(head["x_cls"], fuse_mask_list, layout="bf", cache=split_cache)
        head["x_out"] = self._split_tensor_by_fuse(head["x_out"], fuse_mask_list, layout="logits", cache=split_cache)

        d_activates["dx_head"] = self._split_tensor_by_fuse(d_activates["dx_head"], fuse_mask_list, layout="bf", cache=split_cache)
        d_activates["dx_norm"] = self._split_tensor_by_fuse(d_activates["dx_norm"], fuse_mask_list, layout="bf", cache=split_cache)
        d_activates["dx_patch"] = self._split_tensor_by_fuse(d_activates["dx_patch"], fuse_mask_list, layout="channel", cache=split_cache)

        self._split_nested_bf_by_fuse(tape["blocks"], fuse_mask_list, cache=split_cache)
        self._split_nested_bf_by_fuse(d_activates["blocks"], fuse_mask_list, cache=split_cache)

    def _reshape_cls_token_for_fuse(self, t, Fuse):
        if t is None:
            return None
        if t.ndim == 4 and t.shape[0] == 1 and t.shape[1] == Fuse:
            return t.contiguous()
        if t.ndim == 4 and t.shape[0] == Fuse and t.shape[1] == 1:
            # recover_params from base [1, 1, D] / [1, 1, 1, D] can give [Fuse, 1, 1, D]
            return t.reshape(1, Fuse, 1, t.shape[-1]).contiguous()
        if t.ndim == 3 and t.shape[0] == Fuse:
            # recover_params from base [1, D] or [1, 1, D] usually gives [Fuse, 1, D]
            return t.reshape(1, Fuse, 1, t.shape[-1]).contiguous()
        if t.ndim == 2 and t.shape[0] == Fuse:
            return t.reshape(1, Fuse, 1, t.shape[-1]).contiguous()
        raise AssertionError(f"Cannot reshape cls_token tensor shape={tuple(t.shape)} for Fuse={Fuse}")

    def _reshape_pos_embed_for_fuse(self, t, Fuse):
        if t is None:
            return None
        if t.ndim == 4 and t.shape[0] == 1 and t.shape[1] == Fuse:
            return t.contiguous()
        if t.ndim == 4 and t.shape[0] == Fuse and t.shape[1] == 1:
            # recover_params from base [1, 1, N+1, D] can give [Fuse, 1, N+1, D]
            return t.reshape(1, Fuse, t.shape[-2], t.shape[-1]).contiguous()
        if t.ndim == 3 and t.shape[0] == Fuse:
            # recover_params from base [1, N+1, D] gives [Fuse, N+1, D]
            return t.reshape(1, Fuse, t.shape[-2], t.shape[-1]).contiguous()
        raise AssertionError(f"Cannot reshape pos_embed tensor shape={tuple(t.shape)} for Fuse={Fuse}")

    def pack_recovered_weights(self, tensors_all, Fuse=None):
        """
        Parse active/full recovered forward weights into the dict used by
        run_double_bwd / run_bwd2_1.

        tensors_all order matches d_weights_list_all and pack_recovered_dd:
            cls_token, pos_embed, patch_embed.weight, patch_embed.bias,
            blocks..., norm.weight, norm.bias, head.weight, head.bias
        """
        if Fuse is None:
            Fuse = self.Fuse
        ptr = 0
        cls_token = self._reshape_cls_token_for_fuse(tensors_all[ptr], Fuse); ptr += 1
        pos_embed = self._reshape_pos_embed_for_fuse(tensors_all[ptr], Fuse); ptr += 1
        patchw = tensors_all[ptr]; ptr += 1
        if self.patch_embed.bias is not None:
            patchb = tensors_all[ptr]; ptr += 1
        else:
            patchb = None

        block_weights = []
        for blk in self.get_flat_blocks():
            w_blk = {}
            w_blk["norm1w"] = tensors_all[ptr]; ptr += 1
            w_blk["norm1b"] = tensors_all[ptr]; ptr += 1
            w_blk["attn"] = {}
            w_blk["attn"]["qkvw"] = tensors_all[ptr]; ptr += 1
            if blk.attn.qkv.bias is not None:
                w_blk["attn"]["qkvb"] = tensors_all[ptr]; ptr += 1
            else:
                w_blk["attn"]["qkvb"] = None
            w_blk["attn"]["outprojw"] = tensors_all[ptr]; ptr += 1
            if blk.attn.out_proj.bias is not None:
                w_blk["attn"]["outprojb"] = tensors_all[ptr]; ptr += 1
            else:
                w_blk["attn"]["outprojb"] = None
            w_blk["norm2w"] = tensors_all[ptr]; ptr += 1
            w_blk["norm2b"] = tensors_all[ptr]; ptr += 1
            w_blk["fc1w"] = tensors_all[ptr]; ptr += 1
            if blk.fc1.bias is not None:
                w_blk["fc1b"] = tensors_all[ptr]; ptr += 1
            else:
                w_blk["fc1b"] = None
            w_blk["fc2w"] = tensors_all[ptr]; ptr += 1
            if blk.fc2.bias is not None:
                w_blk["fc2b"] = tensors_all[ptr]; ptr += 1
            else:
                w_blk["fc2b"] = None
            block_weights.append(w_blk)

        normw = tensors_all[ptr]; ptr += 1
        normb = tensors_all[ptr]; ptr += 1
        headw = tensors_all[ptr]; ptr += 1
        if self.head.bias is not None:
            headb = tensors_all[ptr]; ptr += 1
        else:
            headb = None
        assert ptr == len(tensors_all), f"ViT weight tensor parse mismatch: used {ptr}, total={len(tensors_all)}"
        return {
            "cls_token": cls_token,
            "pos_embed": pos_embed,
            "patch": {"patchw": patchw, "patchb": patchb},
            "blocks": block_weights,
            "head": {
                "normw": normw,
                "normb": normb,
                "headw": headw,
                "headb": headb,
            },
        }

    def run_first_bwd(self, tape, target, Fuse=None, fuse_mask_list=None):
        if Fuse is None:
            Fuse = self.Fuse
        patch = tape["patch"]
        head = tape["head"]
        # block_tape = tape["block"]
        block_tapes = tape["blocks"]
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
        # Compact saved tensors only after their last full-Fuse use.
        split_cache = {} if self._active_fuse_range(fuse_mask_list) is not None else None
        if split_cache is not None:
            for key in ("x_norm_in", "x_cls"):
                head[key] = self._split_tensor_by_fuse(head[key], fuse_mask_list, layout="bf", cache=split_cache)
            head["x_out"] = self._split_tensor_by_fuse(x_out, fuse_mask_list, layout="logits", cache=split_cache)
            dx_head = self._split_tensor_by_fuse(dx_head, fuse_mask_list, layout="bf", cache=split_cache)
            dx_norm = self._split_tensor_by_fuse(dx_norm, fuse_mask_list, layout="bf", cache=split_cache)
        del x_norm_in, x_cls, x_out, dx_out, dx_cls
        # dx_block: [B, Fuse, N+1, D]
        # dx_pos, d_block_activates, d_block_weights, d_block_weights_all = self.block.run_first_bwd( block_tape, grad_output=dx_block, Fuse=Fuse )
        flat_blocks = self.get_flat_blocks()
        d_block_activates_list = [None] * len(flat_blocks)
        d_block_weights_list = [None] * len(flat_blocks)
        d_block_weights_all_list = [None] * len(flat_blocks)
        g = dx_block
        del dx_block
        for i in reversed(range(len(flat_blocks))):
            g, d_act_i, d_w_i, d_w_all_i = flat_blocks[i].run_first_bwd(
                tape=block_tapes[i],
                grad_output=g,
                Fuse=Fuse,
            )
            d_block_activates_list[i] = d_act_i
            d_block_weights_list[i] = d_w_i
            d_block_weights_all_list[i] = d_w_all_i
            if split_cache is not None:
                self._split_nested_bf_by_fuse(block_tapes[i], fuse_mask_list, cache=split_cache)
                self._split_nested_bf_by_fuse(d_act_i, fuse_mask_list, cache=split_cache)
            del d_act_i, d_w_i, d_w_all_i
        dx_pos = g
        
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
            "blocks": d_block_activates_list,
        }
        d_weights = {
            "dpatchw": dpatchw,
            "dpatchb": dpatchb,
            "dcls_token": dcls_token,
            "dpos_embed": dpos_embed,
            "blocks": d_block_weights_list,
            "dnormw": dnormw,
            "dnormb": dnormb,
            "dheadw": dheadw,
            "dheadb": dheadb,
        }
        # 这个 list 用来做你后面的  weight_sum = [d.sum() for d in d_weights_list_all]
        d_weights_list_all = []
        # root parameters first
        d_weights_list_all.append(dcls_token)
        d_weights_list_all.append(dpos_embed)
        # then child modules in assignment order
        d_weights_list_all.append(dpatchw)
        if dpatchb is not None:
            d_weights_list_all.append(dpatchb)
        # blocks
        for block_grad_list in d_block_weights_all_list:
            for g_blk in block_grad_list:
                if g_blk is not None:
                    d_weights_list_all.append(g_blk)
        # norm, head
        d_weights_list_all.append(dnormw)
        d_weights_list_all.append(dnormb)
        d_weights_list_all.append(dheadw)
        if dheadb is not None:
            d_weights_list_all.append(dheadb)

        # 完整 Fuse 的 dW 已经收集完；这里只切 double-bwd / bwd2_1 后续会用到的 saved tensors。
        if split_cache is not None:
            patch["x_in"] = self._split_tensor_by_fuse(x_in, fuse_mask_list, layout="channel", cache=split_cache)
            patch["x_patch"] = self._split_tensor_by_fuse(x_patch, fuse_mask_list, layout="channel", cache=split_cache)
            d_activates["dx_patch"] = self._split_tensor_by_fuse(dx_patch, fuse_mask_list, layout="channel", cache=split_cache)
        return dx_in, d_activates, d_weights, d_weights_list_all

    def init_dd_weights(self):
        return {
            "ddpatchw": torch.ones_like(self.patch_embed.weight),
            "ddpatchb": torch.ones_like(self.patch_embed.bias) if self.patch_embed.bias is not None else None,
            "ddcls_token": torch.ones_like(self.cls_token),
            "ddpos_embed": torch.ones_like(self.pos_embed),
            "blocks": [
                blk.init_dd_weights()
                for blk in self.get_flat_blocks()
            ],
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

    def _run_double_bwd_stage(
        self,
        tape,
        d_activates,
        dd_weights,
        ddgrad_in=None,
        Fuse=None,
        weights=None):
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
        if weights is None:
            patch_w = self.patch_embed.weight
            cls_token_w = self.cls_token
            pos_embed_w = self.pos_embed
            block_weights_list = None
            norm_w = self.norm.weight
            head_w = self.head.weight
        else:
            patch_w = weights["patch"]["patchw"]
            cls_token_w = weights["cls_token"]
            pos_embed_w = weights["pos_embed"]
            block_weights_list = weights["blocks"]
            norm_w = weights["head"]["normw"]
            head_w = weights["head"]["headw"]
        patch = tape["patch"]
        head = tape["head"]
        # block_tape = tape["block"]
        block_tapes = tape["blocks"]
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
        x_pos = block_tapes[0]["x_in"]
        x_norm_in = head["x_norm_in"]
        x_cls = head["x_cls"]
        x_out = head["x_out"]
        dx_head = d_activates.pop("dx_head")
        dx_norm = d_activates.pop("dx_norm")
        dx_patch = d_activates.pop("dx_patch")
        # d_block_activates = d_activates.pop("blocks")

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
        dd_dx_patch, x_in_d2, _ = conv_double_bwd( ddgrad_in, ddpatchw, ddpatchb, dx_patch, patch_w,
            x_in, stride_=list(self.patch_embed.stride), padding_=list(self.patch_embed.padding), groups_=Fuse, output_mask=(True, True, False))
        del dx_patch, ddgrad_in, _
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
        del dd_dx_patch, dd_dx_flat, dd_dx_reshape
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
        del dd_dx_tokens
        ddcls_token = get_dd("ddcls_token", "dcls_token")
        if ddcls_token is not None:
            assert ddcls_token.shape == cls_token_w.shape
            dd_dx_pos[:, :, 0:1, :] = dd_dx_pos[:, :, 0:1, :] + ddcls_token.expand(B, -1, -1, -1)

        ddpos_embed = get_dd("ddpos_embed", "dpos_embed")
        if ddpos_embed is not None:
            assert ddpos_embed.shape == pos_embed_w.shape
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
        # dd_dx_block, d_block_activates = self.block.run_double_bwd(
        #     tape=block_tape,
        #     d_activates=d_block_activates,
        #     dd_weights=dd_weights["block"],
        #     ddgrad_in=dd_dx_pos,
        #     Fuse=Fuse,
        # )
        flat_blocks = self.get_flat_blocks()
        d_block_activates_list = d_activates.pop("blocks")
        dd_block_weights_list = dd_weights["blocks"]
        if block_weights_list is None:
            block_weights_list = [None] * len(flat_blocks)
        # double-bwd direction through first-bwd graph is forward block order.
        dd_cur = dd_dx_pos
        del dd_dx_pos
        for i in range(len(flat_blocks)):
            dd_cur, d_block_activates_list[i] = flat_blocks[i].run_double_bwd(
                tape=block_tapes[i],
                d_activates=d_block_activates_list[i],
                dd_weights=dd_block_weights_list[i],
                ddgrad_in=dd_cur,
                Fuse=Fuse,
                weights=block_weights_list[i],
            )
        dd_dx_block = dd_cur
        del dd_cur
        # d_activates["block"] = d_block_activates
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
            weight=norm_w,
            ggX=dd_dx_block,
            ggW=get_dd("ddnormw", "dnormw"),
            ggB=get_dd("ddnormb", "dnormb"),
            gO=dx_norm,
            Fuse=Fuse,
        )
        del dx_norm, dd_dx_block, _
        d_activates["x_norm_in_d2"] = x_norm_in_d2
        d_activates["blocks"] = d_block_activates_list
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
        del dd_dx_norm
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
            w=head_w,
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
        # d_activates["dd_dx_out"] = dd_dx_out
        return dd_dx_out, d_activates

    def run_double_bwd_full(
        self,
        tape,
        d_activates_list,
        dd_weights_list=None,
        d_head_tensors=None,
        dd_head_tensors=None,
        Fuse=None,
        weights=None,
        dd_tensors_all=None,
        d_bwd_state=None,
        weight_tensors_all=None,
    ):
        """
        统一的 top-level double-bwd 接口。

        ViT 内部仍然保留两阶段：_run_double_bwd_stage + run_bwd2_1，
        但主程序只需要调用本函数，并直接得到最终 input gradient。
        """
        if Fuse is None:
            Fuse = self.Fuse
        if dd_tensors_all is not None:
            if dd_weights_list is not None:
                raise ValueError("Pass either dd_tensors_all or dd_weights_list, not both")
            dd_weights_list = self.pack_recovered_dd(dd_tensors_all, Fuse=Fuse)
        if weight_tensors_all is not None:
            if weights is not None:
                raise ValueError("Pass either weight_tensors_all or weights, not both")
            weights = self.pack_recovered_weights(weight_tensors_all, Fuse=Fuse)
        if dd_weights_list is None:
            raise ValueError("run_double_bwd requires dd_tensors_all or dd_weights_list")

        ddgrad_in = None
        if dd_head_tensors is not None:
            ddgrad_in = dd_head_tensors.get("ddgrad_in",
                         dd_head_tensors.get("ddx_input", None))

        dd_dx_out, d_activates_list = self._run_double_bwd_stage(
            tape=tape,
            d_activates=d_activates_list,
            dd_weights=dd_weights_list,
            ddgrad_in=ddgrad_in,
            Fuse=Fuse,
            weights=weights,
        )
        return self.run_bwd2_1(
            tape=tape,
            d_activates=d_activates_list,
            dd_dx_out=dd_dx_out,
            Fuse=Fuse,
            weights=weights,
        )

    def run_bwd2_1(
        self,
        tape,
        d_activates,
        dd_dx_out,
        Fuse=None,
        weights=None,
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
        if weights is None:
            patch_w = self.patch_embed.weight
            block_weights_list = None
            norm_w = self.norm.weight
            head_w = self.head.weight
        else:
            patch_w = weights["patch"]["patchw"]
            block_weights_list = weights["blocks"]
            norm_w = weights["head"]["normw"]
            head_w = weights["head"]["headw"]

        patch = tape["patch"]
        head = tape["head"]
        # block_tape = tape["block"]
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
            head_w,
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
            norm_w,
            Fuse=Fuse,
            grad_output=dx_norm,
        )
        # Inject d2 wrt forward x_norm_in, i.e. block output.
        # x_norm_in_d2 = d_activates.get("x_norm_in_d2", None)
        x_norm_in_d2 = d_activates.pop("x_norm_in_d2")
        # d_block_activates = d_activates.pop("block")
        assert x_norm_in_d2.shape == dx_block.shape
        dx_block = dx_block + x_norm_in_d2
        # ==================================================
        # 5. TransformerBlock bwd2_1
        # ==================================================
        # dx_pos = self.block.run_bwd2_1(
        #     tape=block_tape,
        #     d_activates=d_block_activates,
        #     grad_output=dx_block,
        #     Fuse=Fuse,
        # )
        flat_blocks = self.get_flat_blocks()
        block_tapes = tape["blocks"]
        d_block_activates_list = d_activates.pop("blocks")

        assert len(block_tapes) == len(flat_blocks)
        assert len(d_block_activates_list) == len(flat_blocks)
        if block_weights_list is None:
            block_weights_list = [None] * len(flat_blocks)

        # bwd2_1 follows normal backward direction: reverse block order.
        g = dx_block
        del dx_block, dx_norm, dx_cls, dx_head, dx_out_d1, x_norm_in_d2, x_cls_d2, _
        head.clear()
        del x_norm_in, x_cls, x_out

        for i in reversed(range(len(flat_blocks))):
            g = flat_blocks[i].run_bwd2_1(
                tape=block_tapes[i],
                d_activates=d_block_activates_list[i],
                grad_output=g,
                Fuse=Fuse,
                weights=block_weights_list[i],
            )
            block_tapes[i] = None
            d_block_activates_list[i] = None

        dx_pos = g
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
        dx_in, _, _ = conv_bwd( x_in, patch_w, grad_output=dx_patch,
            stride=self.patch_embed. stride[0], padding=self.patch_embed.padding[0],groups=Fuse )
        # Inject d2 wrt forward input x_in from patch conv double-bwd.
        x_in_d2 = d_activates.pop("x_in_d2")
        assert x_in_d2.shape == dx_in.shape
        dx_in = dx_in + x_in_d2
        patch.clear()
        return dx_in

    def pack_recovered_dd(self, dd_tensors_all, Fuse=None):
        """
        Parse flat recovered dd tensors into the dd_weights dict expected by
        ViT_Fused.run_double_bwd.

        This assumes dd_tensors_all has the same order as d_weights_list_all:

            cls_token, pos_embed,
            patch_embed.weight, patch_embed.bias,
            blocks...
            norm.weight, norm.bias,
            head.weight, head.bias
        """
        if Fuse is None:
            Fuse = self.Fuse
        ptr = 0

        ddcls_token = self._reshape_cls_token_for_fuse(dd_tensors_all[ptr], Fuse)
        ptr += 1

        ddpos_embed = self._reshape_pos_embed_for_fuse(dd_tensors_all[ptr], Fuse)
        ptr += 1

        ddpatchw = dd_tensors_all[ptr]
        ptr += 1

        if self.patch_embed.bias is not None:
            ddpatchb = dd_tensors_all[ptr]
            ptr += 1
        else:
            ddpatchb = None

        dd_blocks = []
        for blk in self.get_flat_blocks():
            dd_blk = {}

            dd_blk["ddnorm1w"] = dd_tensors_all[ptr]
            ptr += 1
            dd_blk["ddnorm1b"] = dd_tensors_all[ptr]
            ptr += 1

            dd_blk["attn"] = {}
            dd_blk["attn"]["ddqkvw"] = dd_tensors_all[ptr]
            ptr += 1

            if blk.attn.qkv.bias is not None:
                dd_blk["attn"]["ddqkvb"] = dd_tensors_all[ptr]
                ptr += 1
            else:
                dd_blk["attn"]["ddqkvb"] = None

            dd_blk["attn"]["ddoutprojw"] = dd_tensors_all[ptr]
            ptr += 1

            if blk.attn.out_proj.bias is not None:
                dd_blk["attn"]["ddoutprojb"] = dd_tensors_all[ptr]
                ptr += 1
            else:
                dd_blk["attn"]["ddoutprojb"] = None

            dd_blk["ddnorm2w"] = dd_tensors_all[ptr]
            ptr += 1
            dd_blk["ddnorm2b"] = dd_tensors_all[ptr]
            ptr += 1

            dd_blk["ddfc1w"] = dd_tensors_all[ptr]
            ptr += 1

            if blk.fc1.bias is not None:
                dd_blk["ddfc1b"] = dd_tensors_all[ptr]
                ptr += 1
            else:
                dd_blk["ddfc1b"] = None

            dd_blk["ddfc2w"] = dd_tensors_all[ptr]
            ptr += 1

            if blk.fc2.bias is not None:
                dd_blk["ddfc2b"] = dd_tensors_all[ptr]
                ptr += 1
            else:
                dd_blk["ddfc2b"] = None

            dd_blocks.append(dd_blk)

        ddnormw = dd_tensors_all[ptr]
        ptr += 1
        ddnormb = dd_tensors_all[ptr]
        ptr += 1

        ddheadw = dd_tensors_all[ptr]
        ptr += 1

        if self.head.bias is not None:
            ddheadb = dd_tensors_all[ptr]
            ptr += 1
        else:
            ddheadb = None

        assert ptr == len(dd_tensors_all), (
            f"ViT dd tensor parse mismatch: used {ptr}, total={len(dd_tensors_all)}"
        )

        return {
            "ddcls_token": ddcls_token,
            "ddpos_embed": ddpos_embed,

            "ddpatchw": ddpatchw,
            "ddpatchb": ddpatchb,

            "blocks": dd_blocks,

            "ddnormw": ddnormw,
            "ddnormb": ddnormb,

            "ddheadw": ddheadw,
            "ddheadb": ddheadb,
        }

class VGG11_Flexfuse(nn.Module):
    CFG = (64, 'A', 128, 'A', 256, 256, 'A',
           512, 512, 'A', 512, 512, 'A')

    def __init__(self, channel=3, num_classes=10, Fuse=1,
                 im_size=(32, 32), v_fuse=False, layout="b_f"):
        super().__init__()
        self.Fuse = int(Fuse)
        self.channel = channel
        self.num_classes = num_classes
        self.im_size = tuple(im_size)
        self.v_fuse = v_fuse
        self.layout = layout
        if layout not in ("b_f", "f_f"):
            raise ValueError("layout must be b_f or f_f")
        self._conv_type = nn.Conv2d if layout == "b_f" else GroupedConv2d

        layers = []
        in_channels = channel
        for ic, out_channels in enumerate(self.CFG):
            if out_channels == 'A':
                layers.append(nn.AvgPool2d(2, 2))
            else:
                pad = 3 if channel == 1 and ic == 0 else 1
                conv = self._conv_type(in_channels * self.Fuse,
                                       out_channels * self.Fuse, 3, padding=pad,
                                       groups=self.Fuse)
                if layout == "b_f":
                    norm = nn.GroupNorm(out_channels * self.Fuse,
                                        out_channels * self.Fuse, affine=True)
                else:
                    # Same existing pure-norm forward as Conv/ResNet.
                    if conv.weight.shape != (out_channels * self.Fuse, in_channels, 3, 3):
                        raise ValueError("Use the flat-weight GroupedConv2d from ConvNet")
                    norm = NormActive(out_channels * self.Fuse, affine=True)
                    with torch.no_grad():
                        norm.weight.fill_(1)
                        norm.bias.zero_()
                layers.extend([conv, norm, nn.ReLU(inplace=False)])
                in_channels = out_channels
        layers.append(nn.AvgPool2d(1, 1))
        self.features = nn.Sequential(*layers)
        self.classifier = GroupedLinear(512, num_classes, self.Fuse, layout=self.layout)

    def forward(self, x, return_tape=False):
        # Keep the original one-tensor forward interface for existing tests.
        if not return_tape and self.layout == "b_f":
            h = self.features(x)
            return self.classifier(h, setup="conv").view(-1, self.num_classes)

        # Record exactly the activation inputs required by the existing
        # manual backward primitives; no autograd graph is needed here.
        blocks, events = [], []
        h = x
        i = 0
        while i < len(self.features):
            layer = self.features[i]
            if isinstance(layer, self._conv_type):
                x_conv = h
                x_norm = layer(x_conv)
                norm = self.features[i + 1]
                h_norm = norm(x_norm) if self.layout == "b_f" else norm(x_norm, layout=self.layout)
                x_pool = self.features[i + 2](h_norm)
                del h_norm
                if return_tape:
                    blocks.append({
                        'feature_idx': i,
                        'x_conv': x_conv,     # input to grouped conv
                        'x_norm': x_norm,     # output of grouped conv
                        'x_pool': x_pool,     # output of Norm + ReLU
                    })
                    events.append(('block', len(blocks) - 1))
                h = x_pool
                i += 3
            elif isinstance(layer, nn.AvgPool2d):
                if return_tape:
                    events.append(('pool', i, h))  # pre-pool input
                h = layer(h)
                i += 1
            else:
                raise RuntimeError(f'Unexpected VGG feature {i}: {type(layer)}')

        x_lin = h
        x_out = self.classifier(x_lin, setup="conv").view(-1, self.num_classes)
        if not return_tape:
            return x_out
        return x_out, {
            'blocks': blocks,
            'events': events,
            'head': {'x_lin': x_lin, 'x_out': x_out},
        }

    def get_weight_pack(self):
        blocks = []
        for b in range(0, len(self.features)):
            conv = self.features[b]
            if isinstance(conv, self._conv_type):
                norm = self.features[b + 1]
                blocks.append({
                    'convw': conv.weight, 'convb': conv.bias,
                    'normw': norm.weight, 'normb': norm.bias,
                })
        return {'blocks': blocks,
                'head': {'linw': self.classifier.weight,
                         'linb': self.classifier.bias}}

    @staticmethod
    def collect_ordered_grads(d_weights_list, dlin_w, dlin_b):
        grads = []
        for dw in d_weights_list:
            grads.extend((dw['dconvw'], dw['dconvb'],
                          dw['dnormw'], dw['dnormb']))
        grads.extend((dlin_w, dlin_b))
        return grads

    def run_first_bwd(self, tape, target, Fuse=None, weights=None,
                      fuse_mask_list=None, keep_input_grad=True,
                      keep_weight_dicts=True):
        """Manual full-Fuse bwd, progressively releasing inactive activations.

        keep_input_grad=True and keep_weight_dicts=True preserve the old return
        contract. For a production MTT step, they may be set False if the caller
        does not use dx_input or the block dictionaries after forming grad_list.
        first-bwd parameter gradients are *always* computed for the entire Fuse.
        """
        fuse = self.Fuse if Fuse is None else int(Fuse)
        if fuse != self.Fuse:
            raise ValueError('First backward must use full Fuse')
        active_range = self._active_fuse_range(fuse_mask_list)
        if weights is None:
            weights = self.get_weight_pack()

        head = tape['head']
        dx_out = crossEntropy_bwd(head['x_out'], target, fuse)
        g, dlin_w, dlin_b = linerFused_bwd(
            head['x_lin'], weights['head']['linw'],
            grad_output=dx_out, Fuse=fuse, layout=self.layout)
        g = g.reshape_as(head['x_lin'])

        # The head is no longer needed at full Fuse after linear first-bwd.
        # This does not change the separately returned full-Fuse logits.
        d_head_tensors = {'dx_out': dx_out}
        split_cache = {}
        if active_range is not None:
            self._split_head(tape, d_head_tensors, active_range, split_cache)
        del dx_out

        d_activates_list = [None] * len(tape['blocks'])
        d_weights_list = [None] * len(tape['blocks'])
        grad_list = [None] * (4 * len(tape['blocks']) + 2)
        grad_list[-2:] = (dlin_w, dlin_b)
        events = tape['events']

        for ev_ix in range(len(events) - 1, -1, -1):
            event = events[ev_ix]
            if event[0] == 'pool':
                _, i, pool_input = event
                pool = self.features[i]
                if pool.kernel_size != 1:
                    g = avgPool_bwd(
                        pool_input, grad_output=g,
                        kernel_size=self._pair(pool.kernel_size),
                        stride=self._pair(pool.stride or pool.kernel_size),
                        padding=self._pair(pool.padding),
                        ceil_mode=pool.ceil_mode,
                        count_include_pad=pool.count_include_pad,
                    )
                # The full pool input has fulfilled its last first-bwd use.
                if active_range is not None:
                    events[ev_ix] = ('pool', i, self._split_tensor(
                        pool_input, active_range, split_cache))
                del pool_input, event
                continue

            idx = event[1]
            block = tape['blocks'][idx]
            w = weights['blocks'][idx]
            dx_pool = g
            dx_norm, dnorm_w, dnorm_b = insNormNRelu_bwd(
                block['x_norm'], w['normw'], block['x_pool'],
                grad_output=dx_pool, v_fuse=self.v_fuse, layout=self.layout)
            conv = self.features[block['feature_idx']]
            g, dconv_w, dconv_b = conv_bwd(
                block['x_conv'], w['convw'],
                grad_output=dx_norm, groups=fuse, layout=self.layout,
                stride=conv.stride, padding=conv.padding, dilation=conv.dilation)

            d_activates_list[idx] = {'dx_pool': dx_pool, 'dx_norm': dx_norm}
            if keep_weight_dicts:
                d_weights_list[idx] = {
                    'dconvw': dconv_w, 'dconvb': dconv_b,
                    'dnormw': dnorm_w, 'dnormb': dnorm_b}
            grad_list[4*idx:4*idx+4] = (dconv_w, dconv_b, dnorm_w, dnorm_b)

            # This block's full-Fuse activations and intermediate derivatives
            # are finished with first-bwd. Split immediately instead of waiting
            # until all 8 blocks have computed full-Fuse gradients.
            if active_range is not None:
                for key in ('x_conv', 'x_norm', 'x_pool'):
                    block[key] = self._split_tensor(block[key], active_range, split_cache)
                act = d_activates_list[idx]
                act['dx_pool'] = self._split_tensor(act['dx_pool'], active_range, split_cache)
                act['dx_norm'] = self._split_tensor(act['dx_norm'], active_range, split_cache)
            del block, w, dx_pool, dx_norm, dconv_w, dconv_b, dnorm_w, dnorm_b

        # dx_input exists only for diagnostic first-bwd comparisons. It is never
        # used by run_double_bwd; production distillation can omit it.
        if keep_input_grad:
            d_head_tensors['dx_input'] = g
        return d_head_tensors, d_activates_list, d_weights_list, grad_list

    @staticmethod
    def _pair(x):
        return [x, x] if isinstance(x, int) else list(x)

    def _active_fuse_range(self, mask):
        if mask is None:
            return None
        if len(mask) != self.Fuse or any(m not in (0, 1) for m in mask):
            raise ValueError('Expected a length-Fuse 0/1 mask')
        active = [i for i, m in enumerate(mask) if m]
        if not active or active != list(range(active[0], active[-1] + 1)):
            raise ValueError('Active branches must be a nonempty contiguous range')
        if len(active) == self.Fuse:
            return None
        return active[0], active[-1] + 1, self.Fuse

    def _split_tensor(self, t, active_range, cache=None, *, batch=False):
        if t is None:
            return None
        if active_range is None:
            return t
        start, stop, full = active_range
        dim = 0 if batch or self.layout == 'f_f' else 1
        key = (id(t), dim)
        if cache is not None and key in cache:
            src_weak, cached = cache[key]
            if src_weak() is t:
                return cached
        unit = t.shape[dim] // full
        view = t.narrow(dim, start * unit, (stop - start) * unit)
        # MUST clone: for B=1, a sliced-channel view is often already contiguous;
        # .contiguous() alone would keep the full-Fuse underlying storage alive.
        result = view.clone(memory_format=torch.contiguous_format)
        if cache is not None:
            import weakref
            cache[key] = (weakref.ref(t), result)
        return result

    def _split_head(self, tape, d_head_tensors, active_range, cache):
        head = tape['head']
        head['x_lin'] = self._split_tensor(head['x_lin'], active_range, cache)
        head['x_out'] = self._split_tensor(head['x_out'], active_range, cache, batch=True)
        d_head_tensors['dx_out'] = self._split_tensor(
            d_head_tensors['dx_out'], active_range, cache, batch=True)

    def _split_saved_for_double_bwd(self, tape, d_activates_list,
                                     d_head_tensors, mask):
        """Retained for callers that split *after* run_first_bwd instead."""
        active_range = self._active_fuse_range(mask)
        if active_range is None:
            return
        cache = {}
        self._split_head(tape, d_head_tensors, active_range, cache)
        for b in tape['blocks']:
            for key in ('x_conv', 'x_norm', 'x_pool'):
                b[key] = self._split_tensor(b[key], active_range, cache)
        for a in d_activates_list:
            for key in ('dx_pool', 'dx_norm'):
                a[key] = self._split_tensor(a[key], active_range, cache)
        tape['events'] = [
            ('pool', ev[1], self._split_tensor(ev[2], active_range, cache))
            if ev[0] == 'pool' else ev for ev in tape['events']]

    def get_active_weight_pack(self, mask):
        """Active-only views of the currently bound (possibly Reparam) weights.

        Call this while the module is bound to the original full-Fuse flat
        parameters, e.g. inside call_with_param. No active flat cat/recover.
        The returned views share the storage of full parameters intentionally.
        """
        active_range = self._active_fuse_range(mask)
        full_pack = self.get_weight_pack()
        if active_range is None:
            return full_pack
        start, stop, full = active_range
        def rows(t):
            if t is None:
                return None
            unit = t.shape[0] // full
            return t.narrow(0, start * unit, (stop - start) * unit)
        return {
            'blocks': [{k: rows(v) for k, v in block.items()}
                       for block in full_pack['blocks']],
            'head': {k: rows(v) for k, v in full_pack['head'].items()},
        }

    def pack_recovered_weights(self, tensors_all):
        """Pack recover_params(flat, base_shapes, bwd_Fuse) in parameter order."""
        if len(tensors_all) != 34:
            raise ValueError(f'Expected 34 VGG11 parameter tensors, got {len(tensors_all)}')
        blocks = [
            dict(convw=tensors_all[4*i], convb=tensors_all[4*i+1],
                 normw=tensors_all[4*i+2], normb=tensors_all[4*i+3])
            for i in range(8)
        ]
        return {'blocks': blocks,
                'head': {'linw': tensors_all[32], 'linb': tensors_all[33]}}

    def pack_recovered_dd(self, dd_tensors_all, x=None, tape=None):
        """Pack the OUTER-loss seed dL/d(dW) into per-layer tensors."""
        if len(dd_tensors_all) != 34:
            raise ValueError(f'Expected 34 dd tensors, got {len(dd_tensors_all)}')
        if x is None:
            x = tape['blocks'][0]['x_conv']
        dd_blocks = [
            dict(ddconvw=dd_tensors_all[4*i], ddconvb=dd_tensors_all[4*i+1],
                 ddnormw=dd_tensors_all[4*i+2], ddnormb=dd_tensors_all[4*i+3])
            for i in range(8)
        ]
        dd_head = {'ddx_input': torch.zeros_like(x),
                   'ddlinw': dd_tensors_all[32], 'ddlinb': dd_tensors_all[33]}
        return dd_head, dd_blocks

    def run_double_bwd_full(self, tape, d_activates_list, dd_weights_list=None,
                       d_head_tensors=None, dd_head_tensors=None, Fuse=None, weights=None,
                       release_intermediates=True, dd_tensors_all=None,
                       d_bwd_state=None, weight_tensors_all=None):
        """Manual active-Fuse double-bwd + bwd2_1.

        release_intermediates=True consumes tape, d_activates_list and
        dd_weights_list; do not reuse them afterward. This is appropriate for
        a single distillation step and reduces peak live activation storage.
        """
        fuse = self.Fuse if Fuse is None else int(Fuse)
        if d_bwd_state is not None:
            if d_head_tensors is not None:
                raise ValueError('Pass only one of d_bwd_state and d_head_tensors')
            d_head_tensors = d_bwd_state
        if dd_tensors_all is not None:
            if dd_weights_list is not None or dd_head_tensors is not None:
                raise ValueError('Pass either dd_tensors_all or packed dd tensors, not both')
            dd_head_tensors, dd_weights_list = self.pack_recovered_dd(
                dd_tensors_all, tape=tape)
        if weight_tensors_all is not None:
            if weights is not None:
                raise ValueError('Pass either weight_tensors_all or weights, not both')
            weights = self.pack_recovered_weights(weight_tensors_all)
        if d_head_tensors is None or dd_head_tensors is None or dd_weights_list is None:
            raise ValueError('run_double_bwd requires first-bwd state and dd parameter tensors')
        if weights is None:
            if fuse != self.Fuse:
                raise ValueError('For split bwd supply active-only weights')
            weights = self.get_weight_pack()

        blocks, events, head = tape['blocks'], tape['events'], tape['head']
        dd_cur = dd_head_tensors['ddx_input']
        d2_cache = [None] * len(blocks)

        # Phase 1: propagate dL/d(dX) forward and save the two additional
        # input-gradient paths which will be merged in phase 3.
        for event in events:
            if event[0] == 'pool':
                pool = self.features[event[1]]
                # In THIS VGG all pools are nonoverlapping AvgPool(2,2) or
                # identity AvgPool(1,1). The adjoint of avgPool backward is
                # just avgPool forward. Avoid avgPool_double_bwd's dummy
                # allocations + grouped convolution + crop/scatter.
                if pool.kernel_size != 1:
                    dd_cur = pool(dd_cur)
                continue

            idx = event[1]
            b, w = blocks[idx], weights['blocks'][idx]
            da, dd = d_activates_list[idx], dd_weights_list[idx]
            conv = self.features[b['feature_idx']]
            ddx_norm, dxconv_d2, _ = conv_double_bwd(
                dd_cur, dd['ddconvw'], dd['ddconvb'],
                da['dx_norm'], w['convw'], b['x_conv'], groups_=fuse,
                stride_=self._pair(conv.stride), padding_=self._pair(conv.padding),
                dilation_=self._pair(conv.dilation), layout=self.layout,
                output_mask=(True, True, False))
            del _
            dd_cur, dx_norm_d2, _ = insNormNRelu_double_bwd(
                ddx_norm, dd['ddnormw'], dd['ddnormb'],
                da['dx_pool'], b['x_pool'], w['normw'], b['x_norm'],
                v_fuse=self.v_fuse, layout=self.layout)
            d2_cache[idx] = (dxconv_d2, dx_norm_d2)
            if release_intermediates:
                d_activates_list[idx] = None
                dd_weights_list[idx] = None
            del b, w, da, dd, ddx_norm, dxconv_d2, dx_norm_d2, _

        # Phase 2: classifier + CE; merge the two paths to its input.
        x_lin, x_out = head['x_lin'], head['x_out']
        dx_out = d_head_tensors['dx_out']
        ddx_out, dx_lin_d2, _ = linearFused_double_bwd(
            x_lin, weights['head']['linw'], dx_out, dd_cur,
            dd_head_tensors['ddlinw'], dd_head_tensors['ddlinb'], fuse, layout=self.layout)
        del _
        dx_out_d1 = crossEntropy_double_bwd(x_out, ddx_out, fuse)
        dx_lin_d1, _, _ = linerFused_bwd(
            x_lin, weights['head']['linw'],
            grad_output=dx_out_d1, Fuse=fuse, layout=self.layout)
        del _
        g = dx_lin_d1.reshape_as(x_lin)
        g += dx_lin_d2.reshape_as(x_lin)
        del dd_cur, ddx_out, dx_lin_d2, dx_out_d1, dx_lin_d1, x_lin, x_out, dx_out
        if release_intermediates:
            head.clear()
            d_head_tensors.clear()
            dd_head_tensors.clear()

        # Phase 3: bwd2_1 traverses pooling and blocks in reverse order.
        for ev_ix in range(len(events) - 1, -1, -1):
            event = events[ev_ix]
            if event[0] == 'pool':
                _, feature_idx, pool_input = event
                pool = self.features[feature_idx]
                if pool.kernel_size != 1:
                    g = avgPool_bwd(
                        pool_input, grad_output=g,
                        kernel_size=self._pair(pool.kernel_size),
                        stride=self._pair(pool.stride or pool.kernel_size),
                        padding=self._pair(pool.padding),
                        ceil_mode=pool.ceil_mode,
                        count_include_pad=pool.count_include_pad)
                if release_intermediates:
                    events[ev_ix] = None
                del pool_input, event
                continue

            idx = event[1]
            b, w = blocks[idx], weights['blocks'][idx]
            dxconv_d2, dx_norm_d2 = d2_cache[idx]
            dx_norm_d1, _, _ = insNormNRelu_bwd(
                b['x_norm'], w['normw'], b['x_pool'],
                grad_output=g, v_fuse=self.v_fuse, layout=self.layout)
            del _
            dx_norm_d1 += dx_norm_d2
            dx_norm_total = dx_norm_d1
            # Unlike conv_bwd(...), conv2d_input ONLY computes dX, NOT dW/dB.
            # This is the only convolution result needed in bwd2_1.
            conv = self.features[b['feature_idx']]
            if self.layout == "b_f":
                g = torch.nn.grad.conv2d_input(
                    b['x_conv'].shape, w['convw'], dx_norm_total,
                    stride=conv.stride, padding=conv.padding,
                    dilation=conv.dilation, groups=fuse)
            else:
                # Existing FF stateless API returns all three gradients.
                # Keep the BF input-only fast path unchanged.
                g, _, _ = conv_bwd(
                    b['x_conv'], w['convw'], grad_output=dx_norm_total,
                    stride=conv.stride, padding=conv.padding,
                    dilation=conv.dilation, groups=fuse, layout=self.layout)
                del _
            g += dxconv_d2
            if release_intermediates:
                blocks[idx] = None
                events[ev_ix] = None
                d2_cache[idx] = None
            del b, w, dxconv_d2, dx_norm_d2, dx_norm_d1, dx_norm_total, conv
        return g


class GroupedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 precision='ieee', device=None, dtype=None):
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.precision = precision

    def forward(self, x):
        g = self.groups
        weight = self.weight.view(
            g,
            self.out_channels // g,
            self.in_channels // g,
            *self.kernel_size,
        )
        bias = None if self.bias is None else self.bias.view(g, -1)

        return grouped_conv2d(
            x, weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=g,
            precision=self.precision,
        )

class Conv_Flexfuse_datalayout(nn.Module):
    # 更新新的datalayout。
    # pool，relu，norm 不需要修改。 conv / linear替换成为最新的代码
    def __init__(self, channel=3, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='maxpooling', im_size = (32,32), Fuse=2, v_fuse=True):
        super(Conv_Flexfuse_datalayout, self).__init__()
        self.Fuse = Fuse
        self.layout = "f_f"
        self.v_fuse = v_fuse
        self.num_classes = num_classes
        self.channel = channel
        self.im_size = im_size
        self.conv1 = GroupedConv2d(in_channels=channel*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        if v_fuse:
            self.norm1 = NormActive(net_width*Fuse, affine=True) 
        else:
            self.norm1 = NormActive(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
            # self.norm1 = nn.GroupNorm(net_width*Fuse,net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = GroupedConv2d(in_channels=net_width*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        if v_fuse:
            self.norm2 = NormActive(net_width*Fuse,affine=True) #BN在channel上单独计算，所以目前不用管。
        else:
            self.norm2 = NormActive(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
            # self.norm2 = nn.GroupNorm(net_width*Fuse,net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = GroupedConv2d(in_channels=net_width*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        if v_fuse:
            self.norm3 = NormActive(net_width*Fuse,affine=True) #BN在channel上单独计算，所以目前不用管。
        else:
            self.norm3 = NormActive(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
            # self.norm3 = nn.GroupNorm(net_width*Fuse,net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool3 = nn.AvgPool2d(kernel_size=2)
        # self.linear = GroupedLinear(net_width * 4 * 4, num_classes,Fuse )
        feat_h = im_size[0] // 8
        feat_w = im_size[1] // 8
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.linear = GroupedLinear(
            net_width * feat_h * feat_w,
            num_classes,
            Fuse, layout="f_f"
        )

        if not self.v_fuse:
            with torch.no_grad():
                for norm in (self.norm1, self.norm2, self.norm3):
                    norm.weight.fill_(1)
                    norm.bias.zero_()
        self.net_width= net_width

    def forward(self, x_conv1):
        # 不管是不是vfuse，都可以加rfelu，因为反正fwd 也不做fuse。
        x_norm1 = self.conv1(x_conv1)          
        x_pool1 = F.relu(self.norm1(x_norm1, layout = 'f_f') )
        x_conv2 = self.pool1(x_pool1)
        x_norm2 = self.conv2(x_conv2)          
        x_pool2 = F.relu(self.norm2(x_norm2, layout = 'f_f')) 
        x_conv3 = self.pool2(x_pool2)
        x_norm3 = self.conv3(x_conv3)          
        x_pool3 = F.relu(self.norm3(x_norm3, layout = 'f_f'))    
        x_lin  = self.pool3(x_pool3)
        x_out = self.linear(x_lin, setup='conv')    # N x 10
        x_out = x_out.view(-1,self.num_classes)
        return  x_conv1,x_norm1, x_pool1,x_conv2,x_norm2, x_pool2,x_conv3,x_norm3, x_pool3, x_lin,x_out
