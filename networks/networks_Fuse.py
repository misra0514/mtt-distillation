# 4.27
# 现在的model 应该都可以把Fuse 写好了。唯一的区别就是 flex fuse 有没有做以及有没有封装到model 里面。所以现在准备重新整理一下这些网络。这里希望可以包含Conv3 、 Resnet18（resnet18test file 复制）
# 这里是完整的大模型, 来源主要有两个： stateless & stateless_basic block 。两个分别是一个封装后的模块，一个是原始的函数接口。



import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import torchvision.transforms as transforms
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

# from networks.networks_stacked import LinearStacked_2 # NOTE: 这里取消注释了。因为Fuse->stacked->stacked_basicblock 太长了。不好。
from networks.networks_basicblock_tritonwrapper import batchNorm2d_backward, batchnorm_double_backwards_fn, batchnorm_double_backwards_fn_new
from networks.networks_basicblock_tritonwrapper import instanceNorm_backward ,instanceNorm_double_backwards_fn
from networks.utils import clear_tensorlists
from networks.networks_stateless import BasicBlock_double_bwd,BasicBlock_bwd,BasicBlock_bwd2_1, conv_norm_relu_bwd
from networks.networks_basicblock_tritonwrapper import Fst_Order_NormActive # fuse+基本优化
from networks.networks_stateless_basicblock import linear_bwd, conv_bwd, insNormNRelu_bwd, \
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

class LinearStacked_2(nn.Module):
    # batch* fusion * In。 --> fusion,Batch ,Out
    def __init__(self ,in_features, out_features, Fuse):
        super(LinearStacked_2, self).__init__()
        self.Fuse = Fuse
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(Fuse* out_features,in_features))
        self.bias = torch.nn.Parameter(torch.randn(self.Fuse* out_features))

    def forward(self, x):
        x = x.view(-1,self.Fuse, self.in_features)
        # TODO: 这里输入如果是BAD/（而不是ABD）的话，可以得到is_contiguous 的结果。那就很简单只要调整target即可
        # 现在是BAD，意味着Fusion，batch的排序，B到了第一位
        # 因为显然有bug，所以后续想慢慢换掉这个接口。加了一个dim==3。遇到问题再说吧
        if x.ndim == 3:
            x = torch.einsum("abc,bcd->bad",x,self.weight.view(self.Fuse,self.out_features,self.in_features).transpose(-1, -2)) 
            x = x+self.bias.view(self.Fuse,1 ,self.out_features)
            x = x.squeeze(0) #加一个squeeze，为了适配在Fuse=1的时候的模型。
            return x

class GroupedLinear(nn.Module):
    # batch* fusion * In。 --> Batch,fusion ,Out
    def __init__(self ,in_features, out_features, Fuse):
        super(GroupedLinear, self).__init__()
        self.Fuse = Fuse
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(Fuse* out_features,in_features))
        self.bias = torch.nn.Parameter(torch.randn(self.Fuse* out_features))

    def forward(self, x):
        W = self.weight.view( self.Fuse, self.out_features, self.in_features ).transpose(-1, -2)
        if x.ndim == 3:
            x = x.view(-1,self.Fuse, self.in_features)
            x = torch.einsum("bfi,fio->bfo", x, W)
            x = x + self.bias.view(1, self.Fuse, self.out_features)
        elif x.ndim == 4:
            # x: [B, Fuse, N, In]
            B, Fs, N, I = x.shape
            x = torch.einsum("bfni,fio->bfno", x, W)
            x = x + self.bias.view(1, self.Fuse, 1, self.out_features)
        return x

class NormActive(nn.Module):
    def __init__(self, channel_num, affine=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn([channel_num]))
        self.bias = nn.Parameter(torch.randn([channel_num]))
    def forward(self, input):
        out = Fst_Order_NormActive.apply(input, self.weight, self.bias)
        return out



class Conv_Flexfuse_backup(nn.Module):
    def __init__(self, channel=3, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='maxpooling', im_size = (32,32), Fuse=2, v_fuse=True):
        super(Conv_Flexfuse_backup, self).__init__()
        self.Fuse = Fuse
        self.v_fuse = v_fuse
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
        self.linear = LinearStacked_2(net_width * 4 * 4, num_classes,Fuse )
        self.net_width= net_width

    def forward(self, x_conv1):
        if self.v_fuse:
            x_conv1 = x_conv1.view(-1,self.Fuse*3 ,32,32)        # 10, 256, 4,4   -> 20, 2048
            x_norm1 = self.conv1(x_conv1)          
            x_pool1 = self.norm1(x_norm1)    
            x_conv2 = self.pool1(x_pool1)
            x_norm2 = self.conv2(x_conv2)          
            x_pool2 = self.norm2(x_norm2)
            x_conv3 = self.pool2(x_pool2)
            x_norm3 = self.conv3(x_conv3)          
            x_pool3 = self.norm3(x_norm3)  
            x_lin  = self.pool3(x_pool3)
            x_out = self.linear(x_lin)    # N x 10
            x_out = x_out.view(-1,10)
        else:
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



class Conv_Flexfuse(nn.Module):
    def __init__(self, channel=3, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='maxpooling', im_size = (32,32), Fuse=2):
        super(Conv_Flexfuse, self).__init__()
        self.Fuse = Fuse
        self.num_classes = num_classes
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
        x_conv1 = x_conv1.view(-1, self.Fuse * 3, 32, 32)
        x_norm1 = self.conv1(x_conv1)
        x_pool1 = F.relu(self.norm1(x_norm1))
        x_conv2 = self.pool1(x_pool1)
        x_norm2 = self.conv2(x_conv2)
        x_pool2 = F.relu(self.norm2(x_norm2))
        x_conv3 = self.pool2(x_pool2)
        x_norm3 = self.conv3(x_conv3)
        x_pool3 = F.relu(self.norm3(x_norm3))
        x_lin = self.pool3(x_pool3)
        x_out = self.linear(x_lin).view(-1, self.num_classes)
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

    def _split_saved_for_double_bwd(self, tape, d_activates_list, d_head_tensors, fuse_mask_list):
        """
        first-bwd 必须用完整 Fuse 算 dW；
        但是 double-bwd 如果只跑 bwd_Fuse，就要把 tape / d_activates / dx_out 切到 bwd_Fuse。
        """
        if fuse_mask_list is None:
            return
        for b in tape["blocks"]:
            b["x_conv"], b["x_norm"], b["x_pool"] = split_half_snd_dim(
                [b["x_conv"], b["x_norm"], b["x_pool"]],
                fuse_mask_list,
            )
        for da in d_activates_list:
            da["dx_norm"], da["dx_pool"] = split_half_snd_dim(
                [da["dx_norm"], da["dx_pool"]],
                fuse_mask_list,
            )
        head = tape["head"]
        head["x_lin"], head["x_out"], d_head_tensors["dx_out"] = split_half_snd_dim(
            [head["x_lin"], head["x_out"], d_head_tensors["dx_out"]],
            fuse_mask_list,
        )

    def _convblock_first_bwd(self, block, weights, grad_output, Fuse):
        dx_conv, dx_norm, dx_pool, dconv_w, dconv_b, dnorm_w, dnorm_b = ConvBlock_bwd1_2(
            block["x_conv"],
            block["x_norm"],
            block["x_pool"],
            weights["convw"],
            weights["normw"],
            grad_output,
            Fuse=Fuse,
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
        dx_out = crossEntropy_bwd(x_out, target, Fuse)
        dx_lin, dlin_w, dlin_b = linerFused_bwd(
            x_lin,
            weights["head"]["linw"],
            grad_output=dx_out,
            Fuse=Fuse,
        )
        dx_lin = dx_lin.view_as(x_lin)

        # block3
        g, d_activates_list[2], d_weights_list[2] = self._convblock_first_bwd(
            blocks[2],
            weights["blocks"][2],
            dx_lin,
            Fuse,
        )
        # block2
        g, d_activates_list[1], d_weights_list[1] = self._convblock_first_bwd(
            blocks[1],
            weights["blocks"][1],
            g,
            Fuse,
        )
        # block1：这里保持你原来的特殊写法，而不是强行用 ConvBlock_bwd1_2
        b1 = blocks[0]
        w1 = weights["blocks"][0]

        dx_pool1 = avgPool_bwd(b1["x_pool"], grad_output=g)
        dx_norm1, dnorm1_w, dnorm1_b = insNormNRelu_bwd(
            b1["x_norm"],
            w1["normw"],
            b1["x_pool"],
            grad_output=dx_pool1,
        )
        _, dconv1_w, dconv1_b = conv_bwd(
            b1["x_conv"],
            w1["convw"],
            grad_output=dx_norm1,
            groups=Fuse,
        )
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
            "dx_out": dx_out,
        }
        # 如果 Fuse != bwd_Fuse，就在 first-bwd 完成之后切 tape 和 d_activates。
        self._split_saved_for_double_bwd(
            tape,
            d_activates_list,
            d_head_tensors,
            fuse_mask_list,
        )
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


    def run_double_bwd(
        self,
        tape,
        d_activates_list,
        dd_weights_list,
        d_head_tensors,
        dd_head_tensors,
        Fuse=1,
        weights=None,
    ):
        """
        返回 dx_conv1_d1，也就是 synthetic image/input 方向的最终梯度。
        这里的 Fuse 应该传 bwd_Fuse。
        """
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
            groups_=Fuse,
        )

        ddx_pool1, dx_norm1_d2, _ = insNormNRelu_double_bwd(
            ddx_norm1,
            dd1["ddnormw"],
            dd1["ddnormb"],
            da1["dx_pool"],
            b1["x_pool"],
            w1["normw"],
            b1["x_norm"],
        )

        dd_cur = avgPool_double_bwd(ddx_pool1)

        d2_cache = [None, None, None]
        d2_cache[0] = {
            "dxconv_d2": dxconv1_d2,
            "dx_norm_d2": dx_norm1_d2,
        }

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
                Fuse,
            )
            d2_cache[idx] = {
                "dxconv_d2": dxconv_d2,
                "dx_norm_d2": dx_norm_d2,
            }
        ddx_lin = dd_cur
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
            Fuse,
        )
        dx_out_d1 = crossEntropy_double_bwd(x_out, ddx_out, Fuse)
        dx_lin_d1, _, _ = linerFused_bwd(
            x_lin,
            weights["head"]["linw"],
            grad_output=dx_out_d1,
            Fuse=Fuse,
        )
        dx_lin_d1 = dx_lin_d1.view_as(x_lin)
        dx_lin_d1 = dx_lin_d1 + dx_lin_d2.view_as(x_lin)
        # ------------------------------------------------------------
        # 3. backward-order bwd2_1:
        #    block3 -> block2 -> block1
        # -----------------------------------------------------------
        g = dx_lin_d1
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
                Fuse=Fuse,
            )
        return g



class BasicBlock_Flexfuse(nn.Module):
    def __init__(self, in_channels, out_channels,   net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='maxpooling', im_size = (32,32), stride=1, Fuse = 1,v_fuse=True):
        # 注意一下这个basic block已经把 fuse隔离在外面了。内部init的时候再扩充in out channel
        super().__init__()
        self.v_fuse = v_fuse
        self.conv1 = nn.Conv2d(in_channels*Fuse, out_channels*Fuse, 3, stride, 1, bias=False, groups=Fuse)
        # self.bn1 = nn.InstanceNorm2d(out_channels*Fuse, affine=True)
        self.bn1 = nn.GroupNorm(out_channels*Fuse, out_channels*Fuse, affine=True) 
        self.conv2 = nn.Conv2d(out_channels*Fuse, out_channels*Fuse, 3, 1, 1, bias=False, groups=Fuse)
        # self.bn2 = nn.InstanceNorm2d(out_channels*Fuse, affine=True)
        self.bn2 = nn.GroupNorm(out_channels*Fuse, out_channels*Fuse, affine=True) 
        if stride != 1 or in_channels != out_channels:
            self.convsc = nn.Conv2d(in_channels*Fuse, out_channels*Fuse, 1, stride, bias=False, groups=Fuse )
            # self.bnsc = nn.InstanceNorm2d(out_channels*Fuse, affine=True)
            self.bnsc = nn.GroupNorm(out_channels*Fuse, out_channels*Fuse, affine=True) 
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

class ResNet18_FlexFuse(nn.Module):
    def __init__(self, channel=3, num_classes=10, Fuse=1,v_fuse=True):
        super().__init__()
        self.Fuse = Fuse
        self.v_fuse = v_fuse
        self.num_classes = num_classes
        blk_in_ch = 64   # 这里仍然是逻辑通道数
        cfg = [ (blk_in_ch,  2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2), ]
        self.conv = nn.Conv2d(channel * Fuse, blk_in_ch * Fuse, kernel_size=3, stride=1, padding=1, bias=False, groups=Fuse )
        # self.bn = nn.InstanceNorm2d(blk_in_ch * Fuse, affine=True)
        self.bn = nn.GroupNorm(blk_in_ch * Fuse, blk_in_ch * Fuse, affine=True) 
        self.stages = nn.ModuleList()
        for stage_id, (out_ch, num_blocks, first_stride) in enumerate(cfg):
            stage = nn.ModuleList()
            for block_id in range(num_blocks):
                stride = first_stride if block_id == 0 else 1
                stage.append(BasicBlock_Flexfuse(blk_in_ch, out_ch, stride=stride, Fuse=Fuse,v_fuse=v_fuse))
                blk_in_ch = out_ch
            self.stages.append(stage)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = LinearStacked_2(512, num_classes, Fuse)
        # self.fc = nn.Linear(512, num_classes)

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

    def _split_tensor_by_fuse(self, p, fuse_mask_list, layout="channel"):
        if p is None or (not torch.is_tensor(p)):
            return p
        active_range = self._active_fuse_range(fuse_mask_list)
        if active_range is None:
            return p
        start, end, Fuse = active_range

        if layout == "batch":
            c = p.shape[0]
            block = c // Fuse
            return p[start * block:end * block, ...].contiguous()

        # ResNet activations normally store fuse in channel / feature dim.
        # 4D: [B, C*Fuse, H, W]
        # x_fc: [B, C*Fuse]
        # 1D norm-like tensors, if ever passed here: [C*Fuse]
        if p.ndim > 2:
            c = p.shape[1]
            block = c // Fuse
            return p[:, start * block:end * block, ...].contiguous()
        if p.ndim == 2:
            c = p.shape[1]
            block = c // Fuse
            return p[:, start * block:end * block].contiguous()
        if p.ndim == 1:
            c = p.shape[0]
            block = c // Fuse
            return p[start * block:end * block].contiguous()
        return p

    def _split_tensor_dict_by_fuse(self, tensor_dict, fuse_mask_list, layout="channel"):
        if tensor_dict is None:
            return None
        if self._active_fuse_range(fuse_mask_list) is None:
            return tensor_dict
        for k in list(tensor_dict.keys()):
            v = tensor_dict[k]
            if torch.is_tensor(v):
                tensor_dict[k] = self._split_tensor_by_fuse(v, fuse_mask_list, layout=layout)
                del v
        return tensor_dict

    def _split_head_saved_for_double_bwd(self, head, d_stem_tensors, fuse_mask_list):
        if self._active_fuse_range(fuse_mask_list) is None:
            return
        # x_pool: [B, C*Fuse, H, W]
        # x_fc:   [B, C*Fuse]
        # x_out/dx_out: [Fuse*B, num_classes]
        head["x_pool"] = self._split_tensor_by_fuse(head["x_pool"], fuse_mask_list, layout="channel")
        head["x_fc"] = self._split_tensor_by_fuse(head["x_fc"], fuse_mask_list, layout="channel")
        head["x_out"] = self._split_tensor_by_fuse(head["x_out"], fuse_mask_list, layout="batch")
        d_stem_tensors["dx_out"] = self._split_tensor_by_fuse(d_stem_tensors["dx_out"], fuse_mask_list, layout="batch")

    def _split_block_saved_for_double_bwd(self, activates_i, d_activates_i, fuse_mask_list):
        if self._active_fuse_range(fuse_mask_list) is None:
            return
        self._split_tensor_dict_by_fuse(activates_i, fuse_mask_list, layout="channel")
        self._split_tensor_dict_by_fuse(d_activates_i, fuse_mask_list, layout="channel")

    def _split_stem_saved_for_double_bwd(self, stem, d_stem_tensors, fuse_mask_list):
        if self._active_fuse_range(fuse_mask_list) is None:
            return
        self._split_tensor_dict_by_fuse(stem, fuse_mask_list, layout="channel")
        d_stem_tensors["dx_bn"] = self._split_tensor_by_fuse(d_stem_tensors["dx_bn"], fuse_mask_list, layout="channel")
        d_stem_tensors["dx_block"] = self._split_tensor_by_fuse(d_stem_tensors["dx_block"], fuse_mask_list, layout="channel")

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
        d_stem_tensors = { "dx_out": dx_out, }
        # dx_fc, dfcw, dfcb = linear_bwd( x_fc, self.fc.weight, grad_output=dx_out)
        dx_fc, dfcw, dfcb = linerFused_bwd( x_fc, self.fc.weight, grad_output=dx_out, Fuse= Fuse)
        dx_fc = dx_fc.view(x_pool.size(0), x_pool.size(1), 1, 1)
        g = adaptivepooling_bwd(x_pool, grad_output=dx_fc)
        self._split_head_saved_for_double_bwd(head, d_stem_tensors, fuse_mask_list)
        del dx_fc, dx_out, x_pool, x_fc, x_out
        for i in reversed(range(len(flat_blocks))):
            blk = flat_blocks[i]
            activates_i = tape["blocks"][i]
            weights_i = self.get_block_weights(blk)
            g, d_activates_i, d_weights_i = BasicBlock_bwd( activates_i, weights_i, grad_output=g, SCstride=blk.conv1.stride[0],Fuse=Fuse,v_fuse=self.v_fuse)
            d_activates_list[i] = d_activates_i
            d_weights_list[i]   = d_weights_i
            self._split_block_saved_for_double_bwd(activates_i, d_activates_i, fuse_mask_list)
            # 这里只删局部引用，不动 tape 里的本体
            del activates_i, weights_i
        dx_block = g
        del g
        dx_bn, dbnw, dbnb, dconvw = conv_norm_relu_bwd( x_conv, x_bn, x_block, self.conv.weight, self.bn.weight, grad_output=dx_block, Fuse=Fuse, v_fuse=self.v_fuse )
        d_stem_tensors["dx_bn"] = dx_bn
        d_stem_tensors["dx_block"] = dx_block
        self._split_stem_saved_for_double_bwd(stem, d_stem_tensors, fuse_mask_list)
        del dx_bn, dx_block, x_conv, x_bn, x_block

        d_weights_list_all = self.collect_ordered_grads( dconvw, dbnw, dbnb, d_weights_list, dfcw, dfcb)
        return  d_stem_tensors, d_activates_list, d_weights_list, d_weights_list_all
        
    def run_double_bwd(
        self,
        tape,
        d_activates_list,
        dd_weights_list,
        d_stem_tensors,
        dd_stem_tensors,
        Fuse = 1,
        weights=None,
    ):
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
        ddx_bn, dx_conv_d2, _ = conv_double_bwd( ddx_conv, ddconvw, None,dx_bn, stem_weights["convw"], x, groups_=Fuse)
        del dx_bn, ddx_conv, ddconvw
        # TODO: instance norm好像根本就不需要考虑fuse的问题。bn再说吧。
        dx_bn_d2, _, dd_cur = instanceNorm_double_backwards_fn( x_bn, stem_weights["bnw"], None, ddx_bn, ddbnw, ddbnb, dx_block)
        del dx_block, ddx_bn,ddbnw, ddbnb
        dd_cur[x_block <= 0] = 0
        for i in range(len(flat_blocks)):
            blk = flat_blocks[i]
            activates_i = tape["blocks"][i]
            d_activates_i = d_activates_list[i]
            weights_i = block_weights_list[i]
            dd_weights_i = dd_weights_list[i]
            dd_cur, _ = BasicBlock_double_bwd(
                activates_i, d_activates_i, weights_i, dd_weights_i, ddgrad_in=dd_cur,
                SCstride=blk.conv1.stride[0], Fuse= Fuse,  v_fuse=self.v_fuse )
            clear_tensorlists(dd_weights_i)
            dd_weights_list[i] = None
            del activates_i, d_activates_i, weights_i, dd_weights_i

        ddx_pool = dd_cur
        del dd_cur
        # head double backward
        ddx_lin = adaptivepooling_double_bwd(ddx_pool)
        del ddx_pool
        ddx_out, dx_lin_d2, _ = linearFused_double_bwd( x_fc, head_weights["fcw"], dx_out, ddx_lin, ddfcw, ddfcb, Fuse )
        del dx_out, ddx_lin,ddfcw,ddfcb
        dx_out_d1 = crossEntropy_double_bwd(x_out, ddx_out, Fuse)
        del ddx_out,x_out
        dx_lin_d1, _, _ = linerFused_bwd( x_fc, head_weights["fcw"], grad_output=dx_out_d1, Fuse=Fuse)
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
            weights_i = block_weights_list[i]
            d2_activates_i = d_activates_list[i]
            if weights_i["convscw"] is None:
                d2_activates_i.setdefault("dx_bnsc_d2", None)
            g = BasicBlock_bwd2_1( activates_i, weights_i, d2_activates_i,grad_output=g, SCstride=blk.conv1.stride[0], Fuse = Fuse, v_fuse=self.v_fuse)
            clear_tensorlists(activates_i, d2_activates_i, weights_i)
            tape["blocks"][i] = None
            d_activates_list[i] = None
            del activates_i, d2_activates_i, weights_i
        dx_block_d1 = g
        del g
        dx_block_d1[x_block <= 0] = 0
        del x_block
        dx_bn_d1, _, _ = instanceNorm_backward( x_bn, stem_weights["bnw"], grad_output=dx_block_d1)
        del dx_block_d1
        dx_bn_d1 += dx_bn_d2
        del dx_bn_d2
        dx_conv, _, _ = conv_bwd( x, stem_weights["convw"], grad_output=dx_bn_d1,groups=Fuse)
        del dx_bn_d1
        dx_conv += dx_conv_d2
        return dx_conv

    def pack_recovered_dd(self, dd_tensors_all, x):
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
        dd_work = {
            "ddconvw": ddconvw,
            "ddbnw": ddbnw,
            "ddbnb": ddbnb,
            "ddfcw": ddfcw,
            "ddfcb": ddfcb,
            # "ddx_conv": torch.zeros_like(x).cuda(), # TODO: 这里可能要改。
        }
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
            w=out_proj_w,
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
            "x_gelu": x_gelu,
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
        x_gelu = tape["x_gelu"]
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
            "dx_dp2": dx_dp2,
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
        dx_dp2 = d_activates.pop("dx_dp2")
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

        d_activates["attn"] = d_attn_activates

        # ==================================================
        # dx_in_from_skip = dx_res1_total
        # dx_attn         = dx_res1_total
        # ==================================================
        dd_dx_res1_total = dd_dx_in_from_skip + dd_dx_attn

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
                "ddnorm1w": torch.ones_like(norm1_w),
                "ddnorm1b": torch.ones_like(self.norm1.bias),
                "attn": {
                    "ddqkvw": torch.ones_like(self.attn.qkv.weight),
                    "ddqkvb": torch.ones_like(self.attn.qkv.bias),
                    "ddoutprojw": torch.ones_like(self.attn.out_proj.weight),
                    "ddoutprojb": torch.ones_like(self.attn.out_proj.bias),
                },
                "ddnorm2w": torch.ones_like(norm2_w),
                "ddnorm2b": torch.ones_like(self.norm2.bias),
                "ddfc1w": torch.ones_like(fc1_w),
                "ddfc1b": torch.ones_like(self.fc1.bias),
                "ddfc2w": torch.ones_like(fc2_w),
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

    def _split_tensor_by_fuse(self, p, fuse_mask_list, layout="bf"):
        if p is None or (not torch.is_tensor(p)):
            return p
        active_range = self._active_fuse_range(fuse_mask_list)
        if active_range is None:
            return p
        start, end, Fuse = active_range

        if layout == "bf":
            # ViT normal activations: [B, Fuse, ...]
            assert p.shape[1] % Fuse == 0 or p.shape[1] == Fuse
            return p[:, start:end, ...].contiguous()

        if layout == "channel":
            # Patch conv input/output: [B, C*Fuse, H, W]
            c = p.shape[1]
            assert c % Fuse == 0, f"Cannot split channel-fused tensor shape={tuple(p.shape)} by Fuse={Fuse}"
            block = c // Fuse
            return p[:, start * block:end * block, ...].contiguous()

        if layout == "logits":
            # ViT logits are reshape(B*Fuse, C), ordered b0f0,b0f1,b1f0,b1f1,...
            assert p.shape[0] % Fuse == 0, f"Cannot split logits shape={tuple(p.shape)} by Fuse={Fuse}"
            B = p.shape[0] // Fuse
            rest = p.shape[1:]
            return p.view(B, Fuse, *rest)[:, start:end, ...].contiguous().view(B * (end - start), *rest)

        if layout == "param_token":
            # cls_token / pos_embed: [1, Fuse, ..., D]
            assert p.shape[0] == 1 and p.shape[1] == Fuse, \
                f"Expected token parameter layout [1,Fuse,...], got shape={tuple(p.shape)}"
            return p[:, start:end, ...].contiguous()

        raise ValueError(f"Unknown fuse split layout: {layout}")

    def _split_nested_bf_by_fuse(self, obj, fuse_mask_list):
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
            return self._split_tensor_by_fuse(obj, fuse_mask_list, layout="bf")
        if isinstance(obj, dict):
            for k in list(obj.keys()):
                obj[k] = self._split_nested_bf_by_fuse(obj[k], fuse_mask_list)
            return obj
        if isinstance(obj, list):
            for i in range(len(obj)):
                obj[i] = self._split_nested_bf_by_fuse(obj[i], fuse_mask_list)
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

        patch = tape["patch"]
        patch["x_in"] = self._split_tensor_by_fuse(patch["x_in"], fuse_mask_list, layout="channel")
        patch["x_patch"] = self._split_tensor_by_fuse(patch["x_patch"], fuse_mask_list, layout="channel")

        head = tape["head"]
        head["x_norm_in"] = self._split_tensor_by_fuse(head["x_norm_in"], fuse_mask_list, layout="bf")
        head["x_cls"] = self._split_tensor_by_fuse(head["x_cls"], fuse_mask_list, layout="bf")
        head["x_out"] = self._split_tensor_by_fuse(head["x_out"], fuse_mask_list, layout="logits")

        d_activates["dx_head"] = self._split_tensor_by_fuse(d_activates["dx_head"], fuse_mask_list, layout="bf")
        d_activates["dx_norm"] = self._split_tensor_by_fuse(d_activates["dx_norm"], fuse_mask_list, layout="bf")
        d_activates["dx_patch"] = self._split_tensor_by_fuse(d_activates["dx_patch"], fuse_mask_list, layout="channel")

        self._split_nested_bf_by_fuse(tape["blocks"], fuse_mask_list)
        self._split_nested_bf_by_fuse(d_activates["blocks"], fuse_mask_list)

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
        # dx_block: [B, Fuse, N+1, D]
        # dx_pos, d_block_activates, d_block_weights, d_block_weights_all = self.block.run_first_bwd( block_tape, grad_output=dx_block, Fuse=Fuse )
        flat_blocks = self.get_flat_blocks()
        d_block_activates_list = [None] * len(flat_blocks)
        d_block_weights_list = [None] * len(flat_blocks)
        d_block_weights_all_list = [None] * len(flat_blocks)
        g = dx_block
        for i in reversed(range(len(flat_blocks))):
            g, d_act_i, d_w_i, d_w_all_i = flat_blocks[i].run_first_bwd(
                tape=block_tapes[i],
                grad_output=g,
                Fuse=Fuse,
            )
            d_block_activates_list[i] = d_act_i
            d_block_weights_list[i] = d_w_i
            d_block_weights_all_list[i] = d_w_all_i
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
        self._split_saved_for_double_bwd(tape, d_activates, fuse_mask_list)
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

    def run_double_bwd(
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

        for i in reversed(range(len(flat_blocks))):
            g = flat_blocks[i].run_bwd2_1(
                tape=block_tapes[i],
                d_activates=d_block_activates_list[i],
                grad_output=g,
                Fuse=Fuse,
                weights=block_weights_list[i],
            )

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

