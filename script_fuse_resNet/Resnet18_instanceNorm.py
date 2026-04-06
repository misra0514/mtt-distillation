# 3.26
# 终于迟迟的调完了正确性。目前虽然只能保证一个instance norm的。但是感觉可以开始搭建resnet18做测试了。
# basic code 复制与basic block

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
from networks.networks_fused3 import batchNorm2d_backward, batchnorm_double_backwards_fn, batchnorm_double_backwards_fn_new
from networks.networks_fused3 import instanceNorm_backward ,instanceNorm_double_backwards_fn, instance_norm_backward_triton,instanceNorm_double_backwards_triton
from networks.networks_stateless_basicblock import linear_bwd, conv_bwd, insNormNRelu_bwd, \
linear_double_bwd, conv_double_bwd, insNormNRelu_double_bwd, avgPool_bwd, crossEntropy_bwd, \
    avgPool_double_bwd,bmm_bwd, linerFused_bwd, linearFused_double_bwd,crossEntropy_double_bwd
    
# from torch.aps.aten import adaptive_avg_pool2d_backward_cuda
from networks.networks_fused3 import NormActive
# from networks_flexFuse import Conv_Flexfused, ConvBlock_double_bwd,ConvBlock_bwd2_1



class MyNormReluFused_SndOrder(torch.autograd.Function):
    '''2 forward: relu+pool+linear.backward '''
    # TODO: 如果做ckpt，那么记得保证forward可以在算完之后全释放掉。 然后backward再重新算一遍。
    # 现在也没有做save ctx，为啥内存消耗还是1483？
    @staticmethod
    def forward(ctx, dLdy, input, weight, out):
        ctx.save_for_backward( input, weight, dLdy, out )
        dLdy[out<=0 ] = 0
        grad_output, dw, db,_,_ = instanceNorm_backward(input, weight, grad_output=dLdy)
        # grad_output, dw, db,mean, std = instanceNorm_backward(input, weight, dLdy)
        return grad_output, dw, db
    @staticmethod
    def backward(ctx, grad_grad_input, grad_grad_w, grad_grad_b):
        input, weight, dLdy, out = ctx.saved_tensors
        # print("grad_grad_input",(grad_grad_input**2).sum().item())
        # print("grad_grad_b",(grad_grad_b**2).sum().item())
        # print("x",(input**2).sum().item())
        # print("dLdy",(dLdy**2).sum().item())
        # print("grad_grad_w",(grad_grad_w**2).sum().item())
        # instance_norm_backward_triton
        gx, gG, ggO =  instanceNorm_double_backwards_fn(input, weight,None, grad_grad_input, grad_grad_w,grad_grad_b, dLdy,1e-5)
        # print("OUTdx",gx.sum().item())
        
        # gx, gG, ggO = batchnorm_double_backwards_fn_new(input, weight, grad_grad_input, grad_grad_w,grad_grad_b, dLdy,1e-5)
        # gx, gG, ggO = instanceNorm_double_backwards_fn_cln(input, weight, grad_grad_input, grad_grad_w,grad_grad_b, dLdy,1e-5,True   )
        # print("OUTddO(pre)",(ggO).sum().item())
        # print("OUTddO(pre)",(ggO**2).sum().item())
        ggO[out <= 0] = 0
        # print("OUTddO",(ggO).sum().item())
        # print("out",out.sum().item())

        ggO = ggO.view_as(dLdy)
        return ggO,gx, gG, None


class MyNormReluFused_FstOrder(torch.autograd.Function):
    '''forward: relu+pool+linear '''
    @staticmethod
    def forward(ctx, input, weight, bias):
        out = F.relu(F.batch_norm(input, running_mean=None, running_var=None,weight= weight, bias = bias, training=True), inplace= True)
        ctx.save_for_backward( input, weight ,out)
        return  out
    @staticmethod
    def backward(ctx, dLdy):
        input, weight, out = ctx.saved_tensors
        return MyNormReluFused_SndOrder.apply(dLdy ,input, weight, out)
        # db = gin.sum(0)
        # return gin, weight, db
# class NormActive_BNRELU(nn.Module):
#     # in_features 应该是1
#     def __init__(self, channel_num, affine=True):
#         super().__init__()
#         self.weight = nn.Parameter(torch.randn([channel_num]))
#         self.bias = nn.Parameter(torch.randn([channel_num]))
#     def forward(self, input):
#         out = MyNormReluFused_FstOrder.apply(input, self.weight, self.bias)
#         return out
class NormActive_BNRELU(nn.Module):
    def __init__(self, channel_num, affine=True):
        super().__init__()
        # BN 默认初始化：gamma=1, beta=0
        self.weight = nn.Parameter(torch.ones(channel_num))
        self.bias   = nn.Parameter(torch.zeros(channel_num))

        # 这三个是 BN2d 默认会有的 buffers（正好“差三个”）
        self.register_buffer("running_mean", torch.zeros(channel_num))
        self.register_buffer("running_var",  torch.ones(channel_num))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, input):
        # 你现在的 fused 实现里用的是 batch 统计（training=True 写死）
        # 这里只是为了 state_dict 对齐；num_batches_tracked 是否加 1 对“加载”不关键
        if self.training:
            self.num_batches_tracked += 1
        out = MyNormReluFused_FstOrder.apply(input, self.weight, self.bias)
        return out


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


def BasicBlock_bwd2_1(x_conv1, x_bn1, x_conv2, x_bn2 , out, conv1w, bn1w, bn1b, conv2w, bn2w, bn2b, \
                       grad_output, dx_conv1_d2, dx_conv2_d2, dx_bn1_d2,dx_bn2_d2):
    # 在2——1，需要做grad加法
    grad_output[out <= 0] = 0
    dx_bn2, dbn2w, dbn2b,_,_ = instanceNorm_backward(x_bn2, bn2w, grad_output=grad_output)
    # print(dx_bn2.shape)
    # print(x_bn2.shape)
    # print(dx_bn2_d2.shape)
    dx_bn2+=dx_bn2_d2
    dx_conv2, dconv2w, _ = conv_bwd(x_conv2, conv2w, grad_output=dx_bn2 )

    dx_conv2[x_conv2 <=0 ] = 0
    dx_conv2 += dx_conv2_d2
    dx_bn1, dbn1w, dbn1b,_,_ = instanceNorm_backward(x_bn1, bn1w, grad_output=dx_conv2)
    dx_bn1+=dx_bn1_d2
    dx_conv1, dconv1w, _ = conv_bwd(x_conv1, conv1w, grad_output=dx_bn1 )
    
    # 因为是inplace操作，最后return的两个实际上是gbno2和gbno1
    return dx_conv1+grad_output+dx_conv1_d2, dconv1w, dbn1w, dbn1b, dconv2w, dbn2w, dbn2b, grad_output, dx_conv2


def BasicBlock_bwd(x_conv1, x_bn1, x_conv2, x_bn2 , out, conv1w, bn1w, bn1b, conv2w, bn2w, bn2b,  grad_output):

    grad_output[out <= 0] = 0
    dx_bn2, dbn2w, dbn2b,_,_ = instanceNorm_backward(x_bn2, bn2w , grad_output=grad_output)
    dx_conv2, dconv2w, _ = conv_bwd(x_conv2, conv2w, grad_output=dx_bn2 )

    dx_conv2[x_conv2 <=0 ] = 0
    dx_bn1, dbn1w, dbn1b,_,_ = instanceNorm_backward(x_bn1, bn1w, grad_output=dx_conv2)
    dx_conv1, dconv1w, _ = conv_bwd(x_conv1, conv1w, grad_output=dx_bn1 )
    
    # 因为是inplace操作，最后return的两个实际上是gbno2和gbno1
    return dx_conv1+grad_output, dconv1w, dbn1w, dbn1b, dconv2w, dbn2w, dbn2b, grad_output, dx_conv2, dx_bn1, dx_bn2

def BasicBlock_bwd_full(
    x_conv1, x_bn1, x_conv2, x_bn2, out,
    conv1w, bn1w, bn1b, conv2w, bn2w, bn2b,
    grad_output,
    # optional downsample branch
    x_scbn=None, scconvw=None, scbnw=None, scbnb=None,
):

    has_downsample = (scconvw is not None)
    g_out = grad_output.clone()
    g_out[out <= 0] = 0

    # ---------------- main branch ----------------
    dx_bn2, dbn2w, dbn2b, _, _ = instanceNorm_backward(
        x_bn2, bn2w, grad_output=g_out
    )
    dx_conv2, dconv2w, _ = conv_bwd(
        x_conv2, conv2w, grad_output=dx_bn2
    )
    dbno1 = dx_conv2.clone()
    dbno1[x_conv2 <= 0] = 0
    dx_bn1, dbn1w, dbn1b, _, _ = instanceNorm_backward(
        x_bn1, bn1w, grad_output=dbno1
    )
    dx_main, dconv1w, _ = conv_bwd(
        x_conv1, conv1w, grad_output=dx_bn1
    )
    dbno2 = g_out.clone()

    if has_downsample:
        # assert x_scbn is not None, "downsample block needs x_scbn"
        # assert scbnw is not None, "downsample block needs scbnw"
        dbnosc = g_out.clone()

        dx_scbn, dscbnw, dscbnb, _, _ = instanceNorm_backward(
            x_scbn, scbnw, grad_output=dbnosc
        )
        dx_short, dscconvw, _ = conv_bwd(
            x_conv1, scconvw, grad_output=dx_scbn
        )

        dx_in = dx_main + dx_short
    else:
        dbnosc = None
        dx_scbn = None
        dscbnw = None
        dscbnb = None
        dscconvw = None

        dx_in = dx_main + g_out

    return (
        dx_in,
        dconv1w, dbn1w, dbn1b,
        dconv2w, dbn2w, dbn2b,
        dscconvw, dscbnw, dscbnb,
        dbno2, dbno1, dbnosc,
        dx_bn1, dx_bn2, dx_scbn
    )


def BasicBlock_double_bwd(x_conv1, x_bn1, x_conv2, x_bn2 , out,\
                          dx_bn1, dx_bn2, dx_out,\
                          dbno1, dbno2,\
                         conv1w, bn1w, conv2w, bn2w, \
                        ddconv1w, ddconv1b,ddbn1w, ddbn1b, ddconv2w,ddconv2b, ddbn2w, ddbn2b, ddx_conv1):
    ddx_bn1, dx_conv1_d2, dconv1w_d2 = conv_double_bwd(ddx_conv1, ddconv1w, None,dx_bn1, conv1w, x_conv1)
    dx_bn1_d2, dbn1w_d2, ddx_conv2 = instanceNorm_double_backwards_fn(x_bn1,bn1w,None, ddx_bn1, ddbn1w, ddbn1b,dbno1,1e-5)
    ddx_conv2[x_conv2<=0] = 0

    ddx_bn2, dx_conv2_d2, dconv2w_d2= conv_double_bwd(ddx_conv2, ddconv2w, None,dx_bn2, conv2w, x_conv2)
    dx_bn2_d2, dbn2w_d2 , ddO= instanceNorm_double_backwards_fn(x_bn2, bn2w, None, ddx_bn2, ddbn2w, ddbn2b,gO=dbno2)   
    ddO += ddx_conv1
    ddO[out<=0] = 0
    return ddO, dx_conv1_d2,dx_bn1_d2,dx_conv2_d2,dx_bn2_d2, dconv1w_d2, dbn1w_d2, dconv2w_d2, dbn2w_d2

# 这个基本上只是为了调准确性的
class BasicBlock_VirticalFuse(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.InstanceNorm2d(channels, affine= True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.InstanceNorm2d(channels, affine= True)

    def forward(self, x):
        identity = x
        x_bn1 = self.conv1(x)
        x_conv2 = F.relu(self.bn1(x_bn1), inplace=True)
        x_bn2 = self.conv2(x_conv2)
        out = self.bn2(x_bn2)
        out = F.relu(out + identity, inplace=True)
        return x_bn1, x_conv2, x_bn2, out
        # identity = x
        # out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        # out = self.bn2(self.conv2(out))
        # out = F.relu(out + identity, inplace=True)
        # return out


class BasicBlock_manuel(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        # self.bn1 = NormActive_BNRELU(channels) # 少了三个参数？？？which is？？
        self.bn1 = nn.InstanceNorm2d(channels, affine= True)

        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.InstanceNorm2d(channels, affine= True)

    def forward(self, x):
        identity = x
        x_bn1 = self.conv1(x)
        x_conv2 = self.bn1(x_bn1)
        x_conv2 = F.relu(x_conv2, inplace=True)
        x_bn2 = self.conv2(x_conv2)
        out = self.bn2(x_bn2)
        out = F.relu(out + identity, inplace=True)
        return x_bn1, x_conv2, x_bn2, out




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.InstanceNorm2d(out_channels, affine=True)
            )

    def forward(self, x):
        identity = x

        # conv1 -> bn1 -> relu
        x_bn1 = self.conv1(x)                       # 其实这是 conv1 output
        x_conv2 = F.relu(self.bn1(x_bn1), inplace=False)

        # conv2 -> bn2
        x_bn2 = self.conv2(x_conv2)                # 其实这是 conv2 output
        out_before_add = self.bn2(x_bn2)

        # residual branch
        if self.downsample is not None:
            identity = self.downsample(identity)

        out_after_add = out_before_add + identity
        out = F.relu(out_after_add, inplace=False)

        # 返回尽量全一点，后面你手写 backward 更方便
        return {
            "input": x,
            "identity": identity,
            "x_bn1": x_bn1,                # conv1 output
            "x_conv2": x_conv2,            # relu(bn1(conv1))
            "x_bn2": x_bn2,                # conv2 output
            "out_before_add": out_before_add,  # bn2(conv2)
            "out_after_add": out_after_add,    # bn2(conv2) + identity
            "out": out
        }



class ResNet18(nn.Module):
    """
    ResNet18-style network:
      stem
      layer1: 2 blocks, channels = base_channels
      layer2: 2 blocks, channels = base_channels * 2
      layer3: 2 blocks, channels = base_channels * 4
      layer4: 2 blocks, channels = base_channels * 8
      avgpool + fc

    这里保留你原来 TinyResNet 的风格：
      - 3x3 stem conv
      - InstanceNorm2d
      - 不用 maxpool
    """

    def __init__(self, flag="original", Fuse=None,
                 in_channels=3, base_channels=64, num_classes=10):
        super().__init__()

        self.flag = flag
        self.Fuse = Fuse

        # 你如果有自己的 fused block，
        # 最好让它也兼容这个构造签名: (in_channels, out_channels, stride=1)
        if flag == "original":
            self.block_cls = BasicBlock
        elif flag == "manuel":
            # 这里假设你自己的 BasicBlock_VirticalFuse 已经定义好了，
            # 并且接口跟 BasicBlock 一致
            self.block_cls = BasicBlock_VirticalFuse
        else:
            raise ValueError(f"Unknown flag: {flag}")

        # stem
        self.conv = nn.Conv2d(
            in_channels, base_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = nn.InstanceNorm2d(base_channels, affine=True)

        # ResNet18 stages: [2, 2, 2, 2]
        self.layer1 = self._make_layer(base_channels,     base_channels,     blocks=2, stride=1)
        self.layer2 = self._make_layer(base_channels,     base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8, num_classes)

    def _build_block(self, in_channels, out_channels, stride):
        # 如果 fused block 也用同样构造函数，这里直接统一建
        return self.block_cls(in_channels, out_channels, stride=stride)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = nn.ModuleList()
        layers.append(self._build_block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(self._build_block(out_channels, out_channels, 1))
        return layers

    def _forward_layer(self, x, layer, layer_name):
        block_outputs = []

        for i, block in enumerate(layer):
            block_dict = block(x)
            x = block_dict["out"]
            block_outputs.append(block_dict)

        return x, block_outputs

    def forward(self, x_conv):
        features = {}

        # stem
        x_bn = self.conv(x_conv)
        x_block = F.relu(self.bn(x_bn), inplace=False)

        features["stem"] = {
            "x_input": x_conv,
            "x_bn": x_bn,         # conv stem output
            "x_block": x_block    # relu(bn(stem))
        }

        # four stages
        x, features["layer1"] = self._forward_layer(x_block, self.layer1, "layer1")
        x, features["layer2"] = self._forward_layer(x,       self.layer2, "layer2")
        x, features["layer3"] = self._forward_layer(x,       self.layer3, "layer3")
        x, features["layer4"] = self._forward_layer(x,       self.layer4, "layer4")

        # head
        x_pool = x
        x_fc = self.pool(x_pool)
        x_fc = torch.flatten(x_fc, 1)
        x_out = self.fc(x_fc)

        features["head"] = {
            "x_pool": x_pool,   # 最后一个 stage 的输出（pool 前）
            "x_fc": x_fc        # flatten 后，fc 前
        }

        return x_out, features


# class TinyResNet(nn.Module):
#     def __init__(self, flag,Fuse,  in_channels=3, base_channels=64, num_classes=10):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn   = nn.InstanceNorm2d(base_channels, affine= True)
#         if flag=='original':
#             # self.block = BasicBlock_manuel(base_channels)
#             self.block = BasicBlock(base_channels)
#         elif flag == 'vertical':
#             self.block = BasicBlock_VirticalFuse(base_channels)
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc   = nn.Linear(base_channels, num_classes)

#     def forward(self, x_conv):
#         x_bn = self.conv(x_conv)
#         x_block = self.bn(x_bn)
#         x_block = F.relu(x_block, inplace=True)
#         x_bn1, x_conv2, x_bn2, x_pool = self.block(x_block)
#         x_fc = self.pool(x_pool)
#         x_fc = torch.flatten(x_fc, 1)
#         x_out = self.fc(x_fc)
#         return x_out, x_fc, x_pool, x_bn2, x_conv2, x_bn1, x_block, x_bn
    

###################################
###################################
# flag = 'flex'        
flag = 'vertical'         # 单纯x为了debug写的
flag = 'original'
Fuse = 1
batch_size = 128
test_iter = 1
set_random_seed() # 仅仅在ACC test的时候使用。会严重影响性能。 
###################################
###################################



if __name__ == "__main__":
    print("flag = " + flag)
    # model1 = ResNet18( flag, Fuse=Fuse).to("cuda")
    model = ResNet18(flag=flag, in_channels=3, base_channels=64, num_classes=10).to("cuda")


    transform = transforms.ToTensor()
    cifar10 = torchvision.datasets.CIFAR10(root='/scratch/yguo25/files/mtt-distillation/data', train=True, download=True, transform=transform)
    # 取一个 batch
    indices = torch.arange(batch_size)   # 或 torch.randperm(len(cifar10))[:batch_size]
    imgs, labels = zip(*(cifar10[i] for i in indices))
    x = torch.stack(imgs).to("cuda").requires_grad_(True)
    target = torch.tensor(labels, device="cuda")


    # torch.save(model.state_dict(), 'model_test_res18_instanceNorm.pt')
    # # exit()
    pretrained_dict = torch.load("model_test_basicblock_instanceNorm.pt")      
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
            x_out, features = model(x)
            loss = criterion(x_out, target)  # compute loss
            print("----CELOSS-----", loss.item())
            dw = torch.torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
            weight = [d.sum() for d in dw]
            grad_loss = sum(weight)
            print("----GRANDLOSS-----", grad_loss.item())
            grad_loss.backward()  
            print("----GRAD-----", x.grad.sum().item())



        elif flag =='vertical':
            # x_out, x_fc, x_pool, x_bn2, x_conv2, x_bn1, x_block, x_bn = model(x)  # forward
            x_out, cache = model(x)
            loss = criterion(x_out, target)  # compute loss
            print("----CELOSS-----", loss.item())
            with torch.no_grad():
                dx_out = torch.autograd.grad(loss,x_out)[0]
                dx_fc, dfcw, dfcb = torch.autograd.grad(x_out, [x_fc, model.fc.weight, model.fc.bias ], grad_outputs=dx_out)

                dx_fc = dx_fc.view(batch_size, 64,1,1)
                dx_pool = adaptivepooling_bwd(x_pool, grad_output= dx_fc)
                dx_block, dconv1w, dbn1w, dbn1b, dconv2w, dbn2w, dbn2b, dbno2, dbno1, dx_bn1, dx_bn2 = BasicBlock_bwd(x_block, x_bn1, x_conv2, x_bn2,x_pool, model.block.conv1.weight, model.block.bn1.weight , model.block.bn1.bias, \
                            model.block.conv2.weight, model.block.bn2.weight , model.block.bn2.bias, grad_output= dx_pool  )
                dx_block[x_block <= 0] = 0
                dx_bn,dbnw, dbnb,_,_ = instanceNorm_backward(x_bn, model.bn.weight, grad_output=dx_block)       
                _, dconvw ,_ = conv_bwd(x, model.conv.weight, grad_output=dx_bn)
                dw = [dconvw,dbnw,dbnb,dconv1w,dbn1w,dbn1b,dconv2w,dbn2w,dbn2b,dfcw,dfcb]
                weight = [d.sum() for d in dw]
                grad_loss = sum(weight)
                print("----GRANDLOSS-----", grad_loss.item())
                ddconvw = torch.ones_like(dconvw).cuda()
                ddbnw = torch.ones_like(dbnw).cuda()
                ddbnb = torch.ones_like(dbnb).cuda()
                ddconv1w = torch.ones_like(dconv1w).cuda()
                ddbn1w = torch.ones_like(dbn1w).cuda()
                ddbn1b = torch.ones_like(dbn1b).cuda()
                ddconv2w = torch.ones_like(dconv2w).cuda()
                ddbn2w = torch.ones_like(dbn2w).cuda()
                ddbn2b = torch.ones_like(dbn2b).cuda()
                ddfcw = torch.ones_like(dfcw).cuda()
                ddfcb = torch.ones_like(dfcb).cuda()
                ddx_conv = torch.zeros_like(x).cuda()

                ddx_bn, dx_conv_d2, dconvw_d2 = conv_double_bwd(ddx_conv,ddconvw,None,dx_bn, model.conv.weight,x )
                dx_bn_d2, dbnw_d2,ddx_block = instanceNorm_double_backwards_fn(x_bn, model.bn.weight, None ,ddx_bn,ddbnw,ddbnb,dx_block)
                ddx_block[x_block<=0] = 0
                ddx_pool, dx_block_d2,dx_bn1_d2,dx_conv2_d2,dx_bn2_d2, dconv1w_d2, dbn1w_d2, dconv2w_d2, dbn2w_d2 = BasicBlock_double_bwd(\
                                            x_block,x_bn1,x_conv2,x_bn2,x_pool,\
                                            dx_bn1, dx_bn2, dx_pool, dbno1,dbno2,\
                                            model.block.conv1.weight,model.block.bn1.weight,  model.block.conv2.weight,model.block.bn2.weight, \
                                             ddconv1w, None, ddbn1w, ddbn1b, ddconv2w, None, ddbn2w,ddbn2b, ddx_block  ) 
                

                ddx_lin = adaptivepooling_double_bwd(ddx_pool)
                ddx_out, dx_lin_d2, _ = linearFused_double_bwd(x_fc,model.fc.weight, dx_out, ddx_lin, ddfcw ,ddfcb ,1)

                dx_out_d1 = crossEntropy_double_bwd(x_out, ddx_out, Fuse) # 没有y。也就是说 CrossEntropy 对 logits 的 Hessian 只和 p = softmax(z) 有关，和 target 无关。                
                dx_lin_d1, _, _ = linerFused_bwd(x_fc, model.fc.weight, grad_output=dx_out_d1, Fuse=1)
                dx_lin_d1 += dx_lin_d2
                dx_lin_d1 = dx_lin_d1.view(batch_size, 64,1,1)

                dx_pool_d1 = adaptivepooling_bwd(x_pool, grad_output= dx_lin_d1)
                dx_block_d1, dconv1w, dbn1w, dbn1b, dconv2w, dbn2w, dbn2b, dbno2, dbno1 = BasicBlock_bwd2_1(x_block, x_bn1, x_conv2, x_bn2,x_pool, model.block.conv1.weight, model.block.bn1.weight , model.block.bn1.bias, \
                            model.block.conv2.weight, model.block.bn2.weight , model.block.bn2.bias, dx_pool_d1 , dx_block_d2, dx_conv2_d2, dx_bn1_d2,dx_bn2_d2 )
                dx_block_d1[x_block <= 0] = 0
                dx_bn_d1,dbnw, dbnb,_,_ = instanceNorm_backward(x_bn, model.bn.weight, grad_output=dx_block_d1)
                dx_bn_d1 +=dx_bn_d2
                dx_conv,_,_ = conv_bwd(x, model.conv.weight, grad_output=dx_bn_d1)
                dx_conv+=dx_conv_d2      
                print("----GRAD-----", dx_conv.sum().item())


        optimizer.step()  # update x

    end = time.time()

    print("当前显存使用:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("峰值显存使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    print("时间占用：", end-start)