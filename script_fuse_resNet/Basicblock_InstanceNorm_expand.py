
# 3.24
# 改成instance norm再试试

# 3.30
# 修正一下没有short cut 1*1 conv的问题
# 为了能做bwd，conv out 也需要保存。(bn out 应该不用存吧... )

# 4.5 修正了在shortcut下的bug。目前后缀为full的接口均可以返回正常结果。（除了性能不太好。）
# 4.7 mem上已经消耗接近。剩下大约150/600 的mem 难以优化因为瓶颈在basic block里面。


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

def BasicBlock_bwd( activates, weights, grad_output,SCstride=1 ):
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

    # 保持和你原来一样的写法：直接改 grad_output
    grad_output[out <= 0] = 0

    # ---------------- main branch ----------------
    dx_bn2, dbn2w, dbn2b, _, _ = instanceNorm_backward(x_bn2, bn2w, grad_output=grad_output)
    dx_conv2, dconv2w, _ = conv_bwd(x_conv2, conv2w, grad_output=dx_bn2)
    dbno1 = dx_conv2
    dbno1[x_conv2 <= 0] = 0

    dx_bn1, dbn1w, dbn1b, _, _ = instanceNorm_backward(x_bn1, bn1w, grad_output=dbno1)
    dx_main, dconv1w, _ = conv_bwd(x_conv1, conv1w, grad_output=dx_bn1, stride=SCstride)

    dbno2 = grad_output
    # ---------------- shortcut branch ----------------
    if has_downsample:
        dbnosc = grad_output
        dx_bnsc, dbnscw, dbnscb, _, _ = instanceNorm_backward(
            x_bnsc, bnscw, grad_output=dbnosc
        )
        dx_short, dconvscw, _ = conv_bwd(
            x_conv1, convscw, grad_output=dx_bnsc, stride=SCstride, padding=0
        )
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
    grad_output, SCstride=1
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
    dx_bnsc_d2 = d2_activates["dx_bnsc_d2"]


    has_downsample = (convscw is not None)
    # g_out = grad_output.clone()
    grad_output[out <= 0] = 0
    dx_bn2, dbn2w, dbn2b, _, _ = instanceNorm_backward(
        x_bn2, bn2w, grad_output=grad_output
    )
    dx_bn2 += dx_bn2_d2
    dx_conv2, dconv2w, _ = conv_bwd(
        x_conv2, conv2w, grad_output=dx_bn2
    )
    # dx_conv2 = dx_conv2.clone()
    dx_conv2 += dx_conv2_d2 # TODO: 应该需要先加法再relu。 每一个bwd2_1结束后需要立刻合并dbwd梯度。
    dx_conv2[x_conv2 <= 0] = 0
    dx_bn1, dbn1w, dbn1b, _, _ = instanceNorm_backward(
        x_bn1, bn1w, grad_output=dx_conv2
    )
    dx_bn1 += dx_bn1_d2
    del dx_bn1_d2
    dx_conv1_main, dconv1w, _ = conv_bwd(
        x_conv1, conv1w, grad_output=dx_bn1, stride=SCstride
    )

    if has_downsample:
        dx_bnsc, dbnscw, dbnscb, _, _ = instanceNorm_backward(
            x_bnsc, bnscw, grad_output=grad_output
        )
        dx_bnsc += dx_bnsc_d2
        dx_conv1_sc, dconvscw, _ = conv_bwd(
            x_conv1, convscw, grad_output=dx_bnsc,
            stride=SCstride, padding=0,
        )
        dx_in = dx_conv1_main + dx_conv1_sc + dx_conv1_d2
    else:
        dx_bnsc = None
        dbnscw = None
        dbnscb = None
        dconvscw = None
        dx_in = dx_conv1_main + grad_output + dx_conv1_d2

    weights["dconv1w"] = dconv1w
    weights["dbn1w"] = dbn1w
    weights["dbn1b"] = dbn1b
    weights["dconv2w"] = dconv2w
    weights["dbn2w"] = dbn2w
    weights["dbn2b"] = dbn2b
    weights["dconvscw"] = dconvscw
    weights["dbnscw"] = dbnscw
    weights["dbnscb"] = dbnscb

    return dx_in
    #     dconv1w, dbn1w, dbn1b,
    #     dconv2w, dbn2w, dbn2b,
    #     dconvscw, dbnscw, dbnscb
    # )


def BasicBlock_double_bwd(
    activates, d_activates, weights, dd_weights,
    ddgrad_in, SCstride=1
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

    # dx_block   = d_activates["dx_conv1"]
    # dbno2   = d_activates["dbno2"]
    # dbno1   = d_activates["dbno1"]
    # dx_bn1  = d_activates["dx_bn1"]
    # dx_bn2  = d_activates["dx_bn2"]
    # dx_bnsc = d_activates["dx_bnsc"]
    #  branch 下还有一些点要再商议一下：
    #  因为最后一个op是先加和再relu；所以dbnosc = dbno2
    has_downsample = (convscw is not None)
    # dbnosc = dbno2
    # ---------------- main branch ----------------
    ddx_bn1, dx_in_d2, dconv1w_d2 = conv_double_bwd(
        ddx_conv1, ddconv1w, None,
        d_activates["dx_bn1"], conv1w, x_conv1, stride_=[SCstride,SCstride]
    )
    d_activates.pop("dx_bn1")
    # del dx_bn1
    d_activates["dx_in_d2"] = dx_in_d2
    dx_bn1_d2, dbn1w_d2, ddx_conv2 = instanceNorm_double_backwards_fn(
        x_bn1, bn1w, None,
        ddx_bn1, ddbn1w, ddbn1b,
        d_activates["dbno1"], 1e-5
    )
    d_activates.pop("dbno1")
    # del dbno1
    d_activates["dx_bn1_d2"] = dx_bn1_d2
    ddx_conv2[x_conv2 <= 0] = 0
    ddx_bn2, dx_conv2_d2, dconv2w_d2 = conv_double_bwd(
        ddx_conv2, ddconv2w, None, d_activates["dx_bn2"], conv2w, x_conv2
    )
    d_activates.pop("dx_bn2")
    # del dx_bn2
    d_activates["dx_conv2_d2"] = dx_conv2_d2
    dx_bn2_d2, dbn2w_d2, ddO_main = instanceNorm_double_backwards_fn(
        x_bn2, bn2w, None,
        ddx_bn2, ddbn2w, ddbn2b,
        gO=d_activates["dbno2"]
    )
    d_activates["dx_bn2_d2"] = dx_bn2_d2


    # ---------------- shortcut branch ----------------
    if has_downsample:
        ddx_scbn, dx_conv1_d2_sc, dconvscw_d2 = conv_double_bwd(
            ddx_conv1, ddconvscw, None,
            d_activates["dx_bnsc"], convscw, x_conv1,
            stride_=[SCstride,SCstride],padding_=[0,0])
        d_activates.pop("dx_bnsc")
        dx_bnsc_d2, dbnscw_d2, ddO_sc = instanceNorm_double_backwards_fn(
            x_bnsc, bnscw, None,
            ddx_scbn, ddbnscw, ddbnscb,
            gO=d_activates["dbno2"]
        )
        d_activates.pop("dbno2")
        d_activates["dx_bnsc_d2"] = dx_bnsc_d2
        d_activates["dx_in_d2"]+= dx_conv1_d2_sc
        # dx_conv1_d2_total = dx_conv1_d2_main
        ddO_main += ddO_sc
    else:
        d_activates.pop("dbno2")
        dx_bnsc_d2 = None
        # dconvscw_d2 = None
        # dbnscw_d2 = None
        # dx_conv1_d2_total = dx_conv1_d2_main
        ddO_main += ddx_conv1

    ddO_main[out <= 0] = 0
    # TODO: 是不是这里少返回了一个dx_convsc d2啊？
    # 不是，因为被累加到dx_conv1_d2_total 里面了，确实也合理，这个是x_conv在两个conv 下面产生的一阶梯度和。
    # dx_conv1_d2_total = dx_conv1_d2_sc(shortcut) + dx_conv1_d2_main(原来的dxconv2d2)

    return ddO_main, d_activates
        # dconv1w_d2,
        # dbn1w_d2,
        # dconv2w_d2,
        # dbn2w_d2,
        # dconvscw_d2,
        # dbnscw_d2,
    # )



def conv_norm_relu_bwd(x, x_bn,x_block,convw, bnw, grad_output  ):
    dx_block = grad_output
    dx_block[x_block <= 0] = 0
    dx_bn, dbnw, dbnb,_,_ = instanceNorm_backward(x_bn, bnw, grad_output=dx_block)
    _, dconvw ,_ = conv_bwd(x, convw, grad_output=dx_bn)
    return dx_bn, dbnw, dbnb, dconvw

def conv_norm_relu_double_bwd(x, x_bn,x_block,dx_bn,dx_block,ddx_conv,  convw ,bnw,ddconvw,ddbnw,ddbnb):
    ddx_bn, dx_conv_d2, dconvw_d2 = conv_double_bwd(ddx_conv,ddconvw,None,dx_bn, convw, x )
    del dx_bn, ddx_conv
    dx_bn_d2, dbnw_d2, ddx_block = instanceNorm_double_backwards_fn(x_bn,bnw, None ,ddx_bn,ddbnw,ddbnb,dx_block)
    del dx_block, ddx_bn # 可以试试用覆盖的话，这里就不用单独del了更加工整内存也更好。
    ddx_block[x_block<=0] = 0
    return  ddx_block, dx_bn_d2, dx_conv_d2


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_channels, affine=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_channels, affine=True)

        if stride != 1 or in_channels != out_channels:
            self.convsc = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            self.bnsc = nn.InstanceNorm2d(out_channels, affine=True)
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
        return activates
        # return x_bn1, x_conv2, x_bn2, out, x_bnsc



class TinyResNet(nn.Module):
    def __init__(
        self,
        flag,
        Fuse,
        in_channels=3,
        stem_channels=64,       # 第一层 conv 输出通道
        block_out_channels=64, # BasicBlock 输出通道
        num_classes=10,
        stride=1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, stem_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = nn.InstanceNorm2d(stem_channels, affine=True)
        self.block = BasicBlock(stem_channels, block_out_channels, stride=stride)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(block_out_channels, num_classes)
    def forward(self, x_conv):
        x_bn = self.conv(x_conv)
        x_block = self.bn(x_bn)
        x_block = F.relu(x_block, inplace=True)
        activates = self.block(x_block)
        x_pool = activates['x_out']
        x_fc = self.pool(x_pool)
        x_fc = torch.flatten(x_fc, 1)
        x_out = self.fc(x_fc)

        return x_out, x_fc, x_pool, x_block, activates, x_bn
    
    



    

###################################
###################################
# flag = 'flex'        
flag = 'original'
flag = 'manuel'         # 单纯x为了debug写的
Fuse = 1
batch_size = 128
out_channel = 128 # in shape是写死了64， 所以out是128的话就是short cut
stride = 2
test_iter = 1
set_random_seed() # 仅仅在ACC test的时候使用。会严重影响性能。 
###################################
###################################



if __name__ == "__main__":
    print("flag = " + flag)
    model1 = TinyResNet( flag, Fuse=Fuse, block_out_channels= out_channel, stride=stride).to("cuda")
    model = model1

    transform = transforms.ToTensor()
    cifar10 = torchvision.datasets.CIFAR10(root='/scratch/yguo25/files/mtt-distillation/data', train=True, download=True, transform=transform)
    # 取一个 batch
    indices = torch.arange(batch_size)   # 或 torch.randperm(len(cifar10))[:batch_size]
    imgs, labels = zip(*(cifar10[i] for i in indices))
    x = torch.stack(imgs).to("cuda").requires_grad_(True)
    target = torch.tensor(labels, device="cuda")


    # torch.save(model.state_dict(), 'model_test_basicblock_instanceNorm.pt')
    # torch.save(model.state_dict(), 'model_test_basicblock_instanceNorm_sc_stride2.pt')
    # torch.save(model.state_dict(), 'model_test_basicblock_instanceNorm_sc.pt')
    # exit()
    if(out_channel==64):
        pretrained_dict = torch.load("model_test_basicblock_instanceNorm.pt")
    else:
        # pretrained_dict = torch.load("model_test_basicblock_instanceNorm_sc.pt")
        pretrained_dict = torch.load("model_test_basicblock_instanceNorm_sc_stride2.pt")
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
            x_out, _, _, _ , _ , _ = model(x)
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
                # 首尾的activate 是需要的x_block，x_pool， 还有一些首尾部分的activates
                x_out, x_fc, x_pool, x_block,activates, x_bn = model(x)  # forward
                bnscw = model.block.bnsc.weight if model.block.bnsc is not None else None
                bnscb = model.block.bnsc.bias if model.block.bnsc is not None else None
                convscw = model.block.convsc.weight if model.block.convsc is not None else None
                weights = {
                    "conv1w": model.block.conv1.weight,
                    "bn1w":  model.block.bn1.weight,
                    "bn1b":  model.block.bn1.bias,
                    "conv2w":  model.block.conv2.weight,
                    "bn2w":  model.block.bn2.weight,
                    "bn2b":  model.block.bn2.bias,
                    "convscw": convscw,
                    "bnscw": bnscw,
                }
                loss = criterion(x_out, target)  
                print("----CELOSS-----", loss.item())

                dx_out = crossEntropy_bwd(x_out,target,1)
                dx_fc,dfcw, dfcb  = linear_bwd(x_fc, model.fc.weight, grad_output=dx_out)
                dx_fc = dx_fc.view(batch_size, out_channel,1,1)
                dx_pool = adaptivepooling_bwd(x_pool, grad_output= dx_fc)
                del dx_fc

                # TODO: check一下activates 里面的x_conv1 到底有多大用。因为i现在既单独返回了x_conv1，又在activate里面保存了。
                dx_block, d_activates, d_weights = BasicBlock_bwd(activates, weights, grad_output=dx_pool, SCstride=stride )
                del dx_pool


                dx_bn, dbnw, dbnb, dconvw = conv_norm_relu_bwd(x, x_bn, x_block ,model.conv.weight, model.bn.weight , grad_output=dx_block )
                dw = [dconvw,dbnw,dbnb,dfcw,dfcb]
                dw += list(d_weights.values())
                weight_sum = [d.sum() for d in dw]
                grad_loss = sum(weight_sum)
                print("----GRANDLOSS-----", grad_loss.item())

                ddconvw = torch.ones_like(dconvw).cuda()
                ddbnw = torch.ones_like(dbnw).cuda()
                ddbnb = torch.ones_like(dbnb).cuda()
                ddconv1w = torch.ones_like(weights["conv1w"]).cuda()
                ddbn1w = torch.ones_like(weights["bn1w"]).cuda()
                ddbn1b = torch.ones_like(weights["bn1b"]).cuda()
                ddconv2w = torch.ones_like(weights["conv2w"]).cuda()
                ddbn2w = torch.ones_like(weights["bn2w"]).cuda()
                ddbn2b = torch.ones_like(weights["bn2b"]).cuda()
                ddfcw = torch.ones_like(dfcw).cuda()
                ddfcb = torch.ones_like(dfcb).cuda()
                if( convscw is not None ): 
                    ddconvscw = torch.ones_like(convscw).cuda()
                    ddbnscw = torch.ones_like(bnscw).cuda()
                    ddbnscb = torch.ones_like(bnscb).cuda()
                else:
                    ddconvscw = None
                    ddbnscw = None
                    ddbnscb = None  
                ddx_conv = torch.zeros_like(x).cuda()
                dd_weights = {
                    "ddconv1w": ddconv1w,
                    "ddbn1w":  ddbn1w,
                    "ddbn1b":  ddbn1b,
                    "ddconv2w":  ddconv2w,
                    "ddbn2w":  ddbn2w,
                    "ddbn2b":  ddbn2b,
                    "ddconvscw": ddconvscw,
                    "ddbnscw": ddbnscw,
                }
                del ddconv1w, ddbn1w, ddbn1b, ddconv2w, ddbn2w, ddbn2b, ddconvscw, ddbnscw


                ddx_bn, dx_conv_d2, dconvw_d2 = conv_double_bwd(ddx_conv,ddconvw,None,dx_bn, model.conv.weight,x )
                del dx_bn, ddx_conv
                dx_bn_d2, dbnw_d2, ddx_block = instanceNorm_double_backwards_fn(x_bn, model.bn.weight, None ,ddx_bn,ddbnw,ddbnb,dx_block)
                del dx_block, ddx_bn # 可以试试用覆盖的话，这里就不用单独del了更加工整内存也更好。
                ddx_block[x_block<=0] = 0

                # 这里的dx_block_d2 已经累计了来自sc的梯度。
                ddx_pool, d2_activates = BasicBlock_double_bwd( activates,d_activates,weights, dd_weights, ddgrad_in = ddx_block, SCstride=stride)
                del ddx_block # 这里是必要的。但是目前瓶颈不在这里
                clear_tensorlists(dd_weights)

                ddx_lin = adaptivepooling_double_bwd(ddx_pool)
                del ddx_pool
                ddx_out, dx_lin_d2, _ = linearFused_double_bwd(x_fc,model.fc.weight, dx_out, ddx_lin, ddfcw ,ddfcb ,1)
                del ddx_lin

                dx_out_d1 = crossEntropy_double_bwd(x_out, ddx_out, Fuse) # 没有y。也就是说 CrossEntropy 对 logits 的 Hessian 只和 p = softmax(z) 有关，和 target 无关。                
                dx_lin_d1, _, _ = linerFused_bwd(x_fc, model.fc.weight, grad_output=dx_out_d1, Fuse=1)
                dx_lin_d1 += dx_lin_d2
                del dx_lin_d2
                dx_lin_d1 = dx_lin_d1.view(batch_size, out_channel,1,1)
                dx_pool_d1 = adaptivepooling_bwd(x_pool, grad_output= dx_lin_d1)

                dx_block_d1 = BasicBlock_bwd2_1(activates, weights, d2_activates, grad_output=dx_pool_d1, SCstride=stride)
                clear_tensorlists(activates, d2_activates,weights)

                del dx_pool_d1, x_pool
                dx_block_d1[x_block <= 0] = 0
                del x_block
                dx_bn_d1,dbnw, dbnb,_,_ = instanceNorm_backward(x_bn, model.bn.weight, grad_output=dx_block_d1)
                dx_bn_d1 +=dx_bn_d2
                dx_conv,_,_ = conv_bwd(x, model.conv.weight, grad_output=dx_bn_d1)
                dx_conv+=dx_conv_d2      
                print("----GRAD-----", dx_conv.sum().item())


        elif flag =='convFuse' or 'convFuse_2':
            output = model(x)  # forward
            if  flag =='convFuse_2':
                output = output[-1]
            output = output.view(-1,10)
            loss = criterion(output, target)  # compute loss
            loss *= Fuse
            print("----CELOSS-----", loss.item())
            dw = torch.torch.autograd.grad(loss, list(model.parameters()), create_graph=True)   
            weight_sum = [d.sum() for d in dw]
            grad_loss = sum(weight_sum)

            print("----GRANDLOSS-----", grad_loss.item())
            grad_loss.backward()  
            print("----GRAD-----", x.grad.sum().item())
            
        optimizer.step()  # update x

    end = time.time()

    print("当前显存使用:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("峰值显存使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    print("时间占用：", end-start)