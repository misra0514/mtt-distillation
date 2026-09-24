
# 1.23
#  沟槽的服务器好像记录没了。这里是准备做resnet的测试。首先测试一个basicblock的double bwd。（但是根本不知道此时conv之类的对不对。。。。。。）

# 还是按照之前的方法，一共是三种model：fused+ flexfuse + original

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
from networks.networks_stateless_basicblock import linear_bwd, conv_bwd, insNormNRelu_bwd, \
linear_double_bwd, conv_double_bwd, insNormNRelu_double_bwd, avgPool_bwd, crossEntropy_bwd, \
    avgPool_double_bwd,bmm_bwd, linerFused_bwd, linearFused_double_bwd,crossEntropy_double_bwd
    
# from torch.aps.aten import adaptive_avg_pool2d_backward_cuda
from networks.networks_basicblock_fused3 import NormActive
# from networks_flexFuse import Conv_Flexfused, ConvBlock_double_bwd,ConvBlock_bwd2_1



class MyNormReluFused_SndOrder(torch.autograd.Function):
    '''2 forward: relu+pool+linear.backward '''
    # TODO: 如果做ckpt，那么记得保证forward可以在算完之后全释放掉。 然后backward再重新算一遍。
    # 现在也没有做save ctx，为啥内存消耗还是1483？
    @staticmethod
    def forward(ctx, dLdy, input, weight, out):
        ctx.save_for_backward( input, weight, dLdy, out )
        dLdy[out<=0 ] = 0
        grad_output, dw, db = batchNorm2d_backward(input, weight, grad_output=dLdy)
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

        gx, gG, ggO =  batchnorm_double_backwards_fn_new(input, weight, grad_grad_input, grad_grad_w,grad_grad_b, dLdy,1e-5)
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
    dx_bn2, dbn2w, dbn2b = batchNorm2d_backward(x_bn2, bn2w, grad_output=grad_output)
    dx_bn2+=dx_bn2_d2
    dx_conv2, dconv2w, _ = conv_bwd(x_conv2, conv2w, grad_output=dx_bn2 )

    dx_conv2[x_conv2 <=0 ] = 0
    dx_conv2 += dx_conv2_d2
    dx_bn1, dbn1w, dbn1b = batchNorm2d_backward(x_bn1, bn1w, grad_output=dx_conv2)
    dx_bn1+=dx_bn1_d2
    dx_conv1, dconv1w, _ = conv_bwd(x_conv1, conv1w, grad_output=dx_bn1 )
    
    # 因为是inplace操作，最后return的两个实际上是gbno2和gbno1
    return dx_conv1+grad_output+dx_conv1_d2, dconv1w, dbn1w, dbn1b, dconv2w, dbn2w, dbn2b, grad_output, dx_conv2


def BasicBlock_bwd(x_conv1, x_bn1, x_conv2, x_bn2 , out, conv1w, bn1w, bn1b, conv2w, bn2w, bn2b,  grad_output):
    # reluout = grad_output * (out > 0)
    grad_output[out <= 0] = 0
    dx_bn2, dbn2w, dbn2b = batchNorm2d_backward(x_bn2, bn2w , grad_output=grad_output)
    # dx_bn2, dbn2w, dbn2b = torch.autograd.grad(out,[x_bn2, bn2w, bn2b ],grad_outputs=grad_output)
    dx_conv2, dconv2w, _ = conv_bwd(x_conv2, conv2w, grad_output=dx_bn2 )

    dx_conv2[x_conv2 <=0 ] = 0
    dx_bn1, dbn1w, dbn1b = batchNorm2d_backward(x_bn1, bn1w, grad_output=dx_conv2)
    # dx_bn1, dbn1w, dbn1b = torch.autograd.grad(x_conv2,[x_bn1, bn1w, bn1b ],grad_outputs=dx_conv2)
    dx_conv1, dconv1w, _ = conv_bwd(x_conv1, conv1w, grad_output=dx_bn1 )
    
    # 因为是inplace操作，最后return的两个实际上是gbno2和gbno1
    return dx_conv1+grad_output, dconv1w, dbn1w, dbn1b, dconv2w, dbn2w, dbn2b, grad_output, dx_conv2, dx_bn1, dx_bn2


def BasicBlock_double_bwd(x_conv1, x_bn1, x_conv2, x_bn2 , out,\
                          dx_bn1, dx_bn2, dx_out,\
                          dbno1, dbno2,\
                         conv1w, bn1w, conv2w, bn2w, \
                        ddconv1w, ddconv1b,ddbn1w, ddbn1b, ddconv2w,ddconv2b, ddbn2w, ddbn2b, ddx_conv1):
    # ddx_conv1 = grad_output
    # print(ddx_conv1.shape)

    ddx_bn1, dx_conv1_d2, dconv1w_d2 = conv_double_bwd(ddx_conv1, ddconv1w, None,dx_bn1, conv1w, x_conv1)
    # ddx_bn, dx_conv_d2, dconvw_d2 = conv_double_bwd(ddx_conv,ddconvw,None,dx_bn, model.conv.weight,x ) ggb 是None？ 

    # 这里，gO=dx_conv2 要的应该是norm输出的梯度。可是根本就没存norm 的输出梯度，只有输出再norm后的。可能需要存一下？
    # print("ddx_bn1", (ddx_bn1**2).sum().item())
    # print("ddbn1w", (ddbn1b**2).sum().item())
    # print("x_bn1", (x_bn1**2).sum().item())
    # print("dbno1", (dbno1**2).sum().item())
    # print("ddbn1w", (ddbn1w**2).sum().item())
    # ddx_bn1 \ ddbn1w \ dbno1 均一样
    dx_bn1_d2, dbn1w_d2, ddx_conv2  = batchnorm_double_backwards_fn_new(x_bn1,bn1w, ddx_bn1, ddbn1w, ddbn1b,dbno1,1e-5)
    # ggO, gx, gG =  batchnorm_double_backwards_fn_new(input, weight, grad_grad_input, grad_grad_w,grad_grad_b, dLdy,1e-5)

    # print("ddx_conv2",dx_bn1_d2.sum().item())
    # print("ddx_conv2(pre)",(ddx_conv2).sum().item())
    # print("ddx_conv2(pre)",(ddx_conv2**2).sum().item())

    ddx_conv2[x_conv2<=0] = 0
    # print("ddx_conv2",(ddx_conv2).sum().item())
    # print("out",x_conv2.sum().item())

    ddx_bn2, dx_conv2_d2, dconv2w_d2= conv_double_bwd(ddx_conv2, ddconv2w, None,dx_bn2, conv2w, x_conv2)
    dx_bn2_d2, dbn2w_d2 , ddO = batchnorm_double_backwards_fn_new(x_bn2, bn2w, ddx_bn2, ddbn2w, ddbn2b,gO=dbno2)   
    ddO += ddx_conv1
    ddO[out<=0] = 0
    return ddO, dx_conv1_d2,dx_bn1_d2,dx_conv2_d2,dx_bn2_d2, dconv1w_d2, dbn1w_d2, dconv2w_d2, dbn2w_d2



class BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        x_bn1 = self.conv1(x)
        x_conv2 = F.relu(self.bn1(x_bn1), inplace=True)
        x_bn2 = self.conv2(x_conv2)
        out = self.bn2(x_bn2)
        out = F.relu(out + identity, inplace=True)
        return x_bn1, x_conv2, x_bn2, out


# 这个基本上只是为了调准确性的
class BasicBlock_VirticalFuse(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

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
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        x_bn1 = self.conv1(x)
        x_conv2 = self.bn1(x_bn1)
        x_conv2 = F.relu(x_conv2, inplace=True)
        x_bn2 = self.conv2(x_conv2)
        out = self.bn2(x_bn2)
        out = F.relu(out + identity, inplace=True)
        return x_bn1, x_conv2, x_bn2, out


class TinyResNet(nn.Module):
    def __init__(self, flag,Fuse,  in_channels=3, base_channels=64, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(base_channels)
        if flag=='original':
            # self.block = BasicBlock_manuel(base_channels)
            self.block = BasicBlock(base_channels)
        elif flag == 'vertical':
            self.block = BasicBlock_VirticalFuse(base_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = nn.Linear(base_channels, num_classes)

    def forward(self, x_conv):
        x_bn = self.conv(x_conv)
        x_block = self.bn(x_bn)
        x_block = F.relu(x_block, inplace=True)
        x_bn1, x_conv2, x_bn2, x_pool = self.block(x_block)
        x_fc = self.pool(x_pool)
        x_fc = torch.flatten(x_fc, 1)
        x_out = self.fc(x_fc)
        return x_out, x_fc, x_pool, x_bn2, x_conv2, x_bn1, x_block, x_bn
    

###################################
###################################
# flag = 'flex'        
flag = 'original'
flag = 'vertical'         # 单纯x为了debug写的
Fuse = 1
batch_size = 128
test_iter = 1
set_random_seed() # 仅仅在ACC test的时候使用。会严重影响性能。 
###################################
###################################



if __name__ == "__main__":
    print("flag = " + flag)
    model1 = TinyResNet( flag, Fuse=Fuse).to("cuda")
    model = model1

    transform = transforms.ToTensor()
    cifar10 = torchvision.datasets.CIFAR10(root='/scratch/yguo25/files/mtt-distillation/data', train=True, download=True, transform=transform)
    # 取一个 batch
    indices = torch.arange(batch_size)   # 或 torch.randperm(len(cifar10))[:batch_size]
    imgs, labels = zip(*(cifar10[i] for i in indices))
    x = torch.stack(imgs).to("cuda").requires_grad_(True)
    target = torch.tensor(labels, device="cuda")


    # torch.save(model.state_dict(), 'model_test_basicblock.pt')
    # exit()
    pretrained_dict = torch.load("model_test_basicblock.pt")

    # if flag =='vertical' or flag == 'flex':
    #     x = x.repeat(1, Fuse, 1, 1).detach().clone().requires_grad_()
    #     target = target.repeat(Fuse)
    #     for i,j in pretrained_dict.items():
    #         if j.ndim  != 0 :
    #             # pretrained_dict[i] = torch.cat([j, j], dim=0)
    #             pretrained_dict[i] = j.repeat((Fuse,) + (1,) * (j.ndim - 1))      

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
            x_out, x_fc, x_pool, x_bn2, x_conv2, x_bn1, x_block, x_bn = model(x)  # forward
            loss = criterion(x_out, target)  # compute loss
            print("----CELOSS-----", loss.item())
            # dx_fc = torch.torch.autograd.grad(loss, x_fc, retain_graph=True)[0]
            dw = torch.torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
            # print(len(dw))
            # print("dx_fc",dx_fc.sum().item())
            # print("dconvw",dw[0].sum().item())
            # print("dbnw",dw[1].sum().item())
            # print("dbnb",dw[2].sum().item())
            # print("dconv1w",dw[3].sum().item())
            # print("dbn1w",dw[4].sum().item())
            # print("dbn1b",dw[5].sum().item())
            # print("dconv2w",dw[6].sum().item())
            # print("dbn2w",dw[7].sum().item())
            # print("dbn2b",dw[8].sum().item())
            # print("dfcw",dw[9].sum().item())
            # print("dfcb",dw[10].sum().item())
            weight = [d.sum() for d in dw]
            grad_loss = sum(weight)
            print("----GRANDLOSS-----", grad_loss.item())
            grad_loss.backward()  
            print("----GRAD-----", x.grad.sum().item())
            # dxout = torch.autograd.grad(loss, x_out )[0]
            # print("----temp-----", dxout.sum().item())


        elif flag =='vertical':
            x_out, x_fc, x_pool, x_bn2, x_conv2, x_bn1, x_block, x_bn = model(x)  # forward
            loss = criterion(x_out, target)  # compute loss
            print("----CELOSS-----", loss.item())
            with torch.no_grad():
                dx_out = torch.autograd.grad(loss,x_out)[0]
                dx_fc, dfcw, dfcb = torch.autograd.grad(x_out, [x_fc, model.fc.weight, model.fc.bias ], grad_outputs=dx_out)

                dx_fc = dx_fc.view(batch_size, 64,1,1)
                dx_pool = adaptivepooling_bwd(x_pool, grad_output= dx_fc)
                # dx_pool = torch.autograd.grad(x_fc, x_pool, grad_outputs=dx_fc, create_graph=True)[0]

                # targets = [x_block, model.block.bn2.weight, model.block.bn2.bias, model.block.conv2.weight,  \
                #            model.block.bn1.weight, model.block.bn1.bias, model.block.conv1.weight]
                # dx_block, dbn2w,dbn2b,dconv2w,dbn1w,dbn1b,dconv1w = torch.autograd.grad(x_pool, targets, grad_outputs=dx_pool, create_graph=True)
                dx_block, dconv1w, dbn1w, dbn1b, dconv2w, dbn2w, dbn2b, dbno2, dbno1, dx_bn1, dx_bn2 = BasicBlock_bwd(x_block, x_bn1, x_conv2, x_bn2,x_pool, model.block.conv1.weight, model.block.bn1.weight , model.block.bn1.bias, \
                            model.block.conv2.weight, model.block.bn2.weight , model.block.bn2.bias, grad_output= dx_pool  )
                
                dx_block[x_block <= 0] = 0
                dx_bn,dbnw, dbnb = batchNorm2d_backward(x_bn, model.bn.weight, grad_output=dx_block)
                # dx_bn,dbnw, dbnb = torch.autograd.grad(x_block, [x_bn, model.bn.weight,  model.bn.bias, ], grad_outputs=dx_block, create_graph=True)
                # dconvw = torch.autograd.grad(x_bn, [ model.conv.weight], grad_outputs=dx_bn) [0]         
                _, dconvw ,_ = conv_bwd(x, model.conv.weight, grad_output=dx_bn)

                dw = [dconvw,dbnw,dbnb,dconv1w,dbn1w,dbn1b,dconv2w,dbn2w,dbn2b,dfcw,dfcb]

                weight = [d.sum() for d in dw]
                grad_loss = sum(weight)
                print("----GRANDLOSS-----", grad_loss.item())
                # exit()


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
                dx_bn_d2, dbnw_d2,ddx_block = batchnorm_double_backwards_fn_new(x_bn, model.bn.weight,ddx_bn,ddbnw,ddbnb,dx_block)
                ddx_block[x_block<=0] = 0
                ddx_pool, dx_block_d2,dx_bn1_d2,dx_conv2_d2,dx_bn2_d2, dconv1w_d2, dbn1w_d2, dconv2w_d2, dbn2w_d2 = BasicBlock_double_bwd(\
                                            x_block,x_bn1,x_conv2,x_bn2,x_pool,\
                                            dx_bn1, dx_bn2, dx_pool, dbno1,dbno2,\
                                            model.block.conv1.weight,model.block.bn1.weight,  model.block.conv2.weight,model.block.bn2.weight, \
                                             ddconv1w, None, ddbn1w, ddbn1b, ddconv2w, None, ddbn2w,ddbn2b, ddx_block  ) 
                

                ddx_lin = adaptivepooling_double_bwd(ddx_pool)
                ddx_out, dx_lin_d2, _ = linearFused_double_bwd(x_fc,model.fc.weight, dx_out, ddx_lin, ddfcw ,ddfcb ,1)

                dx_out_d1 = crossEntropy_double_bwd(x_out, ddx_out, Fuse) # 没有y。也就是说 CrossEntropy 对 logits 的 Hessian 只和 p = softmax(z) 有关，和 target 无关。
                # print("----temp-----", dx_out_d1.sum().item())
                
                dx_lin_d1, _, _ = linerFused_bwd(x_fc, model.fc.weight, grad_output=dx_out_d1, Fuse=1)
                dx_lin_d1 += dx_lin_d2
                dx_lin_d1 = dx_lin_d1.view(batch_size, 64,1,1)

                dx_pool_d1 = adaptivepooling_bwd(x_pool, grad_output= dx_lin_d1)
                dx_block_d1, dconv1w, dbn1w, dbn1b, dconv2w, dbn2w, dbn2b, dbno2, dbno1 = BasicBlock_bwd2_1(x_block, x_bn1, x_conv2, x_bn2,x_pool, model.block.conv1.weight, model.block.bn1.weight , model.block.bn1.bias, \
                            model.block.conv2.weight, model.block.bn2.weight , model.block.bn2.bias, dx_pool_d1 , dx_block_d2, dx_conv2_d2, dx_bn1_d2,dx_bn2_d2 )
                
                dx_block_d1[x_block <= 0] = 0
                # dx_block_d1_new = np.where(x_block <= 0, 0, dx_block_d1)
                
                # dx_bn_d1,dbnw, dbnb = torch.autograd.grad(x_block, [x_bn, model.bn.weight,  model.bn.bias ], grad_outputs=dx_block_d1, create_graph=True)
                dx_bn_d1,dbnw, dbnb = batchNorm2d_backward(x_bn, model.bn.weight, grad_output=dx_block_d1)
                dx_bn_d1 +=dx_bn_d2
                dx_conv,_,_ = conv_bwd(x, model.conv.weight, grad_output=dx_bn_d1)
                # dx_conv = torch.autograd.grad(x_bn, [ x], grad_outputs=dx_bn_d1, create_graph=True) [0]     
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
            dw = torch.torch.autograd.grad(loss, list(model.parameters()), create_graph=True)    # GRANDLOSS 0.6183616518974304 ----GRAD----- -1.0767822265625
            weight = [d.sum() for d in dw]
            grad_loss = sum(weight)

            print("----GRANDLOSS-----", grad_loss.item())
            grad_loss.backward()  
            print("----GRAD-----", x.grad.sum().item())
            
        optimizer.step()  # update x

    end = time.time()

    print("当前显存使用:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("峰值显存使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    print("时间占用：", end-start)