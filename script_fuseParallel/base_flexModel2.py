# 11.16 测试Conv_Flexfused 性能而写的。这次不会把所有东西都写在forward里，而是会封装一部分到stateless model 
# 。希望可以内存消耗小一点。


# 11.13
# 目前已经写完了所有stateless model 。现在这个test code 的目的就是写一个完全flex fuse 后的模型。
# 然后和Fuse=1 的完全bwd去比较一下性能


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
from networks_stacked import LinearStacked_2 # NOTE 这里和flex fuse 不太一样。
from networks_stateless_basicblock import linear_bwd, conv_bwd, insNormNRelu_bwd, \
linear_double_bwd, conv_double_bwd, insNormNRelu_double_bwd, avgPool_bwd, crossEntropy_bwd, \
    avgPool_double_bwd,bmm_bwd, linerFused_bwd, linearFused_double_bwd,crossEntropy_double_bwd
    

from networks_fused3 import NormActive
from networks.networks_flexFuse import Conv_Flexfused, ConvBlock_double_bwd,ConvBlock_bwd2_1

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


class Conv_original(nn.Module):
    def __init__(self, net_width, Fuse):
        super(Conv_original, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=net_width, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(net_width, affine=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.linear = nn.Linear(net_width * 16 * 16, 10)
    def forward(self, x):
        x = self.conv1(x)          # N x net_width x 32 x 32
        x = self.norm1(x)          # N x net_width x 32 x 32
        x = F.relu(x)             # N x net_width x 32 x 32
        x = self.pool1(x)          # N x net_width x 16 x 16
        x = x.view(x.size(0), -1) # Flatten to N x (net_width*16*16)
        x = self.linear(x)    # N x 10
        return x


# 这个基本上只是为了调准确性的
class Conv_Fused(nn.Module):
    def __init__(self, net_width, Fuse):
        super(Conv_Fused, self).__init__()
        self.Fuse = Fuse
        self.conv1 = nn.Conv2d(in_channels=3*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        # self.norm1 = nn.InstanceNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.norm1 = NormActive(net_width*Fuse) #精度有差别
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.linear = LinearStacked_2(net_width * 16 * 16, 10,Fuse )
        self.net_width= net_width

    def forward(self, x):
        x = x.view(-1,self.Fuse*3 ,32,32)        # 10, 256, 4,4   -> 20, 2048
        x = self.conv1(x)          # N x net_width x 32 x 32
        x = self.norm1(x)          # N x net_width x 32 x 32
        # x = F.relu(x)             # N x net_width x 32 x 32
        x = self.pool1(x)          # N x net_width x 16 x 16
        x = x.view(x.size(0), -1) # Flatten to N x (net_width*16*16)
        x = self.linear(x)    # N x 10
        return x





###################################
###################################
# flag = 'conv'
flag = 'convFuse'
flag = 'convFuse_2'         # 单纯为了debug写的
flag = 'convFlexFuse'
Fuse = 2
batch_size = 1024
test_iter = 1
set_random_seed() # 仅仅在ACC test的时候使用。会严重影响性能。 
###################################
###################################



if __name__ == "__main__":
    # print(flag)

    model1 = Conv_original(32, Fuse).to("cuda")
    model2 = Conv_Flexfused(32, Fuse).to("cuda")
    model3 = Conv_Fused(32, Fuse).to("cuda")
    if flag =='conv':
        model = model1
    elif flag=='convFlexFuse':
        model = model2
    elif flag=='convFuse':
        model = model3
    elif flag=='convFuse_2':    
        model = model2
        


    transform = transforms.ToTensor()
    cifar10 = torchvision.datasets.CIFAR10(root='/scratch/yguo25/files/mtt-distillation/data', train=True, download=True, transform=transform)
    # 取一个 batch
    indices = torch.arange(batch_size)   # 或 torch.randperm(len(cifar10))[:batch_size]
    imgs, labels = zip(*(cifar10[i] for i in indices))
    x = torch.stack(imgs).to("cuda").requires_grad_(True)
    target = torch.tensor(labels, device="cuda")


    pretrained_dict = torch.load("model_test_FlexFuse.pt")
    if flag =='convFlexFuse' or flag == 'convFuse' or flag== 'convFuse_2':
        x = x.repeat(1, Fuse, 1, 1).detach().clone().requires_grad_()
        target = target.repeat(Fuse)
        for i,j in pretrained_dict.items():
            if j.ndim  != 0 :
                # pretrained_dict[i] = torch.cat([j, j], dim=0)
                pretrained_dict[i] = j.repeat((Fuse,) + (1,) * (j.ndim - 1))      

    load_state_dict_by_position(model, pretrained_dict)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([x], lr=1e-1)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start = time.time()

    for step in range(test_iter):
        optimizer.zero_grad()

        if flag =='conv':
            output = model(x)  # forward
            loss = criterion(output, target)  # compute loss
            print("----CELOSS-----", loss.item())
            dw = torch.torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
            weight = [d.sum() for d in dw]
            grad_loss = sum(weight)
            print("----GRANDLOSS-----", grad_loss.item())
            grad_loss.backward()  
            print("----GRAD-----", x.grad.sum().item())

        elif flag =='convFlexFuse':
            # with torch.no_grad():
            output= model(x,criterion, target)  # forward+1stbwd+weight op
            x_conv,x_norm, x_pool,x_lin,x_out =output

            celoss = criterion(x_out, target)  # compute loss
            celoss *= Fuse
            print("----CELOSS-----", celoss.item())
            # x_norm.retain_grad()
            # x_pool.retain_grad()
            # x_out.retain_grad()
            # celoss.backward()
            # dx_norm = x_norm.grad
            # dx_pool = x_pool.grad
            # dx_out = x_out.grad
            dx_norm,dx_pool,dx_out   = torch.torch.autograd.grad(celoss,[x_norm,x_pool,x_out]) 
            
            # # TODO: autograd 的内存消耗和手动求解一阶没区别。速度不知道。这样的话不如用loss.bwackward
            # dx_out = crossEntropy_bwd(x_out, target,Fuse)
            # dx_lin, dlin_w, dlin_b = linerFused_bwd(x_lin, model.linear.weight, grad_output=dx_out, Fuse=Fuse)
            # dx_lin = dx_lin.reshape(-1,  32*Fuse, 16,16)
            # dx_pool = avgPool_bwd( x_pool, grad_output= dx_lin )
            # dx_norm, dnorm_w, dnorm_b = insNormNRelu_bwd(x_norm, model.norm1.weight, x_pool, grad_output=dx_pool)
            # _, dconv_w, dconv_b = conv_bwd( x_conv, model.conv1.weight, grad_output=dx_norm, groups=Fuse)


            # # del dconv_w, dconv_b, dnorm_w, dnorm_b, dlin_w, dlin_b # 这个大约也占300 多M。但可能不是非得在这里del 。
            # dweights = [d.grad.sum() for d in model.parameters()]
            # # dweights = [dconv_w, dconv_b, dnorm_w, dnorm_b, dlin_w, dlin_b]
            # # dweights = [d.sum() for d in dweights]
            # grad_loss = sum(dweights)
            # print("----GRANDLOSS-----", grad_loss.sum().item())        
            # # x_conv, x_norm, x_pool,x_lin, x_out, dx_norm, dx_pool,dx_out = graphNodes
            with torch.no_grad():
                ddcon_w = torch.ones_like(model.conv1.weight).cuda()
                ddconv_b = torch.ones_like(model.conv1.bias).cuda()
                ddnorm_w = torch.ones_like(model.norm1.weight).cuda()
                ddnorm_b = torch.ones_like(model.norm1.bias).cuda()
                ddlin_w = torch.ones_like(model.linear.weight).cuda()
                ddlin_b = torch.ones_like(model.linear.bias).cuda()
                ddx_conv = torch.zeros_like(x).cuda()
                # break
                # mem 峰值在这里
                # ddx_lin, dxconv_d2, _,dx_norm_d2,_ = ConvBlock_double_bwd(x_conv, x_norm, x_pool, dx_norm, dx_pool, ddx_conv, model.conv1.weight, model.norm1.weight,ddcon_w, ddconv_b, ddnorm_w, ddnorm_b,Fuse  )
                ddx_norm, dxconv_d2, dconvw_d2 = conv_double_bwd(ddx_conv, ddcon_w, ddconv_b, dx_norm, model.conv1.weight, x_conv, groups_=Fuse )
                del ddx_conv,dx_norm
                x_norm.grad=None
                ddx_pool, dx_norm_d2, dnormw_d2 = insNormNRelu_double_bwd(ddx_norm, ddnorm_w, ddnorm_b, dx_pool, x_pool, model.norm1.weight, x_norm)
                del ddx_norm, dx_pool
                x_pool.grad=None
                ddx_lin = avgPool_double_bwd(ddx_pool)
                del ddx_pool
                # break
                
                ddx_out, dx_lin_d2, _ = linearFused_double_bwd(x_lin,model.linear.weight, dx_out, ddx_lin, ddlin_w,ddlin_b,Fuse)
                del dx_out, ddx_lin
                x_out.grad=None
                # dx_out_d1 = torch.torch.autograd.grad(dx_out, x_out, grad_outputs=ddx_out)[0] # criterion bwd
                dx_out_d1 = crossEntropy_double_bwd(x_out, ddx_out, Fuse) # 没有y。也就是说 CrossEntropy 对 logits 的 Hessian 只和 p = softmax(z) 有关，和 target 无关。
                del ddx_out,x_out
                dx_lin_d1, _, _ = linerFused_bwd(x_lin, model.linear.weight, grad_output=dx_out_d1, Fuse=Fuse)
                del x_lin,dx_out_d1
                dx_lin_d1 = dx_lin_d1.reshape(-1, 32 * Fuse, 16,16) 
                dx_lin_d1 += dx_lin_d2 
                dx_conv_d1 , _ ,_ ,_ ,_ = ConvBlock_bwd2_1(x_conv, x_norm, x_pool, model.conv1.weight, model.norm1.weight, dx_lin_d1,dx_norm_d2,dxconv_d2 ,Fuse=Fuse)
                del dx_norm_d2,dxconv_d2
                print("----GRAD-----", dx_conv_d1.sum().item())


        elif flag =='convFuse' or 'convFuse_2':
            output = model(x)  # forward
            if  flag =='convFuse_2':
                output = output[-1]
            output = output.view(-1,10)
            loss = criterion(output, target)  # compute loss
            loss *= Fuse
            print("----CELOSS-----", loss.item())

            # TODO:            这里有一个很诡异的现象，autograd 和 loss.backward 结果居然不一样。
            # 关键是二阶导数也不受weight 的影响。只能理解成是因为没有梯度清零，导致了梯度累计，干扰了下一次autograd 的结果。
            # loss.backward(create_graph=True) # ----GRANDLOSS----- 0.618267834186554, ----GRAD----- -1.1076405048370361
            # weight = [d.grad.sum() for d in model.parameters()]
            dw = torch.torch.autograd.grad(loss, list(model.parameters()), create_graph=True)    # GRANDLOSS 0.6183616518974304 ----GRAD----- -1.0767822265625
            weight = [d.sum() for d in dw]
            grad_loss = sum(weight)
            # break
            # grad_loss*=Fuse
            print("----GRANDLOSS-----", grad_loss.item())
            grad_loss.backward()  
            print("----GRAD-----", x.grad.sum().item())
            
        optimizer.step()  # update x

    end = time.time()

    print("当前显存使用:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("峰值显存使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    print("时间占用：", end-start)