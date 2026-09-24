# 这个版本专门给linear_doublebwd 写的



import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import torchvision.transforms as transforms
from functools import reduce
from operator import mul
import triton
import triton.language as tl
import time 
import random
import numpy as np


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks_stacked import LinearStacked_2,LinearStacked_2_flexFuse # NOTE 这里和flex fuse 不太一样。
from networks_stateless_basicblock import linear_bwd, conv_bwd, insNormNRelu_bwd, \
linear_double_bwd, conv_double_bwd, insNormNRelu_double_bwd, avgPool_bwd, crossEntropy_bwd, \
    avgPool_double_bwd,bmm_bwd, linerFused_bwd, linearFused_double_bwd,crossEntropy_double_bwd, \
    linear_bwd, linear_double_bwd
    

from networks_fused3 import NormActive
from networks_flexFuse import Conv_Flexfused, ConvBlock_double_bwd,ConvBlock_bwd2_1

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


class Conv_Original(nn.Module):
    def __init__(self, net_width, Fuse):
        super(Conv_Original, self).__init__()
        self.linear = nn.Linear(3 * 32 * 32, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten to N x (net_width*16*16)
        x = self.linear(x)    # N x 10
        return x


class Conv_Flexfuse(nn.Module):
    def __init__(self, net_width, Fuse):
        super(Conv_Flexfuse, self).__init__()
        self.Fuse = Fuse
        self.linear = LinearStacked_2_flexFuse(3 * 32 * 32, 10, Fuse )
        # self.linear = nn.Linear(net_width * 16 * 16, 10*Fuse )
        self.net_width= net_width

    def forward(self, x_lin):
        x_lin = x_lin.view(2048,-1) 
        x_out = self.linear(x_lin)    # N x 10
        return  [x_lin,x_out]




###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
# 只有两个模型但是可以对应到4个flag。Conv_Flexfuse_flex是测试的核心对象。bwd——doublebwd
flag = 'Conv_Bwd+doubleBwd'
flag = 'Conv_Flexfuse_acc'  # 用来对齐精度的
flag = 'Conv_Flexfuse_flex' 

Fuse = 2
batch_size = 1024
test_iter = 1
set_random_seed() # 仅仅在ACC test的时候使用。会严重影响性能。 
###################################
###################################
###################################
###################################
###################################
###################################
###################################






if __name__ == "__main__":
    model1 = Conv_Original(32, Fuse).to("cuda")
    model2 = Conv_Flexfuse(32, Fuse).to("cuda")
    if flag =='Conv_Bwd+doubleBwd':
        model = model1
    elif flag=='Conv_Flexfuse_flex' or 'Conv_Flexfuse_acc':
        model = model2

    transform = transforms.ToTensor()
    cifar10 = torchvision.datasets.CIFAR10(root='/scratch/yguo25/files/mtt-distillation/data', train=True, download=True, transform=transform)
    indices = torch.arange(batch_size)   # 或 torch.randperm(len(cifar10))[:batch_size]
    imgs, labels = zip(*(cifar10[i] for i in indices))
    x = torch.stack(imgs).to("cuda").detach().requires_grad_(True)
    target = torch.tensor(labels, device="cuda")

    # torch.save(model.state_dict(), 'model_test_linearFused.pt')
    # exit()

    pretrained_dict = torch.load("model_test_linearFused.pt")
    if flag =='Conv_Flexfuse_flex' or flag =='Conv_Flexfuse_acc': 
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
            
        if flag =='Conv_Flexfuse_flex':
            x_lin,x_out = model(x)  # forward+1stbwd+weight op

            x_out = x_out.view(-1,10)
            loss = criterion(x_out, target)  # compute loss
            loss *= Fuse
            print("----CELOSS-----", loss.item())
            # dconv_w,dconv_b, dnorm_w,dnorm_b, dlin_w,dlin_b  = torch.autograd.grad(loss, list(model.parameters()), retain_graph=True)
            # dx_norm,dx_pool,dx_out   = torch.torch.autograd.grad(loss,[x_norm,x_pool,x_out]) 

            # dx_out_d1 = crossEntropy_double_bwd(x_out, ddx_out, Fuse)
            dx_out =  torch.autograd.grad(loss, x_out)[0]
            dx_lin_d1, dlin_w, dlin_b = linerFused_bwd(x_lin, model.linear.weight, grad_output=dx_out, Fuse=Fuse)
            dx_lin_d1 = dx_lin_d1.reshape(-1, 32 * Fuse, 16,16)  

            dw = [dlin_w,dlin_b]
            # dw = [d.requires_grad_() for d in dw] # 因为上面没开cg=t。所以默认是不要梯度的
            grand_loss = sum((d**2).sum() for d in dw)
            print("---Dbwd-GRANDLOSS2-----", grand_loss.item()) 

            # ddx_lin, dxconv_d2, _,dx_norm_d2,_ = torch.autograd.grad(grand_loss, )

            # # ddconv_w,ddconv_b,ddnorm_w,ddnorm_b,ddlin_w,ddlin_b  = torch.autograd.grad(grand_loss, dw)
            with torch.no_grad():
            #     ddconv_w = dconv_w.detach()*2
            #     ddconv_b = dconv_b.detach()*2
            #     ddnorm_w = dnorm_w.detach()*2
            #     ddnorm_b = dnorm_b.detach()*2
                ddlin_w =  dlin_w.detach() *2
                ddlin_b =  dlin_b.detach() *2
                ddx_conv = torch.zeros_like(x).cuda()
            #     # mem 峰值在这里
            #     ddx_lin, dxconv_d2, _,dx_norm_d2,_ = ConvBlock_double_bwd(x_conv, x_norm, x_pool, dx_norm, dx_pool, ddx_conv, model.conv1.weight, model.norm1.weight,ddconv_w, ddconv_b, ddnorm_w, ddnorm_b,Fuse  )
            #     del ddx_conv,dx_norm,dx_pool
                
            #     # print("CKPT DDXLIN",ddx_lin.sum().item())
                ddx_out, dx_lin_d2, _ = linearFused_double_bwd(x_lin,model.linear.weight, dx_out, None, ddlin_w,ddlin_b,Fuse)
            #     ddx_out, dx_lin_d2, _ = linear_double_bwd(x_lin,model.linear.weight, dx_out, ddx_lin, ddlin_w,ddlin_b)
            #     del dx_out, ddx_lin
                dx_out_d1 = crossEntropy_double_bwd(x_out, ddx_out, Fuse)

            #     del ddx_out,x_out
                dx_lin_d1, _, _ = linerFused_bwd(x_lin, model.linear.weight, grad_output=dx_out_d1, Fuse=Fuse)
                # dx_lin_d1, _, _ = linear_bwd(x_lin, model.linear.weight, grad_output=dx_out_d1)
            #     # print(x_lin.shape)
            #     del x_lin,dx_out_d1

            #     # print(dx_lin_d1.shape)
            #     # print(dx_lin_d2.shape)
                dx_lin_d1 += dx_lin_d2 
            #     dx_lin_d1 = dx_lin_d1.view(-1, 32 * Fuse, 16,16) 
            #     dx_conv_d1 , _ ,_ ,_ ,_ = ConvBlock_bwd2_1(x_conv, x_norm, x_pool, model.conv1.weight, model.norm1.weight, dx_lin_d1,dx_norm_d2,dxconv_d2 ,Fuse=Fuse)
            #     del dx_norm_d2,dxconv_d2
                print("----GRAD-----", dx_lin_d1.sum().item()) # 0.06954991072416306
 

        elif flag =='Conv_Flexfuse_acc':
            x_lin,x_out = model(x)  # forward+1stbwd+weight op

            x_out = x_out.view(-1,10)
            loss = criterion(x_out, target)  # compute loss
            loss *= Fuse
            print("----CELOSS-----", loss.item())
            # dlin = torch.autograd.grad(loss, x_lin, retain_graph=True)[0]
            dw = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)


            grand_loss = sum((d**2).sum() for d in dw)
            # grand_loss *=Fuse # 不知道这里为啥不用乘了。反正distill要
            print("----GRANDLOSS-----", grand_loss.item())
            grand_loss.backward()  
            print("----GRAD-----", x.grad.sum().item()) # -3.3310112953186035

            # TODO: 测一下具体问题。
            # dxout = torch.autograd.grad(grand_loss, x_out)[0]
            # print("CKPT",dxout.sum().item())
            # dx = torch.autograd.grad(output, x, grad_outputs=dxout)[0]
            # print("----GRAD-----", dx.sum().item()) # -3.3310112953186035


            
        optimizer.step()  # update x

    end = time.time()

    print("当前显存使用:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("峰值显存使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    print("时间占用：", end-start)