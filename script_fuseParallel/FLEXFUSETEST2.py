# 11.23
# 上一个已经对齐了精度。现在想自己写一个Conv3 然后比一下。


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
from networks.networks_stacked import LinearStacked_2 # NOTE 这里和flex fuse 不太一样。
from networks.networks_stateless_basicblock import linear_bwd, conv_bwd, insNormNRelu_bwd, \
linear_double_bwd, conv_double_bwd, insNormNRelu_double_bwd, avgPool_bwd, crossEntropy_bwd, \
    avgPool_double_bwd,bmm_bwd, linerFused_bwd, linearFused_double_bwd,crossEntropy_double_bwd
    

from networks.networks_fused3 import NormActive
from networks.past_version.networks_flexFuse_deleted import Conv_Flexfused, ConvBlock_double_bwd,ConvBlock_bwd2_1
from networks.networks_stateless import conv3_double_bwd,conv3_bwd

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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=net_width, kernel_size=3, padding=1)
        self.norm1 = NormActive(net_width, affine=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=net_width, out_channels=net_width, kernel_size=3, padding=1)
        self.norm2 = NormActive(net_width, affine=True)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=net_width, out_channels=net_width, kernel_size=3, padding=1)
        self.norm3 = NormActive(net_width, affine=True)
        self.pool3 = nn.AvgPool2d(kernel_size=2)
        self.linear = nn.Linear(net_width * 4 * 4, 10)
    def forward(self, x):
        x = self.conv1(x)          # N x net_width x 32 x 32
        x = self.norm1(x)          # N x net_width x 32 x 32
        x = self.pool1(x)          # N x net_width x 16 x 16
        x = self.conv2(x)          # N x net_width x 32 x 32
        x = self.norm2(x)          # N x net_width x 32 x 32
        x = self.pool2(x)          # N x net_width x 16 x 16
        x = self.conv3(x)          # N x net_width x 32 x 32
        x = self.norm3(x)          # N x net_width x 32 x 32
        x = self.pool3(x)          # N x net_width x 16 x 16
        x = x.view(x.size(0), -1) # Flatten to N x (net_width*16*16)
        x = self.linear(x)    # N x 10
        return x


class Conv_Flexfuse(nn.Module):
    def __init__(self, net_width, Fuse):
        super(Conv_Flexfuse, self).__init__()
        self.Fuse = Fuse
        self.conv1 = nn.Conv2d(in_channels=3*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        self.norm1 = NormActive(net_width*Fuse) #BN在channel上单独计算，所以目前不用管。
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=net_width*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        self.norm2 = NormActive(net_width*Fuse) #BN在channel上单独计算，所以目前不用管。
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=net_width*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        self.norm3 = NormActive(net_width*Fuse) #BN在channel上单独计算，所以目前不用管。
        self.pool3 = nn.AvgPool2d(kernel_size=2)
        self.linear = LinearStacked_2(net_width * 4 * 4, 10,Fuse )
        self.net_width= net_width

    def forward(self, x_conv1):
        x_conv1 = x_conv1.view(-1,self.Fuse*3 ,32,32)        # 10, 256, 4,4   -> 20, 2048
        x_norm1 = self.conv1(x_conv1)          
        x_pool1 = self.norm1(x_norm1)    
        x_conv2 = self.pool1(x_pool1)
        # x_conv2 = x_conv1.view(-1,self.Fuse*3 ,32,32)        # 10, 256, 4,4   -> 20, 2048
        x_norm2 = self.conv2(x_conv2)          
        x_pool2 = self.norm2(x_norm2)    
        x_conv3 = self.pool2(x_pool2)
        # x_conv3 = x_conv1.view(-1,self.Fuse*3 ,32,32)        # 10, 256, 4,4   -> 20, 2048
        x_norm3 = self.conv3(x_conv3)          
        x_pool3 = self.norm3(x_norm3)    
        x_lin  = self.pool3(x_pool3)
        x_out = self.linear(x_lin)    # N x 10
        x_out = x_out.view(-1,10)
        return  x_conv1,x_norm1, x_pool1,x_conv2,x_norm2, x_pool2,x_conv3,x_norm3, x_pool3, x_lin,x_out
















###################################
###################################
# 只有两个模型但是可以对应到4个flag。Conv_Flexfuse_flex是测试的核心对象。bwd——doublebwd
# flag = 'Conv_Flexfuse_acc'  # 用来对齐精度的
flag = 'Conv_Bwd+doubleBwd'
flag = 'Conv_Flexfuse_flex' 

Fuse = 2
batch_size = 1024
test_iter = 1
set_random_seed() # 仅仅在ACC test的时候使用。会严重影响性能。 
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
    # torch.save(model.state_dict(), 'model_test_FlexFuse2.pt')
    # exit()
    pretrained_dict = torch.load("model_test_FlexFuse2.pt")
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
        if flag == 'Conv_Bwd+doubleBwd':
            # 那么，就做bwd+double bwd，分开做。 先一阶
            output = model(x)  # forward
            loss = criterion(output, target)  # compute loss
            print("---bwd-CELOSS1-----", loss.item())
            # 似乎不需要清理
            output = model(x)  # forward
            loss = criterion(output, target)  # compute loss
            print("---Dbwd-CELOSS2-----", loss.item())
            dw = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
            grand_loss = sum((d**2).sum() for d in dw)
            print("---Dbwd-GRANDLOSS2-----", grand_loss.item())
            grand_loss.backward()  
            print("---Dbwd-GRAD2-----", x.grad.sum().item())
            optimizer.zero_grad()
            
        elif flag =='Conv_Flexfuse_flex':
            x_conv1,x_norm1, x_pool1,x_conv2,x_norm2, x_pool2,x_conv3,x_norm3, x_pool3, x_lin,x_out = model(x)  # forward+1stbwd+weight op
            celoss = criterion(x_out, target)  # compute loss
            celoss *= Fuse
            print("----CELOSS-----", celoss.item())
            l = list(model.parameters()) + [x_norm1,x_pool1,x_norm2,x_pool2,x_norm3,x_pool3,x_out]
            l  = torch.torch.autograd.grad(celoss,l) 
            dw =l[:-7] 
            dconv1_w,dconv1_b,dnorm1_w,dnorm1_b,dconv2_w,dconv2_b,dnorm2_w,dnorm2_b,dconv3_w,dconv3_b,dnorm3_w,dnorm3_b,dlin_w,dlin_b = dw
            dx_norm1,dx_pool1,dx_norm2,dx_pool2,dx_norm3,dx_pool3,dx_out=l[-7:] 
            # dw = [d.requires_grad_() for d in dw] # 因为上面没开cg=t。所以默认是不要梯度的
            grand_loss = sum((d**2).sum() for d in dw)
            print("---Dbwd-GRANDLOSS2-----", grand_loss.item()) 

            # ddconv_w,ddconv_b,ddnorm_w,ddnorm_b,ddlin_w,ddlin_b  = torch.autograd.grad(grand_loss, dw)
            with torch.no_grad():

                ddx_conv = torch.zeros_like(x).cuda()
                # convB1 
                ddx_conv2, dxconv1_d2, _,dx_norm1_d2,_ = ConvBlock_double_bwd(x_conv1, x_norm1, x_pool1, dx_norm1, dx_pool1, \
                                                                              ddx_conv, model.conv1.weight, model.norm1.weight,dconv1_w*2, dconv1_b*2, dnorm1_w*2, dnorm1_b*2,Fuse  )
                del ddx_conv,dx_norm1,dx_pool1,dconv1_w,dconv1_b,dnorm1_w,dnorm1_b
                ddx_conv3, dxconv2_d2, _,dx_norm2_d2,_ = ConvBlock_double_bwd(x_conv2, x_norm2, x_pool2, dx_norm2, dx_pool2, \
                                                                              ddx_conv2, model.conv2.weight, model.norm2.weight,dconv2_w*2, dconv2_b*2, dnorm2_w*2, dnorm2_b*2,Fuse  )
                del ddx_conv2,dx_norm2,dx_pool2
                ddx_lin, dxconv3_d2, _,dx_norm3_d2,_ = ConvBlock_double_bwd(x_conv3, x_norm3, x_pool3, dx_norm3, dx_pool3, \
                                                                            ddx_conv3, model.conv3.weight, model.norm3.weight,dconv3_w*2, dconv3_b*2, dnorm3_w*2, dnorm3_b*2,Fuse  )
                del ddx_conv3,dx_norm3,dx_pool3
                ddx_out, dx_lin_d2, _ = linearFused_double_bwd(x_lin,model.linear.weight, dx_out, ddx_lin, dlin_w*2 ,dlin_b*2 ,Fuse)
                del dx_out, ddx_lin
                dx_out_d1 = crossEntropy_double_bwd(x_out, ddx_out, Fuse)
                del ddx_out,x_out

                # dx_out_d1, dx_norm1_d2, dx_norm2_d2, dx_norm3_d2, dxconv1_d2, dxconv2_d2, dxconv3_d2 ,dx_lin_d2= conv3_double_bwd(x_conv1,x_norm1, x_pool1,x_conv2,x_norm2, x_pool2,x_conv3,x_norm3, x_pool3, x_lin,x_out ,dx_norm1, dx_norm2,dx_norm3,dx_pool1,dx_pool2,dx_pool3,
                #      dconv1_w,dconv1_b,dnorm1_w,dnorm1_b,dconv2_w,dconv2_b,dnorm2_w,dnorm2_b,dconv3_w,dconv3_b,dnorm3_w,dnorm3_b,dlin_w,ddlin_b,ddx_conv,dx_out,
                #       model, Fuse=2 )

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

                # dx_conv1_d1 = conv3_bwd(x_lin, dx_out_d1, dx_lin_d2, x_conv3, x_norm3, x_pool3, dx_norm3_d2,
                #                          dxconv3_d2, x_conv2, x_norm2, x_pool2, dx_norm2_d2, dxconv2_d2, x_conv1, x_norm1, x_pool1, dx_norm1_d2, dxconv1_d2, model, Fuse)
                print("----GRAD-----", dx_conv1_d1.sum().item()) # 0.06954991072416306
 


        elif flag =='Conv_Flexfuse_acc':
            output = model(x)[-1]  # forward+1stbwd+weight op

            output = output.view(-1,10)
            loss = criterion(output, target)  # compute loss
            loss *= Fuse
            print("----CELOSS-----", loss.item())
            dw = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
            grand_loss = sum((d**2).sum() for d in dw)
            # grand_loss *=Fuse # 不知道这里为啥不用乘了。反正distill要
            print("----GRANDLOSS-----", grand_loss.item())
            grand_loss.backward()  
            print("----GRAD-----", x.grad.sum().item()) # -3.3310112953186035


            
        optimizer.step()  # update x

    end = time.time()

    print("当前显存使用:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("峰值显存使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    print("时间占用：", end-start)