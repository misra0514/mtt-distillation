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
from networks_stacked import LinearStacked_2 as LinearStacked_2_original
from networks_stateless_basicblock import linear_bwd, conv_bwd, insNormNRelu_bwd, \
linear_double_bwd, conv_double_bwd, insNormNRelu_double_bwd, avgPool_bwd, \
    avgPool_double_bwd,bmm_bwd, linerFused_bwd, linearFused_double_bwd, crossEntropy_bwd,crossEntropy_double_bwd

from networks_fused3 import NormActive

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
    # print(pretrained_state_dict.keys())
    # print(len(pretrained_state_dict.keys()))
    # print(len(pretrained_items))
    assert len(model_items) == len(pretrained_items), \
        f"参数数量不一致：当前模型有 {len(model_items)} 个参数，预训练模型有 {len(pretrained_items)} 个参数"
    for (model_key, _), (_, pretrained_val) in zip(model_items, pretrained_items):
        new_state_dict[model_key] = pretrained_val
    model.load_state_dict(new_state_dict)


class LinearStacked_2(nn.Module):
    # 从horuzontal fuse 复制来的。用bmm而不是einsum。虽然没啥区别
    def __init__(self ,in_features, out_features, Fuse):
        super(LinearStacked_2, self).__init__()
        self.Fuse = Fuse
        self.in_features = in_features
        self.out_features = out_features
        # TODO: 在nn实现中，这里是一个转制，也就是说应该是Fuse, out_features, in_features
        self.weight = torch.nn.Parameter(torch.randn(Fuse* out_features,in_features))
        self.bias = torch.nn.Parameter(torch.randn(self.Fuse* out_features))

    def forward(self, x):
        """
        x目前仅支持二维输入： B* STK * In。 B和stk可以view 在一起。 weight  STK*IN*OUT 
        """
        x = x.view(-1,self.Fuse, self.in_features).transpose(0,1)
        # TODO: 这里输入如果是BAD/（而不是ABD）的话，可以得到is_contiguous 的结果。那就很简单只要调整target即可
        # x = torch.einsum("abc,bcd->bad",x,self.weight.view(self.Fuse,self.out_features,self.in_features).transpose(-1, -2)) 
        # x = x+self.bias.view(self.Fuse,1 ,self.out_features)
        # 用bmm试一试？ weight: f*out, in ; x: batch,fuse, in. 输出x fuse，batch，out
        x = torch.bmm(x, self.weight.view(self.Fuse, self.out_features, self.in_features).transpose(-1, -2) )
        x = x+self.bias.view(self.Fuse,1 ,self.out_features)

        return x


class Conv_original(nn.Module):
    def __init__(self, net_width, Fuse):
        super(Conv_original, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=net_width, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(net_width, affine=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.classifier = nn.Linear(net_width * 16 * 16, 10)
    def forward(self, x):
        x = self.conv1(x)          # N x net_width x 32 x 32
        x = self.norm1(x)          # N x net_width x 32 x 32
        x = F.relu(x)             # N x net_width x 32 x 32
        x = self.pool1(x)          # N x net_width x 16 x 16
        x = x.view(x.size(0), -1) # Flatten to N x (net_width*16*16)
        x = self.classifier(x)    # N x 10
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
        self.classifier = LinearStacked_2(net_width * 16 * 16, 10,Fuse )
        self.net_width= net_width

    def forward(self, x):
        x = x.view(-1,self.Fuse*3 ,32,32)        # 10, 256, 4,4   -> 20, 2048
        x = self.conv1(x)          # N x net_width x 32 x 32
        x = self.norm1(x)          # N x net_width x 32 x 32
        # x = F.relu(x)             # N x net_width x 32 x 32
        x = self.pool1(x)          # N x net_width x 16 x 16
        x = x.view(x.size(0), -1) # Flatten to N x (net_width*16*16)
        x = self.classifier(x)    # N x 10
        return x



class Conv_Flexfused(nn.Module):
    def __init__(self, net_width, Fuse):
        super(Conv_Flexfused, self).__init__()
        self.Fuse = Fuse
        self.conv1 = nn.Conv2d(in_channels=3*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        self.norm1 = nn.InstanceNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.classifier = LinearStacked_2(net_width * 16 * 16, 10,Fuse )
        self.net_width= net_width


    def forward(self, x_conv, criterion, target):
        x_conv = x_conv.view(-1,self.Fuse*3 ,32,32)        # 10, 256, 4,4   -> 20, 2048
        x_norm = self.conv1(x_conv)          # N x net_width x 32 x 32
        x_pool = self.norm1(x_norm)          # N x net_width x 32 x 32
        x_pool = F.relu(x_pool)             # N x net_width x 32 x 32
        x_lin = self.pool1(x_pool)          # N x net_width x 16 x 16
        # print(x1.sum().item())
        # x_lin = x_lin.view(x_lin.size(0), -1) # Flatten to N x (net_width*16*16)
        x_out = self.classifier(x_lin)    # N x 10
        # print(x_out.shape) #ze([2, 1024, 10])
        x_out = x_out.view(-1,10)
        loss = criterion(x_out, target)  # compute loss
        loss *= self.Fuse
        print("----CELOSS-----", loss.item())

        # 1st order BWD
        dx_out = crossEntropy_bwd(x_out, target, Fuse)
        # dx_out = torch.autograd.grad(loss, x_out, create_graph=True)[0] # criterion bwd
        # print(dx_out.shape).  # 2048, 10
        # TODO: 从这里开始，就比较麻烦了。 我写的linear和conv的反向。并没有batched linear的反向。 
        # 但是如果不自己写stateless， 就拿不到 d_lin_weight, d_lin_bias 。 conv 和 norm 目前都没问题。
        # dx_lin, d_lin_weight, d_lin_bias = linear_bwd(x_lin, self.classifier.weight, grad_output=dx_out)
        # dx_lin, d_lin_weight, d_lin_bias = torch.autograd.grad(x_out, [x_lin, self.classifier.weight, self.classifier.bias], grad_outputs=dx_out, create_graph= True)
        # print(dx_lin.shape) # ([1024, 64, 16, 16])

        dx_lin, d_lin_weight, d_lin_bias = linerFused_bwd(x_lin, self.classifier.weight, grad_output=dx_out, Fuse=self.Fuse)
        # print(dx_lin.shape) # e([2, 1024, 8192]) 
        # TODO: 这里显然得transpose回去。因为stacked 里面 transpose 出来了。
        # TODO: autograd 好像就不用 (不过为啥对精度没有造成影响？？)应该是因为测试用例都是一样的图片。
        # dx_lin.transpose(0,1)
 
        # print(dx_lin.shape) # ([1024, 2, 8192])
        # TODO: 这里已经不能用view了。有空再确认一下这里的shape是对的
        dx_lin = dx_lin.reshape(-1, self.net_width*self.Fuse, 16,16)
        dx_pool = avgPool_bwd( x_pool, grad_output= dx_lin )
        dx_norm, d_norm_weight, d_norm_bias = insNormNRelu_bwd(x_norm, self.norm1.weight, x_pool, grad_output=dx_pool)
        dx_conv, d_conv_weight, d_conv_bias = conv_bwd( x_conv, self.conv1.weight, grad_output=dx_norm, groups=self.Fuse)

        grad_loss =d_conv_weight.sum() + d_conv_bias.sum() + d_norm_weight.sum() + d_norm_bias.sum() +  d_lin_weight.sum() + d_lin_bias.sum() 
        print("----GRANDLOSS-----", grad_loss.item())
        # exit()

        # 2en order BWD
        # 2.1 loss compute， 需要对gradloss func求梯度。 因为现在的sum是累加，dloss/dweight = 1 所以传入的梯度是。。？
        # 这里好像还必须得手动算。因为One of the differentiated Tensors does not require grad ？ 
        # ddconw, ddconvb, ddlinw, ddlinb = torch.autograd.grad(grad_loss, [grad_lin_weight,d_lin_bias,grad_conv_weight, grad_conv_bias])
        ddcon_w = torch.ones_like(self.conv1.weight).cuda()
        ddconv_b = torch.ones_like(self.conv1.bias).cuda()
        ddnorm_w = torch.ones_like(self.norm1.weight).cuda()
        ddnorm_b = torch.ones_like(self.norm1.bias).cuda()
        ddlin_w = torch.ones_like(self.classifier.weight).cuda()
        ddlin_b = torch.ones_like(self.classifier.bias).cuda()
        ddx_conv = torch.zeros_like(x_conv).cuda()

        # 2.2 double BWD
        # 另外注意这里的dwconv 和 grad_conv_weight 可不一样。虽然都是一阶的。
        ddx_norm, dxconv_d2, dconvw_d2 = conv_double_bwd(ddx_conv, ddcon_w, ddconv_b, dx_norm, self.conv1.weight,x_conv, groups_=self.Fuse )
        ddx_pool, dx_norm_d2, dnormw_d2 = insNormNRelu_double_bwd(ddx_norm, ddnorm_w, ddnorm_b, dx_pool, x_pool, self.norm1.weight, x_norm)
        ddx_lin = avgPool_double_bwd(ddx_pool)

        # ddx_out, dxlin_d2, dlinw_d2 = linear_double_bwd(ddx_lin, ddlin_w, ddlin_b, dx_out, self.classifier.weight, x_lin )
        ddx_out, dx_lin_d2, dlinw_d2 = linearFused_double_bwd(x_lin,self.classifier.weight, dx_out, ddx_lin, ddlin_w,ddlin_b,self.Fuse)
        # print(ddx_out.shape) ([2048, 10])

        # (grad_output, ddx_lin, ddlinw, ddlinb, x_lin, self.classifier.weight,self.classifier.bias)
        # 2.3 re do 1st BWD
        
        dx_out_d1 = crossEntropy_double_bwd(x_out, ddx_out) # 没有y。也就是说 CrossEntropy 对 logits 的 Hessian 只和 p = softmax(z) 有关，和 target 无关。
        # dx_out_d1 = crossEntropy_bwd(x_out, target, grad_output=ddx_out)
        # dx_out_d1 = torch.torch.autograd.grad(dx_out, x_out, grad_outputs=ddx_out)[0] # criterion bwd

        # dx_lin_d1, d_lin_weight_d1, d_lin_bias_d1 = linear_bwd( x_lin, self.classifier.weight, grad_output=dx_out_d1)
        dx_lin_d1, d_lin_weight_d1, d_lin_bias_d1 = linerFused_bwd(x_lin, self.classifier.weight, grad_output=dx_out_d1, Fuse=self.Fuse)

        # 好像只有下面加上，这里加上，结果才比较接近。不懂什么情况
        # print(dx_lin_d1.shape)# ([2, 1024, 8192]) --- > [1024, 2, 8192]
        # print(dx_lin_d2.shape) # [1024, 16384]) —-->[1024, 64, 16, 16])
        # dx_lin_d2 = dx_lin_d2.view(1024,self.Fuse,-1).transpose()
        # dx_lin_d1 += dx_lin_d2 
        # dx_lin_d1 = dx_lin_d1.view(-1, self.net_width, 32,32) 
        dx_lin_d1 = dx_lin_d1.reshape(-1, self.net_width * self.Fuse, 16,16) 
        dx_lin_d1 += dx_lin_d2  

        dx_pool_d1 = avgPool_bwd( x_pool, grad_output= dx_lin_d1 )
        dx_norm_d1, d_norm_weight, d_norm_bias = insNormNRelu_bwd(x_norm, self.norm1.weight, x_pool, grad_output=dx_pool_d1)
        dx_norm_d1 += dx_norm_d2
        dx_conv_d1, d_conv_weight_d1, d_conv_bias_d1 = conv_bwd(x_conv, self.conv1.weight, grad_output=dx_norm_d1, groups=self.Fuse)
        # print(dxconv_d2.sum().item())

        return dx_conv_d1 + dxconv_d2



###################################
###################################
flag = 'conv'
flag = 'convFuse'
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

    transform = transforms.ToTensor()
    cifar10 = torchvision.datasets.CIFAR10(root='/scratch/yguo25/files/mtt-distillation/data', train=True, download=True, transform=transform)
    # img, label = cifar10[0]
    # x = img.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # shape: [batch_size, 3, 32, 32]
    # x = x.clone().detach().to("cuda").requires_grad_(True)
    # target = torch.tensor([label] * batch_size, device="cuda")  # shape: [batch_size]
    # 取一个 batch
    indices = torch.arange(batch_size)   # 或 torch.randperm(len(cifar10))[:batch_size]
    imgs, labels = zip(*(cifar10[i] for i in indices))
    x = torch.stack(imgs).to("cuda").requires_grad_(True)
    target = torch.tensor(labels, device="cuda")


    # torch.save(model.state_dict(), 'model_test_FlexFuse.pt')
    # exit()
    # # model.load_state_dict(torch.load('model_test5.pt'), strict = False)


    pretrained_dict = torch.load("model_test_FlexFuse.pt")
    if flag =='convFlexFuse' or flag == 'convFuse':
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
            output = model(x,criterion, target)  # forward+1stbwd+weight op
            print("----GRAD-----", output.sum().item())            


        elif flag =='convFuse':
            output = model(x)  # forward
            output = output.view(-1,10)
            loss = criterion(output, target)  # compute loss
            loss *= Fuse
            print("----CELOSS-----", loss.item())
            # 因为test code是summation， 而真实场景是AVG。所以这里不用乘法，但是ditill里面需要。
            dw = torch.torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
            weight = [d.sum() for d in dw]
            grad_loss = sum(weight)
            # grad_loss*=Fuse
            print("----GRANDLOSS-----", grad_loss.item())
            grad_loss.backward()  
            print("----GRAD-----", x.grad.sum().item())            
            
        optimizer.step()  # update x

    end = time.time()

    print("当前显存使用:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("峰值显存使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    print("时间占用：", end-start)