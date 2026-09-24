# 10.31
# 主要是想测试一下水平fusion 的加速效果极限在哪里、增加batch size会不会对kernel size产生影响。

# 1 batch size 的增加会影响加速比吗？
# Cifar10 batch    1000相当于cifar100 ipc10. model 是个256的（相比之下cifar10 Conv 128宽，大约大了一倍）

#               conv（无fuse版本）   vs.    conv*2 （无fuse版本）      vs。   batchedconv （fuse=2）

#   16          0.4066252613067627.             0.8                      0.47950220108032227

#   32         0.4629540157318115              0.92                       0.8320863246917725

#   64         0.543942461013794              1.08                       1.0486414432525635

#   124         0.7635984420776367             1.52                       1.449554204940796

#  1024         3.892086982727051              7.6                       7.740631580352783       

#   2048        7.499631404876709               15                        14.864470720291138




#####(错误，因为没有关set seed)-------------------
# 3 分界线 在Conv + cifar10 + ipc6 的时候， fusion 就已经没有效果了。   这个点比想象中的要早很多。
# 4 这个阈值大约是6*原始大小。但是根据最早的测试数据，哪怕Fuskion= 7 都还是有很高。这里有点对不上。
#####------------------------


# 结论：
# 1 增加batch/ model width都会增加kernel size。
# 2 kernel 目前没有感受到负优化。但是其实在size=32的时候就已经没什么太大差别。
# 3 目前不知道会不会有某个阈值代表了fusion的优化界限。 哪怕在2024batch size，fusion也总是快0.1左右。不知道原因。可能是data locality导致的？



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

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭自动优化，确保计算确定性
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 保证 CUDA 计算稳定（仅对 PyTorch 1.8+ 有效）





class LinearStacked_2(nn.Module):
    # batch* fusion * channel * WH。 
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
        # self.weight = self.weight.view(self.Fuse,self.out_features,self.in_features)
        # self.bias = self.bias.view(self.Fuse, self.out_features)

        x = x.view(-1,self.Fuse, self.in_features)
        # print(x.is_contiguous())

        # # # # TODO: 这里输入如果是BAD/（而不是ABD）的话，可以得到is_contiguous 的结果。那就很简单只要调整target即可
        # x = torch.einsum("abc,bcd->abd",x,self.weight.view(self.Fuse,self.out_features,self.in_features).transpose(-1, -2)) 
        # x = x+self.bias.view(self.Fuse,self.out_features)
        # x = x.contiguous()

        # TODO: 这里输入如果是BAD/（而不是ABD）的话，可以得到is_contiguous 的结果。那就很简单只要调整target即可
        x = torch.einsum("abc,bcd->bad",x,self.weight.view(self.Fuse,self.out_features,self.in_features).transpose(-1, -2)) 
        x = x+self.bias.view(self.Fuse,1 ,self.out_features)

        # 用bmm试一试？ weight: f*out, in ; x: batch,fuse, in. 输出x fuse，batch，out
        # x = x.transpose(0,1)
        # x = torch.bmm(x, self.weight.view(self.Fuse, self.out_features, self.in_features).transpose(-1, -2) )
        # x = x+self.bias.view(self.Fuse,1 ,self.out_features)

        return x



class Conv_original(nn.Module):
    def __init__(self, net_width):
        super(Conv_original, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=net_width, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(net_width, affine=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.classifier = nn.Linear(net_width * 16 * 16, 10)
    def forward(self, x):
        # x = x.view(-1, self.num_feat)        # 10, 256, 4,4   -> 20, 2048
        x = self.conv1(x)          # N x net_width x 32 x 32
        x = self.norm1(x)          # N x net_width x 32 x 32
        x = F.relu(x)             # N x net_width x 32 x 32
        x = self.pool1(x)          # N x net_width x 16 x 16

        x = x.view(x.size(0), -1) # Flatten to N x (net_width*16*16)
        # print("CKPT", ( (x@ self.classifier.weight.T) +self.classifier.bias ).sum().item())
        x = self.classifier(x)    # N x 10
        # print("CKPT", x.sum().item())

        return x
    
class ConvNet_bctched(nn.Module):
    def __init__(self, net_width, Fuse):
        self.Fuse = Fuse
        super(ConvNet_bctched, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        self.norm1 = nn.BatchNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.classifier = LinearStacked_2(net_width * 16 * 16, 10,Fuse )
    def forward(self, x):
        x = x.view(-1,self.Fuse*3 ,32,32)        # 10, 256, 4,4   -> 20, 2048
        x = self.conv1(x)          # N x net_width x 32 x 32
        x = self.norm1(x)          # N x net_width x 32 x 32
        x = F.relu(x)             # N x net_width x 32 x 32
        x1 = self.pool1(x)          # N x net_width x 16 x 16
        # print(x1.sum().item())
        x = x1.view(x1.size(0), -1) # Flatten to N x (net_width*16*16)
        x = self.classifier(x)    # N x 10
        # print((x1[:, :32, :, :].view(x1.size(0), -1)).sum().item() )
        # weight : fuse, out, in. 首先切片，然后reshape？
        # w = self.classifier.weight
        # print(w[:10].shape)
        # print((x1[:, :32, :, :].view(x1.size(0), -1) @ self.classifier.weight[:10].T + self.classifier.bias[:10]).sum().item())
        # print("CKPT", x[1:2 ].sum().item())
        return x
    
# class batchedconv(nn.Module):
#     def __init__(self, net_width, Fuse):
#         super(batchedconv, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
#         self.norm1 = nn.BatchNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
#         self.pool1 = nn.AvgPool2d(kernel_size=2)
#         self.classifier = LinearStacked_2(net_width * 16 * 16, 10,2 )
#     def forward(self, x, criterion, target):
#         x = x.view(-1,6 ,32,32)        # 10, 256, 4,4   -> 20, 2048
#         x_norm = self.conv1(x)          # N x net_width x 32 x 32
#         x_relu = self.norm1(x_norm)          # N x net_width x 32 x 32
#         x_pool = F.relu(x_relu)             # N x net_width x 32 x 32
#         x_pool = self.pool1(x_pool)          # N x net_width x 16 x 16
#         x_lin = x_pool.view(x_pool.size(0), -1) # Flatten to N x (net_width*16*16)
#         out = self.classifier(x_lin)    # N x 10
#         # TODO: 2 需要手动实现一阶的loss backward。并且同时把需要的ctx输出。
#         # 2.1 criterion
#         out = out.view(-1,10)
#         loss = criterion(out, target) 

#         # 2.2 求dw，以及所有中间要存ctx的量。 这个地方不知道能不能用autograd
#         # dw = torch.torch.autograd.grad(loss, list(model.parameters()))
#         dw=[]
#         dout = torch.torch.autograd.grad(loss, out) # criterion bwd
#         grad_output, dlinw, dlinb = torch.torch.autograd.grad(out, [x_lin, self.classifier.weight, self.classifier.bias], grad_outputs= dout)  # linear bwd
#         # relu bwd， pooling bwd
#         grad_output = grad_output.view(x_pool.shape)
#         grad_output = F.interpolate(grad_output, scale_factor=2, mode='nearest') /4
#         relu_grad = (x_relu > 0).float()
#         grad_output = grad_output * relu_grad
#         grad_output, d_gamma, d_beta = torch.torch.autograd.grad(x_relu, [x_norm,self.norm1.weight, self.norm1.bias], grad_outputs=grad_output ) # norm bwd
#         grad_output, dconvw, dconvb = torch.torch.autograd.grad(x_norm, [x,self.conv1.weight, self.conv1.bias], grad_outputs=grad_output ) # conv bwd
#         dw = [ dconvw, dconvb, d_gamma, d_beta, dlinw, dlinb]

#         # 2.3 求grand_loss, 这一步根据具体情况调整。
#         weight = [d.sum() for d in dw]
#         grad_loss = sum(weight)
#         return grad_loss, dw

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

#################CONFIG#################
flag = 'batchedconv'
flag = 'conv'
Fuse = 8
batch_size = 32    
test_iter= 1
##################################
# set_random_seed() # 仅仅在ACC test的时候使用。会严重影响性能。 

import argparse

parser = argparse.ArgumentParser(description="Run experiment with configs")
parser.add_argument("--flag", type=str, help="Type of convolution: conv / batchedconv")
parser.add_argument("--batch_size", type=int, help="Batch size")
parser.add_argument("--test_iter", type=int,  help="Number of test iterations")
parser.add_argument("--fuse", type=int,  help="Fusion size")
args = parser.parse_args()
if args.flag is not None:
    flag = args.flag
if args.batch_size is not None:
    batch_size = args.batch_size
if args.test_iter is not None:
    test_iter = args.test_iter
if args.fuse is not None:
    Fuse = args.fuse

if __name__ == "__main__":


    model1 = Conv_original(32).to("cuda")
    model2 = ConvNet_bctched(32, Fuse).to("cuda")
    if flag =='conv':
        model = model1
    else:
        model = model2

    transform = transforms.ToTensor()
    cifar10 = torchvision.datasets.CIFAR10(root='/scratch/yguo25/files/mtt-distillation/data', train=True, download=True, transform=transform)
    img, label = cifar10[0]
    x = img.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # shape: [batch_size, 3, 32, 32]
    x = x.clone().detach().to("cuda").requires_grad_(True)
    target = torch.tensor([label] * batch_size, device="cuda")  # shape: [batch_size]

    def loss_func(grad_real, dw):
        grad_match_loss = sum(((gr - gs) ** 2).sum() for gr, gs in zip(grad_real, dw))
        return grad_match_loss



    # TODO: 1 输入和target、weight先变成两倍
    # torch.save(model.state_dict(), 'model_test7.pt')
    # exit()
    # # model.load_state_dict(torch.load('model_test5.pt'), strict = False)

    pretrained_dict = torch.load("model_test7.pt")
    if flag =='batchedconv':
        x = x.repeat(1, Fuse, 1, 1).detach().clone().requires_grad_()
        target = target.repeat(Fuse)
        
        for i,j in pretrained_dict.items():
            if j.ndim  != 0 :
                # pretrained_dict[i] = torch.cat([j, j], dim=0)
                pretrained_dict[i] = j.repeat((Fuse,) + (1,) * (j.ndim - 1))         

    # 如何复制，取决于原始weight里面是怎么排布的， linear放在最外面，但是weight不知道。（应该也是最外面吧）
    # pretrained_dict = [(torch.cat([x, x], dim=0)) for x in pretrained_dict.items()] 
    load_state_dict_by_position(model, pretrained_dict)

    # x = x.repeat_interleave(Fuse,dim =1)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([x], lr=1e-1)

    for i in range(50): # warmup
        output = model(x)  # forward
        output = output.view(-1,10)
        loss = criterion(output, target)
        dw = torch.torch.autograd.grad(loss, list(model.parameters()))
    optimizer.zero_grad()


    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start = time.time()

    for step in range(test_iter):
        optimizer.zero_grad()

        # if flag =='conv':
        output = model(x)  # forward

        output = output.view(-1,10)
        # print(output[9:19, :].sum().item())
        # print(output[:10, :].sum().item())
        # print(output[10:20, :].sum().item())
        # print(output[5:15, :].sum().item())
        # print(opututput.shape)
        # print(output.shape)
        if flag =='conv':
            loss = criterion(output, target)  # compute loss
        else:
            # mid = int(10*Fuse/2)
            # loss = criterion(output[:mid,:], target[:mid])+ criterion(output[mid:20], target[mid:20])
            loss = criterion(output, target)  # compute loss
            loss*=Fuse # 还是需要乘！
        # print(loss.item())
        dw = torch.torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
        # # weight = list(model.parameters()) 
        # # weight = [(1- p + g).sum() for p, g in zip(weight, dw)]
        # weight = [d.sum() for d in dw]
        # grad_loss = sum(weight)

        grad_real = [torch.ones_like(p) for p in model.parameters()]     # 全 1 梯度（稳定可复现）
        # TODO: 这里，平方项也需要谨慎。
        # print(dw[0].shape)
        grad_loss= loss_func(grad_real, dw)
        

        # TODO: 为了测试速度的话暂时注释掉了。因为这里的求解比较麻烦。暂时我们只考虑kernel上的加速。
        # if flag == 'conv':
        #     grad_loss= loss_func(grad_real, dw)
        # else:
        #     for f in range(Fuse):
        #         # 每个分支取第0维上对应的 slice
        #         start = f * N
        #         end = (f + 1) * N
        #         grad_real_f = grad_real[start:end]  # 当前分支的真实梯度片段
        #         # 把 dw 的每一层都取出该分支对应的部分
        #         dw_f = [d[start:end] for d in dw]
        #         grad_loss += loss_func(grad_real_f, dw_f)


        grad_loss.backward()  

        optimizer.step()  # update x
        # print("----CELOSS-----", loss.item())
        # print("----GRADW-----", dw[0].sum().item())
        # # print(dw[0][:32,:,:,:].sum().item())
        # print("----GRANDLOSS-----", grad_loss.item())
        # print("----GRAD-----",x.grad.sum().item())

    end = time.time()

    print("当前显存使用:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("峰值显存使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    print("时间占用：", end-start)