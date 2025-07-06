# 在3的基础上继续改进，能够做conv
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import torchvision.transforms as transforms
import numpy as np
import random


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    # For CPU & GPU deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Optional: for newer PyTorch versions (>=1.10)
    torch.use_deterministic_algorithms(True)
    # If using torch.compile() in 2.0+, set this env var
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For CUDA reproducibility

class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 256, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(256, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc2 = nn.Linear(256 * 16 * 16, 10)
    def forward(self, x, target):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) 
        logits = self.fc2(x)
        return logits

class MyConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 256, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(256, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc2 = nn.Linear(256 * 16 * 16, 10)
    def forward(self, x3, target):
        x2 = self.conv(x3)
        x1 = self.norm(x2)
        x_poolin = self.relu(x1)
        x = self.pool(x_poolin)
        t1 = x.view(x.size(0), -1) 
        logits = self.fc2(t1)
        # prop back
        grad_output = self.crossEntropy_backward(logits, target)
        # grad_output = criterion(logits, target)
        # grad_output = torch.autograd.grad( grad_output,logits, create_graph=True)[0]

        # return grad_output

        dfc2b = grad_output.sum(dim=0)  
        dfc2w = grad_output.t()@t1
        grad_output = grad_output@self.fc2.weight
        grad_output = grad_output.view(x2.shape[0], 256, 16, 16)

        grad_output = F.interpolate(grad_output, scale_factor=2, mode='nearest') / 4
        # grad_output = self.avg_pool2d_backward(grad_output, x1.shape, 2,2)

        # TODO:  relu out似乎必须得是》=才能结果对的上。下一步 dx1是8817，但是d norm。bias是8845。现在instanceNorm_backward 显然做的不对。
        relu_grad = (x1 >= 0).float()
        grad_output = grad_output * relu_grad


        # grad_output = torch.autograd.grad( logits,x1 , create_graph=True,grad_outputs=grad_output)[0]
        # return grad_output


        grad_output,d_gamma,d_beta = self.instance_norm2d_backward(x2, self.norm.weight, grad_output)
        # grad_output,d_gamma,d_beta = self.instanceNorm_backward(x2, self.norm.weight, grad_output)
        x_grad= grad_output
        # x_grad = self.convLayer_backward(grad_output, relu_in=x1, norm_in=x2, conv_in=x3)[2]

        return grad_output
    
    def crossEntropy_back(self, logits, target):
        N = target.shape[0]
        softmax = F.softmax(logits, dim=1)
        one_hot = torch.zeros_like(logits)
        one_hot[range(N), target] = 1.0
        grad_output = (softmax - one_hot) / N
        return grad_output

    def avg_pool2d_backward(self, dy, input_shape, kernel_size=2, stride=2):
        N, C, H, W = input_shape
        kH, kW = kernel_size, kernel_size
        sH, sW = stride, stride
        # 初始化输入梯度
        dx = torch.zeros(input_shape).to('cuda')
        H_out = dy.shape[2]
        W_out = dy.shape[3]
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * sH
                        h_end = h_start + kH
                        w_start = j * sW
                        w_end = w_start + kW
                        dx[n, c, h_start:h_end, w_start:w_end] += dy[n, c, i, j] / (kH * kW)
        return dx 
    
    def crossEntropy_backward(self, logits, targets):
        N, C = logits.shape
        # 1. Compute log_softmax
        log_probs = F.log_softmax(logits, dim=1)
        # 2. Compute grad of NLLLoss (mean reduction)
        grad = torch.exp(log_probs)  # shape: (N, C)
        grad[range(N), targets] -= 1
        grad = grad / N

        return grad
    
    def instanceNorm_backward(self, x, gamma, grad_output, eps=1e-5):
        N, C, H, W = x.shape
        M = H * W
        x_reshaped = x.view(N, C, M)
        grad_output_reshaped = grad_output.view(N, C, M)
        grad_beta = grad_output_reshaped.sum(dim=(0, 2))             # (C,)
        mean = x_reshaped.mean(dim=2, keepdim=True)  # (N, C, 1)
        var = x_reshaped.var(dim=2, unbiased=False, keepdim=True)  # (N, C, 1)
        std = torch.sqrt(var + eps)  # (N, C, 1)
        x_hat = (x_reshaped - mean) / std  # (N, C, M)
        grad_output_hat = grad_output_reshaped * gamma.view(1, C, 1)  # (N, C, M)
        dx = (1. / M) / std * (
            M * grad_output_hat
            - grad_output_hat.sum(dim=2, keepdim=True)
            - x_hat * (grad_output_hat * x_hat).sum(dim=2, keepdim=True)
        )  # (N, C, M)
        grad_gamma = (grad_output_reshaped * x_hat).sum(dim=(0, 2))  # (C,)
        return dx.view(N, C, H, W), grad_gamma, grad_beta

    def instance_norm2d_backward(self, x, gamma, grad_output, eps=1e-5):
        N, C, H, W = x.shape
        x_reshaped = x.view(N, C, -1)  # (N, C, H*W)
        dy = grad_output.view(N, C, -1)  # same shape

        # Compute mean and variance
        mean = x_reshaped.mean(dim=2, keepdim=True)
        var = x_reshaped.var(dim=2, unbiased=False, keepdim=True)
        std = torch.sqrt(var + eps)

        x_hat = (x_reshaped - mean) / std  # (N, C, H*W)

        # Compute dgamma and dbeta
        d_gamma = torch.sum(dy * x_hat, dim=(0, 2))  # (C,)
        d_beta  = torch.sum(dy, dim=(0, 2))          # ✅ 正确做法

        # dx_hat
        dx_hat = dy * gamma.view(1, C, 1)

        # Backprop through normalization
        HW = H * W
        dx = (1. / HW) / std * (
            HW * dx_hat
            - dx_hat.sum(dim=2, keepdim=True)
            - x_hat * torch.sum(dx_hat * x_hat, dim=2, keepdim=True)
        )
        dx = dx.view(N, C, H, W)
        return dx, d_gamma, d_beta

    def convLayer_backward(self, grad_output, relu_in, norm_in,conv_in, norm, conv, stride=1, padding=1 ):
        grad_output = F.interpolate(grad_output, scale_factor=2, mode='nearest') / 4
        relu_grad = (relu_in > 0).float()
        grad_output = grad_output * relu_grad
        grad_output,d_gamma,d_beta = self.instanceNorm_backward(norm_in, norm.weight, grad_output)
        # TODO: 目前conv层的结果还是有点问题。不知道是累积误差导致的还是什么，结果会差几位
        db = grad_output.sum(dim=(0, 2, 3))
        dw = torch.nn.grad.conv2d_weight(conv_in, conv.weight.shape, grad_output, stride=stride, padding=padding)
        # dw = self.conv2d_weight_grad(conv_in, conv.weight.shape, grad_output, stride=stride, padding=padding)
        dx = torch.nn.grad.conv2d_input(input_size=conv_in.shape, weight=self.conv.weight, grad_output=grad_output, stride=1, padding=1)
        # dx = F.conv_transpose2d(grad_output, conv.weight, stride=stride, padding=padding)         # only when stride == padding
        return [dx, dw, db, d_gamma, d_beta]
    def conv2d_weight_grad(self, input, weight_shape, grad_output, stride=1, padding=0, dilation=1, groups=1):
        N = input.shape[0]
        C_out, C_in_per_group, kH, kW = weight_shape
        # unfold input to im2col
        input_unf = F.unfold(input, kernel_size=(kH, kW), dilation=dilation, padding=padding, stride=stride)
        # shape: (N, C_in * kH * kW, L), where L is number of sliding positions
        grad_output_reshaped = grad_output.reshape(N, C_out, -1)  # (N, C_out, L)
        # 使用 einsum 来做 batch 矩阵乘法 + 求和
        grad_weight = torch.einsum('ncl,nkl->ck', grad_output_reshaped, input_unf)  # (C_out, C_in * kH * kW)
        return grad_weight.view(weight_shape)  # reshape 成权重形式




set_seed(0)
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
    
batch_size = 1
transform = transforms.ToTensor()
cifar10 = torchvision.datasets.CIFAR10(root='/scratch/yguo25/files/mtt-distillation/data', train=True, download=True, transform=transform)
img, label = cifar10[0]
x = img.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # shape: [batch_size, 3, 32, 32]
x = x.clone().detach().to("cuda").requires_grad_(True)
target = torch.tensor([label] * batch_size, device="cuda")  # shape: [batch_size]
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([x], lr=1e-1)


# model = Conv()
model = MyConv()
# torch.save(model.state_dict(), 'model_test4.pth')
model.load_state_dict(torch.load('model_test4.pth'))
model = model.to('cuda')

# model = torch.compile(model, mode="reduce-overhead")
# for param in model.parameters():
#     param.requires_grad = True

import time

start = time.time()
for step in range(100):
    optimizer.zero_grad()
    output = model(x,target)  # forward

    # loss1 = criterion(output, target )  # compute loss
    # output  = torch.autograd.grad(loss1,(model.norm.bias), create_graph=True)[0]

    loss = abs(output.sum())
    print(loss.item())
    exit()
    loss.backward()
    optimizer.step()  # update x

    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
end = time.time()
print(end-start)
print("当前显存使用:", torch.cuda.memory_allocated() / 1024**2, "MB")
print("峰值显存使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")