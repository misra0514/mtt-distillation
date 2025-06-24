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
        x = self.relu(x1)
        x = self.pool(x)
        t1 = x.view(x.size(0), -1) 
        logits = self.fc2(t1)
        # prop back
        # grad_output = self.crossEntropy_backward(logits, target)
        grad_output = criterion(logits, target)
        grad_output = torch.autograd.grad( grad_output,logits, create_graph=True)[0]

        dfc2b = grad_output.sum(dim=0)  
        dfc2w = grad_output.t()@t1
        grad_output = grad_output@self.fc2.weight
        grad_output = grad_output.view(x2.shape[0], 256, 16, 16)

        # x_grad = self.convLayer_backward(grad_output, relu_in=x1, norm_in=x2, conv_in=x3)[2]
        grad_output = F.interpolate(grad_output, scale_factor=2, mode='nearest') / 4
        relu_grad = (x1 > 0).float()
        grad_output = grad_output * relu_grad
        grad_output,d_gamma,d_beta = self.instanceNorm_backward(x2, self.norm.weight, grad_output)
        x_grad= d_beta

        return x_grad

    def crossEntropy_backward(self, logits, target):
        N = target.shape[0]
        softmax = F.softmax(logits, dim=1)
        one_hot = torch.zeros_like(logits)
        one_hot[range(N), target] = 1.0
        grad_output = (softmax - one_hot) / N
        return grad_output
    
    def instanceNorm_backward(self, x, gamma, grad_output, eps=1e-5):
        N, C, H, W = x.shape
        M = H * W
        x_reshaped = x.view(N, C, M)
        grad_output_reshaped = grad_output.view(N, C, M)
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
        grad_beta = grad_output_reshaped.sum(dim=(0, 2))             # (C,)
        return dx.view(N, C, H, W), grad_gamma, grad_beta

    def convLayer_backward(self, grad_output, relu_in, norm_in,conv_in ):
        grad_output = F.interpolate(grad_output, scale_factor=2, mode='nearest') / 4
        relu_grad = (relu_in > 0).float()
        grad_output = grad_output * relu_grad
        grad_output,d_gamma,d_beta = self.instanceNorm_backward(norm_in, self.norm.weight, grad_output)
        # TODO: 目前conv层的结果还是有点问题。不知道是累积误差导致的还是什么，结果会差几位
        db = grad_output.sum(dim=(0, 2, 3))
        dw = torch.nn.grad.conv2d_weight(conv_in, self.conv.weight.shape, grad_output, stride=1, padding=1)
        # dx = torch.nn.grad.conv2d_input(input_size=conv_in.shape, weight=self.conv.weight, grad_output=grad_output, stride=1, padding=1)
        dx = F.conv_transpose2d(grad_output, self.conv.weight, stride=1, padding=1)

        # x_unf = torch.nn.functional.unfold(x, kernel_size=3, padding=1, stride=1)
        # # x_unf shape: (N, C_in*K_h*K_w, L)  where L = H_out * W_out
        # dy_reshaped = dy.reshape(N, C_out, -1)  # shape: (N, C_out, L)
        # # 用 einsum 来实现 batch 矩阵乘积，然后对 batch 求和
        # dw = torch.einsum('ncl,nkl->ck', dy_reshaped, x_unf)  # shape: (C_out, C_in*K_h*K_w)
        # dw = dw.view(C_out, C_in, K_h, K_w)

        return [dx, dw, db, d_gamma, d_beta]


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