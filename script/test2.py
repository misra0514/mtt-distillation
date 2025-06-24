# 想测试一下不存中间结果行不行。或者说，不计算dy/dw。这样就不用存x
# 如果正确的话，意味着在DD中，有些checkpoint就不用存储了。


import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import torchvision.transforms as transforms

class MyLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward( weight)
        return input @ weight.t() + bias
    @staticmethod
    def backward(ctx, grad_output):
        weight = ctx.saved_tensors[0]
        grad_input = grad_output @ weight       
        # grad_weight = grad_output.t() @ input 
        # grad_bias = grad_output.sum(0)         
        return grad_input,None, None

class MyLinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    def forward(self, input):
        return MyLinearFunction.apply(input, self.weight, self.bias)

class MyTwoLayerLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = MyLinearLayer(3 * 32 * 32, 256)
        self.fc2 = MyLinearLayer(256, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class TwoLayerLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model1 = TwoLayerLinearNet().to("cuda")
model2 = MyTwoLayerLinearNet().to("cuda")
model = model1

model.load_state_dict(torch.load('model_test3.pth'))
model = torch.compile(model, mode="reduce-overhead")

batch_size = 1024
transform = transforms.ToTensor()
cifar10 = torchvision.datasets.CIFAR10(root='/scratch/yguo25/files/mtt-distillation/data', train=True, download=True, transform=transform)
img, label = cifar10[0]
x = img.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # shape: [batch_size, 3, 32, 32]
x = x.clone().detach().to("cuda").requires_grad_(True)
target = torch.tensor([label] * batch_size, device="cuda")  # shape: [batch_size]

# x = img.unsqueeze(0).to("cuda").clone().detach().requires_grad_(True)  # shape: [1, 3, 32, 32]
# target = torch.randint(0, 10, (batch_size,))
# target = torch.tensor([0, 1, 2, 3])[:batch_size].to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([x], lr=1e-1)
# for param in model.parameters():
#     param.requires_grad = False

for i in range(3):
    optimizer.zero_grad()
    output = model(x)  # forward
    loss = criterion(output, target)  # compute loss
    loss.backward()  # backprop: will compute dL/dx, not dL/dW
    optimizer.step()  # update x


torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

for step in range(100):
    optimizer.zero_grad()
    output = model(x)  # forward
    loss = criterion(output, target)  # compute loss
    loss.backward()  # backprop: will compute dL/dx, not dL/dW
    optimizer.step()  # update x

    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

print("当前显存使用:", torch.cuda.memory_allocated() / 1024**2, "MB")
print("峰值显存使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")