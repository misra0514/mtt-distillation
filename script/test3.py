# 目标是写一个像DD一样的model，但是展开了backward的部分，这样子方便删除一些中间变量。
# 现在验证成功了是可行的。下一步需要看看内存和计算量上有没有什么优化。比较意外的一点是relu的input还是需要保存。但是除此之外就还不错。


import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import torchvision.transforms as transforms


torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

class MyLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x, target):
        x = x.view(x.size(0), -1)
        relu_grad = self.fc1(x)
        logits = self.fc2(F.relu(relu_grad))

        # TODO: 这里如果用自己写的backward，会有一丢丢精度差别，但是速度更快。
        softmax = F.softmax(logits, dim=1)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, target.unsqueeze(1), 1.0)
        grad_output = (softmax - one_hot) / 1
        # grad_output = criterion(logits, target)
        # grad_output = torch.autograd.grad( grad_output,logits, create_graph=True)[0]

        grad_output = grad_output@self.fc2.weight
        relu_grad = (relu_grad > 0).float()
        grad_output = grad_output * relu_grad
        grad_output = grad_output.t()@x

        return grad_output


class TwoLayerLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, target):
        # Flatten the image: b x 3 x 32 x 32 -> b x (3*32*32)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # x = self.fc1(x)
        x = self.fc2(x)
        return x


model = TwoLayerLinearNet()
batch_size = 1
transform = transforms.ToTensor()
cifar10 = torchvision.datasets.CIFAR10(root='/scratch/yguo25/files/mtt-distillation/data', train=True, download=True, transform=transform)
# 取一张图像和标签，比如第 0 张
img, label = cifar10[0]
# 增加 batch 维度，并设置 requires_grad
x = img.to('cuda').unsqueeze(0).clone().detach().requires_grad_(True)  # shape: [1, 3, 32, 32]
# target = torch.randint(0, 10, (batch_size,))
target = torch.tensor([0, 1, 2, 3])[:batch_size].to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([x], lr=1e-1)
for param in model.parameters():
    param.requires_grad = True
# params_flattened = torch.rand([3110])
# torch.save(model.state_dict(), 'model_test3.pth')

model = MyLinearNet()
model.load_state_dict(torch.load('model_test3.pth'))
model = model.to('cuda')


import time

start = time.time()
for step in range(5000):
    optimizer.zero_grad()
    output = model(x,target)  # forward
    loss = abs(output.sum())

    # loss = criterion(output, target )  # compute loss
    # grad1w , grad1b, grad2w , grad2b = torch.autograd.grad(loss,( model.fc1.weight, model.fc1.bias,model.fc2.weight, model.fc2.bias), create_graph=True)
    # loss = abs(grad1w.sum())


    loss.backward()
    optimizer.step()  # update x

    # if step % 10 == 0:
    #     print(f"Step {step}, Loss: {loss.item():.4f}")
end = time.time()
print(end-start)
print("当前显存使用:", torch.cuda.memory_allocated() / 1024**2, "MB")
print("峰值显存使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")