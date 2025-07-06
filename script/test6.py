# 同test5. 但是转移到conv。为了测试真正的场景。
# 现在的问题是，如果把pooling + linear 合并，实际上的空间消耗还增加了。（ autograd 还偏偏不会记录


# UPDATE: 用with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook 接口之后感觉还不错。但是不知道为什么好像只对norm有效果。（norm 也够了！）
# 然后测试了一下 二阶导数。 发现在换成计算二阶导数的时候pack失效了。Unpacking 还是正常调用，但是内存消耗没有变少。只能理解成norm. backward 在调用的时候引用了和 forward一样的变量，导致本来也没有额外开销。

# 实验结果是，对于Linear、conv这种函数来说，完全没有效果。 目前原因不明。

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import torchvision.transforms as transforms

class MyLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        input = F.avg_pool2d(input,2)
        input = input.view(input.size(0), -1)
        ctx.save_for_backward(input, weight)
        return input @ weight.t() + bias
    @staticmethod
    def backward(ctx, grad_output):
        input1, weight = ctx.saved_tensors
        grad_weight = grad_output.t() @ input1 
        grad_bias = grad_output.sum(0)         
        grad_output = grad_output @ weight   
        grad_output = grad_output.view(grad_output.size(0), -1, 16,16)
        grad_output = F.interpolate(grad_output, scale_factor=2, mode='nearest') / 4
        return grad_output ,grad_weight, grad_bias

class MyLinearLayer(nn.Module):
    def __init__(self, in_features,  out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, input):
        # print(self.weight1.sum())
        # print(self.bias1.sum())
        out = MyLinearFunction.apply(input, self.weight,self.bias)
        return out

class MyLinearSimple(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward( input, weight)
        return input @ weight.t() + bias
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_output @ weight       
        grad_weight = grad_output.t() @ input 
        grad_bias = grad_output.sum(0)         
        return grad_input,grad_weight, grad_bias
class MyLinearLayerSimple(nn.Module):
    def __init__(self, in_features,  out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    def forward(self, input):
        out = MyLinearSimple.apply(input, self.weight,self.bias)
        return out


def pack_hook(x):
    print("Packing", x.shape)
    return x.to('cpu')

def unpack_hook(x):
    print("Unpacking")
    return x.to('cuda')

class Myconv(nn.Module):
    def __init__(self, net_width):
        super(Myconv, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=net_width, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(net_width)
        # self.poolfier = MyLinearLayer(net_width * 16 * 16, 10)
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.classifier = MyLinearLayerSimple(net_width * 16 * 16, 10)
    def forward(self, x):
        x = self.conv(x)          # N x net_width x 32 x 32
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            x = self.norm(x)          # N x net_width x 32 x 32
        x = F.relu(x)             # N x net_width x 32 x 32
        x = self.pool(x)          # N x net_width x 16 x 16
        x = x.view(x.size(0), -1) # Flatten to N x (net_width*16*16)
        x = self.classifier(x)    # N x 10
            # x = torch.matmul(x,self.classifier.weight.t()) + self.classifier.bias
        # with torch.autograd.graph.save_on_cpu(pin_memory=True):
            # x = self.classifier(x)    # N x 10
            # x = x@self.classifier.weight.t() + self.classifier.bias


        # x = self.poolfier(x)
        return x
    
class SimpleConvNet(nn.Module):
    def __init__(self, net_width):
        super(SimpleConvNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=net_width, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(net_width)
        self.pool = nn.AvgPool2d(kernel_size=2)
        # 输入是 N x net_width x 16 x 16 after pooling
        self.classifier = nn.Linear(net_width * 16 * 16, 10)

    def forward(self, x):
        x = self.conv(x)          # N x net_width x 32 x 32
        x = self.norm(x)          # N x net_width x 32 x 32
        x = F.relu(x)             # N x net_width x 32 x 32
        x = self.pool(x)          # N x net_width x 16 x 16
        x = x.view(x.size(0), -1) # Flatten to N x (net_width*16*16)
        x = self.classifier(x)    # N x 10
        return x

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

model1 = SimpleConvNet(32).to("cuda")
model2 = Myconv(32).to("cuda")
model = model2

# torch.save(model.state_dict(), 'model_test5.pt')
# model.load_state_dict(torch.load('model_test5.pt'), strict = False)

# pretrained_dict = torch.load("model_test5.pt")
# load_state_dict_by_position(model, pretrained_dict)

# print(model.fc2.weight.sum())
# print(model.fc2.bias.sum())
# exit()

# model = torch.compile(model, mode="reduce-overhead")

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
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
# for param in model.parameters():
#     param.requires_grad = False

# for i in range(3):
#     optimizer.zero_grad()
#     output = model(x)  # forward
#     loss = criterion(output, target)  # compute loss
#     loss.backward()  # backprop: will compute dL/dx, not dL/dW
#     optimizer.step()  # update x
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

for step in range(1):
    optimizer.zero_grad()
    output = model(x)  # forward
    loss = criterion(output, target)  # compute loss
    # dw = torch.torch.autograd.grad(loss, list(model.parameters()), retain_graph=True)
    # weight = list(model.parameters()) 
    # weight = [(1- p - 0.1*g).sum() for p, g in zip(weight, dw)]
    # grad_loss = sum(weight)
    loss.backward()  
    optimizer.step()  # update x

    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

print("当前显存使用:", torch.cuda.memory_allocated() / 1024**2, "MB")
print("峰值显存使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")