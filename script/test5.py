# 本script主要尝试用autograd optimize ctx。
# 把两个op用torch.autograd.Function 封装之后，似乎可以绕过torch autograd 而控制一些中间变量的缓存行为
# 因为torch的autograd 追踪最多只到torch.autograd.Function级别。在torch.autograd.Function内部的临时变量不会被torch强制记录
# 虽然似乎有些违背torch的设计初衷....
# 另外，用nn.ReLU()好像并不会导致内存消耗上升......?
# 而且，即便可以运行，那么二阶梯度又该如何呢...

# 局限在于，由于torch.autograd.Function 的输入输出还是会被自动统计到计算图中，所以ctx.save的目标如果是input / weight的话，并不能起到缩减内存的效果。【只能用于缩减中间变量】
# 只能把相邻的两层打包放在autograd function里面，把压缩目标当成中间变量。 反向也需要手动实现。

# 对于DD来说，还需要手动实现二阶导。 所以目前只剩下pool+conv / pool+ relu 两个混合方案。

# 但是似乎是唯一方法，如果仅仅封装一个函数的话，需要每一个输入输出都满足对应格式，反而加重计算负担。




import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import torchvision.transforms as transforms

class MyLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight1, weight2, bias1, bias2):
        relu_in = input @ weight1.t() + bias1
        input2 = F.relu(relu_in)
        ctx.save_for_backward( input,input2, weight1, weight2 )
        out2 = input2 @ weight2.t() + bias2
        return  out2
    @staticmethod
    def backward(ctx, grad_output):
        input1, input2, weight1, weight2 = ctx.saved_tensors
        grad_input2 = grad_output @ weight2     
        dw2 =   grad_output.t() @ input2
        db2 = grad_output.sum(0)         
        relu_grad = (grad_input2 > 0).float()
        grad_input1 = grad_input2 * relu_grad
        grad_out = grad_input1 @ weight1      
        dw1 =   grad_input1.t() @ input1
        # dw1 =   weight1+1
        # grad_weight = grad_output.t() @ input 
        db1 = grad_input1.sum(0)         
        return grad_out,dw1, dw2,db1 , db2

class MyLinearLayer(nn.Module):
    def __init__(self, in_features, mid,  out_features):
        super().__init__()
        self.weight1 = nn.Parameter(torch.randn(mid, in_features))
        self.bias1 = nn.Parameter(torch.randn(mid))
        self.weight2 = nn.Parameter(torch.randn(out_features,  mid))
        self.bias2 = nn.Parameter(torch.randn(out_features))
    def forward(self, input):
        # print(self.weight1.sum())
        # print(self.bias1.sum())
        out = MyLinearFunction.apply(input, self.weight1,self.weight2, self.bias1, self.bias2)
        return out
    
class MyTwoLayerLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = MyLinearLayer(3 * 32 * 32, 256,10)
        # self.fc2 = MyLinearLayer(256, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
class TwoLayerLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu= nn.ReLU()
    def forward(self, x):
        x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
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


model1 = TwoLayerLinearNet().to("cuda")
model2 = MyTwoLayerLinearNet().to("cuda")
model = model2

# model.load_state_dict(torch.load('model_test3.pth'), strict=False)

pretrained_dict = torch.load("model_test3.pth")
load_state_dict_by_position(model, pretrained_dict)

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