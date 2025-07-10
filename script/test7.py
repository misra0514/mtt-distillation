# 同test6，但是主要测试ckpt在二阶导数的场景下怎么实现。
# 用一个大的torch.autograd.Function 把relu、pool、linear包起来。
# 在自己定义的反向里面 再嵌套二阶导数的 autograd.Function 通过这个方法把模型中间敲空

# 如果希望ckpt的效果，在二阶导数里需要做forward、create graph、backward
# 如果希望做delta encode， 那就手动写两阶导数。



import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import torchvision.transforms as transforms

class Snd_Order_MyLinearFunction(torch.autograd.Function):
    '''2 forward: relu+pool+linear.backward '''
    # TODO: 如果做ckpt，那么记得保证forward可以在算完之后全释放掉。 然后backward再重新算一遍。
    # 现在也没有做save ctx，为啥内存消耗还是1483？
    @staticmethod
    def forward(ctx, grad_output, input, weight, relu_in):
        ctx.save_for_backward( weight,relu_in )
        dw =   grad_output.t() @ input
        db =   grad_output.sum(0)         
        grad_output = grad_output @ weight 
        grad_output = grad_output.view(grad_output.size(0), -1, 8,8)
        grad_output = F.interpolate(grad_output, scale_factor=2, mode='nearest') / 4
        grad_output = grad_output * relu_in
        return grad_output, dw, db
    @staticmethod
    def backward(ctx, grad_grad_input, grad_grad_w, grad_grad_b):
        weight, relu_in = ctx.saved_tensors
        input = F.relu(relu_in)
        relu_in = (relu_in > 0).float()
        input = F.avg_pool2d(input,2 )
        input = input.view(input.size(0), -1) # Flatten to N x (net_width*16*16)
        ctx.save_for_backward( input, weight,relu_in )
        out = F.linear(input, weight)
        din, dw = torch.torch.autograd.grad()

        return None, None, None, None

class MyLinearFunction(torch.autograd.Function):
    '''forward: relu+pool+linear '''
    @staticmethod
    def forward(ctx, relu_in, weight, bias):
        input = F.relu(relu_in)
        relu_in = (relu_in > 0).float()
        input = F.avg_pool2d(input,2 )
        input = input.view(input.size(0), -1) # Flatten to N x (net_width*16*16)
        ctx.save_for_backward( input, weight,relu_in )
        out = F.linear(input, weight, bias)
        return  out
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, relu_in = ctx.saved_tensors
        return Snd_Order_MyLinearFunction.apply(grad_output,input, weight, relu_in)


class MyLinearLayer(nn.Module):
    def __init__(self, in_features, mid):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(mid, in_features))
        self.bias = nn.Parameter(torch.randn(mid))
    def forward(self, input):
        # print(self.weight1.sum())
        # print(self.bias1.sum())
        out = MyLinearFunction.apply(input, self.weight, self.bias)
        return out

def pack_hook(x):
    print("Packing", x.shape)
    return x.to('cpu')
    # shape = x.shape
    # # x.data=torch.rand([1]).to('cuda')
    # return shape

def unpack_hook(x):
    print("Unpacking", x.sum().item())
    return x.to('cuda')
    # x = torch.ones(x).to('cuda')
    # return x

class Myconv(nn.Module):
    def __init__(self, net_width):
        super(Myconv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=net_width, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(net_width)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=net_width, out_channels=net_width, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(net_width)
        # self.pool2 = nn.AvgPool2d(kernel_size=2)
        # self.classifier = nn.Linear(net_width * 8 * 8, 10)
        self.poolfier = MyLinearLayer(net_width * 8 * 8, 10)

    def forward(self, input):
        # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        x = self.conv1(input)          # N x net_width x 32 x 32
        # del input
        x = self.norm1(x)          # N x net_width x 32 x 32
        x = F.relu(x)             # N x net_width x 32 x 32
        x = self.pool1(x)          # N x net_width x 16 x 16
        x = self.conv2(x)          # N x net_width x 32 x 32
        x = self.norm2(x)          # N x net_width x 32 x 32
        # x = F.relu(x)             # N x net_width x 32 x 32
        # x = self.pool2(x)          # N x net_width x 16 x 16
        # x = x.view(x.size(0), -1) # Flatten to N x (net_width*16*16)
        # out = self.classifier(x)    # N x 10
        out = self.poolfier(x)
        return out
    
class ConvNet(nn.Module):
    def __init__(self, net_width):
        super(ConvNet, self).__init__()
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

model1 = ConvNet(32).to("cuda")
model2 = Myconv(32).to("cuda")
model = model2

# torch.save(model.state_dict(), 'model_test7.pt')
# model.load_state_dict(torch.load('model_test5.pt'), strict = False)

pretrained_dict = torch.load("model_test7.pt")
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
optimizer = torch.optim.SGD([x], lr=1e-1)
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
    # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    output = model(x)  # forward
    loss = criterion(output, target)  # compute loss
    # print(loss.item())
    # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    dw = torch.torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
    weight = list(model.parameters()) 
    weight = [(1- p + g).sum() for p, g in zip(weight, dw)]
    grad_loss = sum(weight)
    # plan a: 0.-3.188770294189453
    grad_loss.backward()  

    # # plan b: 
    # # dw对input 也有梯度，但是没办法把反向分成两半去算。
    # d1w = torch.torch.autograd.grad(grad_loss, weight)
    # ddw = torch.torch.autograd.grad(grad_loss, dw)
    # douts = torch.torch.autograd.grad(dw, output, grad_outputs=ddw)[0]
    # # print(douts)
    # # weight = weight+(output,)
    # weight = list(weight)
    # weight.append(output)
    # # print(len(weight))
    # # d1w = d1w + (douts,)
    # # d1w.append(douts)
    # d1w = list(d1w)
    # d1w.append(douts)
    # dx = torch.torch.autograd.grad(weight, x, grad_outputs=d1w)[0]
    # x.grad = dx

    # # plan c: 只是调整顺序。
    # ins = list(dw) + weight
    # # d1w = torch.torch.autograd.grad(grad_loss, weight)
    # outs = torch.torch.autograd.grad(grad_loss, ins)
    # # grads =  list(ddw) + list(d1w)[::-1]
    # # source = list(dw) + list(weight)[::-1] 
    # # grads = 
    # dx = torch.torch.autograd.grad(ins[::-1], x, grad_outputs=outs[::-1])[0]
    # x.grad = dx

    print(x.grad.sum().item())
    optimizer.step()  # update x

    # if step % 10 == 0:
    #     print(f"Step {step}, Loss: {loss.item():.4f}")

print("当前显存使用:", torch.cuda.memory_allocated() / 1024**2, "MB")
print("峰值显存使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")