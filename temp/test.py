import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFlattenNet(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(SimpleFlattenNet, self).__init__()
        # 定义网络
        self.fc = nn.Linear(input_shape, output_dim)  # （N*F）In * （F*O）Param
        # self.output = nn.Linear(hidden_dim, output_dim)    # 添加最终输出层
    
    def forward(self, x):
        x = self.fc(x)  # 通过 Linear 层
        return x

class LinearStacked(nn.Module):
    # PARAMS: weight=stackNum * batchNum * outFeats, bias = stackNum, x = stackNum * batch * InFeats
    def __init__(self, stackNum ,in_features, out_features):
        super(LinearStacked, self).__init__()
        self.stackNum = stackNum
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(stackNum, in_features, out_features))
        self.bias = torch.nn.Parameter(torch.randn(self.stackNum, out_features))

    
    def forward(self, x):
        #  b应该在每一个batch上对应位置做加法, out = [2,2,1] 后两维公用一个, b广播为2,1,1
        b=self.bias.view(self.stackNum,-1,self.out_features)
        # print(x.shape)
        # print(self.weight.shape)
        x = torch.bmm(x, self.weight)  
        x = x+b
        return x


# LINERAR TESTS
# 原始输入：2，3，32，32 新的输入：4，
input_tensor = torch.randn(4, 3, 32, 32)  # 4张 3×32×32 图片
input_shape = 3* 32* 32
out_shape = 2
# print(input_shape)
input_tensor = input_tensor.flatten(start_dim=1) # flatten之后是n*后面的，3* 32* 32合并了
input_tensor = input_tensor.view(2,2,-1)

print(input_tensor.shape)

model1 = SimpleFlattenNet(input_shape=input_shape, output_dim=out_shape)
model2 = SimpleFlattenNet(input_shape=input_shape, output_dim=out_shape)
model3 = LinearStacked(2,input_shape, out_shape)

param = torch.cat([model1.fc.weight.detach() , model2.fc.weight.detach() ],0  )
bias = torch.cat([model1.fc.bias.detach()  , model2.fc.bias.detach() ],0 )
# bias = model1.fc.bias.detach() + model2.fc.bias.detach() 

model3.weight.data = param.view(2,-1,out_shape)
model3.bias.data = bias

# 如果是2+2 两幅图片分开输入
# 输入应该分成： 2*2* (3* 32* 32) 其中第二维是正常就该有的batch。而第一维是由于多个iter带上来的
output1 = model1(input_tensor[:1].squeeze())
output2 = model2(input_tensor[1:].squeeze())
output3 = model3(input_tensor)



# 目前大概可以对得上，但是好像有点误差？这是为什么
# print("Output 1:", output1)  # ([2, 1])
# print("Output 2:", output2) 
# print("Output 3:", output3)  # ([2, 2, 1])

# output1 = output1.view(1,2,1)
# output2 = output2.view(1,2,1)
# out12 = torch.cat([output1,output2], 0)

# y = torch.rand([2,2,1])
# loss12 = (y-out12).sum()
# loss3 = (y-output3).sum()
# loss12.backward()
# loss3.backward()
# print("-------")
# print(model1.fc.weight.grad.data)
# print(model3.weight.grad.data)

class SimpleCNNEncoder(nn.Module):
    def __init__(self):
        super(SimpleCNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)  # 卷积层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层

    def forward(self, x):
        x= self.conv1(x)
        x = self.pool(F.relu(x))  # 卷积 -> ReLU -> 池化
        # x = torch.flatten(x, start_dim=1)  # 展平成特征向量
        return x

class StackedCNNEncoder(nn.Module):
    def __init__(self):
        super(StackedCNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1, groups=2)  # 卷积层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层

    def forward(self, x):
        x= self.conv1(x)
        x = self.pool(F.relu(x))  # 卷积 -> ReLU -> 池化
        # x = torch.flatten(x, start_dim=1)  # 展平成特征向量
        return x



# # # 测试网络
# if __name__ == "__main__":
#     model1 = SimpleCNNEncoder()
#     model2 = StackedCNNEncoder()
#     sample_input = torch.randn(1, 6, 20, 20)  # 1张 3通道 20x20 的图片
#     # sample_input

#     # param = model1.state_dict()
#     # param = model1.parameters()
#     model2.conv1.weight.data = torch.cat([ model1.conv1.weight.data,  model1.conv1.weight.data], 0)
#     model2.conv1.bias.data = torch.cat([ model1.conv1.bias.data,  model1.conv1.bias.data], 0)
    
#     # for x in param.items():

#         # print(x[1])
#         # print(type(x[1]))

#     output1 = model1(sample_input[:,3:])
#     output2 = model2(sample_input)
#     print("Encoded output shape:", output1)  # 输出形状
#     print("Encoded output shape:", output2)  # 输出形状

#     print(output2[:,6:] == output1)
    
