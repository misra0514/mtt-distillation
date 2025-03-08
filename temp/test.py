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
    # PARAMS: weight=stackSize * batchNum * outFeats, bias = stackSize, x = stackSize * batch * InFeats
    def __init__(self, stackSize ,in_features, out_features):
        super(LinearStacked, self).__init__()
        self.stackSize = stackSize
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(stackSize, in_features, out_features))
        self.bias = torch.nn.Parameter(torch.randn(self.stackSize, out_features))

    
    def forward(self, x):
        #  b应该在每一个batch上对应位置做加法, out = [2,2,1] 后两维公用一个, b广播为2,1,1
        b=self.bias.view(self.stackSize,-1,self.out_features)
        # print(x.shape)
        # print(self.weight.shape)
        x = torch.bmm(x, self.weight)  
        x = x+b
        return x


class LinearStacked_2(nn.Module):
    def __init__(self, stackSize ,in_features, out_features):
        super(LinearStacked_2, self).__init__()
        self.stackSize = stackSize
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(stackSize, in_features, out_features))
        self.bias = torch.nn.Parameter(torch.randn(self.stackSize, out_features))

    def forward(self, x):
        """
        x目前仅支持二维输入： B* STK * In。 B和stk可以view 在一起。 weight  STK*IN*OUT 
        """
        x = x.view(-1,self.stackSize,self.in_features)
        x = torch.einsum("abc,bcd->abd",x,self.weight)
        x = x+self.bias
        return x



# # LINERAR TESTS
# # 原始输入：2，3，32，32 新的输入：4，
# input_tensor = torch.randn(4, 3, 32, 32)  # 4张 3×32×32 图片
input_shape = 3* 32* 32
out_shape = 2
# # print(input_shape)
# input_tensor = input_tensor.flatten(start_dim=1) # flatten之后是n*后面的，3* 32* 32合并了
# input_tensor = input_tensor.view(2,2,-1)

# print(input_tensor.shape)

# model1 = SimpleFlattenNet(input_shape=input_shape, output_dim=out_shape)
# model2 = SimpleFlattenNet(input_shape=input_shape, output_dim=out_shape)
input_tensor_2 = torch.randn(4, 3*32, 32)  # 4张 3×32×32 图片
model3 = LinearStacked_2(2,input_shape, out_shape)
output3 = model3(input_tensor_2)

exit()

# param = torch.cat([model1.fc.weight.detach() , model2.fc.weight.detach() ],0  )
# bias = torch.cat([model1.fc.bias.detach()  , model2.fc.bias.detach() ],0 )
# # bias = model1.fc.bias.detach() + model2.fc.bias.detach() 

# model3.weight.data = param.view(2,-1,out_shape)
# model3.bias.data = bias

# # 如果是2+2 两幅图片分开输入
# # 输入应该分成： 2*2* (3* 32* 32) 其中第二维是正常就该有的batch。而第一维是由于多个iter带上来的
# output1 = model1(input_tensor[:1].squeeze())
# output2 = model2(input_tensor[1:].squeeze())
# output3 = model3(input_tensor)



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


class Conv2D_Stacked(nn.Module):
    def __init__(self, in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1, stackSize=1):
        super(Conv2D_Stacked, self).__init__()
        self.stackSize = stackSize
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = torch.nn.Parameter(torch.randn(stackSize, out_channels, in_channels * kernel_size * kernel_size))
        self.bias_param = torch.nn.Parameter(torch.randn(stackSize, 1, out_channels, 1))  # 用于加到最终的 feature map

        # TODO: 似乎可以直接调用上面写好的LinearStacked？？

    def forward(self, x):
        """
        x: 输入形状 (batch_size, in_channels, height, width)
        输入和以前一样，但是在batch维度上放了一个更大的Stach Num维度
        """
        batch_size, _, height, width = x.shape

        # 使用 unfold 进行 im2col 操作，展开窗口
        x_unfolded = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # x_unfolded: (batch_size, in_channels * kernel_size * kernel_size, output_height * output_width)
        # in 1*3*20*20 / k=3 ---> 1,27,400

        # out = self.weight @ x_unfolded  # (batch_size, out_channels, output_height * output_width)
        x_unfolded = x_unfolded.view(self.stackSize, -1, self.out_channels)
        out = torch.bmm(x_unfolded, self.weight)

        # if self.bias_param is not None:
        # out += self.bias_param  # (batch_size, out_channels, output_height * output_width)

        # 计算输出的 feature map 尺寸
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        out = out.view(batch_size*self.stackSize, self.out_channels, out_height, out_width)
        return out






class SimpleCNNEncoder(nn.Module):
    def __init__(self):
        super(SimpleCNNEncoder, self).__init__()
        self.conv1 = Conv2D_Stacked(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)  # 卷积层
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)  # 卷积层
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





# # 测试网络
if __name__ == "__main__":
    model1 = SimpleCNNEncoder()
    model2 = StackedCNNEncoder()

    sample_input = torch.randn(1,6, 20, 20)  # 1张 3通道 20x20 的图片
    output1 = model1(sample_input[:,3:])

    # output2 = model2(sample_input)


    # sample_input = torch.randn(1, 6, 20, 20)  # 1张 3通道 20x20 的图片
    # sample_input

    # # param = model1.state_dict()
    # # param = model1.parameters()
    # model2.conv1.weight.data = torch.cat([ model1.conv1.weight.data,  model1.conv1.weight.data], 0)
    # model2.conv1.bias.data = torch.cat([ model1.conv1.bias.data,  model1.conv1.bias.data], 0)
    
    # # for x in param.items():

    #     # print(x[1])
    #     # print(type(x[1]))

    # output1 = model1(sample_input[:,3:])
    output2 = model2(sample_input)
    # # print("Encoded output shape:", output1)  # 输出形状
    # # print("Encoded output shape:", output2)  # 输出形状

    # # print(output2[:,6:] == output1)

    # # print(output1.shape)
    # # print(output2.shape)

    # loss1 = (torch.rand([1,6,10,10])-output1).sum()
    # loss2 = (torch.rand([1,12,10,10])-output2).sum()

    # loss1.backward(retain_graph=True)
    # loss2.backward(retain_graph=True)


    # import time
    # a = time.time()
    # loss1.backward()
    # b1 = time.time()
    # loss2.backward()
    # b2 = time.time()

    # print(b1-a)
    # print(b2-b1)
