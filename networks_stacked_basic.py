import torch.nn as nn
import torch.nn.functional as F
import torch

class LinearStacked(nn.Module):
    # PARAMS: weight=stack_size * batchNum * outFeats, bias = stack_size, x = stack_size * batch * InFeats
    def __init__(self ,in_features, out_features, stack_size=2):
        super(LinearStacked, self).__init__()
        self.stack_size = stack_size
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(stack_size, in_features, out_features))
        self.bias = torch.nn.Parameter(torch.randn(stack_size,1, out_features))

    def forward(self, x):
        """
          STK*B * In。 B和stk可以view 在一起。 使用BMM
        """
        x = x.view(self.stack_size, -1, self.in_features)
        # b=self.bias.view(self.stack_size,-1,self.out_features)
        x = torch.bmm(x, self.weight)  
        x = x+self.bias
        return x


class LinearStacked_2(nn.Module):
    def __init__(self ,in_features, out_features, stack_size=2):
        super(LinearStacked_2, self).__init__()
        self.stack_size = stack_size
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(stack_size, in_features, out_features))
        self.bias = torch.nn.Parameter(torch.randn(self.stack_size, out_features))

    def forward(self, x):
        """
        x目前仅支持二维输入： B* STK * In。 B和stk可以view 在一起。 weight  STK*IN*OUT 
        """
        x = x.view(-1,self.stack_size,self.in_features)
        x = torch.einsum("abc,bcd->abd",x,self.weight) # 10,2,2048 * 2,2048,10
        x = x+self.bias
        return x

class Conv2d_Stacked(nn.Module):
    def __init__(self, in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1, stackSize=1):
        super(Conv2d_Stacked, self).__init__()
        self.stackSize = stackSize
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = torch.nn.Parameter(torch.randn(stackSize, out_channels, in_channels * kernel_size * kernel_size))
        self.bias_param = torch.nn.Parameter(torch.randn(stackSize, 1, out_channels, 1))  # 用于加到最终的 feature map

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

        # 计算输出的 feature map 尺寸
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        out = out.view(batch_size*self.stackSize, self.out_channels, out_height, out_width)
        return out




