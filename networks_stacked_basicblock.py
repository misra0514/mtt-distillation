import torch.nn as nn
import torch.nn.functional as F
import torch

class LinearStacked(nn.Module):
    # PARAMS: weight=stack_size * batchNum * outFeats, bias = stack_size, x = stack_size * batch * InFeats
    # 用来做Fusion * batch 的形式。
    def __init__(self ,in_features, out_features, stack_size):
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
    # batch* fusion * channel * WH。 
    def __init__(self ,in_features, out_features, Fuse):
        super(LinearStacked_2, self).__init__()
        self.Fuse = Fuse
        self.in_features = in_features
        self.out_features = out_features
        # TODO: 在nn实现中，这里是一个转制，也就是说应该是Fuse, out_features, in_features
        self.weight = torch.nn.Parameter(torch.randn(Fuse* out_features,in_features))
        self.bias = torch.nn.Parameter(torch.randn(self.Fuse* out_features))

    def forward(self, x):
        """
        x目前仅支持二维输入： B* STK * In。 B和stk可以view 在一起。 weight  STK*IN*OUT 
        """
        # self.weight = self.weight.view(self.Fuse,self.out_features,self.in_features)
        # self.bias = self.bias.view(self.Fuse, self.out_features)

        x = x.view(-1,self.Fuse, self.in_features)
        # print(x.is_contiguous())

        # # # # TODO: 这里输入如果是BAD/（而不是ABD）的话，可以得到is_contiguous 的结果。那就很简单只要调整target即可 
        # x = torch.einsum("abc,bcd->abd",x,self.weight.view(self.Fuse,self.out_features,self.in_features).transpose(-1, -2)) 
        # x = x+self.bias.view(self.Fuse,self.out_features)
        # x = x.contiguous()

        # 其实也可以用吧bmm。view一下即可。

        # TODO: 这里输入如果是BAD/（而不是ABD）的话，可以得到is_contiguous 的结果。那就很简单只要调整target即可
        # 现在是BAD，意味着Fusion，batch的排序，B到了第一位
        x = torch.einsum("abc,bcd->bad",x,self.weight.view(self.Fuse,self.out_features,self.in_features).transpose(-1, -2)) 
        x = x+self.bias.view(self.Fuse,1 ,self.out_features)

        return x
    
class LinearStacked_2_flexFuse(nn.Module):
    # 从horuzontal fuse 复制来的。用bmm而不是einsum。虽然没啥区别
    def __init__(self ,in_features, out_features, Fuse):
        super(LinearStacked_2_flexFuse, self).__init__()
        self.Fuse = Fuse
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(Fuse* out_features,in_features))
        self.bias = torch.nn.Parameter(torch.randn(self.Fuse* out_features))

    def forward(self, x):
        """
        x目前仅支持二维输入： B* STK * In。 B和stk可以view 在一起。 weight  STK*IN*OUT 
        """
        x = x.view(-1,self.Fuse, self.in_features).transpose(0,1)
        x = torch.bmm(x, self.weight.view(self.Fuse, self.out_features, self.in_features).transpose(-1, -2) )
        x = x+self.bias.view(self.Fuse,1 ,self.out_features)

        return x



class Conv2d_Stacked(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, stackSize=1):
        super(Conv2d_Stacked, self).__init__()
        self.stackSize = stackSize
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.temp = in_channels * kernel_size * kernel_size
        self.weight = torch.nn.Parameter(torch.randn(stackSize,out_channels, self.temp ))
        self.bias_param = torch.nn.Parameter(torch.randn(stackSize, out_channels))  # 用于加到最终的 feature map

    def forward(self, x):
        """
        x: 输入形状 (batch_size, in_channels, height, width)
        输入和以前一样，但是在batch维度上放了一个更大的Stach Num维度
        """
        _, _, height, width = x.shape
        x = x.view(-1, self.in_channels,  height, width )

        # 使用 unfold 进行 im2col 操作，展开窗口
        # x = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # x_unfolded: (batch_size, in_channels * kernel_size * kernel_size, output_height * output_width)
        # in 1*3*20*20 / k=3 ---> 1,27,400

        # out = self.weight @ x_unfolded  # (batch_size, out_channels, output_height * output_width)
        # x = x.view(self.stackSize, self.temp, -1)
        out = torch.bmm(self.weight,F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).view(self.stackSize, self.temp, -1))

        # 计算输出的 feature map 尺寸
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # TODO: 因为norm的问题，现在输出之前需要把stackSize 移动到C这里，仅作为临时参考。
        out = out.view(-1, self.out_channels*self.stackSize, out_height, out_width)
        return out




