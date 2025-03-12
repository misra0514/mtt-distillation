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
