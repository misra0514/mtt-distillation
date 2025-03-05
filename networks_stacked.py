import torch.nn as nn
import torch.nn.functional as F
import torch

class LinearStacked(nn.Module):
    # TODO: 实际测试的时候，总感觉好像差了一个转制？？
    # PARAMS: weight=stack_num * batchNum * outFeats, bias = stack_num, x = stack_num * batch * InFeats
    def __init__(self ,in_features, out_features, stack_num=2):
        super(LinearStacked, self).__init__()
        self.stack_num = stack_num
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(stack_num, in_features, out_features))
        self.bias = torch.nn.Parameter(torch.randn(stack_num, out_features))

    def forward(self, x):
        #  b应该在每一个batch上对应位置做加法, out = [2,2,1] 后两维公用一个, b广播为2,1,1
        b=self.bias.view(self.stack_num,-1,self.out_features)
        x = torch.bmm(x, self.weight)  
        x = x+b
        return x



class ConvNetStacked(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32),stack_num=2):
        super(ConvNetStacked, self).__init__()

        self.stack_num = stack_num
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.num_feat = num_feat
        self.classifierStacked = LinearStacked(num_feat,num_classes, stack_num)
        # self.classifier = nn.Linear(num_feat, num_classes)

        # self.features2, shape_feat2 = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        # num_feat2 = shape_feat2[0]*shape_feat2[1]*shape_feat2[2]
        # self.classifier2 = nn.Linear(num_feat, num_classes)


    def forward(self, x):
        # Input x: B*B*M 大小。普通linear的shape 为B*input。新的维度在最外面
        # TODO: 额外的输入可以以channel的形式直接cat在新的维度上，这样子group conv可能比较好做，linear还需要在变换一下
        # TODO: 可以在输入的时候就把X在Channel 维度排列好。反正group对Batch和channel 都是独立计算。输入没有batch即可
        batch_size = x.size(0)
        out = self.features(x)
        # out2 = self.features2(x)
        # 10, 256, 4,4   -> 20, 2048
        # size = out.size(1)
        # out1 = out[:,:size ]
        # out2 = out[:,size:]
        # out, out2 = torch.chunk(out, 2, dim=1)
        out = torch.cat(torch.chunk(out, self.stack_num, dim=1), 0).contiguous()

        # out = out.view(10, -1)
        # out2 = out.view(10, -1)
        # out = self.classifier(out)
        # out2 = self.classifier2(out2)

        # TODO: out = 【10,4096】, 两张图片在channel 维度cat
        
        out = out.view(self.stack_num, batch_size, -1)
        out = self.classifierStacked(out)

        out = torch.unbind(out, dim=0)

        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            # FIXED: 目前仅仅更改了instance norm
            return nn.GroupNorm(shape_feat[0]*self.stack_num, shape_feat[0]*self.stack_num, affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        stak_num = self.stack_num
        for d in range(net_depth):
            # FIXED: 把去全部conv2d改成带group即可（注意in，out channel 也要翻倍）
            layers += [nn.Conv2d(in_channels*stak_num, net_width*stak_num , kernel_size=3, padding=3 if channel == 1 and d == 0 else 1, groups=stak_num)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2
        return nn.Sequential(*layers), shape_feat
