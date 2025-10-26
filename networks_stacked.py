import torch.nn as nn
import torch.nn.functional as F
import torch
from networks_stacked_basicblock import LinearStacked, LinearStacked_2 , Conv2d_Stacked

class ConvNetStacked(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32),stack_size=None):
        super(ConvNetStacked, self).__init__()

        self.stack_size = stack_size
        self.l = "BS"
        if(self.l=="BS"): # BS : Batch* fusionsize *rest,  group conv + einsum
            self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
            num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
            self.num_feat = num_feat
            self.classifierStacked2 = LinearStacked_2(num_feat,num_classes, stack_size)
        else: # based on torch.bmm (conv & linear)
            self.features2, shape_feat2 = self._make_layers_2(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
            num_feat = shape_feat2[0]*shape_feat2[1]*shape_feat2[2]
            self.num_feat = num_feat
            self.classifierStacked = LinearStacked(num_feat,num_classes, stack_size)


        # self.stack_size = stack_size
        # self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        # num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        # self.num_feat = num_feat
        # self.classifier = nn.Linear(num_feat,num_classes, stack_size)




    def forward(self, x):
        # # 额外的输入可以以channel的形式直接cat在新的维度上，这样子group conv可能比较好做，linear还需要在变换一下
        # out = self.features(x)
        # out = out.view(-1, self.num_feat)        # 10, 256, 4,4   -> 20, 2048

        # # 两张图片在channel 维度cat 转为0维（转置）
        # # 20, 2048
        # out = out.view(self.stack_size, -1, self.num_feat )
        # out = out.permute(1,0,2).contiguous()
        # # out = torch.cat(torch.chunk(out, self.stack_size, dim=1), 0).contiguous()
        
        # out = self.classifierStacked(out)
        # out = torch.unbind(out, dim=0)# 如果是走的linear2需要在dim1上unbind
        # return out


        # TODO: 现在是 groupconv+ bmm。中间做了一个contiguous。 下面用branch 重新写两种Dayout
        if(self.l=="BS"): # B,S, else
            out = self.features(x)
            print("out1",out.sum().item()) # stk=1这里还一致，后面好像也有点出入
            out = out.view(-1, self.num_feat)        # 10, 256, 4,4   -> 20, 2048
            # 20, 2048
            # out = out.view(-1, self.stack_size, self.num_feat )
            # print(out.shape)
            out = self.classifierStacked2(out)
            # print("out",out.sum().item()) 
            # out = torch.unbind(out, dim=1)# 如果是走的linear2需要在dim1上unbind
            # out 10 * 4 *10
            # Update: 不要unbind了，直接和target 做loss，注意stk在一维就可以
            return out
        else: # Stk, Batch ,esle 
            out = self.features2(x)
            out = out.view(-1, self.num_feat)        # 10, 256, 4,4   -> 20, 2048
            # 20, 2048
            out = self.classifierStacked(out)
            out = torch.unbind(out, dim=0)# 如果是走的linear2需要在dim1上unbind
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
            return nn.GroupNorm(shape_feat[0]*self.stack_size, shape_feat[0]*self.stack_size, affine=True)
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
        stak_num = self.stack_size
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

    # Used to gen by conv2
    def _make_layers_2(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        stak_num = self.stack_size
        for d in range(net_depth):
            # 用了自己写的conv
            layers += [Conv2d_Stacked(in_channels, net_width , kernel_size=3, padding=3 if channel == 1 and d == 0 else 1, stackSize=stak_num)]
            shape_feat[0] = net_width
            if net_norm != 'none':
            # TODO: 在F,B,C格式下
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2
        return nn.Sequential(*layers), shape_feat

