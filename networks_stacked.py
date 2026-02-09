# 11.29 
# 做了一次更新。只保留了group conv 这一个分支。
from networks_fused3 import NormActive # fuse+基本优化
import torch.nn as nn
import torch.nn.functional as F
import torch
from networks_stacked_basicblock import LinearStacked, LinearStacked_2 , Conv2d_Stacked, LinearStacked_2_flexFuse

class ConvNetStacked(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32),stack_size=None):
        super(ConvNetStacked, self).__init__()

        self.stack_size = stack_size
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.num_feat = num_feat
        self.classifierStacked2 = LinearStacked_2(num_feat,num_classes, stack_size)

    def forward(self, x):
        out = self.features(x)
        out = out.view(-1, self.num_feat)        # 10, 256, 4,4   -> 20, 2048
        out = self.classifierStacked2(out)
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



''' ConvNet '''
class ConvNet_virticalfuse(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet_virticalfuse, self).__init__()

        if im_size[0] == 28:
            im_size = (32, 32)
        self.shape_feat = [net_width, im_size[0], im_size[1]]

        # --- Layer 1 ---
        padding = 3 if channel == 1 else 1
        self.conv1 = nn.Conv2d(channel, net_width, kernel_size=3, padding=padding)
        # self.norm1 = self._get_normlayer(net_norm, [net_width, im_size[0], im_size[1]]) if net_norm != 'none' else nn.Identity()
        self.norm1 = NormActive(net_width)
        # self.act1 = self._get_activation(net_act)
        self.pool1 = self._get_pooling(net_pooling) if net_pooling != 'none' else nn.Identity()
        if net_pooling != 'none':
            self.shape_feat[1] //= 2
            self.shape_feat[2] //= 2

        # --- Layer 2 ---
        self.conv2 = nn.Conv2d(net_width, net_width, kernel_size=3, padding=1)
        # self.norm2 = self._get_normlayer(net_norm, [net_width , self.shape_feat[1], self.shape_feat[2]]) if net_norm != 'none' else nn.Identity()
        self.norm2 = NormActive(net_width)
        # self.act2 = self._get_activation(net_act)
        self.pool2 = self._get_pooling(net_pooling) if net_pooling != 'none' else nn.Identity()
        if net_pooling != 'none':
            self.shape_feat[1] //= 2
            self.shape_feat[2] //= 2

        # --- Layer 3 ---
        self.conv3 = nn.Conv2d(net_width, net_width, kernel_size=3, padding=1)
        # self.norm3 = self._get_normlayer(net_norm, [net_width, self.shape_feat[1], self.shape_feat[2]]) if net_norm != 'none' else nn.Identity()
        self.norm3 = NormActive(net_width)
        # self.act3 = self._get_activation(net_act)
        self.pool3 = self._get_pooling(net_pooling) if net_pooling != 'none' else nn.Identity()
        if net_pooling != 'none':
            self.shape_feat[1] //= 2
            self.shape_feat[2] //= 2
        num_feat = self.shape_feat[0]*self.shape_feat[1]*self.shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        # print("MODEL DATA ON: ", x.get_device(), "MODEL PARAMS ON: ", self.classifier.weight.data.get_device())
        x = self.conv1(x)
        x = self.norm1(x)
        # x = nn.functional.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        # x = nn.functional.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        # x = nn.functional.relu(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
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
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
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
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
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




class Conv_Flexfuse(nn.Module):
    def __init__(self, channel=3, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='maxpooling', im_size = (32,32), Fuse=2):
        super(Conv_Flexfuse, self).__init__()
        self.Fuse = Fuse
        self.conv1 = nn.Conv2d(in_channels=channel*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        # self.norm1 = nn.InstanceNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.norm1 = nn.GroupNorm(net_width*Fuse,net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=net_width*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        # self.norm2 = nn.InstanceNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.norm2 = nn.GroupNorm(net_width*Fuse,net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=net_width*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        # self.norm3 = nn.InstanceNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.norm3 = nn.GroupNorm(net_width*Fuse,net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool3 = nn.AvgPool2d(kernel_size=2)
        self.linear = LinearStacked_2(net_width * 4 * 4, num_classes,Fuse )
        self.net_width= net_width

    def forward(self, x_conv1):
        x_conv1 = x_conv1.view(-1,self.Fuse*3 ,32,32)        # 10, 256, 4,4   -> 20, 2048
        x_norm1 = self.conv1(x_conv1)          
        x_pool1 = F.relu(self.norm1(x_norm1)    )
        x_conv2 = self.pool1(x_pool1)
        x_norm2 = self.conv2(x_conv2)          
        x_pool2 = F.relu(self.norm2(x_norm2) )   
        x_conv3 = self.pool2(x_pool2)
        x_norm3 = self.conv3(x_conv3)          
        x_pool3 = F.relu(self.norm3(x_norm3))    
        x_lin  = self.pool3(x_pool3)
        x_out = self.linear(x_lin)    # N x 10
        x_out = x_out.view(-1,10)
        return  x_conv1,x_norm1, x_pool1,x_conv2,x_norm2, x_pool2,x_conv3,x_norm3, x_pool3, x_lin,x_out

