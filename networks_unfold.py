import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.checkpoint import checkpoint


# 手写反向传播的版本

''' MLP '''
class MLP(nn.Module):
    def __init__(self, channel, num_classes):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(28*28*1 if channel==1 else 32*32*3, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out



''' ConvNet '''
class ConvNet(nn.Module):
    # def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
    #     super(ConvNet, self).__init__()

    #     self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
    #     num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
    #     self.classifier = nn.Linear(num_feat, num_classes)

    # def forward(self, x):
    #     # print("MODEL DATA ON: ", x.get_device(), "MODEL PARAMS ON: ", self.classifier.weight.data.get_device())
    #     out = self.features(x)
    #     out = out.view(out.size(0), -1)
    #     out = self.classifier(out)
    #     return out
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet, self).__init__()

        if im_size[0] == 28:
            im_size = (32, 32)
        self.shape_feat = [net_width, im_size[0], im_size[1]]

        # --- Layer 1 ---
        padding = 3 if channel == 1 else 1
        self.conv1 = nn.Conv2d(channel, net_width, kernel_size=3, padding=padding)
        self.norm1 = self._get_normlayer(net_norm, [net_width, im_size[0], im_size[1]]) if net_norm != 'none' else nn.Identity()
        self.act1 = self._get_activation(net_act)
        self.pool1 = self._get_pooling(net_pooling) if net_pooling != 'none' else nn.Identity()
        if net_pooling != 'none':
            self.shape_feat[1] //= 2
            self.shape_feat[2] //= 2

        # --- Layer 2 ---
        self.conv2 = nn.Conv2d(net_width, net_width, kernel_size=3, padding=1)
        self.norm2 = self._get_normlayer(net_norm, [net_width, self.shape_feat[1], self.shape_feat[2]]) if net_norm != 'none' else nn.Identity()
        self.act2 = self._get_activation(net_act)
        self.pool2 = self._get_pooling(net_pooling) if net_pooling != 'none' else nn.Identity()
        if net_pooling != 'none':
            self.shape_feat[1] //= 2
            self.shape_feat[2] //= 2

        # --- Layer 3 ---
        self.conv3 = nn.Conv2d(net_width, net_width, kernel_size=3, padding=1)
        self.norm3 = self._get_normlayer(net_norm, [net_width, self.shape_feat[1], self.shape_feat[2]]) if net_norm != 'none' else nn.Identity()
        self.act3 = self._get_activation(net_act)
        self.pool3 = self._get_pooling(net_pooling) if net_pooling != 'none' else nn.Identity()
        if net_pooling != 'none':
            self.shape_feat[1] //= 2
            self.shape_feat[2] //= 2
        num_feat = self.shape_feat[0]*self.shape_feat[1]*self.shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x_conv1, target=None, criterion = None):
        # print("MODEL DATA ON: ", x.get_device(), "MODEL PARAMS ON: ", self.classifier.weight.data.get_device())
        x_norm1 = self.conv1(x_conv1)
        x_relu1 = self.norm1(x_norm1)
        x = self.act1(x_relu1)
        x_conv2 = self.pool1(x)

        x_norm2 = self.conv2(x_conv2)
        x_relu2 = self.norm2(x_norm2)
        x = self.act2(x_relu2)
        x_conv3 = self.pool2(x)

        x_norm3 = self.conv3(x_conv3)
        x_relu3 = self.norm3(x_norm3)
        x3 = self.act3(x_relu3)
        x3 = self.pool3(x3)

        x = x3.view(x3.size(0), -1)
        logits = self.classifier(x)

        # Backprop
        # loss = nn.CrossEntropyLoss()(logits, target)
        grad_output = self.crossEntropy_backward(logits, target)
        dfcb = grad_output.sum(dim=0)  
        dfcw = grad_output.t()@x
        grad_output = grad_output@self.classifier.weight
        grad_output = grad_output.view(x_conv1.shape[0], self.shape_feat[0], self.shape_feat[1], self.shape_feat[2])
        # # TODO: 为了做局部的ckpt，哪怕都用autograd也可以
        # def custom_forward(x_conv3, target):
            # # grad_output = torch.torch.autograd.grad(loss, logits, create_graph=True)[0]
            # x_norm3 = self.conv3(x_conv3)
            # x_relu3 = self.norm3(x_norm3)
            # x3 = self.act3(x_relu3)
            # x3 = self.pool3(x3)
            # x3 = x3.view(x3.size(0), -1)
            # logits = self.classifier(x3)
            # grad_output = self.crossEntropy_backward(logits, target)

            # dfcb = grad_output.sum(dim=0)  
            # dfcw = grad_output.t()@x3
            # grad_output = grad_output@self.classifier.weight
            # grad_output = grad_output.view(x_conv1.shape[0], self.shape_feat[0], self.shape_feat[1], self.shape_feat[2])
            # grad_output, dw3, db3, d_gamma3, d_beta3 = self.convLayer_backward(grad_output,x_relu3, x_norm3, x_conv3, self.norm3, self.conv3 )

        #     return grad_output, dw3, db3, d_gamma3, d_beta3,dfcw, dfcb
        # grad_output, dw3, db3, d_gamma3, d_beta3,dfcw, dfcb = checkpoint(custom_forward, x_conv3, target)

        # grad_output, dw3, db3, d_gamma3, d_beta3 = torch.torch.autograd.grad(x3, [x_conv3,self.conv3.weight, self.conv3.bias, self.norm3.weight, self.norm3.bias], grad_outputs=grad_output, create_graph=True )
        # grad_output, dw2, db2, d_gamma2, d_beta2 = torch.torch.autograd.grad(x_conv3, [x_conv2,self.conv2.weight, self.conv2.bias, self.norm2.weight, self.norm2.bias], grad_outputs=grad_output, create_graph=True )
        # dfcw = self.classifier.weight
        # dfcb = self.classifier.bias
        # dw3=self.conv3.weight
        # db3=self.conv3.bias
        # d_gamma3=self.norm3.weight
        # d_beta3=self.norm3.bias
        # dw2=self.conv3.weight
        # db2=self.conv3.bias
        # d_gamma2=self.norm3.weight
        # d_beta2=self.norm3.bias
        # grad_output=torch.zeros_like(x_conv2).cuda()
        # _, dw, db, d_gamma, d_beta = torch.torch.autograd.grad(x_conv2, [x_conv1,self.conv1.weight, self.conv1.bias, self.norm1.weight, self.norm1.bias], grad_outputs=grad_output, create_graph=True )
        # dw=torch.ones_like(self.conv1.weight)
        # db=torch.ones_like(self.conv1.bias)
        # d_gamma=torch.ones_like(self.norm1.weight)
        # d_beta=torch.ones_like(self.norm1.bias)
        grad_output, dw3, db3, d_gamma3, d_beta3 = self.convLayer_backward(grad_output,x_relu3, x_norm3, x_conv3, self.norm3, self.conv3 )
        grad_output, dw2, db2, d_gamma2, d_beta2 = self.convLayer_backward(grad_output,x_relu2, x_norm2, x_conv2, self.norm2, self.conv2 )
        _, dw, db, d_gamma, d_beta = self.convLayer_backward(grad_output,x_relu1, x_norm1, x_conv1, self.norm1, self.conv1 )

        l= [dw,db,d_gamma,d_beta,dw2,db2,d_gamma2,d_beta2,dw3,db3,d_gamma3,d_beta3,dfcw,dfcb]
        grad = torch.cat([p.reshape(-1) for p in l], 0)

        return grad


    def crossEntropy_backward(self, logits, target):
        N = target.shape[0]
        softmax = F.softmax(logits, dim=1)
        one_hot = torch.zeros_like(logits)
        one_hot[range(N), target] = 1.0
        grad_output = (softmax - one_hot) / N
        return grad_output
        # N, C = logits.shape
        # # 1. Compute log_softmax
        # log_probs = F.log_softmax(logits, dim=1)
        # # 2. Compute grad of NLLLoss (mean reduction)
        # grad = torch.exp(log_probs)  # shape: (N, C)
        # grad[range(N), target] -= 1
        # grad = grad / N
        # return grad
    
    def instanceNorm_backward(self, x, gamma, grad_output, eps=1e-5):
        N, C, H, W = x.shape
        M = H * W
        x_reshaped = x.view(N, C, M)
        grad_output_reshaped = grad_output.view(N, C, M)
        mean = x_reshaped.mean(dim=2, keepdim=True)  # (N, C, 1)
        var = x_reshaped.var(dim=2, unbiased=False, keepdim=True)  # (N, C, 1)
        std = torch.sqrt(var + eps)  # (N, C, 1)
        x_hat = (x_reshaped - mean) / std  # (N, C, M)
        grad_output_hat = grad_output_reshaped * gamma.view(1, C, 1)  # (N, C, M)
        dx = (1. / M) / std * (
            M * grad_output_hat
            - grad_output_hat.sum(dim=2, keepdim=True)
            - x_hat * (grad_output_hat * x_hat).sum(dim=2, keepdim=True)
        )  # (N, C, M)
        grad_gamma = (grad_output_reshaped * x_hat).sum(dim=(0, 2))  # (C,)
        grad_beta = grad_output_reshaped.sum(dim=(0, 2))             # (C,)
        return dx.view(N, C, H, W), grad_gamma, grad_beta

    def convLayer_backward(self, grad_output, relu_in, norm_in,conv_in, norm, conv, stride=1, padding=1 ):
        grad_output = F.interpolate(grad_output, scale_factor=2, mode='nearest') / 4
        relu_grad = (relu_in > 0).float()
        grad_output = grad_output * relu_grad
        grad_output,d_gamma,d_beta = self.instanceNorm_backward(norm_in, norm.weight, grad_output)
        # TODO: 目前conv层的结果还是有点问题。不知道是累积误差导致的还是什么，结果会差几位
        db = grad_output.sum(dim=(0, 2, 3))
        # dw = torch.nn.grad.conv2d_weight(conv_in, conv.weight.shape, grad_output, stride=stride, padding=padding)
        dw = self.conv2d_weight_grad(conv_in, conv.weight.shape, grad_output, stride=stride, padding=padding)
        # dx = torch.nn.grad.conv2d_input(input_size=conv_in.shape, weight=self.conv.weight, grad_output=grad_output, stride=1, padding=1)
        dx = F.conv_transpose2d(grad_output, conv.weight, stride=stride, padding=padding)         # only when stride == padding
        return [dx, dw, db, d_gamma, d_beta]

    def conv2d_weight_grad(self, input, weight_shape, grad_output, stride=1, padding=0, dilation=1, groups=1):
        N = input.shape[0]
        C_out, C_in_per_group, kH, kW = weight_shape
        # unfold input to im2col
        input_unf = F.unfold(input, kernel_size=(kH, kW), dilation=dilation, padding=padding, stride=stride)
        # shape: (N, C_in * kH * kW, L), where L is number of sliding positions
        grad_output_reshaped = grad_output.reshape(N, C_out, -1)  # (N, C_out, L)
        # 使用 einsum 来做 batch 矩阵乘法 + 求和
        grad_weight = torch.einsum('ncl,nkl->ck', grad_output_reshaped, input_unf)  # (C_out, C_in * kH * kW)
        return grad_weight.view(weight_shape)  # reshape 成权重形式



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

