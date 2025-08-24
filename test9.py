# 7.23
# 实验二阶ckpt，手动实现。
# 准备工作： 需要把network改成        return out, xin 。（xin是任意一个断电）
        # xin = self.pool2(x)
# 梯度还是有问题：5.908347033465548e-16 vs 6.5 e-16

# 因为dw1对x1也有梯度，所以不能采取原先的分步梯度下降方式。
# 需要一种办法能够



import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import wandb
import copy
import random
from reparam_module import ReparamModule
import torch.profiler

import time
import warnings


''' ConvNet '''
class ConvNet(nn.Module):
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
        self.norm2 = self._get_normlayer(net_norm, [net_width , self.shape_feat[1], self.shape_feat[2]]) if net_norm != 'none' else nn.Identity()
        # self.act2 = self._get_activation(net_act)
        self.pool2 = self._get_pooling(net_pooling) if net_pooling != 'none' else nn.Identity()
        if net_pooling != 'none':
            self.shape_feat[1] //= 2
            self.shape_feat[2] //= 2

        # --- Layer 3 ---
        self.conv3 = nn.Conv2d(net_width, net_width, kernel_size=3, padding=1)
        self.norm3 = self._get_normlayer(net_norm, [net_width, self.shape_feat[1], self.shape_feat[2]]) if net_norm != 'none' else nn.Identity()
        # self.act3 = self._get_activation(net_act)
        self.pool3 = self._get_pooling(net_pooling) if net_pooling != 'none' else nn.Identity()
        if net_pooling != 'none':
            self.shape_feat[1] //= 2
            self.shape_feat[2] //= 2
        num_feat = self.shape_feat[0]*self.shape_feat[1]*self.shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x11):
        # print("MODEL DATA ON: ", x.get_device(), "MODEL PARAMS ON: ", self.classifier.weight.data.get_device())
        x12 = self.conv1(x11)
        x13 = self.norm1(x12)
        x14 = nn.functional.relu(x13)
        xin1 = self.pool1(x14)

        x = self.conv2(xin1)
        x = self.norm2(x)
        x = nn.functional.relu(x)
        xin2 = self.pool2(x)

        x = self.conv3(xin2)
        x = self.norm3(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)
        x3 = x.view(x.size(0), -1)
        out = self.classifier(x3)
        handle = x11.register_hook(lambda grad: print("x grad shape:", grad.sum().item()))
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


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭自动优化，确保计算确定性
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 保证 CUDA 计算稳定（仅对 PyTorch 1.8+ 有效）

def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling


set_random_seed(42)
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--zca',default=False, action='store_true', help="do ZCA whitening")
    parser.add_argument('--device',default='cuda', action='store_true', help="do ZCA whitening")
    parser.add_argument('--ipc',default=10, action='store_true', help="do ZCA whitening")
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    num_params=504420
    args = parser.parse_args()

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset('CIFAR10','/scratch/yguo25/files/mtt-distillation/dataset', args=args)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()
    # student_net = get_network('ConvNet', channel, num_classes, im_size, dist=False).to('cuda')  # get a random model
    student_net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size).to('cuda')
    image_syn = torch.load("./script/in.pt").cuda()
    image_syn = torch.repeat_interleave(image_syn, repeats=10, dim=0)
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    label_syn = torch.tensor([np.ones(args.ipc,dtype=np.int_)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    y_hat = label_syn.to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)

    expert_files = []
    n = 0
    expert_dir = './buffer/CIFAR10/ConvNet'
    while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
        expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
        n += 1
    buffer = torch.load(expert_files[0])
    file_idx = 0
    expert_idx = 0

    expert_trajectory = buffer[0]
    starting_params = expert_trajectory[0]
    target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)
    # student_net.load_state_dict(starting_params)
    for (name, param), t in zip(student_net.named_parameters(), expert_trajectory[1]):
        t= t.cuda()
        param.data.copy_(t)


    # 对照版本：
    out = student_net(image_syn)
    ce_loss = criterion(out, y_hat )
    ce_loss.backward()
    # grad = torch.torch.autograd.grad(ce_loss , list(student_net.parameters()), create_graph=True)

    # fin_param = [i.cuda()-g*1e-05 for i,g in zip(starting_params,grad)]
    # fin_param =  torch.cat([p.reshape(-1) for p in fin_param], 0)
    # param_loss = torch.tensor(0.0).to(args.device)
    # param_dist = torch.tensor(0.0).to(args.device)
    # param_loss += torch.nn.functional.mse_loss(fin_param, target_params, reduction="sum")
    # param_loss /= num_params
    # # param_loss.backward()
    



    # # # TODO: 1 拿到一个起点的值，用autograd分段计算终点
    # # # 2 把反向也分成多步
    # # # 这里有个潜在的问题是，list中的tensor还是会保持引用不会被释放 = = 
    # out , xin1,xin, x11,x12,x13,x14 = student_net(image_syn)
    # ce_loss = criterion(out, y_hat )
    # target_list1 = [
    #                 student_net.conv3.weight,student_net.conv3.bias,student_net.norm3.weight,student_net.norm3.bias,
    #                student_net.classifier.weight , student_net.classifier.bias,xin ]
    # dlist1 = torch.torch.autograd.grad(ce_loss , target_list1, create_graph=True)
    # dxin = dlist1[-1]
    # # dxin = dxin.detach()
    # target_list2 = [student_net.conv1.weight,student_net.conv1.bias,student_net.norm1.weight,student_net.norm1.bias,
    #                 student_net.conv2.weight,student_net.conv2.bias,student_net.norm2.weight,student_net.norm2.bias,]
    # dlist2 = torch.torch.autograd.grad(target_list1 , target_list2, create_graph=True, grad_outputs=dlist1)
    # grad = dlist2+dlist1[:-1]
    # fin_param = [i.cuda()-g*1e-05 for i,g in zip(starting_params,grad)]
    # fin_param =  torch.cat([p.reshape(-1) for p in fin_param], 0)
    # param_loss = torch.tensor(0.0).to(args.device)
    # param_dist = torch.tensor(0.0).to(args.device)
    # param_loss += torch.nn.functional.mse_loss(fin_param, target_params, reduction="sum")
    # param_loss /= num_params


    # # 这里好像写错了，应该首先对所有参数求导，先拿第1、2层的算出x的梯度，再加上第三层的往前传播。
    # ddw, ddb, dd_gamma, dd_beta,ddw2, ddb2, dd_gamma2, dd_beta2,ddw3, ddb3, dd_gamma3, dd_beta3,ddlinw,ddlinb  = torch.torch.autograd.grad(param_loss,grad,retain_graph=True )
    # # dlist2 = [student_net.conv1.weight,student_net.conv1.bias,student_net.norm1.weight,student_net.norm1.bias,
    # #                 student_net.conv2.weight,student_net.conv2.bias,student_net.norm2.weight,student_net.norm2.bias, xin]
    # # 因为下面这两行的结果一样，所以d2xin是包算对了的。
    # # d2xin = torch.torch.autograd.grad(param_loss,dxin ,retain_graph=True )[0]
    # # d2xin =  torch.torch.autograd.grad(dlist2 , dxin, grad_outputs=[ddw, ddb, dd_gamma, dd_beta,ddw2, ddb2, dd_gamma2, dd_beta2])[0]
    # # TODO: 关键是求d2xin的时候，还不能同时求d2xin\dx13， 因为dx13还是会沿着xin的链路做累加。（）

    # d2xin  =  torch.torch.autograd.grad(dlist2 , [dxin], grad_outputs=[ddw, ddb, dd_gamma, dd_beta,ddw2, ddb2, dd_gamma2, dd_beta2])[0]
    # # exit()
    # # dtemp = list(dlist1) + [xin, x11,x12,x13,x14]
    # dx =  torch.torch.autograd.grad(dlist1 , image_syn, grad_outputs=[ddw3, ddb3, dd_gamma3, dd_beta3 ,ddlinw, ddlinb, d2xin])[0]


    # dxin = dxin.detach()
    # xin.requires_grad= False
    # xin.grad_fn = None

    # param_loss.backward()

    # TODO: 为了修复这个问题，在第一轮的时候把对中间x对梯度也一并返回。 现在需要conv1-conv2间所有中间变量在二区产生的梯度。也就是Dgrad2 * X2这一部分。可以在detach掉一部分链路之后重新反传一遍
    # dx =  torch.torch.autograd.grad(dlist1 , image_syn, grad_outputs=[ddw3, ddb3, dd_gamma3, dd_beta3 ,ddlinw, ddlinb, d2xin])[0]

    # dout =  torch.torch.autograd.grad(grad , out, grad_outputs=[ddw, ddb, dd_gamma, dd_beta,ddw2, ddb2, dd_gamma2, dd_beta2,ddw3, ddb3, dd_gamma3, dd_beta3,ddlinw,ddlinb])[0]
    # dx = torch.torch.autograd.grad(out, image_syn, grad_outputs=dout)[0]
    # dx = torch.torch.autograd.grad(out, image_syn, grad_outputs=dout)[0]
    # dout =  torch.torch.autograd.grad(grad, out,grad_outputs=[ddw, ddb, dd_gamma, dd_beta,ddw2, ddb2, dd_gamma2, dd_beta2,ddw3, ddb3, dd_gamma3, dd_beta3,ddlinw,ddlinb ])[0]
    # dx = torch.torch.autograd.grad(out, image_syn, grad_outputs=dout)[0]


    # TODO: 分析：
    # 使用din前向传播，得到4.645536846744041e-16
    # 使用din+loss 前向传播，得到1152964343418546e-15
    # 中间正好差了需要的梯度6.5e-16
    # 这个问题说明，当autograd提供了两个起点，且重复路径的时候，两条路径都按照各自的计算图向前累计梯度。
    # 所以d2xin对x对梯度被重复累加了一次。
    # dx =  torch.torch.autograd.grad(dxin , image_syn, grad_outputs=d2xin)[0]
    # dx =  torch.torch.autograd.grad([dxin, param_loss] , image_syn, grad_outputs=[torch.zeros_like(dxin),None])[0]
    # 只用grad 就求出了dx，说明与w无关。
    # dx =  torch.torch.autograd.grad(grad, image_syn,grad_outputs=[ddw, ddb, dd_gamma, dd_beta,ddw2, ddb2, dd_gamma2, dd_beta2,ddw3, ddb3, dd_gamma3, dd_beta3,ddlinw,ddlinb ])[0]

    # print("---LOSS---")
    # print(param_loss.sum().item())
    # print("---GRAD---")
    # print(dx.sum().item())
    # # print(image_syn.grad.sum().item())

    print("峰值cache使用:(nvidia-smi)", torch.cuda.max_memory_reserved() / 1024**2, "MB")
    print("峰值tensor使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
