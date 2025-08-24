# 研究二阶导数的内存占用问题。
# 在一阶导数的中间放个断点（例如，conv2的梯度），先对这个断点求导，然后再传播回去
# 结果显示，如果改变一下顺序，可以省出大概20M左右的内存（unfold上）
# 但是正确定好像还没有测试....

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

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭自动优化，确保计算确定性
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 保证 CUDA 计算稳定（仅对 PyTorch 1.8+ 有效）

set_random_seed(42)
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--zca',default=False, action='store_true', help="do ZCA whitening")
    parser.add_argument('--device',default='cuda', action='store_true', help="do ZCA whitening")
    parser.add_argument('--ipc',default=1, action='store_true', help="do ZCA whitening")
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    num_params=504420
    args = parser.parse_args()

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset('CIFAR10','/scratch/yguo25/files/mtt-distillation/dataset', args=args)
    student_net = get_network('ConvNet_unfold', channel, num_classes, im_size, dist=False).to('cuda')  # get a random model
    image_syn = torch.load("./script/in.pt").cuda()
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
    # for (name, param), t in zip(student_net.named_parameters(), expert_trajectory[1]):
    #     t= t.cuda()
    #     param.data.copy_(t)


    grad, grad_output1, dw, db, d_gamma, d_beta ,dw2, db2, d_gamma2, d_beta2,dw3, db3, d_gamma3, d_beta3  = student_net(image_syn,target =y_hat, criterion=criterion)
    fin_param = [i.cuda()-g*1e-05 for i,g in zip(starting_params,grad)]
    fin_param =  torch.cat([p.reshape(-1) for p in fin_param], 0)
    # fin_param = starting_params - grad*1e-05

    param_loss = torch.tensor(0.0).to(args.device)
    param_dist = torch.tensor(0.0).to(args.device)
    param_loss += torch.nn.functional.mse_loss(fin_param, target_params, reduction="sum")
    # param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
    param_loss /= num_params
    # param_dist /= num_params
    # param_loss /= param_dist
    grand_loss = param_loss
    # optimizer_img.zero_grad()
    # optimizer_lr.zero_grad()
    # grand_loss.backward()
    # print(image_syn.grad.sum().item())
    # grad = torch.torch.autograd.grad(grand_loss, image_syn)
    # # TODO: 替换backward
    ddw, ddb, dd_gamma, dd_beta,ddw2, ddb2, dd_gamma2, dd_beta2,ddw3, ddb3, dd_gamma3, dd_beta3  = torch.torch.autograd.grad(grand_loss,[dw, db, d_gamma, d_beta,dw2, db2, d_gamma2, d_beta2,dw3, db3, d_gamma3, d_beta3],retain_graph=True )
    dgrad_output1 = torch.torch.autograd.grad([dw, db, d_gamma, d_beta],grad_output1, grad_outputs=[ ddw, ddb, dd_gamma, dd_beta]  )[0]
    grad = torch.torch.autograd.grad([grad_output1,dw2, db2, d_gamma2, d_beta2,dw3, db3, d_gamma3, d_beta3 ],image_syn, grad_outputs=[dgrad_output1,ddw2, ddb2, dd_gamma2, dd_beta2,ddw3, ddb3, dd_gamma3, dd_beta3] )[0]
    print(grad.sum().item())

    print("峰值cache使用:(nvidia-smi)", torch.cuda.max_memory_reserved() / 1024**2, "MB")
    print("峰值tensor使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
