import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import torchvision.transforms as transforms


import torch
from functools import reduce
from operator import mul


import triton
import triton.language as tl
import time 
import random
import numpy as np
import os

import sys
import os

def clear_tensorlists(*dicts):
    for d in dicts:
        d.clear()

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭自动优化，确保计算确定性
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 保证 CUDA 计算稳定（仅对 PyTorch 1.8+ 有效）

def load_state_dict_by_position(model, pretrained_state_dict):
    model_state_dict = model.state_dict()
    new_state_dict = {}
    # 取出当前模型的参数名字和值（有顺序）
    model_items = list(model_state_dict.items())
    pretrained_items = list(pretrained_state_dict.items())
    assert len(model_items) == len(pretrained_items), \
        f"参数数量不一致：当前模型有 {len(model_items)} 个参数，预训练模型有 {len(pretrained_items)} 个参数"
    for (model_key, _), (_, pretrained_val) in zip(model_items, pretrained_items):
        new_state_dict[model_key] = pretrained_val
    model.load_state_dict(new_state_dict)


def repeat_params_for_fuse(pretrained_dict,Fuse ):
    for i, j in pretrained_dict.items():
        if j.ndim == 0:
            continue
        # cls_token:
        # [1, 1, D] -> [1, Fuse, 1, D]
        if "cls_token" in i:
            pretrained_dict[i] = j.unsqueeze(1).repeat(1, Fuse, 1, 1)
        # pos_embed:
        # [1, N, D] -> [1, Fuse, N, D]
        elif "pos_embed" in i:
            pretrained_dict[i] = j.unsqueeze(1).repeat(1, Fuse, 1, 1)
        # 其他参数默认还是沿第 0 维 repeat
        else:
            pretrained_dict[i] = j.repeat((Fuse,) + (1,) * (j.ndim - 1))
    return pretrained_dict