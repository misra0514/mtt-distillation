# 5.2 整理一下目前的一些helpers

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random 
import warnings


def build_global_group_mask(starting_params, fuse_mask_list):
    """
    starting_params: 已经 repeat 完成后的参数列表
                     每个 p 的 shape[0] = 原来的 * Fuse
    fuse_mask_list: 例如 [1,0]，长度 = Fuse
    返回：
        一个全局 1D bool mask，可直接作用于 flatten 后的 student_params
    """

    Fuse = len(fuse_mask_list)
    flat_masks = []

    for p in starting_params:
        if p.ndim == 0:
            # 标量可直接全 True
            flat_masks.append(torch.ones_like(p, dtype=torch.bool).reshape(-1))
            continue

        B = p.shape[0]        # 这是 repeat 后的总长度
        block = B // Fuse     # 每一组大小，例如 60020 // 2 = 30010

        # 构造这个参数自己的 mask
        mask = torch.zeros(B, dtype=torch.bool, device=p.device)
        for i, m in enumerate(fuse_mask_list):
            if m == 1:
                s = i * block
                e = (i + 1) * block
                mask[s:e] = True

        # 展成和 flatten 一致的一维
        mask = mask.reshape(-1).repeat_interleave(p[0].numel())

        flat_masks.append(mask)

    # 合并所有参数的 mask
    return torch.cat(flat_masks, 0)

def fuse_params_with_mask(starting_params, Fuse, mask_list):
    assert len(mask_list) == Fuse, "mask_list 长度必须等于 Fuse"
    fused_params = []
    fused_mask   = []
    for layer_idx, p in enumerate(starting_params):
        if p.ndim == 0:
            # 不复制标量（可以按你的需求修改，这里直接跳过）
            fused_params.append(p)
            fused_mask.append(torch.ones_like(p) * mask_list[0])
            continue
        repeat_shape = (int(Fuse),) + (1,) * (p.ndim - 1)
        fused_p = p.repeat(repeat_shape)
        fused_params.append(fused_p)
        masks = []
        for m in mask_list:
            masks.append(torch.ones_like(p) * m)
        fused_m = torch.cat(masks, dim=0)
        fused_mask.append(fused_m)
    student_params = [
        torch.cat([fp.reshape(-1) for fp in fused_params], dim=0).requires_grad_(True).cuda()
    ]
    mask = torch.cat([fm.reshape(-1) for fm in fused_mask], dim=0).cuda()
    return  student_params, mask

def split_half_second_dim(param_list, fuse_mask_list):
    output = []
    Fuse = len(fuse_mask_list)
    # 找连续 1 的起点和终点
    start = fuse_mask_list.index(1)
    end = start + fuse_mask_list.count(1)   # 不包括 end（Python 切片习惯）

    for p in param_list:
        if p.ndim > 2:
            c = p.shape[1]
            block = c // Fuse               # 每一份的长度
            # 计算切片范围
            s = start * block
            e = end * block
            # 切分并保证连续（减少显存峰值）
            output.append(p[:, s:e, ...].contiguous())
            del p
        else:
            # 似乎只有x_out 系列是2维,在第一维切
            c = p.shape[0]
            block = c // Fuse               # 每一份的长度
            # 计算切片范围
            s = start * block
            e = end * block
            # 切分并保证连续（减少显存峰值）
            output.append(p[ s:e, ...].contiguous())
            # output.append(p)

    return output

def recover_params(flat_tensor, base_shapes, fusion):
    """
    flat_tensor: 一维的总参数向量 (例如 starting_params[0])
    shapes: 每一层参数的形状，按顺序排列
    fusion: 重复几次。 fusion永远是在第一维度
    return: 一个 list，元素为每层恢复出的 weight 参数
    """
    recovered = []
    pointer = 0
    for shape in base_shapes:
        # 当前参数 flatten 后的长度
        new_shape = (shape[0] * fusion,) + tuple(shape[1:])
        numel = torch.prod(torch.tensor(new_shape)).item()
        chunk = flat_tensor[pointer: pointer + numel]
        recovered.append(chunk.reshape(new_shape))
        pointer += numel
    return recovered

warnings.filterwarnings("ignore", category=DeprecationWarning)
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭自动优化，确保计算确定性
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 保证 CUDA 计算稳定（仅对 PyTorch 1.8+ 有效）


