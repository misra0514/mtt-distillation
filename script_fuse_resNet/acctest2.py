import copy
import torch

import torch
import torch.nn as nn
# 按你的实际 import 路径改
# from your_fuse_file import ResNet18_FlexFuse
# from your_orig_file import ResNet18
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.networks_Fuse  import Conv_Flexfuse, ResNet18_FlexFuse
from networks.networks import ResNet18
from reparam_module import ReparamModule

def _extract_output(y):
    if isinstance(y, tuple):
        return y[0]
    return y

def compare_plain_vs_reparam(
    channel=3,
    num_classes=10,
    fusion=1,
    batch_size=4,
    image_size=32,
    device="cuda",
):
    assert fusion == 1, "这个脚本先只比较 Fuse=1"

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # 1) 先建 plain 模型
    plain = ResNet18_FlexFuse(
        channel=channel,
        num_classes=num_classes,
        Fuse=fusion,
    ).to(device).eval()

    # 2) 在 wrap 之前，先拿到 plain 的参数
    plain_named_params = list(plain.named_parameters())
    assert len(plain_named_params) > 0, "plain.named_parameters() 为空，模型本身有问题"

    flat = torch.cat([p.detach().reshape(-1) for _, p in plain_named_params], dim=0)

    print("=" * 80)
    print("Plain model param count:", len(plain_named_params))
    print("Flat numel:", flat.numel())
    print("=" * 80)

    # 3) 用 plain 的一个深拷贝去构造 ReparamModule
    #    不要直接 ReparamModule(plain)，因为那会改 plain 本体
    reparam_base = copy.deepcopy(plain)
    reparam = ReparamModule(reparam_base).to(device).eval()

    # 4) 随机输入
    x = torch.randn(batch_size, channel, image_size, image_size, device=device)

    # 5) forward 对比
    with torch.no_grad():
        y_plain = _extract_output(plain(x))
        y_reparam = _extract_output(reparam(x, flat_param=flat))

    print("plain output shape :", tuple(y_plain.shape))
    print("reparam output shape:", tuple(y_reparam.shape))
    print("max abs diff       :", (y_plain - y_reparam).abs().max().item())
    print("mean abs diff      :", (y_plain - y_reparam).abs().mean().item())
    print("sum plain          :", y_plain.sum().item())
    print("sum reparam        :", y_reparam.sum().item())

    # 6) CE 对比
    target = torch.randint(0, num_classes, (batch_size,), device=device)
    ce = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        loss_plain = ce(y_plain, target)
        loss_reparam = ce(y_reparam, target)

    print("ce plain           :", loss_plain.item())
    print("ce reparam         :", loss_reparam.item())
    print("ce abs diff        :", abs(loss_plain.item() - loss_reparam.item()))
    print("=" * 80)

    return {
        "max_abs_diff": (y_plain - y_reparam).abs().max().item(),
        "mean_abs_diff": (y_plain - y_reparam).abs().mean().item(),
        "ce_abs_diff": abs(loss_plain.item() - loss_reparam.item()),
    }



import torch

# 按你的实际 import 改
# from networks.networks import ResNet18
# from networks.networks_Fuse import ResNet18_FlexFuse

def _extract_output(y):
    if isinstance(y, tuple):
        return y[0]
    return y

def load_param_list_by_position(model, param_list, device="cuda"):
    named = list(model.named_parameters())
    assert len(named) == len(param_list), (
        f"param count mismatch: model {len(named)} vs list {len(param_list)}"
    )

    with torch.no_grad():
        for i, ((name, p), src) in enumerate(zip(named, param_list)):
            assert tuple(p.shape) == tuple(src.shape), (
                f"[{i}] {name}: model {tuple(p.shape)} vs src {tuple(src.shape)}"
            )
            p.copy_(src.to(device))

def compare_expert_loaded_forward(
    expert_trajectory,
    start_epoch=0,
    channel=3,
    num_classes=10,
    batch_size=4,
    image_size=32,
    device="cuda",
):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # 1) 取 expert 的 starting params
    starting_params = expert_trajectory[start_epoch]

    # 2) 建两个 plain 模型
    orig = ResNet18(channel=channel, num_classes=num_classes).to(device).eval()
    fuse = ResNet18_FlexFuse(channel=channel, num_classes=num_classes, Fuse=1).to(device).eval()

    # 3) 按位置把同一份 starting_params load 进去
    load_param_list_by_position(orig, starting_params, device=device)
    load_param_list_by_position(fuse, starting_params, device=device)

    # 4) 同一个输入做 forward
    x = torch.randn(batch_size, channel, image_size, image_size, device=device)

    with torch.no_grad():
        y_orig = _extract_output(orig(x))
        y_fuse = _extract_output(fuse(x))

    print("=" * 80)
    print("Expert-loaded plain models forward comparison")
    print("y_orig.shape:", tuple(y_orig.shape))
    print("y_fuse.shape:", tuple(y_fuse.shape))
    print("max abs diff :", (y_orig - y_fuse).abs().max().item())
    print("mean abs diff:", (y_orig - y_fuse).abs().mean().item())
    print("sum orig     :", y_orig.sum().item())
    print("sum fuse     :", y_fuse.sum().item())

    target = torch.randint(0, num_classes, (batch_size,), device=device)
    ce = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        loss_orig = ce(y_orig, target)
        loss_fuse = ce(y_fuse, target)

    print("ce orig      :", loss_orig.item())
    print("ce fuse      :", loss_fuse.item())
    print("ce abs diff  :", abs(loss_orig.item() - loss_fuse.item()))
    print("=" * 80)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compare_plain_vs_reparam(
        channel=3,
        num_classes=10,
        fusion=1,
        batch_size=4,
        image_size=32,
        device=device,
    )