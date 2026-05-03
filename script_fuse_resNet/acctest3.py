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


def print_expert_vs_model_order(starting_params, model):
    named = list(model.named_parameters())

    print("=" * 100)
    print("Expert tensor list vs model.named_parameters()")
    print("expert count:", len(starting_params))
    print("model count :", len(named))
    print("=" * 100)

    assert len(starting_params) == len(named), (
        f"count mismatch: expert {len(starting_params)} vs model {len(named)}"
    )

    for i, (src, (name, p)) in enumerate(zip(starting_params, named)):
        ok = tuple(src.shape) == tuple(p.shape)
        mark = "OK " if ok else "BAD"
        print(f"[{i:02d}] {mark} | expert {tuple(src.shape)!s:<20} | model {tuple(p.shape)!s:<20} | {name}")


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

    # 3) 先打印顺序/shape 对照
    print("\n[Orig model order check]")
    print_expert_vs_model_order(starting_params, orig)

    print("\n[FlexFuse model order check]")
    print_expert_vs_model_order(starting_params, fuse)

    # 4) 按位置把同一份 starting_params load 进去
    load_param_list_by_position(orig, starting_params, device=device)
    load_param_list_by_position(fuse, starting_params, device=device)

    # 5) 同一个输入做 forward
    x = torch.randn(batch_size, channel, image_size, image_size, device=device)

    with torch.no_grad():
        y_orig = _extract_output(orig(x))
        y_fuse = _extract_output(fuse(x))

    print("\n" + "=" * 100)
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
    print("=" * 100)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== 这里按你的实际情况改 =====
    buffer_path = "/scratch/yguo25/files/mtt-distillation/buffer/CIFAR10_NO_ZCA/ResNet18/replay_buffer_0.pt"
    expert_idx = 0
    start_epoch = 0
    channel = 3
    num_classes = 10
    image_size = 32
    batch_size = 4
    # ==============================

    assert os.path.exists(buffer_path), f"buffer file not found: {buffer_path}"

    print(f"loading buffer: {buffer_path}")
    buffer = torch.load(buffer_path, map_location="cpu")

    print("num expert trajectories in file:", len(buffer))
    expert_trajectory = buffer[expert_idx]
    print("selected expert_idx:", expert_idx)
    print("trajectory length   :", len(expert_trajectory))
    print("selected start_epoch:", start_epoch)

    compare_expert_loaded_forward(
        expert_trajectory=expert_trajectory,
        start_epoch=start_epoch,
        channel=channel,
        num_classes=num_classes,
        batch_size=batch_size,
        image_size=image_size,
        device=device,
    )