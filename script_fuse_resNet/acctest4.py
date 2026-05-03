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


def compare_expert_plain_vs_reparam(
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

    starting_params = expert_trajectory[start_epoch]

    # 1) plain model，按位置 load expert params
    plain = ResNet18_FlexFuse(
        channel=channel,
        num_classes=num_classes,
        Fuse=1,
    ).to(device).eval()

    load_param_list_by_position(plain, starting_params, device=device)

    # 2) 直接把同一份 expert params flatten 成 flat_param
    flat_from_expert = torch.cat(
        [p.detach().to(device).reshape(-1) for p in starting_params],
        dim=0
    )

    print("=" * 100)
    print("expert tensor count :", len(starting_params))
    print("flat_from_expert numel:", flat_from_expert.numel())
    print("=" * 100)

    # 3) 用 plain 的深拷贝包一层 ReparamModule
    reparam_base = copy.deepcopy(plain)
    reparam = ReparamModule(reparam_base).to(device).eval()

    # 4) 同一个输入
    x = torch.randn(batch_size, channel, image_size, image_size, device=device)

    with torch.no_grad():
        y_plain = _extract_output(plain(x))
        y_reparam = _extract_output(reparam(x, flat_param=flat_from_expert))

    print("plain shape         :", tuple(y_plain.shape))
    print("reparam shape       :", tuple(y_reparam.shape))
    print("max abs diff        :", (y_plain - y_reparam).abs().max().item())
    print("mean abs diff       :", (y_plain - y_reparam).abs().mean().item())
    print("sum plain           :", y_plain.sum().item())
    print("sum reparam         :", y_reparam.sum().item())

    target = torch.randint(0, num_classes, (batch_size,), device=device)
    ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        loss_plain = ce(y_plain, target)
        loss_reparam = ce(y_reparam, target)

    print("ce plain            :", loss_plain.item())
    print("ce reparam          :", loss_reparam.item())
    print("ce abs diff         :", abs(loss_plain.item() - loss_reparam.item()))
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

    compare_expert_plain_vs_reparam(
        expert_trajectory=expert_trajectory,
        start_epoch=start_epoch,
        channel=channel,
        num_classes=num_classes,
        batch_size=batch_size,
        image_size=image_size,
        device=device,
    )