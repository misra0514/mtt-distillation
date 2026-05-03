# script_fuse_resNet/compare_manual_grad_vs_autograd.py
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

def compare_manual_grad_vs_autograd(
    buffer_path,
    expert_idx=0,
    start_epoch=0,
    channel=3,
    num_classes=10,
    batch_size=4,
    image_size=32,
    device="cuda",
):
    assert os.path.exists(buffer_path), f"buffer file not found: {buffer_path}"

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print(f"loading buffer: {buffer_path}")
    buffer = torch.load(buffer_path, map_location="cpu")

    print("num expert trajectories in file:", len(buffer))
    expert_trajectory = buffer[expert_idx]
    print("selected expert_idx:", expert_idx)
    print("trajectory length   :", len(expert_trajectory))
    print("selected start_epoch:", start_epoch)

    # 1) 取 starting params
    starting_params = expert_trajectory[start_epoch]

    # 2) 建 plain + reparam model
    plain = ResNet18_FlexFuse(
        channel=channel,
        num_classes=num_classes,
        Fuse=1,
    ).to(device).eval()

    reparam_base = copy.deepcopy(plain)
    reparam = ReparamModule(reparam_base).to(device).eval()

    # 3) expert param list -> flat_param
    flat = torch.cat(
        [p.detach().to(device).reshape(-1) for p in starting_params],
        dim=0
    ).clone().detach().requires_grad_(True)

    print("=" * 100)
    print("expert tensor count:", len(starting_params))
    print("flat numel         :", flat.numel())
    print("reparam param numel:", reparam.param_numel)
    print("=" * 100)

    assert flat.numel() == reparam.param_numel, (
        f"flat numel mismatch: {flat.numel()} vs {reparam.param_numel}"
    )

    # 4) 构造输入
    x = torch.randn(batch_size, channel, image_size, image_size, device=device)
    target = torch.randint(0, num_classes, (batch_size,), device=device)
    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------------------------------
    # A. autograd flat grad
    # --------------------------------------------------------------------------------
    y = reparam(x, flat_param=flat)
    if isinstance(y, tuple):
        y_out = y[0]
    else:
        y_out = y

    loss = criterion(y_out, target)
    g_auto = torch.autograd.grad(loss, flat, retain_graph=False, create_graph=False)[0].detach()

    print("\n[AUTOGRAD]")
    print("loss      :", loss.item())
    print("grad numel:", g_auto.numel())
    print("grad sum  :", g_auto.sum().item())

    # --------------------------------------------------------------------------------
    # B. manual grad from run_first_bwd
    # --------------------------------------------------------------------------------
    with torch.no_grad():
        y2, tape = reparam(x, flat_param=flat.detach())
        d_stem_tensors, d_activates_list, d_weights_list, d_weights_list_all = reparam.call_with_param(
            flat.detach(),
            reparam.module.run_first_bwd,
            tape=tape,
            target=target,
            Fuse=1,
        )

    g_manual = torch.cat([g.reshape(-1) for g in d_weights_list_all], dim=0)

    print("\n[MANUAL]")
    print("grad numel:", g_manual.numel())
    print("grad sum  :", g_manual.sum().item())

    # --------------------------------------------------------------------------------
    # C. global compare
    # --------------------------------------------------------------------------------
    diff = (g_auto - g_manual).abs()

    print("\n[GLOBAL COMPARE]")
    print("max abs diff :", diff.max().item())
    print("mean abs diff:", diff.mean().item())
    print("sum auto     :", g_auto.sum().item())
    print("sum manual   :", g_manual.sum().item())

    # --------------------------------------------------------------------------------
    # D. per-parameter compare, 用 Reparam 的 param_infos 做定位
    # --------------------------------------------------------------------------------
    print("\n[PER-PARAM COMPARE]")
    ptr = 0
    first_bad = None

    for i, ((mn, n), numel, shape) in enumerate(
        zip(reparam._param_infos, reparam._param_numels, reparam._param_shapes)
    ):
        full_name = f"{mn}.{n}" if mn else n

        ga = g_auto[ptr:ptr + numel].view(shape)
        gm = g_manual[ptr:ptr + numel].view(shape)
        d = (ga - gm).abs()

        maxd = d.max().item()
        meand = d.mean().item()

        print(f"[{i:02d}] {full_name:<40} shape={tuple(shape)!s:<20} max={maxd:.8e} mean={meand:.8e}")

        if first_bad is None and maxd > 1e-6:
            first_bad = (i, full_name, tuple(shape), maxd, meand)

        ptr += numel

    print("=" * 100)
    if first_bad is None:
        print("All params match within threshold 1e-6")
    else:
        i, full_name, shape, maxd, meand = first_bad
        print("FIRST BAD PARAM:")
        print(f"idx={i}, name={full_name}, shape={shape}, max_diff={maxd:.8e}, mean_diff={meand:.8e}")
    print("=" * 100)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    compare_manual_grad_vs_autograd(
        buffer_path="/scratch/yguo25/files/mtt-distillation/buffer/CIFAR10_NO_ZCA/ResNet18/replay_buffer_0.pt",
        expert_idx=0,
        start_epoch=0,
        channel=3,
        num_classes=10,
        batch_size=4,
        image_size=32,
        device=device,
    )