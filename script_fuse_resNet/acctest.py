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


def compare_models(channel=3, num_classes=10, fusion=1, batch_size=4, image_size=32, device="cuda"):
    assert fusion == 1, "这个脚本是拿 Fuse=1 和原始 ResNet18 做严格对比的"

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # 1) 建模
    m_fuse = ResNet18_FlexFuse(channel=channel, num_classes=num_classes, Fuse=fusion).to(device).eval()
    m_orig = ResNet18(channel=channel, num_classes=num_classes).to(device).eval()

    # 2) 拿 named_parameters
    p_fuse = list(m_fuse.named_parameters())
    p_orig = list(m_orig.named_parameters())

    print("=" * 80)
    print("Parameter count")
    print("FlexFuse:", len(p_fuse))
    print("Orig    :", len(p_orig))
    print("=" * 80)

    if len(p_fuse) != len(p_orig):
        print("参数个数不一样，先别往下看了。")
        for i, (name, p) in enumerate(p_fuse):
            print(f"[Fuse {i:02d}] {name:<40} {tuple(p.shape)}")
        print("-" * 80)
        for i, (name, p) in enumerate(p_orig):
            print(f"[Orig {i:02d}] {name:<40} {tuple(p.shape)}")
        return

    # 3) 比较每个位置的 shape
    print("\nParameter order + shape comparison")
    same_shape_all = True
    for i, ((name_f, param_f), (name_o, param_o)) in enumerate(zip(p_fuse, p_orig)):
        same_shape = tuple(param_f.shape) == tuple(param_o.shape)
        if not same_shape:
            same_shape_all = False
        mark = "OK " if same_shape else "BAD"
        print(
            f"[{i:02d}] {mark} | "
            f"Fuse: {name_f:<35} {tuple(param_f.shape)!s:<20} || "
            f"Orig: {name_o:<35} {tuple(param_o.shape)}"
        )

    print("=" * 80)
    print("All shapes aligned by position:", same_shape_all)
    print("=" * 80)

    if not same_shape_all:
        print("shape 顺序没对齐，forward 对比就没意义了。")
        return

    # 4) 把 Orig 的参数按“位置”拷到 Fuse 里
    with torch.no_grad():
        for (name_f, param_f), (name_o, param_o) in zip(p_fuse, p_orig):
            param_f.copy_(param_o)

    # 5) 再检查 copy 后参数是否逐项一致
    print("\nParameter value equality check after positional copy")
    all_equal = True
    for i, ((name_f, param_f), (name_o, param_o)) in enumerate(zip(p_fuse, p_orig)):
        max_diff = (param_f - param_o).abs().max().item()
        same = max_diff == 0.0
        if not same:
            all_equal = False
        mark = "OK " if same else "BAD"
        print(f"[{i:02d}] {mark} | max abs diff = {max_diff:.8e} | {name_f}  <->  {name_o}")

    print("=" * 80)
    print("All copied params exactly equal:", all_equal)
    print("=" * 80)

    # 6) forward 对比
    x = torch.randn(batch_size, channel, image_size, image_size, device=device)

    with torch.no_grad():
        y_fuse = m_fuse(x)
        y_orig = m_orig(x)

    if isinstance(y_fuse, tuple):
        y_fuse = y_fuse[0]
    if isinstance(y_orig, tuple):
        y_orig = y_orig[0]

    print("y_fuse.shape:", tuple(y_fuse.shape))
    print("y_orig.shape:", tuple(y_orig.shape))
    print("max abs diff :", (y_fuse - y_orig).abs().max().item())
    print("mean abs diff:", (y_fuse - y_orig).abs().mean().item())
    print("sum fuse     :", y_fuse.sum().item())
    print("sum orig     :", y_orig.sum().item())

    # 7) 如果你想更细一点，还可以比较 CE loss
    target = torch.randint(0, num_classes, (batch_size,), device=device)
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        loss_fuse = ce(y_fuse, target)
        loss_orig = ce(y_orig, target)

    print("\nCE comparison")
    print("loss_fuse:", loss_fuse.item())
    print("loss_orig:", loss_orig.item())
    print("loss diff:", abs(loss_fuse.item() - loss_orig.item()))

    print("=" * 80)

import torch

def compare_plain_vs_reparam(channel=3, num_classes=10, fusion=1, batch_size=4, image_size=32, device="cuda"):
    assert fusion == 1

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    plain = ResNet18_FlexFuse(channel=channel, num_classes=num_classes, Fuse=fusion).to(device).eval()
    reparam = ReparamModule(plain).to(device).eval()

    x = torch.randn(batch_size, channel, image_size, image_size, device=device)

    # 用 plain 当前参数按顺序拼成 flat
    flat = torch.cat([p.detach().reshape(-1) for _, p in plain.named_parameters()], 0)

    with torch.no_grad():
        y_plain = plain(x)
        y_reparam = reparam(x, flat_param=flat)

    if isinstance(y_plain, tuple):
        y_plain = y_plain[0]
    if isinstance(y_reparam, tuple):
        y_reparam = y_reparam[0]

    print("plain shape  :", tuple(y_plain.shape))
    print("reparam shape:", tuple(y_reparam.shape))
    print("max abs diff :", (y_plain - y_reparam).abs().max().item())
    print("mean abs diff:", (y_plain - y_reparam).abs().mean().item())
    print("sum plain    :", y_plain.sum().item())
    print("sum reparam  :", y_reparam.sum().item())


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # compare_models(
    #     channel=3,
    #     num_classes=10,
    #     fusion=1,
    #     batch_size=4,
    #     image_size=32,
    #     device=device,
    # )
    compare_plain_vs_reparam()