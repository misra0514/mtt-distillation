# G-first Conv2d forward 续跑

## 补跑现有日志

把四份原始 `fwd_v2_part*.txt` 放在解压目录，或者把下方输入路径改成它们的实际位置。每条命令只补本 shard 中缺少成功记录的点，结果写到单独的文件。

```bash
python benchmark.py --parts 4 --part 1 --layout-stages --resume-from fwd_v2_part1.txt --output fwd_v2_resume_part1.txt
python benchmark.py --parts 4 --part 2 --layout-stages --resume-from fwd_v2_part2.txt --output fwd_v2_resume_part2.txt
python benchmark.py --parts 4 --part 3 --layout-stages --resume-from fwd_v2_part3.txt --output fwd_v2_resume_part3.txt
python benchmark.py --parts 4 --part 4 --layout-stages --resume-from fwd_v2_part4.txt --output fwd_v2_resume_part4.txt
```

四份已收到的日志依次留下 27、30、35、33 个有效点。续跑依次补 32、30、26、27 个点。各 shard 仍保留自身的 FP64 边界检查。可以加 `--list-tasks` 预览续跑范围。

现有逐元素容差和全局相对 L2 门槛用于首轮 FP32 比较。逐元素比较失败、全局门槛通过时，脚本会用直接 FP64 卷积重算至多 4096 个超差位置。候选和 Torch 各自对照该参考，日志记录它们的超差数量和最大绝对误差。候选所有复核位置都满足原容差时记录 `fp64_resolved`。确实超差时记录 `validation_error` 并继续扫描，成功验证的实现仍会计时。脚本结束时用非零退出码提示仍有数值验证失败。

## 调用前向接口

`group_first_conv.py` 提供可以直接调用的前向接口。输入和输出都采用连续的 G-first 布局，扁平首维的索引是 `g * B + b`。每组的输入通道与卷积权重独立。权重是 `[G,O,C,R,S]`，可选 bias 是 `[G,O]`。

```python
import torch
from group_first_conv import grouped_conv2d

g, b = 4, 10
x = torch.randn(g*b, 128, 8, 8, device="cuda")
weight = torch.randn(g, 128, 128, 3, 3, device="cuda")
bias = torch.randn(g, 128, device="cuda")

with torch.no_grad():
    y = grouped_conv2d(x, weight, bias, padding=1, groups=g)
# y.shape == (40, 128, 8, 8)
```

`image_tile` 是默认调度，沿用原版单图 implicit GEMM。传入 `algorithm="batch_tile"` 可尝试把同组的图片合入空间维。两条路径都直接写回 G-first 输出。FP32 默认使用 `ieee`，也接受 FP16、BF16，以及显式选择的 FP32 `tf32` 或 `tf32x3`。本次 benchmark 固定 FP32 IEEE，只测 forward。接口不记录 autograd。首次调用会编译并搜索分块配置，正式计时前会完成这一过程。

## 从头运行测试

在解压后的 `fwd_v2_group_first` 目录，每台机器执行一条。四份使用相同的参数，只改 shard 编号和输出文件名。

```bash
python benchmark.py --parts 4 --part 1 --layout-stages --output fwd_v2_part1.txt
python benchmark.py --parts 4 --part 2 --layout-stages --output fwd_v2_part2.txt
python benchmark.py --parts 4 --part 3 --layout-stages --output fwd_v2_part3.txt
python benchmark.py --parts 4 --part 4 --layout-stages --output fwd_v2_part4.txt
```

默认扫描 CIFAR10、TinyImageNet 和 ImageNet10 对应的 Conv3 与 ResNet18 卷积形状。Conv3 使用宽度 128，卷积后做 2×2 池化。ResNet18 在 CIFAR10 使用小图 stem，其余使用 7×7 stride=2 stem。总计 30 个唯一卷积形状 × 8 个 `(B,G)` 组合。每条 `RESULT` 的 `sources` 保留层名，能按网络累计卷积时间。四份分别包含 59、60、61、60 个配置和各自的 FP64 边界检查。每个性能点在计时前都和同轮 Torch 输出做正确性比较。

`--sweep full` 含 `B=1000` 等 14 个 `(B,G)` 组合。运行整套时，四条命令都加上这个选项。超出索引限制或显存不足的点会记录为跳过。下面两条命令分别只跑边界正确性和 ResNet 后段的少量形状。

```bash
python benchmark.py correctness --output fwd_v2_correctness.txt
python benchmark.py --suite resnet18 --filter cifar10/layer4 --pairs 1x64 10x4
```

| 输出字段 | 计时范围 |
|---|---|
| `image_tile` | 原调度，G-first 输入和输出，包含输出分配 |
| `batch_tile` | 同组跨图片 tile，性能扫描在 `B>1、P*Q<=64` 时测 |
| `torch_native` | 已准备好的 Torch NCHW grouped 输入，直接调用 `F.conv2d` |
| `torch_layer` | G-first 输入转换、`F.conv2d`、输出转回 G-first |
| `layout.stages` | 加 `--layout-stages` 时单独计时实际发生的输入或输出拷贝 |

`full` 记录 CUDA event 和 wall 时间的中位数、变异系数、每轮数据和额外分配峰值。`layout` 还记录实际是否拷贝和拷贝的有效载荷字节数。分步时间单独计量，端到端以 `torch_layer` 为准。两种 Triton 调度使用同一组 12 个分块候选，日志记录最终分块、CTA 数、空间占用率和 K 维占用率。不同 GPU 的结果可分别分析。

## 性能假设

- 单图调度的最小空间 tile 是 64。2×2 输出的有效位置占 4/64，4×4 占 16/64。跨图片 tile 提高有效位置比例，并允许一个程序块将同一权重 tile 用到多张图。它也会减少 CTA 数，且访存可能跨图片边界，所以小 G 需要实测判断。
- 两种调度都在 kernel 内生成窗口地址，不分配全局 im2col。较大空间图上，各个空间 tile 仍会请求同一组权重。对于密集 3×3，直接卷积的乘加量和窗口取数会继续影响速度。
- RGB 7×7 的有效 K 是 147。当 `TK=32` 时，逐滤波位置分块有 1568 个 K 槽位，展平 CRS 只需要 160 个。两者的 K 利用率和访存模式不同，结果中保留实际选择。
