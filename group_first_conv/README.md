# Group-first convolution: Triton candidate implementation

## 状态与范围

已实现 forward 内核和 correctness-gated benchmark。当前开发环境无 PyTorch、Triton 或 CUDA GPU：通过 Python 语法检查，以及直接执行 kernel AST 的 96 项 NumPy 地址/掩码/调度验证；**尚未完成 Triton GPU 编译、GPU 数值验证、性能实测，不保证快于 cuDNN**。NumPy 验证不替代 GPU 验证。

接口：连续的 X[G,B,C,H,W]、Weight[G,O,C,R,S]、可选 bias[G,O]，输出连续 Y[G,B,O,P,Q]。每组有独立输入和权重，组内是 dense conv，不额外嵌套 depthwise/groups。语义与 PyTorch cross-correlation 一致，不翻转权重。支持 FP32/FP16/BF16、二维 stride/padding/dilation；padding 是对称零填充，不支持字符串 same、非连续输入、自动求导、backward 或 double backward。requires_grad 输入在 grad mode 下报错；手写 backward 集成时请显式 no_grad。

当前实现使用 int32 地址，拒绝任一输入/权重/输出元素数达到 2^31 的形状；G*B <= 65535。目标为 NVIDIA Ampere 及后续架构，实际支持取决于安装的 Triton/设备；不是已验证的跨版本兼容承诺。优先使用现有 CUDA PyTorch 环境及其匹配的 Triton，不要单独强行升级 Triton。

## 计算与设计

每个 (g,b) 做 Y[O,PQ] = Weight[O,CRS] @ patches[CRS,PQ]。数学上是等尺寸 batched GEMM；不需要 MoE 式不等尺寸 grouped-GEMM 调度。这里的 G 对应你融合的独立网络分支数，不是 batch。

普通 PyTorch grouped conv 接收 [B,G*C,H,W]。G-first 连续张量通常不能仅靠 view 得到它：G>1 且 B>1 时通常需要搬运。该内核直接使用 G-first 地址，结果也直接写入 G-first 布局。

不物化全局 im2col：程序块生成所需窗口的地址并加载数据，然后 tl.dot 累加，最后融合 bias 写回。以 G=2,B=100,C=128,P=Q=16,R=S=3、FP32 为例，完整 patch 张量约 225 MiB，仅物化的写入加后续读取就约 450 MiB；implicit 路线避免这一中间张量，但不同 CTA 的重复读取及索引计算仍有成本，不代表零额外开销。

实现包含 12 个 autotune 配置：6 组 tile/warp/stage 设置 × 2 种 iterator。

- CRS_FLAT：K=C*R*S，减少小 C 时的 padding 浪费，适合尝试首层 C=3。
- RS_LOOP：每个 r,s 固定，沿 C 分块，改善 NCHW 相邻空间访问；小 C 时可能明显浪费 Tensor Core 运算。
- 输出 tile 为 [O tile, PQ tile]，B 单独放入 grid.y，输出空间连续；避免跨 batch 合并 M/N 带来的 store 跳跃。
- 同一组、同一 batch 中，以最多 4 个输出通道 tile 为一簇，尝试复用 activation 的 L2 cache。
- 1x1 在编译期退化为无窗口展开的矩阵乘法地址计算。
- 所有路径 FP32 accumulator。FP16/BF16 使用其原始输入精度；FP32 默认 ieee，tf32/tf32x3 需要显式选择。

这些是优化候选，尚无本机性能证据。严格 FP32、不允许 TF32 时，cuDNN 的算法选择（包括某些形状的变换卷积）可能占优；手写直接 GEMM 并不天然更快。大 IPC 时转置成本可能相对下降，也不能由小 IPC 的收益推断所有形状都能赢。

## 使用

```python
import torch
from group_first_conv import conv2d_group_first

x = torch.randn(2, 10, 128, 16, 16, device='cuda', dtype=torch.float32)
w = torch.randn(2, 128, 128, 3, 3, device='cuda', dtype=torch.float32)
b = torch.randn(2, 128, device='cuda', dtype=torch.float32)
y = conv2d_group_first(x, w, b, padding=1, precision='ieee')
# y.shape == [2, 10, 128, 16, 16]
```

第一次调用包括编译和 autotune，可能需要较长时间；不应计入稳态时间。autotune 按形状和卷积参数、bias、精度区分，Triton 还区分输入 dtype。返回新输出张量，未提供复用 out 参数。

## 测试命令

在解压目录中运行。B 是每组图片数；CIFAR10 IPC=1 对应 B=10，IPC=10 对应 B=100。

```bash
# 先运行小规模 FP32 严格精度：边界测试 + ConvNet 三层与 pointwise
python benchmark.py --groups 1 2 4 8 --batch 10 --dtype fp32 --precision ieee --cuda-graph --output fp32_ieee.csv

# 如果原实验启用 cuDNN TF32，采用这一组同策略对比
python benchmark.py --groups 1 2 4 8 --batch 10 --dtype fp32 --precision tf32 --cuda-graph --output fp32_tf32.csv

# 单独看 FP16，不与 FP32 baseline 混比
python benchmark.py --groups 1 2 4 8 --batch 10 --dtype fp16 --cuda-graph --output fp16.csv

# 固定 G=2 扫 B，例：只测 conv2
python benchmark.py --case conv2 --groups 2 --batch 10 100 200 --dtype fp32 --precision ieee --output ipc_sweep.csv

# 自定义 C O H W R S；非默认卷积参数也可指定
python benchmark.py --groups 2 --batch 10 --shape 64 128 32 32 3 3 --stride 2 --padding 1 --output custom.csv
```

默认先进行 ragged channels、非方形、边界、stride/dilation、无 bias 等 GPU 测试。每个性能案例也先与逐组 FP64 conv 的参考值比较，失败立即停止，不输出该案例的性能结论。随机种子固定；CSV 保存 max_abs 与 rel_l2，容差在代码中明确列出。GPU 验证容差是筛查门槛，应用中高阶梯度精度仍需独立验证。大形状的 FP64 reference、autotune 和 cuDNN workspace 可能占较多显存，OOM 会明确报错，不静默跳过。

## 公平比较及列解释

| 列前缀 | 实际测量范围 |
|---|---|
| triton | G-first 输入到连续 G-first 输出，包含输出分配/包装调用 |
| serial | G 次 F.conv2d 加 torch.stack，独立网络串行基线 |
| grouped_e2e | G-first 输入转换 + groups=G cuDNN + 连续 G-first 输出转换 |
| channels_last_e2e | channels_last 输入/权重转换 + cuDNN + 连续 G-first 输出转换 |
| grouped_prepacked | 输入已转为 NCHW grouped，权重已就绪，只测 F.conv2d，输出是 grouped 布局 |
| channels_last_prepacked | 输入/权重已转 channels_last，只测 F.conv2d，不恢复 G-first 输出 |

prepacked 两行用于观察 cuDNN 原生布局下的性能，**不是相同输入输出布局契约的端到端比较**。不要把这两种口径混为一个加速比。脚本没有穷尽 torch.compile/vmap 等所有可能的 PyTorch 实现，因此结果只能表明优于所列 baseline。

- event_ms：Triton do_bench 的 CUDA event 稳态计时，预热后取 median；包含函数触发的 GPU 操作，短 kernel 仍可能受 CPU 发射间隙影响。
- wall_ms：重复调用前后同步的墙钟时间，每轮 iters 次，3 轮取 median，包含 CPU 包装/分配/发射成本。
- graph_ms（--cuda-graph）：CUDA Graph replay，排除重复 Python 发射间隙；完整调用中的转换 kernel 仍被捕获，但主机分配行为不能与普通调用等同。
- *_speedup = 对应 baseline 时间 / Triton 时间，>1 才算加速。没有乘 G：所有方法已经完成相同 G 组工作。
- *.csv.json 保存 GPU、PyTorch、Triton、CUDA、cuDNN 版本与精度设置。CSV 逐案例刷新保存，同时记录 autotune 选中的配置。

cuDNN benchmark=True；正式计时前预热。所有 FP32 baseline 的 TF32 开关明确设置。tf32x3 是可选近似模式，cuDNN 没有对应开关：此时 baseline 保持严格 FP32，应同时报告误差，不能称为完全相同的数值算法。

## 为什么暂不改成 CUDA

Triton 能表达这一地址映射和 tiled GEMM，先获得逐层实测是决定是否转 CUDA 的依据。若大 C 的 iterator/访存开销成为主要瓶颈，再考虑 CUDA/CUTLASS 的 optimized iterator、显式 shared memory staging 和更精细的流水线。换语言本身不会自动带来加速。

若目标仍为 RTX A5000，它是 Ampere；不能直接套用 Hopper/Blackwell 的 TMA im2col/WGMMA 教程。若换成这些较新硬件，且允许内部 channels-last，应另做对应实现并把布局转换成本计入。

## 开源参考与借鉴范围

以下为算法/实现模式参考，本包代码是针对该接口独立编写，没有直接复制其源码文件。

1. Triton 官方 matrix multiplication 教程：tl.dot、tile、L2 program ordering、autotune。
   https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
2. NVIDIA CUTLASS implicit GEMM convolution：运行时生成 patch tile、analytic/optimized iterator、避免全局 im2col。
   https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html
3. Triton Gluon convolution tutorial：TMA im2col / NHWC / Hopper与Blackwell。用于判断架构适配，不是本实现依赖。
   https://triton-lang.org/main/getting-started/tutorials/gluon/conv-im2col.html
4. Triton tl.dot 精度接口：
   https://triton-lang.org/main/python-api/generated/triton.language.dot.html

## 本地验证记录

- python -m py_compile：通过。
- test_indexing_cpu.py：96 项通过；执行实际 kernel AST，模拟两类 iterator 的指针和掩码，对比独立直接卷积，含多 G/B、odd C/O、跨调度簇、bias、stride/dilation。
- GPU 编译、GPU correctness、速度、资源占用：未运行；必须由目标设备执行 benchmark.py 才能下结论。
