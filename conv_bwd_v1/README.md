# Group-first conv_bwd v1

目标环境为 RTX A5000、Torch 2.5.1、Triton 3.1.0、CUDA 12.4。输入为同一 CUDA 设备上的 FP32 张量，点积采用 IEEE FP32。把 `conv_bwd.py` 放进项目即可使用。全部 kernel 和调度逻辑都在这个文件内。

```python
from conv_bwd import conv_bwd
dx, dw, db = conv_bwd(x, weight, grad_output, stride=2, padding=1)
```

| 张量 | 布局 |
|---|---|
| x, dx | [G,B,C,H,W] |
| weight, dw | [G,O,C/I,R,S] |
| grad_output | [G,B,O,P,Q] |
| db | [G,O] |

每个外层组拥有独立的权重和 bias。通常 `groups=1` 即可，G 由输入推断。若权重包含每组全部 C 个通道，也接受 `groups=G`。对于每个外组内部还包含 I 个 channel groups 的卷积，传 `groups=I`，此时 C 和 O 必须能被 I 整除。普通 4D NCHW 调用保留 Torch 的 groups 含义，输出为 4D、4D、1D。stride、padding、dilation 支持整数或两个整数。非连续输入会在函数内部复制。函数手写一阶梯度，不构建高阶计算图，不持有输入、中间张量或全局 workspace。

## 首版调度

- 直接法以外组、channel tile 和空间位置分配工作。3×3 stride=2、dilation=1 的输入梯度按四种行列奇偶分块，跳过不能贡献的 kernel tap。
- dW 和 db 共用 dY load。一个指定的输入通道 tile 计算每个输出通道的 bias 梯度。大的 B×P×Q 支持 split-K 并行归约，中间结果在一次 kernel launch 中合并。无需浮点原子加。
- 小 B、大 G 的密集 3×3 stride=1、padding=dilation=1 提供 F(2×2,3×3) Winograd 候选。dx 与 dw 复用同一批临时 buffer。dw 的矩阵乘法从 dY 变换后的 (1,1) 分量获取每个 2×2 输出块的和，同时输出 db。
- 直接法的归约临时空间及 Winograd 工作区各自以 128 MiB 为上限。自动调度是首版启发式。仅 Winograd 单组 tiles 在 16–1024、输入与输出通道均至少 32 且满足工作区预算时自动选用。可在 benchmark 中强制比较两条路径。没有使用 double backward 的整套 API 或其性能表。

## 在 A5000 上运行

先做 CUDA 编译和正确性检查，再运行扫描。每项扫描可单独执行。所有命令从本目录运行，不需要模型文件或下载数据集。

```bash
python benchmark.py --check --output check.txt
python benchmark.py --sweep small --candidates --output small_b.txt
python benchmark.py --sweep large --output large_b.txt
```

只看少量点、同时观察布局变换和热点 kernel：

```bash
python benchmark.py --models resnet18 --datasets cifar10 --batch 1 10 --groups 1 4 --candidates --e2e --profile --output resnet_sample.txt
```

| 预设 | 每组 B | 外组 G |
|---|---|---|
| small | 1, 4 | 1, 2, 4, 8, 16, 32 |
| large | 1, 10, 32, 100, 256, 512, 1000 | 1, 2, 4 |
| 默认 both | 上述两组扫描取并集 | 相同点去重 |

测试直接将 stdout/stderr 同时写入屏幕和 --output 指定的 UTF-8 TXT。默认文件为 conv_bwd.txt。文件在运行期间逐条刷新，包含参数、环境、每层实际张量形状、所有候选的耗时/CV/峰值显存/误差/扩展效率、摘要和异常 traceback。RESULT 行是可解析的 JSON 文本。每次运行会覆盖指定输出文件，不同扫描请使用不同文件名。

按提供的 ConvNet，采用 channel=3、net_width=128、net_depth=3、每层后有 2×2 stride=2 池化。norm 和 activation 不改变这里的卷积尺寸。

| 数据集 | 卷积层 | 输入 [G,B,C,H,W] | 权重 | grad_output |
|---|---|---|---|---|
| CIFAR10 | conv1 | [G,B,3,32,32] | [G,128,3,3,3] | [G,B,128,32,32] |
| CIFAR10 | conv2 | [G,B,128,16,16] | [G,128,128,3,3] | [G,B,128,16,16] |
| CIFAR10 | conv3 | [G,B,128,8,8] | [G,128,128,3,3] | [G,B,128,8,8] |
| TinyImageNet | conv1 | [G,B,3,64,64] | [G,128,3,3,3] | [G,B,128,64,64] |
| TinyImageNet | conv2 | [G,B,128,32,32] | [G,128,128,3,3] | [G,B,128,32,32] |
| TinyImageNet | conv3 | [G,B,128,16,16] | [G,128,128,3,3] | [G,B,128,16,16] |

这六种卷积都是 kernel=3、stride=1、padding=1、dilation=1，且有 bias。最后一次池化之后的 feature shape 分别为 [G,B,128,4,4] 和 [G,B,128,8,8]。当前配置的卷积形状与之前测试相同。此测试中 B 指每组实际送入卷积的图片数。

Conv3 width=128。CIFAR10 是 32×32，TinyImageNet 是 64×64。ResNet18 的 auto stem 与卷积参考 benchmark 相同，CIFAR10 采用 3×3 s1，TinyImageNet 采用 7×7 s2 加池化；如果项目中的 TinyImageNet 也用了 3×3 stem，加 `--stem small`。重复 shape 只测一次，layer 名和调用次数保存在 TXT。

## 怎么读结果

- `--check` 在 GPU 上逐组比较三项输出，参考为用户原始 Torch 函数的 FP64 结果。覆盖奇数边长、非整 tile、RGB stem、group 隔离、非连续输入、内层 groups、4D 和 B=1000。性能测试另外对每个 shape 的候选与 FP32 grouped Torch 检查，错误会中止。
- 所有实现使用相同数据。cuDNN benchmark 开启、两处 TF32 关闭。每个实现先 warmup，5 轮按固定种子打乱顺序。分别记录 CUDA Event 和同步墙钟的中位数、CV、额外峰值显存（包括输出）。首次编译及 cuDNN 搜索在计时之前。
- `torch_given` 在原生 [B,G*C,H,W] 调用 conv2d_input、conv2d_weight 和 sum。`aten` 在同一布局单次调用 convolution_backward。`aten_serial` 对原生分组视图逐组调用 ATen，保留独立结果，组间没有同步。`vs_best_torch` 为这三者中最快的 CUDA Event 中位数除以本行中位数。`--e2e` 另计从 group-first 连续输入到 group-first 连续输出的一次 grouped ATen，包含所有必要布局转换。
- `auto` 是实际入口，`direct`、`winograd` 是显式候选。超过 50 µs（含）的最快 Torch 参考视为热点。若候选或参考的 event/wall CV 超过 5%，标记 unstable；稳定时超过 1.03× 为 hot_win，0.97–1.03× 为 near_tie。更快的小算子允许 20% 或 10 µs 中较宽的偏差。模型摘要按该 shape 的调用次数加权，列出 Torch 占比最高的三种 shape。加权总和是隔离测得的各层卷积反传时间之和。
- 固定 B 时，S_G=G·T(1,B)/T(G,B)，E_G=S_G/G。固定 G 时，S_B=B·T(G,1)/T(G,B)，E_B=S_B/B，衡量批处理吞吐扩展（dW、db 沿 B 归约）。相应的参照点必须实测，缺失时留空。时间使用同一实现、同一层的 CUDA Event 中位数。稳定且效率达到 `--eff-floor`（默认 0.8）标为接近线性。逐组串行 ATen 延迟也单独列出。
- OOM 会写入 TXT，未覆盖的模型组合报告 partial coverage。--profile 在正式计时之外列出 CUDA kernel 耗时。带宽曲线需要设备计数器另测。额外峰值显存并非 DRAM 流量。
