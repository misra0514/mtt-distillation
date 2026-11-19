Group-first stateless convolution double backward — Triton v1

交付内容
conv_double_bwd.py：手写内核及无状态接口。
test_double_bwd.py：只有 correctness 和 bench 两种测试模式。
VALIDATION.txt：本次实际完成的验证记录与局限。

接口（位置参数顺序与用户给出的 ATen 接口一致）
ggO, gI, gW = convolution_double_backward(
    ggI_opt, ggW_r_opt, ggb_opt, gO_r, weight_r, input,
    stride_, padding_, dilation_, transposed_, output_padding_,
    groups_, output_mask)

数据布局
input、ggI：[G,F,C,H,W]，F 是每个独立分支的 batch。
weight、ggW：[G,O,C,R,S]。
gO、ggO：[G,F,O,P,Q]。
ggb：[G,O]；gI 与 input 同形，gW 与 weight 同形。
所有输入要求 contiguous、同 CUDA device、同 dtype。输出 contiguous。
外层 G 是 fuse 数，自动从第一维读取，可以是奇数，不要求整除 tile。
groups_=1 指每个分支内部的 dense conv，绝不能因为 fuse=8 就把此参数改为8。
ATen reference 打包成 [F,G*C,H,W] 后才使用 groups=G；这是布局转换后的另一种表示。

当前支持范围
非转置 2D 卷积：transposed_=False，output_padding_=[0,0]，groups_=1。
支持正数 stride/dilation，对称非负 padding，非方形输入和卷积核。
FP32、FP16、BF16；默认 FP32 ieee，不偷偷转换 FP16 或 TF32。
本版不支持转置卷积、分支内部 grouped/depthwise conv、空张量或非连续张量。
使用保守索引界限：任一输入张量元素数 <2^30，G*F<=65535。
这是手写二阶导数本身，不依赖 forward tape，也不构建三阶 autograd 图。
带 requires_grad 的参数应在 torch.no_grad() 上下文中调用。

ATen 2.5.1 的兼容细节
已核对 v2.5.1 Convolution.cpp，并在 CPU ATen 上验证：此算子的 output_mask 参数并未被使用。
为了数值/None 语义直接对齐，本实现也接受但忽略 output_mask，而不是自行改变其行为。
缺少某个 optional 输入时：
- ggO = conv(ggI,weight) + conv(input,ggW) + broadcast(ggb)，只计算存在的项。
- gI 仅在 ggW 存在时返回，否则 None。
- gW 仅在 ggI 存在时返回，否则 None。
- 三个 optional 输入全部为 None 时，返回 (None,None,None)。
只有 ggb 时 ATen 可返回 expanded view；本实现返回连续张量，数值相同，不保证别名/stride相同。
不同 PyTorch 版本可能改变 output_mask 行为，因此正确性测试专门覆盖全 False mask。

为什么这样实现
1. ggO：两个 implicit GEMM 在同一 kernel 中完成，共用坐标生成并融合 bias。
   只分配最终 ggO，没有两个完整 conv 输出和后续相加的全尺寸临时张量。
   两个乘积各用 FP32 accumulator；FP16/BF16 输出在求和前保留 ATen 中间 dtype 的舍入步骤。
2. gI：按照每个 input 像素直接 gather 对应 gO 和 ggW 并做矩阵乘。
   不物化插零的 gO，不构造 col2im，不使用 atomic scatter，不生成翻转权重拷贝。
3. gW：直接计算 sum_{f,p,q} gO[g,f,o,p,q] * ggI[g,f,c,p*stride-padding+r*dilation,q*stride-padding+s*dilation]。
   不交换 batch/channel、不把多个 G 放到 Python 循环中调用卷积、不生成完整 im2col。
4. gW split-K：当输出 tile 数不足时按形状及 SM 数启发式选择1/2/4/8/16个分片；
   每片写独立FP32结果，第二个kernel按固定顺序归约，不用原子加法。
   split=1时直接写gW，临时空间为0；split>1时临时空间为4*G*split*O*C*R*S字节。
   默认临时空间预算32MiB，超预算降低split。预算不包含输入、最终输出、编译器运行时开销。
   可通过 workspace_limit_mb=0 强制零归约临时空间，通过 split_k=1/2/4/8/16 手动覆盖。
   显式 split_k 超预算时直接报错，不悄悄超额分配。
5. 每类矩阵乘各有5组tile/warp/stage候选；按形状、精度和optional输入自动选择。
   autotune缓存编译代码/配置，不持有数据张量、历史计算图或共享临时tensor。
   常见完整输出需要3次kernel launch；split-K启用时4次。None输入按编译期分支省略。

限制与性能判断
这些设计减少了ATen特定double-backward路径中的中间张量和逐组操作，具有明确的优化目标，
但没有目标GPU上的实测，因此没有已证实的加速倍数，也不承诺所有形状比cuDNN快。
默认严格FP32的GEMM可能不如cuDNN某些卷积算法；大G和小F可能更容易从少发射/少拷贝中受益，
这只是待验证假设。split-K目前是受内存约束的启发式，不是全局最优搜索。
通用stride>1的gI包含整除/取模检查；当前重点形状stride=1由编译期常量消除相关检查。
目标Ampere(sm80/sm86)及后续NVIDIA架构；使用常规tl.dot，不依赖Hopper/Blackwell专属TMA/WGMMA。
没有GPU时不能完成最终的真实设备正确性、速度、访存安全、寄存器占用或竞争检查。

使用
from conv_double_bwd import convolution_double_backward
with torch.no_grad():
    ggO, gI, gW = convolution_double_backward(
        ggI, ggW, ggb, gO, weight, input,
        [1,1], [1,1], [1,1], False, [0,0], 1, [True,True,True])

测试（在解压目录运行，复用现有匹配版本的torch/triton CUDA环境）
python test_double_bwd.py correctness
python test_double_bwd.py bench --groups 1 2 4 8 16 32 --batch 10 --output double_bwd_times.csv

correctness：逐项对比真实 aten._convolution_double_backward，覆盖三类形状、split=1/4、
odd C/O/G、stride/dilation、8种optional输入组合以及output_mask兼容性。
bench：默认测ConvNet三层（3->128@32，128->128@16，128->128@8，均3x3）。
每个性能形状也先检查正确性，未通过即终止，不会跳过报错后给出速度结果。
F=10在CIFAR10每次使用全部类别时对应IPC=1；F=100对应IPC=10。

可选参数
--case conv1 / conv2 / conv3 / all
--batch 10 100 1000
--shape C O H W R S
--stride 1 --padding 1 --dilation 1
--dtype fp32 / fp16 / bf16
--precision ieee / tf32 / tf32x3
--workspace-mb 32
--split-k 1 / 2 / 4 / 8 / 16
--iters 100
--output xxx.csv

计时口径
triton：完整无状态调用，包含输出和临时tensor分配及所有kernel。
aten_prepacked：提前将输入转换为ATen的grouped布局，计时只包含该ATen double-backward接口；
               输出保留ATen原生布局，代表不计外部转置成本的较强基线。
aten_group_first_e2e：每次计时包含输入打包、同一ATen接口和连续group-first输出恢复。
两者都不是G次普通conv的串行弱基线，而是同一个ATen grouped double backward接口。
加速比为对应ATen时间/Triton时间，相同G和F，不再乘G。只有大于1才加速。
所有路径同dtype，同TF32开关，cuDNN benchmark=True；首次编译/autotune不计入。
每个函数预热10次，然后3轮测量，每轮iters次，结果取3轮平均值。
event_ms：CUDA events测量重复调用区间，短kernel仍可能包含CPU发射间隙。
wall_ms：同步前后墙钟，包括Python参数检查、分配和调度。
peak_extra_bytes：预热后单次调用相对现有输入的PyTorch allocator峰值增量，包含输出/临时tensor；
                  不等同nvidia-smi进程显存，不包含不可见的driver/library内存。
CSV逐案例写入；另输出环境JSON，保存torch/triton/CUDA/cuDNN/GPU/精度设置。
FP32 tf32x3没有对应cuDNN开关，因此对比严格FP32 ATen并报告误差，不能称完全相同算术。

参考资料（算法/代码模式参考；本包未复制第三方源码文件）
PyTorch 2.5.1 ATen公式与实现：
https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/Convolution.cpp
Triton官方Matmul：tile、tl.dot、program ordering、autotune：
https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
NVIDIA CUTLASS implicit GEMM convolution：避免物化patch矩阵：
https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html
Triton官方Conv Dgrad：gather/隐式卷积、split-K FP32 workspace + reduction设计参考；
其TMA/WGMMA主体面向更新架构，未直接用于本Ampere实现：
https://triton-lang.org/main/getting-started/examples/gluon/conv-dgrad.html
