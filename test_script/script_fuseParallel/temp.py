import torch
import torch.nn.functional as F

torch.manual_seed(0)

# ======= 设定卷积超参 =======
N, Cin, Cout = 2, 3, 4
H, W = 8, 8
kH = kW = 3
stride = [1, 1]
padding = [1, 1]
dilation = [1, 1]
transposed = False
output_padding = [0, 0]
groups = 1

device = "cpu"  # 建议先在 CPU 上验证；如需 GPU，改成 "cuda" 并注意设置 cudnn 为确定性
# 如果你改用 GPU，建议加：
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

# ======= 构造前向需要的张量（都放到同一设备）======
x      = torch.randn(N, Cin, H, W, device=device, requires_grad=True)
w      = torch.randn(Cout, Cin, kH, kW, device=device, requires_grad=True)
b      = torch.randn(Cout, device=device, requires_grad=True)   # 带 bias，方便 ggb 路线验证
target = torch.randn(N, Cout, H, W, device=device)

# ======= 前向 + 一阶 loss（用简单的 MSE）======
y = F.conv2d(x, w, b, stride=stride, padding=padding, dilation=dilation, groups=groups)
# 0.5 * ||y - target||^2 的梯度 w.r.t y 就是 (y - target)
loss = 0.5 * (y - target).pow(2).sum()

# 用 create_graph=True 保留图，为二阶做准备
gx, gw, gb = torch.autograd.grad(loss, (x, w, b), create_graph=True)

# ======= 准备二阶的“向量”（别用 None，全都随机）======
ggI = torch.randn_like(x)   # 对 grad_input 的 VJP 种子
ggW = torch.randn_like(w)   # 对 grad_weight 的 VJP 种子（我们最终会对比 d2w）
ggb = torch.randn_like(b)   # 对 grad_bias 的 VJP 种子

# ======= 在 autograd 路线下：构造二阶标量并对 weight 求导，得到 d2w_auto =======
# s = <gx, ggI> + <gw, ggW> + <gb, ggb>
s = (gx * ggI).sum() + (gw * ggW).sum() + (gb * ggb).sum()
d2w_auto = torch.autograd.grad(s, w, retain_graph=True)[0]  # 这是 H_ww · ggW + 其它项对 w 的影响（由 ggI、ggb 贡献）

# ======= 计算 grad_output（作为 _convolution_double_backward 的输入）======
# 对于上面的 MSE：grad_output = d loss / d y = y - target
grad_output = (y - target).detach()  # 作为“已知”的一阶输出梯度传给双反算子

# ======= 调用无状态的底层双反算子 =======
# 返回 (ggO, gI, gW)，我们关心第三个 gW
output_mask = [True, True, True]
ggO_at, gI_at, gW_at = torch.ops.aten._convolution_double_backward(
    ggI, ggW, ggb,
    grad_output, w, x,
    stride, padding, dilation,
    transposed, output_padding, groups,
    output_mask
)

# ======= 对比 d2w（autograd 路线 vs. _convolution_double_backward）======
atol, rtol = 1e-6, 1e-5
allclose = torch.allclose(d2w_auto, gW_at, atol=atol, rtol=rtol)
max_abs_diff = (d2w_auto - gW_at).abs().max().item()

print(f"autograd d2w shape: {d2w_auto.shape}")
print(f"aten     gW  shape: {gW_at.shape}")
print(f"allclose(d2w_auto, gW_at): {allclose}  (rtol={rtol}, atol={atol})")
print(f"max |diff| = {max_abs_diff:.3e}")
