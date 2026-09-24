# 11.2 测试einsum 性能

# 结论：如果没有转制的话就差不多。但如果算上转制，那么明显是bad要快。
# bmm 和 einsum倒是没什么区别。推测可能最后底层走的都是同一个接口，只要没有contiguous就行。

import torch
import time

# 设置随机种子（可选，保证可重复）
torch.manual_seed(0)

# 在 GPU 上测试（如果有 CUDA）
device = "cuda" 
print(f"Using device: {device}")

# 构造大张量
a = torch.randn(128, 1024, 512, device=device)
b = torch.randn(1024, 512, 256, device=device)

# 预热 GPU（避免第一次运行慢）
for _ in range(5):
    torch.einsum("abc,bcd->abd", a, b)
    torch.einsum("abc,bcd->bad", a, b)
torch.cuda.synchronize() if device == "cuda" else None

# 定义计时函数
def benchmark(expr, a, b, n_iter=1000):
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()
    for _ in range(n_iter):
        c = torch.einsum(expr, a, b)
        c = c.contiguous()
    torch.cuda.synchronize() if device == "cuda" else None
    end = time.time()
    return (end - start) / n_iter

def bmm(a,b, n_iter=1000):
    start = time.time()
    for i in range(n_iter):
        d = a.transpose(0, 1)
        c = torch.bmm(d,b)
    torch.cuda.synchronize() if device == "cuda" else None
    end = time.time()
    return (end - start) / n_iter

# 运行测试
time_abd = benchmark("abc,bcd->abd", a, b)
time_bad = benchmark("abc,bcd->bad", a, b)
time_bmm = bmm(a,b)

print(f"Average time per run:")
print(f'  "abc,bcd->abd": {time_abd:.6f} s')
print(f'  "abc,bcd->bad": {time_bad:.6f} s')
print(f'  "bmm": {time_bmm:.6f} s')

# 对比结果
ratio = time_bad / time_abd if time_abd > 0 else float("inf")
print(f"\nRatio (bad / abd): {ratio:.3f}x")
