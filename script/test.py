# 这个小程序测试了一下在torch中实现异步数据拷贝
# 异步数据拷贝有两个条件：a. pinned memory;    b.non_blocking=True
# 如果不满足，就会退化成同步拷贝。（当然还是调用async接口）
# 注意调用.pin_memory() 本身也有开销。这导致x.pin_memory().to('cuda', non_blocking=True)反而更慢。
# 使用x_cpu = torch.empty_like(x_gpu, device='cpu', pin_memory=True) 来保证D2H也可以享受异步拷贝。
# 还没有测试搭配不同流的代码。


import torch
import time

x = torch.rand([100000,10000])

model = torch.nn.Linear(10000, 1000).cuda()
model.eval()

x = x.pin_memory()

start = time.time()
# x_gpu = x.to('cuda')  
x_gpu = x.to('cuda', non_blocking=True)  
# torch.cuda.synchronize()  
end = time.time()

y = model(x_gpu)
print(f"[CPU -> GPU] Transfer time: {(end - start)*1000:.2f} ms")
