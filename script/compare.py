import torch
import os

# 设置路径
folder = "/scratch/yguo25/files/mtt-distillation/buffer/ImageNet/imagenette/128/ConvNet"

# 获取所有 .pt 文件（仅数字命名），并按数字排序
pt_files = [f for f in os.listdir(folder)]

# pt_files = sorted(pt_files, key=lambda x: int(x.replace(".pt", "")))


# print(pt_files)
file_a = "/scratch/yguo25/files/mtt-distillation/buffer/ImageNet/imagenette/128/ConvNet/replay_buffer_2.pt"
# file_a = "/scratch/yguo25/files/mtt-distillation/buffer/CIFAR100/ResNet18/replay_buffer_9.pt"
file_b = "/scratch/yguo25/files/mtt-distillation/buffer/ImageNet/imagenette/128/ConvNet/replay_buffer_1.pt"
# 遍历相邻对
# for f in range(len(pt_files) - 1):
    # file_a = os.path.join(folder, pt_files[f])
    # file_b = os.path.join(folder, pt_files[f + 1])

# 加载 tensor, 10*50*14
tensor_a = torch.load(file_a)
tensor_b = torch.load(file_b)

# # 检查 shape
# assert tensor_a.shape == tensor_b.shape, f"Shape mismatch: {file_a} vs {file_b}"


# 计算差值
# print(tensor_a.shape)
for i in range(len(tensor_a)-1):   # 10
# i=9
    for j in range(len(tensor_a[i])  ):   # 50
    # j1 =0
    # j2=49
        for k in range(len(tensor_a[i][0])): # 14

            diff = torch.abs(tensor_a[0][0][k] - tensor_a[i][j][k])
            max_idx = torch.argmax(diff)             # 展平索引
            max_pos = torch.unravel_index(max_idx, diff.shape)  # 多维索引

            # 获取原始值
            a_val = tensor_a[0][0][k][max_pos]
            b_val = tensor_a[i][j][k][max_pos]
            diff_val = torch.max(diff).item()

            # TODO: 这里应该有一个保留最大值的操作。现在相当于永远记录最后一层的最大值。
            # 同时现在这个统计方法还有瑕疵，因为精度损失最大的地方和绝对值差最大的地方不是一个地方。

        ratio = diff_val / a_val
        # print(f"  → 误差比: {abs(ratio)}")
        # print(f"  → 最大绝对差值: {diff_val}, 误差比: {abs(ratio)}")
        print(ratio.item(), " ")
        # print(f"  → a,b: {a_val},  {b_val}")
        del diff, diff_val, a_val, b_val

