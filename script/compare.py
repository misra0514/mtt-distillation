import torch
import os
import struct

import numpy as np

def float_to_bin32(x):
    """将 float32 转换为 IEEE 754 二进制字符串（32 位）"""
    x32 = np.float32(x)
    [packed] = struct.unpack('>I', struct.pack('>f', x32))
    return f'{packed:032b}'

def extract_mantissa32(x):
    bin_str = float_to_bin32(x)
    mantissa = bin_str[9:]  # 去掉 1 位符号 + 8 位指数
    return mantissa




def count_leading_zeros(diff: int, bit_length: int = 32) -> int:
    # 限制为无符号32位数（模拟补码）
    diff &= (1 << bit_length) - 1
    bin_str = bin(diff)[2:].zfill(bit_length)
    leading_zeros = len(bin_str) - len(bin_str.lstrip('0'))
    return leading_zeros
def compute_difference_and_leading_zeros(a: int, b: int):
    diff = abs(a - b)
    lz = count_leading_zeros(diff)
    # print(f"Difference (a - b): {diff}")
    # print(f"Binary (32-bit): {bin(diff & 0xFFFFFFFF)[2:].zfill(32)}")
    # print(f"Leading zeros (32-bit): {lz}")
    return lz
def floats_to_exponents(float_list):
    list = []
    for f in float_list:
        # packed = struct.pack('>f', f)  # big-endian float
        # as_int = struct.unpack('>I', packed)[0]  # interpret as unsigned 32-bit int

        # 提取 exponent（位 23~30）
        exponent_bits = (f >> 23) & 0xFF  # 只保留 8 位 exponent
        # return exponent_bits
        list.append(exponent_bits)
    return list

def computeExpo(ta, tb):
    # 先导0数量 的list
    a_bits = ta.view(torch.int32).flatten().tolist()
    b_bits = tb.view(torch.int32).flatten().tolist()
    # a_bits = floats_to_exponents(a_bits)
    # b_bits = floats_to_exponents(b_bits)
    for i in range(len(a_bits)):
        a_bits[i] = compute_difference_and_leading_zeros(a_bits[i], b_bits[i])
    return a_bits



def list_to_freq_dict(input_list):
    freq_dict = {}
    for item in input_list:
        if item in freq_dict:
            freq_dict[item] += 1
        else:
            freq_dict[item] = 1
    return freq_dict

def MakeOrder(buffer):
    temp = [i[0] for i in buffer] # i 10*14*buffer
    # temp = [  for i in temp for item in i]
    newlist = []
    for i in temp:
        sum = 0
        for j in i:
            sum+=j.sum().item()
        newlist.append(sum)
    sorted_indices = np.argsort(newlist)[::-1]  # 从大
    buffer = [buffer[i] for i in sorted_indices]

    return buffer
        




if __name__=='__main__':
    folder = "/scratch/yguo25/files/mtt-distillation/buffer/ImageNet/imagenette/128/ConvNet"
    pt_files = [f for f in os.listdir(folder)] # 获取所有 .pt 文件（仅数字命名），并按数字排序
    file_a = "/scratch/yguo25/files/mtt-distillation/buffer/ImageNet/imagenette/128/ConvNet/replay_buffer_2.pt"
    # file_a = "/scratch/yguo25/files/mtt-distillation/buffer/CIFAR100/ResNet18/replay_buffer_9.pt"
    file_b = "/scratch/yguo25/files/mtt-distillation/buffer/ImageNet/imagenette/128/ConvNet/replay_buffer_1.pt"
    # 遍历相邻对
    # for f in range(len(pt_files) - 1):
        # file_a = os.path.join(folder, pt_files[f])
        # file_b = os.path.join(folder, pt_files[f + 1])

    # 加载 tensor, 10*50*14
    tensor_a = torch.load(file_a)
    # tensor_b = torch.load(file_b)
    # tensor_a = MakeOrder(tensor_a)
    
    # for iter in range(9):
        # iter = 9
    iter = 9
    for j in range(3):
        l = []
        for i in range(len(tensor_a[0][0])):
            ta = tensor_a[1][j][i]
            tb = tensor_a[2][j][i]
            # l += computeExpo(ta,tb)


            tc = tensor_a[1][j+1][i]
            td = tensor_a[2][j+1][i]
            a_bits = ta.view(torch.int32).flatten().tolist()
            b_bits = tb.view(torch.int32).flatten().tolist()
            c_bits = tc.view(torch.int32).flatten().tolist()            
            d_bits = td.view(torch.int32).flatten().tolist()            
            ta = ta.view(torch.int32).flatten().tolist()
            tb = tb.view(torch.int32).flatten().tolist()
            for i in range(len(a_bits)):
                ta[i] = abs(a_bits[i]-c_bits[i])
                tb[i] = abs(b_bits[i]-d_bits[i])
                l.append(compute_difference_and_leading_zeros(ta[i], tb[i])) 

            
        c = list_to_freq_dict(l)
        ordered_dict = {k: c[k] for k in sorted(c,reverse=True)}
        print(ordered_dict)
        print("---")







    exit()

    # 计算差值
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
                # 1 minnus 2 


            ratio = diff_val / a_val
            # print(f"  → 误差比: {abs(ratio)}")
            # print(f"  → 最大绝对差值: {diff_val}, 误差比: {abs(ratio)}")
            print(ratio.item(), " ")
            # print(f"  → a,b: {a_val},  {b_val}")
            del diff, diff_val, a_val, b_val

