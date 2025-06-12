import math
import torch
import struct

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
    
# a = torch.rand([2,3])
# 
b = torch.tensor([0.0316])
a = torch.tensor([0.0315])

a = torch.load("/scratch/yguo25/files/mtt-distillation/script/x0.pt")
b = torch.load("/scratch/yguo25/files/mtt-distillation/script/x1.pt")
c = torch.load("/scratch/yguo25/files/mtt-distillation/script/x1.pt")

c= computeExpo(a,b)
c = list_to_freq_dict(c)
ordered_dict = {k: c[k] for k in sorted(c,reverse=True)}
print(ordered_dict)

exit()
# print(a)
# # print(b)

# a_bits = a.view(torch.int32)
# # b_bits = b.view(torch.int32)
# print(a_bits)
# flat_list = a_bits.flatten().tolist()

# exponent_list = floats_to_exponents(flat_list)


# print(exponent_list)



# a = 1
# b = 2
# c = compute_difference_and_leading_zeros(a, b)
# print(c)
# # # 示例
# # float_list = [1.0, 0.5, 3.14, 0.0, -8.0, 1024.0]
# # exponent_list = floats_to_exponents(float_list)



# # 00111110100000000110000000000000
# # 00111101000000010110111100000000 0.00001000000101101111 ： 0.0316
# # 00111111001110011001100110011010 : 0.72
# # 相减后的先导0是：0011110 （7位，其中包含一个sign）
# # exponent: 01111010

# # def count_leading_zeros(n: int, bit_width: int = 32) -> int:
# #     if n < 0:
# #         raise ValueError("Only non-negative integers are supported")
# #     if n == 0:
# #         return bit_width
# #     bin_len = n.bit_length()
# #     return bit_width - bin_len

# # # 示例
# # print(count_leading_zeros(5))       # 输出: 29 （因为 5 的二进制是 101，占 3 位）
# # print(count_leading_zeros(0))       # 输出: 32
# # print(count_leading_zeros(1023))    # 输出: 22 （1023 的二进制是 10 个 1）