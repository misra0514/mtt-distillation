# from networks import ResNet
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import argparse
os.environ["PATH"] = "/scratch/yguo25/files/mtt-distillation"+f"{os.pathsep}"+os.environ.get("PATH", "")
from utils import *
from reparam_module import ReparamModule

def get_images(c, n):  # get random n images from class c
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    return images_all[idx_shuffle]


# 本file主要测试x的距离。
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

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--detachNum', type=int, default=0, help='discard grad before this syn')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--res', type=int, default=128, help='resolution for imagenet')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')
    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')
    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')
    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")
    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')
    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')
    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')
    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')
    args = parser.parse_args()


    folder = "/scratch/yguo25/files/mtt-distillation/buffer/ImageNet/imagenette/128/ConvNet"
    pt_files = [f for f in os.listdir(folder)] # 获取所有 .pt 文件（仅数字命名），并按数字排序
    # file_a = "/scratch/yguo25/files/mtt-distillation/buffer/CIFAR10/ConvNet/replay_buffer_2.pt"
    file_a = "/scratch/yguo25/files/mtt-distillation/buffer/ImageNet/imagenette/128/ConvNet/replay_buffer_1.pt"
    ipc = 1
    model = 'ConvNet'
    device='cuda'
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    # for c in range(10):
    #     image_syn.data[c * ipc:(c + 1) * ipc] = get_images(c, ipc).detach().data


    iter = 9
    student_net = get_network(model, channel, num_classes, im_size, dist=False).to(device)  # get a random model
    student_net = ReparamModule(student_net)

    student_net.train()
    num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])
    tensor_a = torch.load(file_a)
    
    student_params = tensor_a[iter][42]
    student_params = [torch.cat([p.data.to(device).reshape(-1) for p in student_params], 0).requires_grad_(True)]

    image_syn = torch.randn(size=(10 * ipc, channel, im_size[0], im_size[1]), dtype=torch.float).to(device)

    l = []
    for j in range(20):
        
        # print(student_params[0].shape)
        x = student_net(image_syn, flat_param=student_params[0])
        # print(x.shape)
        l.append(x)
        # torch.save(x, "script/x/x"+str(j)+".pt")
        if(j>=1):
            c= computeExpo(x,l[-2])
            c = list_to_freq_dict(c)
            ordered_dict = {k: c[k] for k in sorted(c,reverse=True)}
            print(ordered_dict)
        

