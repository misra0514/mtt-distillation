# 5.3 尝试写一个Forward 可以fuse 的code。 目前先对照原版和fused（autograd）
# 原始版本：
# ----CELOSS----- 2.288494825363159
# ----GRANDLOSS----- 0.03257177770137787
# ----GRAD----- -0.02666577696800232

#  本代码版本（Fuse1）：
# ----CELOSS----- 2.288494825363159
# ----GRANDLOSS----- 0.032537974417209625
# ----GRAD----- -0.026661895215511322
#  本代码版本（Fuse2 ）：
# ----CELOSS----- 2.288494825363159
# ----GRANDLOSS----- 0.03257175534963608
# ----GRAD----- -0.026666179299354553
#  本代码版本（Fuse2，GroupedLinear ）：（注意target生成方式不一样了）
# ----CELOSS----- 2.28849458694458
# ----GRANDLOSS----- 0.032537974417209625
# ----GRAD----- -0.026661895215511322


import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import torchvision.transforms as transforms


import torch
from functools import reduce
from operator import mul


import triton
import triton.language as tl
import time 
import random
import numpy as np
import os

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.nn.attention import sdpa_kernel, SDPBackend
from testing_utils import clear_tensorlists, set_random_seed,load_state_dict_by_position, repeat_params_for_fuse

from networks.networks_Fuse import LinearStacked_2,GroupedLinear,GroupedLayerNorm


class MultiHeadSelfAttention_Fused(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, Fuse=1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.Fuse = Fuse
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.qkv = GroupedLinear(embed_dim, 3 * embed_dim, Fuse)
        self.out_proj = GroupedLinear(embed_dim, embed_dim, Fuse)

    def forward(self, x):
        B, Fs, N, C = x.shape
        qkv = self.qkv(x)
        # [B, Fuse, N, 3C]
        qkv = qkv.view(B, Fs, N, 3, self.num_heads, self.head_dim)
        # [B, Fuse, N, 3, H, D]
        qkv = qkv.permute(3, 0, 1, 4, 2, 5).contiguous()
        # [3, B, Fuse, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        # [B, Fuse, H, N, D]
        with sdpa_kernel(SDPBackend.MATH):
            out = F.scaled_dot_product_attention(q, k, v)
        # [B, Fuse, H, N, D]
        out = out.permute(0, 1, 3, 2, 4).contiguous()
        # [B, Fuse, N, H, D]
        out = out.view(B, Fs, N, C)
        # [B, Fuse, N, C]
        out = self.out_proj(out)
        # [B, Fuse, N, C]

        return out

class TransformerBlock_Fused(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0, Fuse=1):
        super().__init__()
        self.Fuse = Fuse
        self.embed_dim = embed_dim
        self.norm1 = GroupedLayerNorm(embed_dim,Fuse)
        self.attn = MultiHeadSelfAttention_Fused( embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, Fuse=Fuse, )
        self.norm2 = GroupedLayerNorm(embed_dim,Fuse)  # 这个之前是一个Linear。或许可以用linear1.
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = GroupedLinear(embed_dim, hidden_dim, Fuse)
        self.act = nn.GELU()
        self.fc2 = GroupedLinear(hidden_dim, embed_dim, Fuse) # 下一步也是一个LayerNorm。可以用新的linear。

    def forward(self, x):
        """
        x: [B, Fuse, N, C]
        return: [B, Fuse, N, C]
        """
        x = x + self.attn(self.norm1(x))
        y = self.norm2(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        x = x + y
        return x
    
    
class ViT_Fused(nn.Module):
    def __init__( self, image_size=32, patch_size=4, in_channels=3, num_classes=10, embed_dim=128, num_heads=4, mlp_ratio=4.0, dropout=0.0, Fuse=1,
    ):
        super().__init__()
        self.Fuse = Fuse
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d( in_channels=in_channels * Fuse, out_channels=embed_dim * Fuse,  kernel_size=patch_size, stride=patch_size, groups=Fuse, )
        self.cls_token = nn.Parameter(torch.zeros(1, Fuse, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, Fuse, self.num_patches + 1, embed_dim))
        self.block = TransformerBlock_Fused( embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout, Fuse= Fuse )
        # 注意：如果 forward 里面的 shape 是 [B, Fuse, N, D]，
        # 那么 LayerNorm 应该是 embed_dim，而不是 embed_dim * Fuse。
        self.norm = GroupedLayerNorm(embed_dim,Fuse)
        # self.head = LinearStacked_2(embed_dim, num_classes, Fuse)
        self.head = GroupedLinear(embed_dim, num_classes, Fuse)
        self._init_weights()

    def forward(self, x):
        Fuse = self.Fuse
        B = x.shape[0]
        # input:
        # [B, C*Fuse, H, W]
        x = self.patch_embed(x)
        # [B, D*Fuse, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)
        # [B, N, D*Fuse], N =  H/P* W/P
        x = x.reshape(B, self.num_patches, Fuse, self.embed_dim) # 这步首先存疑，因为Fuse已经到最后了（不知道是Fuseemb还是embFuse）
        x = x.permute(0, 2, 1, 3).contiguous()
        # [B, Fuse, N, D]
        cls_token = self.cls_token.expand(B, -1, -1, -1)
        # [B, Fuse, 1, D]
        x = torch.cat((cls_token, x), dim=2)
        # [B, Fuse, N+1, D]
        x = x + self.pos_embed
        # [B, Fuse, N+1, D]
        x = self.block(x)
        # [B, Fuse, N+1, D]
        x = self.norm(x)
        # [B, Fuse, N+1, D]
        x = x[:, :, 0]
        # [B, Fuse, D]
        x = self.head(x)
        # [B, Fuse, num_classes]
        # x = x.permute(1, 0, 2).reshape(B * Fuse, self.num_classes)
        # [Fuse*B, num_classes]
        return x, {}
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, GroupedLayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    

###################################
###################################
flag = 'ManuelBwd'          
flag = 'VFuse'
Fuse = 1
batch_size = 128
num_class=10
out_channel = 128 # in shape是写死了64， 所以out是128的话就是short cut
stride = 2
test_iter = 1
set_random_seed() # 仅仅在ACC test的时候使≈≈≈≈≈≈≈≈≈
###################################



if __name__ == "__main__":
    print("flag = " + flag)
    model1 = ViT_Fused( image_size=32, patch_size=4, in_channels=3, num_classes=10, embed_dim=128, num_heads=4, Fuse=Fuse, ).to("cuda")
    model = model1
    transform = transforms.ToTensor()
    cifar10 = torchvision.datasets.CIFAR10(root='/scratch/yguo25/files/mtt-distillation/data', train=True, download=True, transform=transform)
    # 取一个 batch
    indices = torch.arange(batch_size)   # 或 torch.randperm(len(cifar10))[:batch_size]
    imgs, labels = zip(*(cifar10[i] for i in indices))
    x = torch.stack(imgs).to("cuda").requires_grad_(True)
    target = torch.tensor(labels, device="cuda")

    # torch.save(model.state_dict(), 'model_test_Vit.pt')
    # exit()
    pretrained_dict = torch.load("model_test_Vit.pt")
    x = x.repeat(1, Fuse, 1, 1).detach().clone().requires_grad_()
    # target = target.repeat(Fuse) # 这个只能对应Linear stacked 的输出。否则BFC还是FBC顺序不同。
    target = target.repeat_interleave(Fuse) # 这步也很关键。因为之前是直接把batch维度重复了，所以target也要对应地重复。repeat_interleave 可以把每个元素重复Fuse次。


    pretrained_dict = repeat_params_for_fuse(pretrained_dict,Fuse)
    load_state_dict_by_position(model, pretrained_dict)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([x], lr=1e-1)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    # print(model.block.bn1.track_running_stats, model.block.bn2.track_running_stats)
    start = time.time()

    for step in range(test_iter):
        optimizer.zero_grad()
        if flag =='VFuse':
            x_out,_ = model(x)
            x_out = x_out.reshape(-1, x_out.shape[-1])
            loss = criterion(x_out, target)  # compute loss
            print("----CELOSS-----", loss.item())
            dw = torch.torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
            weight_sum = [d.sum() for d in dw]
            grad_loss = sum(weight_sum)
            print("----GRANDLOSS-----", grad_loss.item())
            grad_loss.backward()  
            print("----GRAD-----", x.grad.sum().item())
        elif flag =='ManuelBwd':
            x_out,_ = model(x)
            