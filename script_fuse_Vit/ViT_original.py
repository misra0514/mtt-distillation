# 5.3 尝试写一个Forward 可以fuse 的code。 目前先对照原版和fused（autograd）
# 原始版本：
# ----CELOSS----- 2.288494825363159
# ----GRANDLOSS----- 0.03257177770137787
# ----GRAD----- -0.02666577696800232
#  下面开始加Fuse


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
from testing_utils import clear_tensorlists, set_random_seed,load_state_dict_by_position,repeat_params_for_fuse







class SimpleMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()

        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, N, 3C]
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # TODO: scaled_dot_product_attention 因为不支持二阶导被禁用了。如果只是fwd应该不影响。
        # out = F.scaled_dot_product_attention(
        #     q, k, v,
        #     dropout_p=self.dropout if self.training else 0.0
        # )
        with sdpa_kernel(SDPBackend.MATH):
            out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.out_proj(out)

        return out


class SimpleTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SimpleMultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        # Pre-norm Transformer
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SimpleViT(nn.Module):
    def __init__( self, image_size=32, patch_size=4, in_channels=3, num_classes=10, embed_dim=128, num_heads=4, mlp_ratio=4.0, dropout=0.0, Fuse=1,
    ):
        super().__init__()

        self.Fuse = Fuse
        self.in_channels = in_channels * Fuse
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.block = SimpleTransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

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

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, 3 * Fuse, 32, 32]
        x = self.patch_embed(x)          # [B, C, H', W']
        x = x.flatten(2).transpose(1, 2) # [B, N, C]
        B = x.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, C]
        x = torch.cat([cls_token, x], dim=1)          # [B, N+1, C]
        x = x + self.pos_embed
        x = self.block(x)
        x = self.norm(x)
        cls_feature = x[:, 0]       # [B, C]
        logits = self.head(cls_feature)

        return logits, cls_feature
    

    

###################################
###################################
# flag = 'flex'        
flag = 'fused'          # Fuse 大小可以控制。
flag = 'manuel'         # 只是一个写开bwd的版本。fuse 永远是1
flag = 'original'
Fuse = 1
batch_size = 128
num_class=10
out_channel = 128 # in shape是写死了64， 所以out是128的话就是short cut
stride = 2
test_iter = 1
set_random_seed() # 仅仅在ACC test的时候使用。会严重影响性能。 
###################################
###################################



if __name__ == "__main__":
    print("flag = " + flag)
    if flag == 'manuel' :
        Fuse =1
    model1 = SimpleViT( image_size=32, patch_size=4, in_channels=3, num_classes=10, embed_dim=128, num_heads=4, Fuse=Fuse, ).to("cuda")
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
    if flag =='fused':
        pretrained_dict = repeat_params_for_fuse(pretrained_dict,Fuse)
        x = x.repeat(1, Fuse, 1, 1).detach().clone().requires_grad_()
        target = target.repeat(Fuse)


    load_state_dict_by_position(model, pretrained_dict)



    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([x], lr=1e-1)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    # print(model.block.bn1.track_running_stats, model.block.bn2.track_running_stats)
    start = time.time()

    for step in range(test_iter):
        optimizer.zero_grad()

        if flag =='original':
            x_out,_ = model(x)
            # x_out, x_fc,x_bnsc, x_pool, x_bn2, x_conv2, x_bn1, x_block, x_bn= model(x)  # forward
            loss = criterion(x_out, target)  # compute loss
            print("----CELOSS-----", loss.item())
            dw = torch.torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
            weight_sum = [d.sum() for d in dw]
            grad_loss = sum(weight_sum)
            print("----GRANDLOSS-----", grad_loss.item())
            grad_loss.backward()  
            print("----GRAD-----", x.grad.sum().item())
