import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
# from networks_fused3 import NormActive # fuse+基本优化
# from networks_fused import NormActive # 无fuse
from networks_fused2 import GeluDrop # 无fuse


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_size=128, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, emb_size, H/patch, W/patch)
        x = x.flatten(2)  # (B, emb_size, N)
        x = x.transpose(1, 2)  # (B, N, emb_size)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, emb_size=128, heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(emb_size)

        self.mlp = nn.Sequential(
            nn.Linear(emb_size, mlp_dim),
            # nn.GELU(),
            # nn.Dropout(dropout),
            GeluDrop(dropout),
            nn.Linear(mlp_dim, emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_ln = self.ln1(x)
        attn_output, _ = self.attn(x_ln, x_ln, x_ln)
        x = x + attn_output
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, emb_size=128, num_classes=10, depth=6, heads=4, mlp_dim=256):
        super().__init__()
        self.patch_embed = PatchEmbedding(3, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, emb_size))

        self.transformer = nn.Sequential(*[
            TransformerEncoder(emb_size, heads, mlp_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_size)
        self.mlp_head = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, emb_size)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+N, emb_size)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x[:, 0])  # 只取CLS token
        return self.mlp_head(x)


import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

# 可选：把归一化层抽出来，想换 BatchNorm 只需改这里
def norm2d(num_feats):
    return nn.InstanceNorm2d(num_feats, affine=True)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        # 1x1 -> 3x3 -> 1x1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = norm2d(planes)

        # 注意：降采样放到 3x3 这层，通过 stride 控制
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = norm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = norm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 需要时匹配通道/步幅
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet50_CIFAR(nn.Module):
    """ResNet-50 for CIFAR (32x32):
       - stem: 3x3 stride=1，无 maxpool
       - stages: [3, 4, 6, 3] 个 bottleneck，首个 block 做降采样
       - 归一化：InstanceNorm2d(affine=True)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.inplanes = 64

        # stem（保持你原来的 3x3 / stride=1 / padding=1）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = norm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        # ResNet-50 配置：每个 stage 的 block 数
        layers = [3, 4, 6, 3]

        # stage2 对应输出通道 256（planes=64, expansion=4）
        self.layer1 = self._make_layer(planes=64,  blocks=layers[0], stride=1)  # 32x32
        # stage3: 输出 512（planes=128）
        self.layer2 = self._make_layer(planes=128, blocks=layers[1], stride=2)  # 16x16
        # stage4: 输出 1024（planes=256）
        self.layer3 = self._make_layer(planes=256, blocks=layers[2], stride=2)  # 8x8
        # stage5: 输出 2048（planes=512）
        self.layer4 = self._make_layer(planes=512, blocks=layers[3], stride=2)  # 4x4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, planes, blocks, stride):
        """构建一个 stage：
           - 第一个 bottleneck 可能需要 downsample（步幅=stride）
           - 其余 bottleneck 步幅=1
        """
        downsample = None
        outplanes = planes * Bottleneck.expansion
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                norm2d(outplanes),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, stride=1, downsample=None))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # stages
        x = self.layer1(x)  # 64 -> 256
        x = self.layer2(x)  # 256 -> 512，/2
        x = self.layer3(x)  # 512 -> 1024，/2
        x = self.layer4(x)  # 1024 -> 2048，/2

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# if __name__ == "__main__":
#     net = ResNet50_CIFAR(num_classes=10)
#     x = torch.randn(4, 3, 32, 32)
#     y = net(x)
#     print(y.shape)  # torch.Size([4, 10])
