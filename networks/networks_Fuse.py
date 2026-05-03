# 4.27
# 现在的model 应该都可以把Fuse 写好了。唯一的区别就是 flex fuse 有没有做以及有没有封装到model 里面。所以现在准备重新整理一下这些网络。这里希望可以包含Conv3 、 Resnet18（resnet18test file 复制）
# 这里是完整的大模型, 来源主要有两个： stateless & stateless_basic block 。两个分别是一个封装后的模块，一个是原始的函数接口。



import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import torchvision.transforms as transforms
import torch

# from networks.networks_stacked import LinearStacked_2 # NOTE: 这里取消注释了。因为Fuse->stacked->stacked_basicblock 太长了。不好。
from networks.networks_basicblock_fused3 import batchNorm2d_backward, batchnorm_double_backwards_fn, batchnorm_double_backwards_fn_new
from networks.networks_basicblock_fused3 import instanceNorm_backward ,instanceNorm_double_backwards_fn, instance_norm_backward_triton,instanceNorm_double_backwards_triton
from networks.networks_stateless_basicblock import linear_bwd, conv_bwd, insNormNRelu_bwd, \
linear_double_bwd, conv_double_bwd, insNormNRelu_double_bwd, avgPool_bwd, crossEntropy_bwd, \
    avgPool_double_bwd,bmm_bwd, linerFused_bwd, linearFused_double_bwd,crossEntropy_double_bwd, \
    adaptivepooling_bwd, adaptivepooling_double_bwd
from networks.utils import clear_tensorlists
from networks.networks_stateless import BasicBlock_double_bwd,BasicBlock_bwd,BasicBlock_bwd2_1, conv_norm_relu_bwd
from networks.networks_basicblock_fused3 import Fst_Order_NormActive # fuse+基本优化


class LinearStacked_2(nn.Module):
    # batch* fusion * channel * WH。 
    def __init__(self ,in_features, out_features, Fuse):
        super(LinearStacked_2, self).__init__()
        self.Fuse = Fuse
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(Fuse* out_features,in_features))
        self.bias = torch.nn.Parameter(torch.randn(self.Fuse* out_features))

    def forward(self, x):
        """
        x目前仅支持二维输入： B* STK * In。 B和stk可以view 在一起。 weight  STK*IN*OUT 
        """
        x = x.view(-1,self.Fuse, self.in_features)
        # TODO: 这里输入如果是BAD/（而不是ABD）的话，可以得到is_contiguous 的结果。那就很简单只要调整target即可
        # 现在是BAD，意味着Fusion，batch的排序，B到了第一位
        x = torch.einsum("abc,bcd->bad",x,self.weight.view(self.Fuse,self.out_features,self.in_features).transpose(-1, -2)) 
        x = x+self.bias.view(self.Fuse,1 ,self.out_features)
        x = x.squeeze(0) #加一个squeeze，为了适配在Fuse=1的时候的模型。
        return x


class NormActive(nn.Module):
    def __init__(self, channel_num, affine=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn([channel_num]))
        self.bias = nn.Parameter(torch.randn([channel_num]))
    def forward(self, input):
        out = Fst_Order_NormActive.apply(input, self.weight, self.bias)
        return out



class Conv_Flexfuse(nn.Module):
    def __init__(self, channel=3, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='maxpooling', im_size = (32,32), Fuse=2):
        super(Conv_Flexfuse, self).__init__()
        self.Fuse = Fuse
        self.conv1 = nn.Conv2d(in_channels=channel*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        self.norm1 = nn.InstanceNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        # self.norm1 = nn.GroupNorm(net_width*Fuse,net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=net_width*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        self.norm2 = nn.InstanceNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        # self.norm2 = nn.GroupNorm(net_width*Fuse,net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=net_width*Fuse, out_channels=net_width*Fuse, kernel_size=3, padding=1, groups=Fuse)  #conv是N，G，C。其中G替换成FUse
        self.norm3 = nn.InstanceNorm2d(net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        # self.norm3 = nn.GroupNorm(net_width*Fuse,net_width*Fuse, affine=True) #BN在channel上单独计算，所以目前不用管。
        self.pool3 = nn.AvgPool2d(kernel_size=2)
        self.linear = LinearStacked_2(net_width * 4 * 4, num_classes,Fuse )
        self.net_width= net_width

    def forward(self, x_conv1):
        x_conv1 = x_conv1.view(-1,self.Fuse*3 ,32,32)        # 10, 256, 4,4   -> 20, 2048
        x_norm1 = self.conv1(x_conv1)          
        x_pool1 = F.relu(self.norm1(x_norm1)    )
        x_conv2 = self.pool1(x_pool1)
        x_norm2 = self.conv2(x_conv2)          
        x_pool2 = F.relu(self.norm2(x_norm2) )   
        x_conv3 = self.pool2(x_pool2)
        x_norm3 = self.conv3(x_conv3)          
        x_pool3 = F.relu(self.norm3(x_norm3))    
        x_lin  = self.pool3(x_pool3)
        x_out = self.linear(x_lin)    # N x 10
        x_out = x_out.view(-1,10)
        return  x_conv1,x_norm1, x_pool1,x_conv2,x_norm2, x_pool2,x_conv3,x_norm3, x_pool3, x_lin,x_out



class BasicBlock_Flex_fuse(nn.Module):
    def __init__(self, in_channels, out_channels,   net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='maxpooling', im_size = (32,32), stride=1, Fuse = 1):
        # 注意一下这个basic block已经把 fuse隔离在外面了。内部init的时候再扩充in out channel
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels*Fuse, out_channels*Fuse, 3, stride, 1, bias=False, groups=Fuse)
        # self.bn1 = nn.InstanceNorm2d(out_channels*Fuse, affine=True)
        self.bn1 = nn.GroupNorm(out_channels*Fuse, out_channels*Fuse, affine=True) 
        self.conv2 = nn.Conv2d(out_channels*Fuse, out_channels*Fuse, 3, 1, 1, bias=False, groups=Fuse)
        # self.bn2 = nn.InstanceNorm2d(out_channels*Fuse, affine=True)
        self.bn2 = nn.GroupNorm(out_channels*Fuse, out_channels*Fuse, affine=True) 
        if stride != 1 or in_channels != out_channels:
            self.convsc = nn.Conv2d(in_channels*Fuse, out_channels*Fuse, 1, stride, bias=False, groups=Fuse )
            # self.bnsc = nn.InstanceNorm2d(out_channels*Fuse, affine=True)
            self.bnsc = nn.GroupNorm(out_channels*Fuse, out_channels*Fuse, affine=True) 
        else:
            self.convsc = None
            self.bnsc = None
    def forward(self, x):
        identity = x
        x_bnsc = None
        if self.convsc is not None:
            x_bnsc = self.convsc(x)
            identity = self.bnsc(x_bnsc)
        x_bn1 = self.conv1(x)
        x_conv2 = F.relu(self.bn1(x_bn1), inplace=True)
        x_bn2 = self.conv2(x_conv2)
        out = self.bn2(x_bn2)
        out = F.relu(out + identity, inplace=True)
        activates = {
            "x_conv1": x,
            "x_bn1": x_bn1,
            "x_conv2": x_conv2,
            "x_bn2": x_bn2,
            "x_bnsc": x_bnsc,
            "x_out": out,
        }
        return out, activates

class ResNet18_FlexFuse(nn.Module):
    def __init__(self, channel=3, num_classes=10, Fuse=1):
        super().__init__()
        self.Fuse = Fuse
        self.num_classes = num_classes
        blk_in_ch = 64   # 这里仍然是逻辑通道数
        cfg = [ (blk_in_ch,  2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2), ]
        self.conv = nn.Conv2d(channel * Fuse, blk_in_ch * Fuse, kernel_size=3, stride=1, padding=1, bias=False, groups=Fuse )
        # self.bn = nn.InstanceNorm2d(blk_in_ch * Fuse, affine=True)
        self.bn = nn.GroupNorm(blk_in_ch * Fuse, blk_in_ch * Fuse, affine=True) 
        self.stages = nn.ModuleList()
        for stage_id, (out_ch, num_blocks, first_stride) in enumerate(cfg):
            stage = nn.ModuleList()
            for block_id in range(num_blocks):
                stride = first_stride if block_id == 0 else 1
                stage.append(BasicBlock_Flex_fuse(blk_in_ch, out_ch, stride=stride, Fuse=Fuse))
                blk_in_ch = out_ch
            self.stages.append(stage)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = LinearStacked_2(512, num_classes, Fuse)
        # self.fc = nn.Linear(512, num_classes)

    def forward(self, x_conv):
        tape = { "stem": {}, "blocks": [], "head": {} }
        x_bn = self.conv(x_conv)
        x_block = self.bn(x_bn)
        x_block = F.relu(x_block, inplace=True)
        tape["stem"] = {  "x_conv": x_conv, "x_bn": x_bn,  "x_block": x_block}
        h = x_block
        for stage_id, stage in enumerate(self.stages):
            for block_id, blk in enumerate(stage):
                h, activates = blk(h)
                activates["stage_id"] = stage_id
                activates["block_id"] = block_id
                tape["blocks"].append(activates)
        x_pool = h
        x_fc = self.pool(x_pool)
        x_fc = torch.flatten(x_fc, 1)
        x_out = self.fc(x_fc)
        x_out = x_out.view(-1,self.num_classes)
        tape["head"] = { "x_pool": x_pool, "x_fc": x_fc, "x_out": x_out, }
        return x_out, tape
    
    def get_flat_blocks(self):
        return [blk for stage in self.stages for blk in stage]
    
    def get_block_weights(self, blk):
        bnscw = blk.bnsc.weight if blk.bnsc is not None else None
        bnscb = blk.bnsc.bias if blk.bnsc is not None else None
        convscw = blk.convsc.weight if blk.convsc is not None else None
        weights = {
            "conv1w": blk.conv1.weight,
            "bn1w": blk.bn1.weight,
            "bn1b": blk.bn1.bias,
            "conv2w": blk.conv2.weight,
            "bn2w": blk.bn2.weight,
            "bn2b": blk.bn2.bias,
            "convscw": convscw,
            "bnscw": bnscw,
            "bnscb": bnscb,
        }
        return weights

    def init_dd_block_weights(self, blk):
        convscw = blk.convsc.weight if blk.convsc is not None else None
        bnscw = blk.bnsc.weight if blk.bnsc is not None else None
        bnscb = blk.bnsc.bias if blk.bnsc is not None else None
        dd_weights = {
            "ddconv1w": torch.ones_like(blk.conv1.weight),
            "ddbn1w": torch.ones_like(blk.bn1.weight),
            "ddbn1b": torch.ones_like(blk.bn1.bias),
            "ddconv2w": torch.ones_like(blk.conv2.weight),
            "ddbn2w": torch.ones_like(blk.bn2.weight),
            "ddbn2b": torch.ones_like(blk.bn2.bias),
            "ddconvscw": torch.ones_like(convscw) if convscw is not None else None,
            "ddbnscw": torch.ones_like(bnscw) if bnscw is not None else None,
            "ddbnscb": torch.ones_like(bnscb) if bnscb is not None else None,
        }
        return dd_weights
    
    def collect_ordered_grads(self, dconvw, dbnw, dbnb, d_weights_list, dfcw, dfcb):
        grads_all = [dconvw, dbnw, dbnb]

        block_keys_main = ["dconv1w", "dbn1w", "dbn1b", "dconv2w", "dbn2w", "dbn2b"]
        block_keys_sc = ["dconvscw", "dbnscw", "dbnscb"]

        for d_weights_i in d_weights_list:
            if d_weights_i is None:
                continue

            for k in block_keys_main:
                grads_all.append(d_weights_i[k])

            for k in block_keys_sc:
                v = d_weights_i.get(k, None)
                if v is not None:
                    grads_all.append(v)

        grads_all.extend([dfcw, dfcb])
        return grads_all

        
    def run_double_bwd(
        self,
        tape,
        d_activates_list,
        dd_weights_list,
        d_stem_tensors,
        dd_stem_tensors,
        Fuse = 1,
    ):
        flat_blocks = self.get_flat_blocks()
        stem = tape["stem"]
        head = tape["head"]
        x = stem["x_conv"]
        x_bn = stem["x_bn"]
        x_block = stem["x_block"]
        x_pool = head.pop("x_pool")
        x_fc = head.pop("x_fc")
        x_out = head.pop("x_out")

        clear_tensorlists(stem, head)
        tape["stem"] = None
        tape["head"] = None
        dx_bn = d_stem_tensors.pop("dx_bn")
        dx_block = d_stem_tensors.pop("dx_block")
        dx_out = d_stem_tensors.pop("dx_out")
        ddx_conv = dd_stem_tensors.pop("ddx_conv")
        ddconvw = dd_stem_tensors.pop("ddconvw")
        ddbnw = dd_stem_tensors.pop("ddbnw")
        ddbnb = dd_stem_tensors.pop("ddbnb")
        ddfcw = dd_stem_tensors.pop("ddfcw")
        ddfcb = dd_stem_tensors.pop("ddfcb")
        ddx_bn, dx_conv_d2, _ = conv_double_bwd( ddx_conv, ddconvw, None,dx_bn, self.conv.weight, x, groups_=Fuse)
        del dx_bn, ddx_conv, ddconvw
        # TODO: instance norm好像根本就不需要考虑fuse的问题。bn再说吧。
        dx_bn_d2, _, dd_cur = instanceNorm_double_backwards_fn( x_bn, self.bn.weight, None, ddx_bn, ddbnw, ddbnb, dx_block)
        del dx_block, ddx_bn,ddbnw, ddbnb
        dd_cur[x_block <= 0] = 0
        for i in range(len(flat_blocks)):
            blk = flat_blocks[i]
            activates_i = tape["blocks"][i]
            d_activates_i = d_activates_list[i]
            weights_i = self.get_block_weights(blk)
            dd_weights_i = dd_weights_list[i]
            dd_cur, _ = BasicBlock_double_bwd(
                activates_i, d_activates_i, weights_i, dd_weights_i, ddgrad_in=dd_cur,
                SCstride=blk.conv1.stride[0], Fuse= Fuse )
            clear_tensorlists(dd_weights_i)
            dd_weights_list[i] = None
            del activates_i, d_activates_i, weights_i, dd_weights_i

        ddx_pool = dd_cur
        del dd_cur
        # head double backward
        ddx_lin = adaptivepooling_double_bwd(ddx_pool)
        del ddx_pool
        ddx_out, dx_lin_d2, _ = linearFused_double_bwd( x_fc, self.fc.weight, dx_out, ddx_lin, ddfcw, ddfcb, Fuse )
        del dx_out, ddx_lin,ddfcw,ddfcb
        dx_out_d1 = crossEntropy_double_bwd(x_out, ddx_out, Fuse)
        del ddx_out,x_out
        dx_lin_d1, _, _ = linerFused_bwd( x_fc, self.fc.weight, grad_output=dx_out_d1, Fuse=Fuse)
        del dx_out_d1,x_fc
        dx_lin_d1 += dx_lin_d2
        del dx_lin_d2
        dx_lin_d1 = dx_lin_d1.view(x_pool.size(0), x_pool.size(1), 1, 1)
        g = adaptivepooling_bwd(x_pool, grad_output=dx_lin_d1)
        del dx_lin_d1, x_pool

        # blocks: bwd2_1, backward direction
        for i in reversed(range(len(flat_blocks))):
            blk = flat_blocks[i]
            activates_i = tape["blocks"][i]
            weights_i = self.get_block_weights(blk)
            d2_activates_i = d_activates_list[i]
            if weights_i["convscw"] is None:
                d2_activates_i.setdefault("dx_bnsc_d2", None)
            g = BasicBlock_bwd2_1( activates_i, weights_i, d2_activates_i,grad_output=g, SCstride=blk.conv1.stride[0], Fuse = Fuse)
            clear_tensorlists(activates_i, d2_activates_i, weights_i)
            tape["blocks"][i] = None
            d_activates_list[i] = None
            del activates_i, d2_activates_i, weights_i
        dx_block_d1 = g
        del g
        dx_block_d1[x_block <= 0] = 0
        del x_block
        dx_bn_d1, _, _, _, _ = instanceNorm_backward( x_bn, self.bn.weight, grad_output=dx_block_d1)
        del dx_block_d1
        dx_bn_d1 += dx_bn_d2
        del dx_bn_d2
        dx_conv, _, _ = conv_bwd( x, self.conv.weight, grad_output=dx_bn_d1,groups=Fuse)
        del dx_bn_d1
        dx_conv += dx_conv_d2

        return dx_conv


    def run_first_bwd(self, tape, target, Fuse = 1):
        flat_blocks = self.get_flat_blocks()
        # 这里只读，不 pop。double-bwd 还要继续用 tape
        stem = tape["stem"]
        head = tape["head"]
        x_conv  = stem["x_conv"]
        x_bn    = stem["x_bn"]
        x_block = stem["x_block"]
        x_pool = head["x_pool"]
        x_fc   = head["x_fc"]
        x_out  = head["x_out"]
        d_activates_list = [None] * len(flat_blocks)
        d_weights_list   = [None] * len(flat_blocks)

        dx_out = crossEntropy_bwd(x_out, target, Fuse=Fuse) 
        # dx_fc, dfcw, dfcb = linear_bwd( x_fc, self.fc.weight, grad_output=dx_out)
        dx_fc, dfcw, dfcb = linerFused_bwd( x_fc, self.fc.weight, grad_output=dx_out, Fuse= Fuse)
        dx_fc = dx_fc.view(x_pool.size(0), x_pool.size(1), 1, 1)
        g = adaptivepooling_bwd(x_pool, grad_output=dx_fc)
        del dx_fc
        for i in reversed(range(len(flat_blocks))):
            blk = flat_blocks[i]
            activates_i = tape["blocks"][i]
            weights_i = self.get_block_weights(blk)
            g, d_activates_i, d_weights_i = BasicBlock_bwd( activates_i, weights_i, grad_output=g, SCstride=blk.conv1.stride[0],Fuse=Fuse )
            d_activates_list[i] = d_activates_i
            d_weights_list[i]   = d_weights_i
            # 这里只删局部引用，不动 tape 里的本体
            del activates_i, weights_i
        dx_block = g
        del g
        dx_bn, dbnw, dbnb, dconvw = conv_norm_relu_bwd( x_conv, x_bn, x_block, self.conv.weight, self.bn.weight, grad_output=dx_block, Fuse=Fuse )
        d_stem_tensors = { "dx_bn": dx_bn, "dx_block": dx_block, "dx_out": dx_out, }

        print(d_weights_list[-1]['dbn2w'].sum().item())
        d_weights_list_all = self.collect_ordered_grads( dconvw, dbnw, dbnb, d_weights_list, dfcw, dfcb)
        return  d_stem_tensors, d_activates_list, d_weights_list, d_weights_list_all
            
    def pack_recovered_dd(self, dd_tensors_all, x):
        ptr = 0

        # ---- stem ----
        ddconvw = dd_tensors_all[ptr]; ptr += 1
        ddbnw   = dd_tensors_all[ptr]; ptr += 1
        ddbnb   = dd_tensors_all[ptr]; ptr += 1

        # ---- blocks ----
        dd_weights_list = []
        flat_blocks = self.get_flat_blocks()

        for blk in flat_blocks:
            dd_weights_i = {
                "ddconv1w": dd_tensors_all[ptr],
                "ddbn1w":   dd_tensors_all[ptr + 1],
                "ddbn1b":   dd_tensors_all[ptr + 2],
                "ddconv2w": dd_tensors_all[ptr + 3],
                "ddbn2w":   dd_tensors_all[ptr + 4],
                "ddbn2b":   dd_tensors_all[ptr + 5],
                "ddconvscw": None,
                "ddbnscw":   None,
                "ddbnscb":   None,
            }
            ptr += 6

            if blk.convsc is not None:
                dd_weights_i["ddconvscw"] = dd_tensors_all[ptr]; ptr += 1
                dd_weights_i["ddbnscw"]   = dd_tensors_all[ptr]; ptr += 1
                dd_weights_i["ddbnscb"]   = dd_tensors_all[ptr]; ptr += 1

            dd_weights_list.append(dd_weights_i)

        # ---- head ----
        ddfcw = dd_tensors_all[ptr]; ptr += 1
        ddfcb = dd_tensors_all[ptr]; ptr += 1

        assert ptr == len(dd_tensors_all), (
            f"dd tensor parse mismatch: used {ptr}, total {len(dd_tensors_all)}"
        )

        dd_work = {
            "ddconvw": ddconvw,
            "ddbnw": ddbnw,
            "ddbnb": ddbnb,
            "ddfcw": ddfcw,
            "ddfcb": ddfcb,
            # "ddx_conv": torch.zeros_like(x).cuda(), # TODO: 这里可能要改。
        }

        return dd_work, dd_weights_list