# 11.12
# 一步一步来吧。这里准备逐层验证一下double backward 的精度问题。
# 目前准备把conv的一阶导二阶导数 函数先写出来。之后放在forward里面直接调用。


import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import torchvision.transforms as transforms
import torch
from typing import Optional, Sequence, Tuple

import torch
from functools import reduce
from operator import mul


import triton
import triton.language as tl
import time 
import random
import numpy as np
import os
# from my_linear_double_backward import linear_double_backward

def load_state_dict_by_position(model, pretrained_state_dict):
    model_state_dict = model.state_dict()
    new_state_dict = {}
    # 取出当前模型的参数名字和值（有顺序）
    model_items = list(model_state_dict.items())
    pretrained_items = list(pretrained_state_dict.items())
    assert len(model_items) == len(pretrained_items), \
        f"参数数量不一致：当前模型有 {len(model_items)} 个参数，预训练模型有 {len(pretrained_items)} 个参数"

    for (model_key, _), (_, pretrained_val) in zip(model_items, pretrained_items):
        new_state_dict[model_key] = pretrained_val

    model.load_state_dict(new_state_dict)

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭自动优化，确保计算确定性
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 保证 CUDA 计算稳定（仅对 PyTorch 1.8+ 有效）



#########################
# 一阶导数：

def crossEntropy_backward( logits, target):
    N = target.shape[0]
    softmax = F.softmax(logits, dim=1)
    one_hot = torch.zeros_like(logits)
    one_hot[range(N), target] = 1.0
    grad_output = (softmax - one_hot) / N
    return grad_output
    # N, C = logits.shape
    # # 1. Compute log_softmax
    # log_probs = F.log_softmax(logits, dim=1)
    # # 2. Compute grad of NLLLoss (mean reduction)
    # grad = torch.exp(log_probs)  # shape: (N, C)
    # grad[range(N), target] -= 1
    # grad = grad / N
    # return grad
    
def avgPool_bwd(
    x,
    grad_output,
    kernel_size=[2, 2],
    stride=[2, 2],
    padding=[0, 0],
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    # TODO:  精度不太一样找不到原因。目前只能认为是累加误差。
    return torch.ops.aten.avg_pool2d_backward(grad_output,x,kernel_size,stride, padding,ceil_mode,count_include_pad, divisor_override)


def linear_bwd( x, weight, grad_output):
    grad_x = grad_output @ weight                     # (N, in_features)
    grad_weight = grad_output.t() @ x                # (out_features, in_features)
    grad_bias = grad_output.sum(0)                      # (out_features,)

    return grad_x, grad_weight, grad_bias

def conv_bwd( x, weight,grad_output, stride=1, padding=1, dilation=1, groups=1):
    input_shape = x.shape
    weight_size = weight.shape
    grad_x = torch.nn.grad.conv2d_input(input_shape, weight, grad_output, stride, padding, dilation, groups)
    grad_weight = torch.nn.grad.conv2d_weight(x, weight_size, grad_output, stride, padding, dilation, groups)
    grad_bias = grad_output.sum(0)

    return grad_x, grad_weight, grad_bias


#########################
# 二阶导数：


def avgPool_double_bwd(
    grad_grad_input,
    kernel_size=[2, 2],
    stride=[2, 2],
    padding=[0, 0],
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    """
    grad_grad_input: d^2 L / d (input) d(...)
    返回: grad_grad_output（对应 avg_pool 的输出位置），形状 [N,C,OH,OW]
    只要 grad_grad_input 在 CUDA 上，这里全程走 GPU。
    """
    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    else:
        kh, kw = kernel_size

    if isinstance(stride, int):
        sh = sw = stride
    else:
        sh, sw = stride

    if isinstance(padding, int):
        ph = pw = padding
    else:
        ph, pw = padding

    N, C, H, W = grad_grad_input.shape
    device = grad_grad_input.device
    dtype = grad_grad_input.dtype

    # 1) 先用一个 dummy 输入算出 OH, OW（这样 ceil_mode 等细节由 avg_pool2d 自己处理）
    with torch.no_grad():
        dummy = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
        y_dummy = F.avg_pool2d(
            dummy,
            kernel_size=(kh, kw),
            stride=(sh, sw),
            padding=(ph, pw),
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
        OH, OW = y_dummy.shape[2], y_dummy.shape[3]

    # 2) 构造 divisor map（和 backward 的一样）
    if divisor_override is not None:
        div = torch.full((1, 1, OH, OW), float(divisor_override), device=device, dtype=dtype)
    else:
        if count_include_pad:
            div = torch.full((1, 1, OH, OW), float(kh * kw), device=device, dtype=dtype)
        else:
            oh_idx = torch.arange(OH, device=device)
            ow_idx = torch.arange(OW, device=device)

            hstart = oh_idx * sh - ph
            hend   = torch.clamp(hstart + kh, max=H)
            hstart_clamped = torch.clamp(hstart, min=0)
            eff_h = (hend - hstart_clamped).clamp(min=0)

            wstart = ow_idx * sw - pw
            wend   = torch.clamp(wstart + kw, max=W)
            wstart_clamped = torch.clamp(wstart, min=0)
            eff_w = (wend - wstart_clamped).clamp(min=0)

            div_hw = eff_h[:, None] * eff_w[None, :]
            div_hw = div_hw.clamp(min=1)

            div = div_hw.unsqueeze(0).unsqueeze(0).to(dtype=dtype)

    # 3) 计算 conv_transpose2d 的完整输出尺寸，用 dummy 走一遍
    weight = torch.ones((C, 1, kh, kw), device=device, dtype=dtype)
    with torch.no_grad():
        dummy_grad_scaled = torch.zeros(N, C, OH, OW, device=device, dtype=dtype)
        full_out = F.conv_transpose2d(
            dummy_grad_scaled,
            weight,
            bias=None,
            stride=(sh, sw),
            padding=(ph, pw),
            groups=C,
        )
        full_H, full_W = full_out.shape[2], full_out.shape[3]

    # 4) 反裁剪：把 grad_grad_input 填回到“大” tensor 里
    grad_grad_input_full = torch.zeros(
        (N, C, full_H, full_W), device=device, dtype=dtype
    )
    grad_grad_input_full[:, :, :H, :W] = grad_grad_input

    # 5) conv_transpose2d 对输入的梯度是 conv2d
    #    对 weight=ones 的情况，就是用同一个 ones kernel 做 conv2d
    grad_S = F.conv2d(
        grad_grad_input_full,
        weight,
        bias=None,
        stride=(sh, sw),
        padding=(ph, pw),
        groups=C,
    )  # 形状 [N,C,OH,OW]

    # 6) 最后一步：对应 forward 里的 `/ div`，对 G 的梯度 = grad_S / div
    grad_grad_output = grad_S / div  # broadcast 到 [N,C,OH,OW]

    return grad_grad_output


def linear_double_bwd(
    # grads: Sequence[Optional[torch.Tensor]],  # [grad_grad_input, grad_grad_weight, grad_grad_bias]
    ggi: torch.Tensor,
    ggw: torch.Tensor,
    ggb: torch.Tensor,
    grad_output: torch.Tensor,               # 一阶 backward 里的 grad_output, shape [..., out_features]
    weight: torch.Tensor,                    # forward 的 weight, shape [out_features, in_features]
    x: torch.Tensor,                         # forward 的 input, shape [..., in_features]
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

    # ggi, ggw, ggb = grads    # grads[0], grads[1], grads[2]

    # 这里直接用 weight 推出 in/out 维度，避免靠 dim 猜
    out_features, in_features = weight.shape

    # -------- 把所有有关的张量 flatten 到 [B, *] --------
    # B = 所有 batch 维的乘积，方便统一用 matmul
    x_flat = x.reshape(-1, in_features)                    # [B, in]
    go_flat = grad_output.reshape(-1, out_features)        # [B, out]

    ggi_flat = None
    if ggi is not None:
        ggi_flat = ggi.reshape(-1, in_features)            # [B, in]

    ggw_2d = None
    if ggw is not None:
        # 线性层的 grad_grad_weight 应该就是 [out, in]，这里强行 reshape 一下更保险
        ggw_2d = ggw.reshape(out_features, in_features)    # [out, in]

    ggb_flat = None
    if ggb is not None:
        # bias 方向的高阶梯度，最后一维必须是 out_features
        ggb_ = ggb
        if ggb_.dim() == 1:
            ggb_ = ggb_.unsqueeze(0)                       # [1, out]
        last_dim = ggb_.shape[-1]
        assert last_dim == out_features, \
            f"ggb last dim ({last_dim}) != out_features ({out_features})"
        ggb_flat = ggb_.reshape(-1, out_features)          # [B_ggb, out]
        # 如果只给了一份，就 broadcast 到所有 batch
        if ggb_flat.shape[0] == 1:
            ggb_flat = ggb_flat.expand_as(go_flat)         # [B, out]
        else:
            assert ggb_flat.shape[0] == go_flat.shape[0], \
                f"ggb batch dim {ggb_flat.shape[0]} != grad_output batch dim {go_flat.shape[0]}"

    # -------- 计算 dx2, dw2, dgradout2（全部在 flat 空间）--------
    dx2_flat: Optional[torch.Tensor] = None
    dw2:     Optional[torch.Tensor] = None
    dgo2_flat: Optional[torch.Tensor] = None

    # 1) dx2_flat = grad_output @ ggw
    if ggw_2d is not None:
        # [B, out] @ [out, in] = [B, in]
        dx2_flat = go_flat.matmul(ggw_2d)

    # 2) dw2 = grad_output^T @ ggi
    if ggi_flat is not None:
        # [B, out]^T @ [B, in] = [out, in]
        dw2 = go_flat.transpose(0, 1).matmul(ggi_flat)

    # 3) dgo2_flat = ggi @ W^T + x @ ggw^T + ggb
    need = (ggi_flat is not None) or (ggw_2d is not None) or (ggb_flat is not None)
    if need:
        dgo2_flat = torch.zeros_like(go_flat)              # [B, out]

        if ggi_flat is not None:
            # [B, in] @ [in, out] = [B, out]
            dgo2_flat = dgo2_flat + ggi_flat.matmul(weight.transpose(0, 1))

        if ggw_2d is not None:
            # [B, in] @ [in, out] = [B, out]
            dgo2_flat = dgo2_flat + x_flat.matmul(ggw_2d.transpose(0, 1))

        if ggb_flat is not None:
            dgo2_flat = dgo2_flat + ggb_flat

    # -------- 把 flat 结果 reshape 回原状 --------
    dx2 = dx2_flat.reshape_as(x) if dx2_flat is not None else None
    dgrad_out = dgo2_flat.reshape_as(grad_output) if dgo2_flat is not None else None

    # 返回顺序按你现在的写法：grad_input, dw_weight, grad_bias_like
    # 注意：第三个其实是 d(grad_output)，不是 d(bias)，只是你现在变量名叫 grad_bias
    # gg0 gI gw
    return dgrad_out,dx2, dw2,



def conv_double_bwd(ggI_opt, ggW_r_opt, ggb_opt, gO_r, weight_r, input, stride_=1, padding_=1, groups_=1, output_mask=None):
    stride_ = [1,1]
    padding_        = [1, 1]
    dilation_       = [1, 1]
    transposed_     = False
    output_padding_ = [0, 0]
    groups_         = 1
    output_mask     = [True, True, True]   # 返回 ggO, gI, gW
    # ggI_opt输入梯度的梯度； ggW_r_opt： w的二阶梯度？； ggb_opt：b的二阶梯度？； gO_r一阶导数的输入gradoutput； input 是conv的input？
    # 返回：输出tensor（grad_output)的二阶梯度。 gI：x的梯度累计； gw weight的梯度累计； bias 应该没有梯度。
    # const std::optional<Tensor>& ggI_opt, const std::optional<Tensor>& ggW_r_opt, const std::optional<Tensor>& ggb_opt,
    # const Tensor& gO_r, const Tensor& weight_r, const Tensor& input,
    # IntArrayRef stride_, IntArrayRef padding_, IntArrayRef dilation_,
    # bool transposed_, IntArrayRef output_padding_, int64_t groups_,
    # std::array<bool, 3> output_mask
    ggO, gI, gW = torch.ops.aten._convolution_double_backward(ggI_opt,ggW_r_opt, ggb_opt, gO_r, weight_r, input,     stride_,         # [1, 1]
    padding_,        # [1, 1]
    dilation_,       # [1, 1]
    transposed_,     # False
    output_padding_, # [0, 0]
    groups_,         # 1
    output_mask      # [True, True, True]
    )
    return ggO, gI, gW



class Myconv(nn.Module):
    def __init__(self, net_width):
        super(Myconv, self).__init__()
        self.net_width = net_width
        self.halfIMG = 16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=net_width, kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(net_width * self.halfIMG * self.halfIMG, 10)

    def forward(self, x_conv, criterion, target):
        # TODO: 这里要用state less。 分别做一阶和二阶导数。
        # TODO: 符号需要统一一下； torch 的那套规则是 I*W = O ，所有命名都是根据fwd来的。感觉用这个比较好。
        # 比如grad_output 就不好

        x_pool = self.conv1(x_conv)          # N x net_width x 32 x 32
        x_lin = self.pool1(x_pool)
        x_lin = x_lin.view(x_lin.size(0), -1) # Flatten to N x (net_width*16*16)
        x_out = self.linear(x_lin)    # N x 10

        loss = criterion(x_out, target)  # compute loss
        print("----CELOSS-----", loss.item())

        # 1st order BWD
        # grad_output = crossEntropy_backward(output, target)
        dx_out = torch.torch.autograd.grad(loss, x_out, create_graph=True)[0] # criterion bwd

        dx_lin, d_lin_weight, d_lin_bias = linear_bwd(x_lin, self.linear.weight, grad_output=dx_out)
        dx_lin = dx_lin.view(-1, self.net_width, self.halfIMG,self.halfIMG)
        dx_pool = avgPool_bwd( x_pool, grad_output= dx_lin )
        dx_conv, d_conv_weight, d_conv_bias = conv_bwd( x_conv, self.conv1.weight,grad_output=dx_pool)

        grad_loss = d_lin_weight.sum() + d_lin_bias.sum() + d_conv_weight.sum() + d_conv_bias.sum()
        print("----GRANDLOSS-----", grad_loss.item())

        # 2en order BWD
        # 2.1 loss compute， 需要对gradloss func求梯度。 因为现在的sum是累加，dloss/dweight = 1 所以传入的梯度是。。？
        # 这里好像还必须得手动算。因为One of the differentiated Tensors does not require grad ？ 
        # ddconw, ddconvb, ddlinw, ddlinb = torch.autograd.grad(grad_loss, [grad_lin_weight,d_lin_bias,grad_conv_weight, grad_conv_bias])
        ddcon_w = torch.ones_like(self.conv1.weight).cuda()
        ddconv_b = torch.ones_like(self.conv1.bias).cuda()
        ddlin_w = torch.ones_like(self.linear.weight).cuda()
        ddlin_b = torch.ones_like(self.linear.bias).cuda()
        ddx_conv = torch.zeros_like(x_conv).cuda()

        # 2.2 double BWD
        # 另外注意这里的dwconv 和 grad_conv_weight 可不一样。虽然都是一阶的。
        # 二阶的这两个kernel都生成 ggO, gI, gW。这个gI不确定要不要累加到前面去。
        # TODO: 这个dx_lin对吗？
        ddx_pool, dxconv_d2, dconvw_d2 = conv_double_bwd(ddx_conv, ddcon_w, ddconv_b, dx_pool, self.conv1.weight,x_conv )
        # ddx_out, dxlin_d2, dlinw_d2 = linear_double_bwd([ddx_lin, ddlin_w, ddlin_b], x_out, self.linear.weight, x_lin )
        ddx_lin = avgPool_double_bwd(ddx_pool)

        ddx_out, dxlin_d2, dlinw_d2 = linear_double_bwd(ddx_lin, ddlin_w, ddlin_b, dx_out, self.linear.weight, x_lin )
        # (grad_output, ddx_lin, ddlinw, ddlinb, x_lin, self.linear.weight,self.linear.bias)
        # 2.3 re do 1st BWD
        # 这里遇到了一个命名问题。。 主要是涉及到要不要合并这几个同名梯度
        dx_out_d1 = torch.torch.autograd.grad(dx_out, x_out, grad_outputs=ddx_out)[0] # criterion bwd
        dx_lin_d1, d_lin_weight_d1, d_lin_bias_d1 = linear_bwd( x_lin, self.linear.weight, grad_output=dx_out_d1)
        # 
        dx_lin_d1 += dxlin_d2 
        dx_lin_d1 = dx_lin_d1.view(-1, self.net_width, self.halfIMG,self.halfIMG) 

        dx_pool_d1 = avgPool_bwd( x_pool, grad_output= dx_lin_d1 )
        dx_conv_d1, d_conv_weight_d1, d_conv_bias_d1 = conv_bwd(x_conv, self.conv1.weight, grad_output=dx_pool_d1)

        return dx_conv_d1 + dxconv_d2
    

class Conv_original(nn.Module):
    def __init__(self, net_width=8):
        super(Conv_original, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=net_width, kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(net_width * 16 * 16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.linear(x)
        return x



###################################
###################################
flag = 'myconv'
flag = 'conv'
Fuse = 2
batch_size = 1024
set_random_seed() # 仅仅在ACC test的时候使用。会严重影响性能。 
###################################
###################################



if __name__ == "__main__":
    print(flag)

    model1 = Conv_original(32).to("cuda")
    model2 = Myconv(32).to("cuda")
    if flag =='conv':
        model = model1
    else:
        model = model2

    transform = transforms.ToTensor()
    cifar10 = torchvision.datasets.CIFAR10(root='/scratch/yguo25/files/mtt-distillation/data', train=True, download=True, transform=transform)
    img, label = cifar10[0]
    x = img.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # shape: [batch_size, 3, 32, 32]
    x = x.clone().detach().to("cuda").requires_grad_(True)
    target = torch.tensor([label] * batch_size, device="cuda")  # shape: [batch_size]


    # torch.save(model.state_dict(), 'model_test_doublebwd_pool.pt')
    # exit()
    # # model.load_state_dict(torch.load('model_test5.pt'), strict = False)

    pretrained_dict = torch.load("model_test_doublebwd_pool.pt")
    load_state_dict_by_position(model, pretrained_dict)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([x], lr=1e-1)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start = time.time()

    for step in range(1):
        optimizer.zero_grad()

        if flag =='conv':
            output = model(x)  # forward
            loss = criterion(output, target)  # compute loss
            print("----CELOSS-----", loss.item())

            # print(loss.item())
            dw = torch.torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
            # weight = list(model.parameters()) 
            # weight = [(1- p + g).sum() for p, g in zip(weight, dw)]
            weight = [d.sum() for d in dw]
            grad_loss = sum(weight)
            print("----GRANDLOSS-----", grad_loss.item())
            grad_loss.backward()  
            print("----GRAD-----", x.grad.sum().item())
        else: 
            output = model(x,criterion, target)  # forward+1stbwd+weight op
            print("----GRAD-----", output.sum().item())
            
        optimizer.step()  # update x

    end = time.time()

    print("当前显存使用:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("峰值显存使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    print("时间占用：", end-start)