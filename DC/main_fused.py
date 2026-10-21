import os
import time
import copy
import argparse
import torch.nn.functional as F
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torchvision.utils import save_image
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils_dc import get_loops, get_dataset, get_network as _get_original_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, ViT
from reparam_module import ReparamModule
from networks.networks_Fuse import Conv_Flexfuse, ResNet18_FlexFuse, ViT_FlexFuse, VGG11_Flexfuse
from utils_flex import set_random_seed
# In this experiment Fuse refers to consecutive NETWORK CHECKPOINTS, not classes.
# The project's utils_dc.get_network currently constructs original networks.
# Keep a single local factory so the caller gets the appropriate interface.
# 新的接口，原始的get_network改为_get_original_network，
def get_network(model, channel, num_classes, im_size, Fuse=1, v_fuse=False):
    if Fuse == 1:
        if model == 'ViT':
            return ViT(img_size=im_size[0], in_channels=channel, num_classes=num_classes)
        return _get_original_network(model, channel, num_classes, im_size)
    return get_fused_network(model, channel, num_classes, im_size, Fuse=Fuse, v_fuse=v_fuse)
def get_fused_network(model, channel, num_classes, im_size, Fuse=1, v_fuse=False):
    if model == 'ConvNet':
        return Conv_Flexfuse(channel=channel, num_classes=num_classes, net_width=128, im_size=im_size, Fuse=Fuse, v_fuse=v_fuse)
    if model == 'ResNet18':
        return ResNet18_FlexFuse(channel=channel, num_classes=num_classes, Fuse=Fuse, v_fuse=v_fuse)
    if model == 'ViT':
        return ViT_FlexFuse(img_size=im_size[0], in_channels=channel, num_classes=num_classes, Fuse=Fuse, v_fuse=v_fuse)
    if model == 'VGG11':
        return VGG11_Flexfuse(channel=channel, num_classes=num_classes, Fuse=Fuse, im_size=im_size, v_fuse=v_fuse)
    raise ValueError(f'Unsupported fused-model configuration: {model}, Fuse={Fuse}')
def snapshot_params(net):
    return tuple(p.detach().clone() for p in net.parameters())
def pack_checkpoints(checkpoints):
    """Parameter-major order; within each parameter, checkpoint 0..Fuse-1."""
    if not checkpoints:
        raise ValueError('No checkpoints to pack')
    n_params = len(checkpoints[0])
    if any(len(theta) != n_params for theta in checkpoints):
        raise ValueError('Inconsistent checkpoint parameter counts')
    return torch.cat([
        torch.cat([theta[i].reshape(-1) for theta in checkpoints], dim=0)
        for i in range(n_params)
    ])
def checkpoint_mean_match_loss(gw_syn, gw_real, base_shapes, args):
    """Match each replica independently, with DC's original distance metric.
    'ours': sum cosine distances per output channel, except 1D params.
    'mse': per-parameter squared Euclidean distance.
    'cos': whole-network cosine per replica, not one cosine across Fuse.
    """
    losses = []
    for f in range(args.Fuse):
        gs = [g.reshape((args.Fuse, *shape))[f] for g, shape in zip(gw_syn, base_shapes)]
        gr = [g.reshape((args.Fuse, *shape))[f] for g, shape in zip(gw_real, base_shapes)]
        losses.append(match_loss(gs, gr, args))
    return torch.stack(losses).mean()
class FusedLogitsWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        out = self.net(x)
        return out[0] if isinstance(out, tuple) else out
    # 计算gw + loss
def manual_fused_class_gradient(fused, flat_params, base_shapes, real, syn, labels_real,
                                labels_syn, args, time_stats=None):
    """Run manual first-bwd on real/syn; VJP through syn first-bwd via double-bwd.
    Inputs to all checkpoints are identical. Result is the mean GM gradient over Fuse branches.
    """
    with torch.no_grad():
        Fuse = args.Fuse
        xr = real.repeat(1, Fuse, 1, 1)
        xs = syn.detach().repeat(1, Fuse, 1, 1)
        if args.model == 'ViT':
            yr = labels_real.repeat_interleave(Fuse)
            ys = labels_syn.repeat_interleave(Fuse)
        else:
            yr = labels_real.repeat(Fuse)
            ys = labels_syn.repeat(Fuse)
        if args.use_barrier and torch.cuda.is_available(): torch.cuda.synchronize()
        fwd_start = time.time()
        if flat_params is None:
            if args.model == 'VGG11':
                _, tape_real = fused(xr, return_tape=True)
            else:
                _, tape_real = fused(xr)
            gw_real = fused.run_first_bwd(tape=tape_real, target=yr, Fuse=Fuse)[-1]
        else:
            if args.model == 'VGG11':
                _, tape_real = fused(xr, flat_param=flat_params, return_tape=True)
            else:
                _, tape_real = fused(xr, flat_param=flat_params)
            gw_real = fused.call_with_param(flat_params, fused.module.run_first_bwd, tape=tape_real, target=yr, Fuse=Fuse)[-1]
        gw_real = [g.detach() for g in gw_real]
        del tape_real, xr
        if flat_params is None:
            if args.model == 'VGG11':
                _, tape_syn = fused(xs, return_tape=True)
            else:
                _, tape_syn = fused(xs)
            d_bwd_state, dacts, dweights, gw_syn = fused.run_first_bwd(tape=tape_syn, target=ys, Fuse=Fuse)
        else:
            if args.model == 'VGG11':
                _, tape_syn = fused(xs, flat_param=flat_params, return_tape=True)
            else:
                _, tape_syn = fused(xs, flat_param=flat_params)
            d_bwd_state, dacts, dweights, gw_syn = fused.call_with_param(flat_params, fused.module.run_first_bwd, tape=tape_syn, target=ys, Fuse=Fuse)
        del dweights
        if args.model == 'ViT':
            del d_bwd_state
            d_bwd_state = None
        if len(gw_syn) != len(base_shapes) or len(gw_real) != len(base_shapes):
            raise RuntimeError('Fused first-bwd returned unexpected gradient count')
        for i, (g, shape) in enumerate(zip(gw_syn, base_shapes)):
            expected_numel = Fuse * int(np.prod(shape))
            if g.numel() != expected_numel:
                raise RuntimeError(f'Fused first-bwd parameter {i} has {g.numel()} elements, expected {expected_numel}')
    with torch.enable_grad():
        gs_leaf = [g.detach().requires_grad_(True) for g in gw_syn]
        dist = checkpoint_mean_match_loss(gs_leaf, gw_real, base_shapes, args)
        if args.use_barrier and torch.cuda.is_available(): torch.cuda.synchronize()
        fwd_end = time.time()
        time_stats['fwd'] += fwd_end - fwd_start
        time_stats['bwd_start'] = fwd_end
        seeds = torch.autograd.grad(dist, gs_leaf, allow_unused=True)
        dd_tensors = [torch.zeros_like(g) if dd is None else dd.detach() for dd, g in zip(seeds, gw_syn)]
    with torch.no_grad():
        if flat_params is None:
            dx_fused = fused.run_double_bwd(tape=tape_syn, d_activates_list=dacts, d_bwd_state=d_bwd_state, dd_tensors_all=dd_tensors, Fuse=Fuse)
        else:
            dx_fused = fused.call_with_param(flat_params, fused.module.run_double_bwd, tape=tape_syn, d_activates_list=dacts, d_bwd_state=d_bwd_state, dd_tensors_all=dd_tensors, Fuse=Fuse)
        if dx_fused.shape != xs.shape:
            raise RuntimeError(f'Double backward input gradient shape {tuple(dx_fused.shape)} != {tuple(xs.shape)}')
        dx = dx_fused.reshape(syn.shape[0], Fuse, *syn.shape[1:]).sum(dim=1)
    return dx, dist.detach()
def network_train_once(net, optimizer_net, criterion, image_syn, label_syn, args):
    """ONE original network-training block; input images kept fixed and detached."""
    # Original DC creates a shuffled DataLoader from a detached snapshot.
    image_syn_train = image_syn.detach().clone()
    label_syn_train = label_syn.detach().clone()
    loader = torch.utils.data.DataLoader(
        TensorDataset(image_syn_train, label_syn_train),
        batch_size=args.batch_train, shuffle=True, num_workers=0)
    net.train()
    for _ in range(args.inner_loop):
        epoch('train', loader, net, optimizer_net, criterion, args,
              aug=True if args.dsa else False)
def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', choices=['ConvNet', 'ResNet18', 'ViT', 'VGG11'], help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='/scratch/yguo25/files/mtt-distillation/dataset/', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--Fuse', type=int, default=2, help='number of consecutive network checkpoints')
    parser.add_argument('--fuse_mask_list', type=int, nargs='+', default=None, help='all-one fuse mask, e.g. 1 1 1')
    parser.add_argument('--v_fuse', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--use-barrier', dest='use_barrier', action=argparse.BooleanOptionalAction, default=False, help='use explicit cuda.synchronize timing')
    parser.add_argument('--skip_eval', action='store_true', help='skip expensive synthetic dataset evaluation')
    parser.add_argument('--AccTest', type=bool, default=False, help='num of models being stacked')
    args = parser.parse_args()
    if args.fuse_mask_list is not None:
        if any(m != 1 for m in args.fuse_mask_list):
            raise ValueError('Current DC fused implementation only supports all-one --fuse_mask_list')
        args.Fuse = len(args.fuse_mask_list)
    if args.Fuse < 1:
        raise ValueError('--Fuse must be >= 1')
    # 这个目前就先开着。等到真的训练的时候再说。
    args.skip_eval = True
    if args.AccTest:
        set_random_seed(42)
        args.Iteration = 0
        args.num_exp = 1
        args.ipc = 10
        args.skip_eval = True
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    if args.AccTest:
        args.outer_loop = 1
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False
    time_stats = {'total': 0.0, 'fwd': 0.0, 'bwd': 0.0}
    peak_reserved = 0
    peak_allocated = 0
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path, exist_ok=True)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    eval_it_pool = np.arange(0, args.Iteration+1, 500).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []
    data_save = []
    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)
        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))
        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]
        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))
        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')
        if args.AccTest:
            file_path = Path(__file__).resolve().parent.parent / "test_script/script/in_ip10.pt"
            image_syn = torch.load(file_path, weights_only=True).detach().to(args.device).requires_grad_(True)
        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        if args.Fuse > 1:
            # Multi-checkpoint case stays stateless: ordinary net owns SGD state and ReparamModule receives packed checkpoints.
            init_rng = torch.get_rng_state()
            fused_raw = get_fused_network(args.model, channel, num_classes, im_size, Fuse=args.Fuse, v_fuse=args.v_fuse).to(args.device)
            torch.set_rng_state(init_rng)
            fused_shapes = [tuple(p.shape) for p in fused_raw.parameters()]
            fused = ReparamModule(fused_raw).train()
        else:
            fused_shapes = None
            fused = None
        if args.model == 'ConvNet' and (channel != 3 or tuple(im_size) != (32, 32)):
            raise ValueError('Conv_Flexfuse expects RGB 32x32')
        print('%s training begins'%get_time())
        for it in range(args.Iteration+1):
            ''' Evaluate synthetic data '''
            if (not args.skip_eval) and it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    if args.dsa:
                        args.epoch_eval_train = 1000
                        args.dc_aug_param = None
                        print('DSA augmentation strategy: \n', args.dsa_strategy)
                        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                        print('DC augmentation parameters: \n', args.dc_aug_param)
                    if args.dsa or args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                    else:
                        args.epoch_eval_train = 300
                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs
                ''' visualize and save '''
                save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.
            if args.Fuse == 1:
                net = get_fused_network(args.model, channel, num_classes, im_size, Fuse=1, v_fuse=args.v_fuse).to(args.device)
                net.train()
                fused_for_gm = net
                train_net = FusedLogitsWrapper(net)
                optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
                base_shapes = [tuple(p.shape) for p in net.parameters()]
            else:
                # Fuse>1 still uses one ordinary network to generate consecutive checkpoints.
                net = get_network(args.model, channel, num_classes, im_size, Fuse=1).to(args.device)
                if args.model == 'VGG11':
                    for i, m in enumerate(net.features):
                        if isinstance(m, nn.MaxPool2d):
                            net.features[i] = nn.AvgPool2d(kernel_size=m.kernel_size, stride=m.stride, padding=m.padding, ceil_mode=m.ceil_mode)
                net.train()
                train_net = net
                fused_for_gm = fused
                optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
                base_shapes = [tuple(p.shape) for p in net.parameters()]
                if any(isinstance(m, nn.BatchNorm2d) for m in net.modules()):
                    raise ValueError('Current manual fused models do not support BatchNorm')
                if any(isinstance(m, nn.GroupNorm) and m.num_groups != m.num_channels for m in net.modules()):
                    raise ValueError('Current manual fused models require channelwise GroupNorm')
                if len(base_shapes) != len(fused_shapes):
                    raise RuntimeError('Original and fused model parameter counts differ')
                for i, (shape, fs) in enumerate(zip(base_shapes, fused_shapes)):
                    if int(np.prod(fs)) != args.Fuse * int(np.prod(shape)):
                        raise RuntimeError(f'Original/fused parameter {i} size mismatch: {shape} -> {fs}')
                    if args.model != 'ViT' and len(shape) > 0 and fs != (args.Fuse * shape[0], *shape[1:]):
                        raise RuntimeError(f'Original/fused parameter {i} shape mismatch: {shape} -> {fs}')
            loss_avg = 0.0
            args.dc_aug_param = None
            for ol in range(args.outer_loop):
                # All-one Fuse scheme: collect Fuse consecutive network checkpoints.
                # With S_t unchanged, evaluate GM at every checkpoint and average them.
                # BN的部分暂时注释，后续补全。
                # BN_flag = False
                # BNSizePC = 16  # for batch normalization
                # for module in net.modules():
                #     if 'BatchNorm' in module._get_name(): #BatchNorm
                #         BN_flag = True
                # if BN_flag:
                #     img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                #     net.train() # for updating the mu, sigma of BatchNorm
                #     output_real = net(img_real) # get running mu, sigma
                #     for module in net.modules():
                #         if 'BatchNorm' in module._get_name():  #BatchNorm
                #             module.eval() # fix mu and sigma of every BatchNorm layer
                if args.Fuse == 1:
                    flat_fused = None
                else:
                    theta0 = snapshot_params(net)
                    if args.AccTest:
                        # Correctness test: every fused branch uses exactly the same model weights.
                        checkpoints = [theta0] * args.Fuse
                    else:
                        checkpoints = [theta0]
                        for _ in range(args.Fuse - 1):
                            network_train_once(train_net, optimizer_net, criterion, image_syn, label_syn, args)
                            checkpoints.append(snapshot_params(net))
                    flat_fused = pack_checkpoints(checkpoints)
                # Real/Syn sampling is ONCE per class, shared across checkpoints.
                optimizer_img.zero_grad(set_to_none=True)
                image_syn.grad = torch.zeros_like(image_syn)  # one shared image gradient buffer
                loss_ol = 0.0
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                total_start = time.time()
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    lab_real = torch.full((img_real.shape[0],), c, device=args.device, dtype=torch.long)
                    original_syn = image_syn[c*args.ipc:(c+1)*args.ipc]
                    img_syn = original_syn
                    lab_syn = label_syn[c*args.ipc:(c+1)*args.ipc]
                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    # 以manual_fused_class_gradient 取代
                    # dx_class：当前类别的 Synthetic Image Gradient，用于更新合成图片。
                    # d_class：当前类别的 Gradient Matching Loss，主要用于记录 Loss，不直接参与更新。
                    dx_class, loss_class = manual_fused_class_gradient(
                        fused_for_gm, flat_fused, base_shapes,
                        img_real, img_syn, lab_real, lab_syn, args, time_stats)
                    # DSA: dx_class is dD/d(augmented image), not dD/d(S).
                    # Preserve augmentation VJP via torch.autograd.grad.
                    if args.dsa:
                        dx_orig = torch.autograd.grad(
                            img_syn, original_syn, grad_outputs=dx_class,
                            allow_unused=False)[0]
                    else:
                        dx_orig = dx_class
                    image_syn.grad[c*args.ipc:(c+1)*args.ipc].add_(dx_orig)
                    if args.use_barrier and torch.cuda.is_available(): torch.cuda.synchronize()
                    image_bwd_end = time.time()
                    time_stats['bwd'] += image_bwd_end - time_stats.pop('bwd_start')
                    loss_ol += loss_class #只是记录loss用。已经将梯度累积完毕了
                if torch.cuda.is_available(): torch.cuda.synchronize()
                total_end = time.time()
                time_stats['total'] += total_end - total_start
                if torch.cuda.is_available():
                    peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
                    peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
                if args.AccTest:
                    print("----GRAD L2-----", image_syn.grad.norm().item())
                optimizer_img.step()
                if args.Fuse == 1 and ol != args.outer_loop - 1:
                    network_train_once(train_net, optimizer_net, criterion, image_syn, label_syn, args)
                loss_avg += loss_ol.item()
                if args.Fuse > 1:
                    del checkpoints
                del flat_fused
            loss_avg /= (num_classes*args.outer_loop)
            if it%10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))
            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))
    print('------------FIN TIME-------------')
    print('time sumation: ', time_stats['total'])
    print('fwd_time:      ', time_stats['fwd'])
    print('backward_time: ', time_stats['bwd'])
    print('峰值cache使用:', peak_reserved / 1024**2, 'MB')
    print('峰值tensor使用:', peak_allocated / 1024**2, 'MB')
    # print('\n==================== Final Results ====================\n')
    if args.skip_eval:
        print('Evaluation skipped (skip_eval=True).')
    else:
        for key in model_eval_pool:
            accs = accs_all_exps[key]
            print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))
if __name__ == '__main__':
    main()
