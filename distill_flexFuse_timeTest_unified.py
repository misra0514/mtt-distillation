# 11.11: 合并了conv。vgg，resnet，vit四个模型。目前还没有测试。


"""Unified FlexFuse MTT timing entry: ConvNet / ResNet18 / ViT / VGG11.

--datalayout b_f (default) or f_f; f_f is supported by ConvNet, ResNet18 and VGG11.

Place this file in the original mtt-distillation project, alongside utils.py.
Example (replace data/buffer paths as needed):
  python distill_flexFuse_timeTest_unified.py --model ResNet18 \
    --fuse_mask_list 1 0 --v_fuse --load_all --syn_steps 1 \
    --Iteration 30 --expert_epochs 1 --max_files 1 --max_experts 1
Add --use-barrier for synchronized system/kernel times; additionally add
--split-fwd-barrier to split syn/backward. Without barrier, only total time is
synchronized; the legacy 'kernel time( without barrier)' is host elapsed time.

Contract:
* syn_steps=1 only (the user's current sweep); no multi-step unroll is claimed.
* --Iteration=N measures N+1 iterations, matching all supplied inclusive loops.
* Three warmup iterations (zero under AccTest) are excluded from elapsed times.
* Peak memory includes preparation and warmup, matching the new Conv script.
* --load_all / --loadall / --preload pack CPU expert groups in advance, using
  shuffle seed 42 and wrap-padding. Active lanes must be a prefix of ones.
* Without preload, expert selection/loading follows each supplied model's code;
  Conv consumes Fuse experts and reloads the first file at every boundary.
* Conv denominator is active squared distance / bwd_Fuse; VGG uses the sum;
  ResNet/ViT retain their mean of per-layer/per-lane MSEs.
* Gradient writeback is unchanged: VGG assigns image_syn.grad and syn_lr.grad;
  the supplied Conv/ResNet/ViT do not. Those three are timing/gradient-compute
  paths and their optimizer steps do not update the synthetic data.
* Texture/augmentation flags retain the source behavior (no new augmentation
  or crop implementation is introduced in this timing-only merge).
"""
# Uses the supplied Conv unified model interface; preload and CUDA timing follow
# the newer scattered Conv implementation. Requires the original project modules.
# Normalization and gradient-update behavior remain model-specific (see below).
import os
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, ParamDiffAug
from reparam_module import ReparamModule
from utils_flex import recover_params, set_random_seed


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('expected True or False')


def validate_mask(mask):
    active = [i for i, keep in enumerate(mask) if keep == 1]
    if any(keep not in (0, 1) for keep in mask) or not active:
        raise ValueError('fuse_mask_list must contain 0/1 and at least one 1')
    if active != list(range(active[0], active[-1] + 1)):
        raise ValueError('active fuse lanes must be contiguous')
    return active


def active_slices(shape_list, mask):
    active = validate_mask(mask)
    offset = 0
    slices = []
    for shape in shape_list:
        n = int(np.prod(shape))
        if len(shape) == 0:
            slices.append(slice(offset, offset + n))
            offset += n
        else:
            slices.append(slice(offset + active[0] * n,
                                offset + (active[-1] + 1) * n))
            offset += len(mask) * n
    return slices


def select_active(flat, slices, all_active):
    return flat if all_active else torch.cat([flat[sl] for sl in slices], 0)


def pack_snapshots(snapshots):
    # Layer-major, then lane-major. ViT cls/pos flatten identically even though
    # the fused model recovers them with the lane axis in dim=1.
    return torch.cat([p.detach().reshape(-1) for layer in zip(*snapshots) for p in layer], 0)


def pack_expert_groups(trajectories, fuse_mask_list, target_epoch):
    # Same shuffle / wrap-padding / CPU residency as the latest Conv preload.
    active = validate_mask(fuse_mask_list)
    fuse = len(fuse_mask_list)
    if fuse_mask_list != [1] * len(active) + [0] * (fuse - len(active)):
        raise ValueError('--load_all requires leading ones followed by zeros')
    if not trajectories:
        raise ValueError('--load_all selected no expert trajectories')
    order = list(range(len(trajectories)))
    random.Random(42).shuffle(order)
    count = (len(order) + fuse - 1) // fuse
    padding = count * fuse - len(order)
    order = [order[i % len(order)] for i in range(count * fuse)]
    shapes = [p.shape for p in trajectories[order[0]][0]]
    if any(len(s) == 0 for s in shapes):
        raise ValueError('Prepacked fused experts require non-scalar parameters')
    groups = []
    for offset in range(0, len(order), fuse):
        group = [trajectories[i] for i in order[offset:offset + fuse]]
        starts = [trajectory[0] for trajectory in group]
        targets = [group[i][target_epoch] for i in active]
        for params in starts + targets:
            if [p.shape for p in params] != shapes:
                raise ValueError('Expert snapshots must have matching parameter shapes')
        groups.append((pack_snapshots(starts), pack_snapshots(targets)))
    print(f'Preloaded {count} expert groups of {fuse}; padded {padding} lanes by reusing experts')
    return groups, shapes


class ExpertSource:
    def __init__(self, args, expert_dir):
        self.args = args
        self.files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, f'replay_buffer_{n}.pt')):
            self.files.append(os.path.join(expert_dir, f'replay_buffer_{n}.pt'))
            n += 1
        self.files = self.files[:args.max_files]
        if not self.files:
            raise ValueError(f'No replay buffers selected at {expert_dir}')
        self.index = self.file_index = 0
        if args.load_all:
            trajectories = []
            for path in self.files:
                trajectories.extend(torch.load(path, map_location='cpu')[:args.max_experts])
            self.buffer, self.shapes = pack_expert_groups(
                trajectories, args.fuse_mask_list, args.expert_epochs)
        else:
            self.buffer = self.load(self.files[0])

    def load(self, path):
        buffer = torch.load(path)[:self.args.max_experts]
        if not buffer:
            raise ValueError(f'No expert trajectories selected in {path}')
        return buffer

    def next(self):
        args = self.args
        if args.load_all:
            start, target = self.buffer[self.index]
            self.index = (self.index + 1) % len(self.buffer)
            return (start.to(args.device, copy=True).requires_grad_(True),
                    target.to(args.device), self.shapes)
        trajectory = self.buffer[self.index]
        if args.model == 'ConvNet':
            # Latest Conv's no-preload benchmark consumes Fuse experts and
            # reloads the FIRST file at each boundary. Preserve this policy.
            self.index += len(args.fuse_mask_list)
            while self.index >= len(self.buffer):
                self.index -= len(self.buffer)
                self.buffer = self.load(self.files[0])
        else:
            # Original ResNet / ViT / VGG: one selected expert, cycle files.
            self.index += 1
            if self.index == len(self.buffer):
                self.index = 0
                self.file_index = (self.file_index + 1) % len(self.files)
                if args.max_files != 1:
                    self.buffer = self.load(self.files[self.file_index])
        starts = list(trajectory[0])
        targets = list(trajectory[args.expert_epochs])
        shapes = [p.shape for p in starts]
        if [p.shape for p in targets] != shapes:
            raise ValueError('Start/target parameter shapes differ')
        fuse = len(args.fuse_mask_list)
        active = validate_mask(args.fuse_mask_list)
        # Same repeated-expert workload when preload is disabled; never mutate
        # the trajectory lists (the old Conv/ResNet/ViT paths did so in place).
        def repeat_flat(params, lanes):
            return torch.cat([p.detach().repeat((lanes,) + (1,) * (p.ndim - 1)).reshape(-1)
                              if p.ndim else p.detach().reshape(-1) for p in params], 0).to(args.device)
        return repeat_flat(starts, fuse).requires_grad_(True), repeat_flat(targets, len(active)), shapes


def make_student(args, channel, num_classes, im_size):
    if args.model == 'ConvNet':
        from networks.networks_Fuse import Conv_Flexfuse
        model = Conv_Flexfuse(channel=channel, num_classes=num_classes,
                             im_size=im_size, Fuse=int(args.Fuse), v_fuse=args.v_fuse,
                             layout=args.datalayout)
    elif args.model == 'VGG11':
        from networks.networks_Fuse import VGG11_Flexfuse
        model = VGG11_Flexfuse(channel, num_classes, Fuse=int(args.Fuse),
                              im_size=im_size, v_fuse=args.v_fuse, layout=args.datalayout)
    elif args.model == 'ResNet18':
        from networks.networks_Fuse import ResNet18_FlexFuse
        model = ResNet18_FlexFuse(channel=channel, num_classes=num_classes,
                                  Fuse=int(args.Fuse), v_fuse=args.v_fuse,
                                  layout=args.datalayout)
    else:
        model = get_network('ViTFlexFuse' + args.Fuse, channel, num_classes, im_size,
                            dist=False, v_fuse=args.v_fuse)
    return ReparamModule(model.to(args.device))


def parameter_distance(start_active, target, shapes, bwd_fuse, model):
    # Preserve the supplied files' DIFFERENT normalization conventions.
    if model == 'ConvNet':
        return (start_active - target).square().sum() / bwd_fuse
    if model == 'VGG11':
        return (start_active - target).square().sum()
    terms = []
    offset = 0
    for shape in shapes:
        n = int(np.prod(shape))
        lanes = bwd_fuse if len(shape) else 1
        end = offset + lanes * n
        terms.append((start_active[offset:end] - target[offset:end]).view(lanes, n).square().mean(dim=1))
        offset = end
    return torch.cat(terms).mean()


def run_iteration(args, student, image_syn, label_syn, syn_lr, optimizer_img,
                  optimizer_lr, start, target, shapes, channel, measured):
    fuse = int(args.Fuse)
    bwd_fuse = sum(args.fuse_mask_list)
    all_active = fuse == bwd_fuse
    slices = active_slices(shapes, args.fuse_mask_list)
    fuse_mask = None if all_active else args.fuse_mask_list
    with torch.no_grad():
        # Conv computes this inside the kernel interval, as in the new Conv.
        if args.model != 'ConvNet':
            param_dist = parameter_distance(select_active(start, slices, all_active),
                                            target, shapes, bwd_fuse, args.model)
    indices_chunks = []
    params = start
    if measured and args.use_barrier:
        torch.cuda.synchronize()
    syn_start = time.time()
    for step in range(args.syn_steps):
        if not indices_chunks:
            indices_chunks = list(torch.split(torch.arange(len(image_syn)), args.batch_syn))
        these_indices = indices_chunks.pop()
        if args.datalayout == 'f_f':
            # [B,C,H,W] -> [Fuse*B,C,H,W], group-major order.
            x = image_syn[these_indices].repeat(fuse, 1, 1, 1).requires_grad_(True)
        else:
            # Original channel-packed layout [B,Fuse*C,H,W].
            x = image_syn[these_indices].repeat(1, fuse, 1, 1).requires_grad_(True)
        labels = label_syn[these_indices]
        labels = labels.repeat_interleave(fuse) if args.model == 'ViT' else labels.repeat(fuse)
        forward_params = params
        with torch.no_grad():
            kwargs = {'return_tape': True} if args.model == 'VGG11' else {}
            x_out, tape = student(x, flat_param=forward_params, **kwargs)
            ce_loss = nn.functional.cross_entropy(x_out, labels) * fuse
            kwargs = {'keep_input_grad': False, 'keep_weight_dicts': False} if args.model == 'VGG11' else {}
            state, activates, weights, grad_list = student.call_with_param(
                forward_params, student.module.run_first_bwd, tape=tape,
                target=labels, Fuse=fuse, fuse_mask_list=fuse_mask, **kwargs)
            grad = torch.cat([g.reshape(-1) for g in grad_list], 0)
        params = params - syn_lr * grad
        if args.model == 'VGG11' and step == args.syn_steps - 1:
            last_grad_active = select_active(grad, slices, all_active)
        del weights, grad_list, grad, x_out
        # The current sweep uses a single synthetic step.
        if args.model == 'ViT':
            del state
    with torch.no_grad():
        final_active = select_active(params, slices, all_active)
        forward_active = select_active(forward_params, slices, all_active)
        if measured and args.use_barrier and args.split_fwd_barrier:
            torch.cuda.synchronize()
        syn_end = time.time()
        if args.model == 'ConvNet':
            param_dist = parameter_distance(select_active(start, slices, all_active),
                                            target, shapes, bwd_fuse, args.model)
        raw_loss = nn.functional.mse_loss(final_active, target, reduction='sum')
        grand_loss = raw_loss / param_dist
        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()
        ddw = 2 * (final_active - target) / param_dist
        if args.model == 'VGG11':
            syn_lr_grad = -torch.dot(ddw, last_grad_active)
        ddw *= -syn_lr
        dd_tensors = recover_params(ddw, shapes, bwd_fuse)
        active_weights = None if all_active else recover_params(forward_active, shapes, bwd_fuse)
        kwargs = {} if args.model == 'ViT' else {'d_bwd_state': state}
        if args.model == 'VGG11':
            kwargs['release_intermediates'] = True
        dx = student.call_with_param(
            forward_params, student.module.run_double_bwd_full, tape=tape,
            d_activates_list=activates, dd_tensors_all=dd_tensors,
            weight_tensors_all=active_weights, Fuse=bwd_fuse, **kwargs)
        # Only the supplied VGG script assigns these optimizer gradients.
        # Preserve Conv/ResNet/ViT behavior rather than silently changing training.
        if args.model == 'VGG11':
            if args.datalayout == 'f_f':
                dx_image = dx.reshape(bwd_fuse, -1, channel, *dx.shape[2:]).sum(dim=0)
            else:
                dx_image = dx.reshape(dx.shape[0], bwd_fuse, channel, *dx.shape[2:]).sum(dim=1)
            image_syn.grad = torch.zeros_like(image_syn)
            image_syn.grad.index_add_(0, these_indices.to(image_syn.device), dx_image)
            syn_lr.grad = syn_lr_grad.detach().reshape_as(syn_lr)
    if args.AccTest:
        if args.model == 'VGG11':
            print('--ParamLossRaw--', raw_loss.item())
            print('--ParamDist--', param_dist.item())
        print('--Celoss--', ce_loss.item())
        print('--GradLoss--', grand_loss.item())
        print('----GRAD-----', dx.sum().item())
    optimizer_img.step()
    optimizer_lr.step()
    # Release tape/parameter references before the next iteration starts.
    return syn_start, syn_end


def main(args):
    validate_mask(args.fuse_mask_list)
    if args.datalayout not in ('b_f', 'f_f'):
        raise ValueError("datalayout must be 'b_f' or 'f_f'")
    if args.datalayout == 'f_f' and args.model not in ('ConvNet', 'ResNet18', 'VGG11'):
        raise ValueError('--datalayout f_f currently supports ConvNet, ResNet18 and VGG11 only')
    args.Fuse = str(len(args.fuse_mask_list))
    if args.split_fwd_barrier and not args.use_barrier:
        raise ValueError('--split-fwd-barrier requires --use-barrier')
    if args.syn_steps < 1 or args.Iteration < 0:
        raise ValueError('syn_steps must be positive and Iteration nonnegative')
    if args.syn_steps != 1:
        raise ValueError('This unified entry supports syn_steps=1, matching the current sweep. Multi-step hypergradients require a separate unroll implementation.')
    if not torch.cuda.is_available():
        raise RuntimeError('This CUDA timing script requires a CUDA GPU')
    if args.AccTest:
        set_random_seed(42)
        args.Iteration = 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    if args.zca and args.texture:
        raise AssertionError('Cannot use zca and texture together')
    if args.texture and args.pix_init == 'real':
        print('WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.')
    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    args.im_size = im_size
    if args.dsa:
        args.dc_aug_param = None
    args.dsa_param = ParamDiffAug()
    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None
    args.dsa_param = dsa_params
    args.zca_trans = zca_trans
    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc
    args.distributed = False
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])
    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to('cpu')
    labels_all = torch.tensor(labels_all, dtype=torch.long, device='cpu')

    def get_images(c, n):
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]
    label_syn = torch.tensor([np.ones(args.ipc, dtype=np.int_) * i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)
    if args.texture:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0] * args.canvas_size, im_size[1] * args.canvas_size), dtype=torch.float)
    else:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)
    syn_lr = torch.tensor(args.lr_teacher).to(args.device)
    if args.pix_init == 'real':
        if args.texture:
            for c in range(num_classes):
                for i in range(args.canvas_size):
                    for j in range(args.canvas_size):
                        image_syn.data[c * args.ipc:(c + 1) * args.ipc, :, i * im_size[0]:(i + 1) * im_size[0], j * im_size[1]:(j + 1) * im_size[1]] = torch.cat([get_images(c, 1).detach().data for s in range(args.ipc)])
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print('initialize synthetic data from random noise')
    if args.AccTest:
        if args.dataset == 'Tiny':
            image_syn = torch.load('./test_script/script/in_Tiny.pt')
        else:
            args.ipc = 10
            label_syn = torch.arange(num_classes, device=args.device).repeat_interleave(args.ipc)
            image_syn = torch.load('./test_script/script/in_ip10.pt')
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr)
    optimizer_img.zero_grad()
    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == 'ImageNet':
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ['CIFAR10', 'CIFAR100'] and (not args.zca):
        expert_dir += '_NO_ZCA'
    expert_dir = os.path.join(expert_dir, args.model)

    source = ExpertSource(args, expert_dir)
    student_net = make_student(args, channel, num_classes, im_size)
    warmup = 0 if args.AccTest else 10
    torch.cuda.synchronize()  # preparation must finish before warmup
    measured_count = args.Iteration + 1  # preserve the original inclusive loop
    system_time = kernel_time = syn_time = bwd_time = 0.0
    print(f'MODEL={args.model} | PRELOAD={args.load_all} | FUSE={args.Fuse} | '
          f'BARRIER={args.use_barrier} | SPLIT_FWD_BARRIER={args.split_fwd_barrier} | '
          f'WARMUP={warmup} | MEASURED_ITERATIONS={measured_count} | '
          f'DATALAYOUT={args.datalayout}')
    for it in range(warmup + measured_count):
        measured = it >= warmup
        if measured:
            if args.use_barrier:
                torch.cuda.synchronize()
            if it == warmup:
                torch.cuda.synchronize()  # exclude pending warmup, even without barrier
                total_start = time.time()
            iter_start = time.time()
        student_net.train()
        start, target, shapes = source.next()
        syn_start, syn_end = run_iteration(
            args, student_net, image_syn, label_syn, syn_lr, optimizer_img,
            optimizer_lr, start, target, shapes, channel, measured)
        del start, target
        if measured:
            if args.use_barrier:
                torch.cuda.synchronize()
            iter_end = time.time()
            system_time += syn_start - iter_start
            kernel_time += iter_end - syn_start
            if args.use_barrier and args.split_fwd_barrier:
                syn_time += syn_end - syn_start
                bwd_time += iter_end - syn_end
    torch.cuda.synchronize()  # include all queued work in wall time
    total_time = time.time() - total_start
    print('------------TESTING FINISHED-------------')
    if not args.use_barrier:
        print('total time: ', total_time)
        print('kernel time( without barrier): ', kernel_time)
    elif args.split_fwd_barrier:
        print('prepare time (', args.Iteration, '): ', system_time)
        print('syn_time     (', args.Iteration, '): ', syn_time)
        print('backward_time(', args.Iteration, '): ', bwd_time)
    else:
        print('system time  (', args.Iteration, '): ', system_time)
        print('Kernel time  (', args.Iteration, '): ', kernel_time)
    print('峰值cache使用:', torch.cuda.max_memory_reserved() / 1024**2, 'MB')
    print('峰值tensor使用:', torch.cuda.max_memory_allocated() / 1024**2, 'MB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument(   '--fuse_mask_list',   type=int,   nargs='+',       required=True,        help='fuse mask list, e.g. 1 1 0')
    parser.add_argument('--datalayout', type=str, choices=('b_f', 'f_f'),
                        default='b_f', help='activation layout (f_f: ConvNet/ResNet18 only)')
    parser.add_argument('--Fuse', type=str, default="1", help='num of models being stacked')
    parser.add_argument('--v_fuse', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--use-barrier', '--use_barrier', dest='use_barrier', action=argparse.BooleanOptionalAction, default=False, help='use explicit cuda.synchronize timing')
    parser.add_argument('--split-fwd-barrier', '--split_fwd_barrier', dest='split_fwd_barrier', action=argparse.BooleanOptionalAction, default=False, help='split timing of fwd & bwd')
    parser.add_argument('--AccTest', type=parse_bool, default=False, help='num of models being stacked')
    parser.add_argument('--detachNum', type=int, default=0, help='discard grad before this syn')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')
    parser.add_argument('--model', type=str, default='ConvNet', choices=('ConvNet', 'ResNet18', 'ViT', 'VGG11'), help='model')
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
    parser.add_argument('--syn_steps', type=int, default=1, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')
    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--load_all', '--loadall', '--preload', dest='load_all', action='store_true', help="prepack shuffled expert groups in CPU RAM before the loop")
    parser.add_argument('--no_aug', type=parse_bool, default=False, help='this turns off diff aug during distillation')
    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')
    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')
    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')
    args = parser.parse_args()
    if args.split_fwd_barrier and not args.use_barrier:
        parser.error('--split-fwd-barrier requires --use-barrier')
    main(args)
