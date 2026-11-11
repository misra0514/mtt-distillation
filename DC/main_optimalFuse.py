# DC ConvNet interleaved simulation.
# One grouped iteration creates Fuse independent random networks.
# Odd/even jobs form two cohorts. Each class alternates cohort 0 and cohort 1.
# Prime 123, repeat 1 -> fused(2,4) -> 3, and drain the final 4.
# Each outer loop uses a shared image snapshot and applies ONE mean-gradient SGD step.
# Real-data branches need only stages 1 and 2; synthetic branches need all four.
# With --AccTest, compare each completed synthetic gradient to serial Fuse=1 jobs.
# Timing follows the supplied script: warmup=3 groups; kernel covers class work;
# total also includes network creation, parameter packing, and inner network training.
import time
import copy
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torchvision.utils import save_image
import os
# Support placement either at the project root or under test_script/.
PROJECT_ROOT = next((parent for parent in Path(__file__).resolve().parents
                     if (parent / 'utils_dc.py').is_file() and (parent / 'networks').is_dir()), None)
if PROJECT_ROOT is None:
    raise RuntimeError('Place this script inside the DC project containing utils_dc.py and networks/')
sys.path.insert(0, str(PROJECT_ROOT))
from utils_dc import get_loops, get_dataset, get_network as _get_original_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
from networks.networks_Fuse import Conv_Flexfuse
from networks.networks_stateless import ConvBlock_bwd2_1
from networks.networks_stateless_basicblock import linerFused_bwd
from contextlib import contextmanager
from utils_flex import set_random_seed
# Fuse counts independent random-network jobs per grouped iteration.
def get_network(model, channel, num_classes, im_size, Fuse=1, v_fuse=False):
    return _get_original_network(model, channel, num_classes, im_size)


def get_fused_network(model, channel, num_classes, im_size, Fuse=1, v_fuse=False):
    return Conv_Flexfuse(channel=channel, num_classes=num_classes, net_width=128,
                         im_size=im_size, Fuse=Fuse, v_fuse=v_fuse)


@contextmanager
def stage_timer(args, stats, name):
    enabled = stats is not None and args.use_barrier and args.split_fwd_barrier
    if enabled:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
    try:
        yield
    finally:
        if enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            stats[name] += time.perf_counter() - started


class FusedLogitsWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)[0]


def load_cohort_weights(cohort, networks):
    """Copy independent Fuse=1 networks into one read-only grouped network."""
    parameter_lists = [tuple(net.parameters()) for net in networks]
    with torch.no_grad():
        for index, destination in enumerate(cohort.parameters()):
            packed = torch.cat([params[index].detach() for params in parameter_lists], dim=0)
            if packed.shape != destination.shape:
                raise RuntimeError(f'Cannot pack parameter {index}: {packed.shape} != {destination.shape}')
            destination.copy_(packed)


def gradient_matching_seeds(gw_syn, gw_real, base_shapes, lanes, args):
    """Each lane contributes 1/Fuse of its original DC matching objective."""
    if len(gw_syn) != len(base_shapes) or len(gw_real) != len(base_shapes):
        raise RuntimeError('Unexpected first-bwd parameter count')
    for grad, shape in zip(gw_syn, base_shapes):
        if grad.numel() != lanes * int(np.prod(shape)):
            raise RuntimeError(f'Invalid gradient shape {grad.shape} for {lanes} lanes and {shape}')
    with torch.enable_grad():
        leaves = [grad.detach().requires_grad_(True) for grad in gw_syn]
        losses = []
        for lane in range(lanes):
            gs = [grad.reshape(lanes, *shape)[lane] for grad, shape in zip(leaves, base_shapes)]
            gr = [grad.detach().reshape(lanes, *shape)[lane] for grad, shape in zip(gw_real, base_shapes)]
            losses.append(match_loss(gs, gr, args))
        loss = torch.stack(losses).sum() / args.Fuse
        seeds = torch.autograd.grad(loss, leaves, allow_unused=True)
        seeds = [torch.zeros_like(grad) if seed is None else seed.detach()
                 for seed, grad in zip(seeds, gw_syn)]
    return seeds, loss.detach()


def concatenate_packs(current, pending, weights=False):
    # [current first-bwd lanes, pending bwd2-1 lanes], hence mask = [1...1, 0...0].
    def cat(left, right, dim):
        if left is None and right is None:
            return None
        return torch.cat((left, right), dim=dim)
    return {
        'blocks': [
            {key: cat(a[key], b[key], 0 if weights else 1) for key in a}
            for a, b in zip(current['blocks'], pending['blocks'])
        ],
        'head': {
            key: cat(current['head'][key], pending['head'][key],
                     0 if weights or key == 'x_out' else 1)
            for key in current['head']
        },
    }


@torch.no_grad()
def standalone_bwd2_1(task, lanes, v_fuse):
    """Drain the last task using the same kernels as run_double_bwd_full."""
    tape = task['tape']
    weights = task['model'].get_weight_pack()
    state = task.pop('corrections')
    head = tape['head']
    x_lin = head['x_lin']
    dx_out = state.pop('dx_out_d1')
    g, dlin_w, dlin_b = linerFused_bwd(x_lin, weights['head']['linw'], dx_out, Fuse=lanes)
    g = g.reshape_as(x_lin)
    g.add_(state.pop('dx_lin_d2').reshape_as(x_lin))
    head.clear()
    del x_lin, dx_out, dlin_w, dlin_b
    corrections = state.pop('blocks')
    for index in (2, 1, 0):
        block, weight = tape['blocks'][index], weights['blocks'][index]
        correction = corrections[index]
        g, dconv_w, dconv_b, dnorm_w, dnorm_b = ConvBlock_bwd2_1(
            block['x_conv'], block['x_norm'], block['x_pool'],
            weight['convw'], weight['normw'], g,
            correction.pop('dx_norm_d2'), correction.pop('dxconv_d2'),
            Fuse=lanes, v_fuse=v_fuse)
        tape['blocks'][index] = None
        corrections[index] = None
        del block, weight, correction, dconv_w, dconv_b, dnorm_w, dnorm_b
    return g


@torch.no_grad()
def serial_lane_reference(net, real, syn, labels_real, labels_syn, base_shapes, args):
    real_tape = net(real)[1]
    gw_real = net.run_first_bwd(real_tape, labels_real, Fuse=1)[-1]
    del real_tape
    tape = net(syn.detach())[1]
    head, acts, dweights, gradients = net.run_first_bwd(tape, labels_syn, Fuse=1)
    del dweights
    seeds, loss = gradient_matching_seeds(gradients, gw_real, base_shapes, 1, args)
    dx = net.run_double_bwd_full(tape, acts, d_bwd_state=head, dd_tensors_all=seeds, Fuse=1)
    return dx.detach(), loss


def prepare_class_task(model, networks, class_id, image_syn, label_syn, get_images, base_shapes, args):
    real_images, syn_images, real_labels, syn_labels, augmentation_links = [], [], [], [], []
    reference_dx, reference_loss = [], []
    for net in networks:
        real = get_images(class_id, args.batch_real)
        labels_real = torch.full((len(real),), class_id, device=args.device, dtype=torch.long)
        original = image_syn[class_id * args.ipc:(class_id + 1) * args.ipc]
        syn = original
        labels_syn = label_syn[class_id * args.ipc:(class_id + 1) * args.ipc]
        if args.dsa:
            seed = int(time.time() * 1000) % 100000
            real = DiffAugment(real, args.dsa_strategy, seed=seed, param=args.dsa_param)
            syn = DiffAugment(original, args.dsa_strategy, seed=seed, param=args.dsa_param)
            augmentation_links.append((syn, original))
        if args.AccTest:
            dx, loss = serial_lane_reference(net, real, syn, labels_real, labels_syn, base_shapes, args)
            reference_dx.append(dx.cpu())
            reference_loss.append(loss.cpu())
        real_images.append(real)
        syn_images.append(syn.detach())
        real_labels.append(labels_real)
        syn_labels.append(labels_syn)
    lanes = len(networks)
    with torch.no_grad():
        real = torch.cat(real_images, dim=1)
        real_tape = model(real)[1]
        gw_real = model.run_first_bwd(real_tape, torch.cat(real_labels), Fuse=lanes)[-1]
        del real_tape, real
        syn = torch.cat(syn_images, dim=1)
        tape = model(syn)[1]
    return {
        'model': model, 'class_id': class_id, 'tape': tape,
        'labels': torch.cat(syn_labels), 'gw_real': gw_real,
        'augmentation_links': augmentation_links,
        'reference_dx': torch.cat(reference_dx, dim=1) if args.AccTest else None,
        'reference_loss': torch.stack(reference_loss).sum() if args.AccTest else None,
    }


def advance_double_bwd(task, first_result, base_shapes, lanes, args, stats):
    head, acts, dweights, gradients = first_result
    del first_result, dweights
    with stage_timer(args, stats, 'fwd'):
        seeds, loss = gradient_matching_seeds(gradients, task.pop('gw_real'), base_shapes, lanes, args)
    with stage_timer(args, stats, 'bwd'), torch.no_grad():
        task['corrections'] = task['model'].run_double_bwd(
            task['tape'], acts, d_bwd_state=head, dd_tensors_all=seeds, Fuse=lanes)
    task['loss'] = loss


def accumulate_class_gradient(task, dx, image_syn, lanes, args):
    expected_shape = (args.ipc, lanes * image_syn.shape[1], *image_syn.shape[2:])
    if tuple(dx.shape) != expected_shape:
        raise RuntimeError(f'Input gradient shape {tuple(dx.shape)} != {expected_shape}')
    if args.AccTest:
        reference = task.pop('reference_dx').to(dx.device)
        torch.testing.assert_close(dx, reference, rtol=3e-3, atol=1e-5)
        reference_norm = reference.norm().item()
        relative = (dx - reference).norm().item() / max(reference_norm, 1e-30)
        if reference_norm and relative > 3e-3:
            raise AssertionError(f'Input gradient relative L2 error {relative:.6e}')
        torch.testing.assert_close(task['loss'].cpu(), task.pop('reference_loss'), rtol=3e-3, atol=1e-5)
        print(f"[CHECK] class={task['class_id']} input-gradient rel_l2={relative:.6e}")
    channels = image_syn.shape[1]
    if args.dsa:
        # Keep every lane's augmentation graph until that lane's stage 4 finishes.
        dx_original = torch.zeros_like(image_syn[:args.ipc])
        for lane, (augmented, original) in enumerate(task.pop('augmentation_links')):
            grad = torch.autograd.grad(
                augmented, original,
                grad_outputs=dx[:, lane * channels:(lane + 1) * channels],
                allow_unused=False)[0]
            dx_original.add_(grad)
    else:
        dx_original = dx.reshape(args.ipc, lanes, channels, *image_syn.shape[2:]).sum(dim=1)
    class_id = task['class_id']
    with torch.no_grad():
        image_syn.grad[class_id * args.ipc:(class_id + 1) * args.ipc].add_(dx_original)
    return task['loss']


def interleaved_class_gradients(cohort_models, cohort_networks, image_syn, label_syn,
                                get_images, base_shapes, num_classes, args, stats):
    """Prime 123, alternate 1 -> fused(2,4) -> 3, then drain the final 4.

    Rows within each cohort are independent random networks. The two cohorts
    alternate across classes. No image or network update occurs while a task
    is pending. All matching seeds already include the global 1/Fuse factor.
    """
    lanes = args.Fuse // 2
    mask = [1] * lanes + [0] * lanes
    pending = None
    loss = torch.zeros((), device=image_syn.device)
    for class_id in range(num_classes):
        for model, networks in zip(cohort_models, cohort_networks):
            with stage_timer(args, stats, 'fwd'):
                current = prepare_class_task(model, networks, class_id, image_syn, label_syn,
                                             get_images, base_shapes, args)
            if pending is None:
                with stage_timer(args, stats, 'fwd'), torch.no_grad():
                    first_result = model.run_first_bwd(current['tape'], current['labels'], Fuse=lanes)
            else:
                with stage_timer(args, stats, 'mixed'), torch.no_grad():
                    packed_tape = concatenate_packs(current['tape'], pending['tape'])
                    packed_weights = concatenate_packs(model.get_weight_pack(), pending['model'].get_weight_pack(), weights=True)
                    # The new packed storage now owns both tasks' forward activations.
                    pending.pop('tape').clear()
                    current['tape'] = packed_tape
                    result = model.run_fused_bwd(packed_tape, current['labels'], pending.pop('corrections'),
                                                 mask, weights=packed_weights)
                    first_result, dx_pending = result[:4], result[4]
                    del result, packed_tape, packed_weights
                with stage_timer(args, stats, 'bwd'):
                    loss += accumulate_class_gradient(pending, dx_pending, image_syn, lanes, args)
                del pending, dx_pending
            advance_double_bwd(current, first_result, base_shapes, lanes, args, stats)
            del first_result
            pending = current
            del current
    with stage_timer(args, stats, 'bwd'):
        dx_pending = standalone_bwd2_1(pending, lanes, args.v_fuse)
        loss += accumulate_class_gradient(pending, dx_pending, image_syn, lanes, args)
    return loss


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
    parser.add_argument('--model', type=str, default='ConvNet', choices=['ConvNet'], help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='grouped iterations; each creates Fuse independent random networks, using the original inclusive 0..Iteration loop')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='/scratch/yguo25/files/mtt-distillation/dataset/', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--Fuse', type=int, default=2, help='even number of independent network jobs, split into two interleaved cohorts')
    parser.add_argument('--fuse_mask_list', type=int, nargs='+', default=None, help='optional all-one list whose length specifies the even job count')
    parser.add_argument('--v_fuse', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--use-barrier', dest='use_barrier', action=argparse.BooleanOptionalAction, default=False, help='use explicit cuda.synchronize timing')
    parser.add_argument('--split-fwd-barrier', action=argparse.BooleanOptionalAction, default=False, help='split timing of first-order work and image backward')
    parser.add_argument('--skip_eval', action='store_true', help='skip expensive synthetic dataset evaluation')
    parser.add_argument('--AccTest', type=bool, default=False, help='compare every class gradient to independent serial Fuse=1 jobs')
    args = parser.parse_args()
    if args.fuse_mask_list is not None:
        if any(m != 1 for m in args.fuse_mask_list):
            raise ValueError('--fuse_mask_list specifies job count and must contain only ones')
        args.Fuse = len(args.fuse_mask_list)
    if args.Fuse < 2 or args.Fuse % 2:
        raise ValueError('--Fuse must be an even integer >= 2')
    # TODO: 这个目前就先开着。等到真的训练的时候再说。
    args.skip_eval = True
    if args.AccTest:
        set_random_seed(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        args.Iteration = 0
        args.num_exp = 1
        args.ipc = 10
        args.skip_eval = True
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    if args.AccTest:
        args.outer_loop = 1
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[SCHEDULE] Fuse={args.Fuse}, cohort lanes={args.Fuse // 2}; mean image gradient over all jobs')
    print(f'[WORKLOAD] measured groups={args.Iteration + 1}, random-network jobs per experiment={(args.Iteration + 1) * args.Fuse}')
    for method in ('run_first_bwd', 'run_double_bwd', 'run_double_bwd_full', 'run_fused_bwd'):
        if not hasattr(Conv_Flexfuse, method):
            raise RuntimeError(f'Update networks.networks_Fuse.Conv_Flexfuse: missing {method}')
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False
    warmup = 0 if args.AccTest else 3
    final_iteration = args.Iteration + warmup
    time_stats = {'total': 0.0, 'kernel': 0.0, 'fwd': 0.0, 'bwd': 0.0, 'mixed': 0.0}
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
        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        if channel != 3 or tuple(im_size) != (32, 32):
            raise ValueError('Conv_Flexfuse expects RGB 32x32')
        # Allocate read-only execution models outside the measured class loops.
        cohort_models = [get_fused_network(args.model, channel, num_classes, im_size,
                         Fuse=args.Fuse // 2, v_fuse=args.v_fuse).to(args.device).train()
                         for _ in range(2)]
        print('%s training begins'%get_time())
        for it in range(final_iteration+1):
            measured = it >= warmup
            ''' Evaluate synthetic data '''
            if (not args.skip_eval) and (it - warmup) in eval_it_pool and measured:
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

                    if it == final_iteration: # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.
            if it == warmup:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                total_start = time.time()
            # Every row in the drawing owns its random network and SGD state.
            networks = [get_fused_network(args.model, channel, num_classes, im_size,
                        Fuse=1, v_fuse=args.v_fuse).to(args.device).train()
                        for _ in range(args.Fuse)]
            train_nets = [FusedLogitsWrapper(net) for net in networks]
            optimizers_net = [torch.optim.SGD(net.parameters(), lr=args.lr_net) for net in networks]
            cohort_networks = [networks[0::2], networks[1::2]]
            base_shapes = [tuple(param.shape) for param in networks[0].parameters()]
            loss_avg = 0.0
            args.dc_aug_param = None
            for ol in range(args.outer_loop):
                for model, members in zip(cohort_models, cohort_networks):
                    load_cohort_weights(model, members)
                del model, members
                optimizer_img.zero_grad(set_to_none=True)
                image_syn.grad = torch.zeros_like(image_syn)
                if measured and args.use_barrier:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    kernel_start = time.time()
                loss_ol = interleaved_class_gradients(
                    cohort_models, cohort_networks, image_syn, label_syn,
                    get_images, base_shapes, num_classes, args,
                    time_stats if measured else None)
                if measured and args.use_barrier:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    time_stats['kernel'] += time.time() - kernel_start
                if args.AccTest:
                    print('----GRAD L2-----', image_syn.grad.norm().item())
                # All pending stage-4 work has drained before either kind of update.
                optimizer_img.step()
                if ol != args.outer_loop - 1:
                    for train_net, optimizer_net in zip(train_nets, optimizers_net):
                        network_train_once(train_net, optimizer_net, criterion, image_syn, label_syn, args)
                    del train_net, optimizer_net
                loss_avg += loss_ol.item()
            del networks, train_nets, optimizers_net, cohort_networks
            loss_avg /= (num_classes*args.outer_loop)
            if it == final_iteration:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
                    peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
                time_stats['total'] += time.time() - total_start
            if it%10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))
            if it == final_iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_interleaved_F%d_%s_%s_%s_%dipc.pt'%(args.Fuse, args.method, args.dataset, args.model, args.ipc)))
    print('------------TESTING FINISHED-------------')
    print('Measured grouped iterations per experiment:', args.Iteration + 1)
    print('Independent random-network jobs per experiment:', (args.Iteration + 1) * args.Fuse)
    if args.use_barrier:
        print('Kernel time: ', time_stats['kernel'])
        if args.split_fwd_barrier:
            print('fwd_time: ', time_stats['fwd'])
            print('backward_time: ', time_stats['bwd'])
            print('mixed_bwd_time: ', time_stats['mixed'])
    else:
        print('total time: ', time_stats['total'])
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


