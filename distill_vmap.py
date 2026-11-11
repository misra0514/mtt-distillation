# 11.8: vmap 为distill的对照版本。
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, ParamDiffAug
from utils_flex import set_random_seed
from torch.func import functional_call, grad_and_value, vmap
import random
import time

def flat(snapshot):
    return torch.cat([p.detach().reshape(-1) for p in snapshot])

def make_step(model, shapes, criterion):
    names = [name for name, _ in model.named_parameters()]
    if list(shapes) != [p.shape for p in model.parameters()]:
        raise ValueError("Expert parameter shapes/order do not match the single ConvNet")
    buffers = dict(model.named_buffers())
    if buffers:
        raise ValueError("Expected buffer-free ConvNet (no running statistics)")
    sizes = [shape.numel() for shape in shapes]
    def ce(w, x, y):
        params = {n: t.view(sh) for n, t, sh in zip(names, w.split(sizes), shapes)}
        out = functional_call(model, (params, buffers), (x,), strict=True)
        return criterion(out, y)
    return ce, vmap(grad_and_value(ce), in_dims=(0, None, None))

def serial_reference(ce, initial, target, images, labels, lr, batches):
    x = images.detach().requires_grad_()
    eta = lr.detach().requires_grad_()
    ends, ces = [], []
    for row in initial:
        w = row.detach().requires_grad_()
        for idx in batches:
            loss = ce(w, x[idx], labels[idx])
            g, = torch.autograd.grad(loss, w, create_graph=True)
            w = w - eta * g
        ends.append(w)
        ces.append(loss)
    ends = torch.stack(ends)
    m = len(target)
    obj = (ends[:m]-target).square().sum() / ((initial[:m]-target).square().sum()/m)
    dx, de = torch.autograd.grad(obj, (x, eta))
    return ends, torch.stack(ces), obj, dx, de

def preload_expert_groups(expert_files, max_experts, fuse_mask_list, target_epoch):
    fuse = len(fuse_mask_list)
    active_count = sum(fuse_mask_list)
    if not active_count or fuse_mask_list != [1] * active_count + [0] * (fuse - active_count):
        raise ValueError("--load_all requires a fuse mask with leading ones followed by zeros")

    snapshots = []
    shape_list = None
    for expert_file in expert_files:
        trajectories = torch.load(expert_file, map_location="cpu")
        for trajectory in trajectories[:max_experts]:
            start, target = trajectory[0], trajectory[target_epoch]
            if shape_list is None:
                shape_list = [param.shape for param in start]
                if any(len(shape) == 0 for shape in shape_list):
                    raise ValueError("Fused ConvNet expects non-scalar expert parameters")
            if any([param.shape for param in params] != shape_list for params in (start, target)):
                raise ValueError("Expert snapshots must have matching parameter shapes")
            snapshots.append((flat(start), flat(target)))
            del trajectory, start, target
        del trajectories

    if not snapshots:
        raise ValueError("--load_all selected no expert trajectories")

    order = list(range(len(snapshots)))
    random.Random(42).shuffle(order)
    group_count = (len(order) + fuse - 1) // fuse
    padding = group_count * fuse - len(order)
    order = [order[index % len(order)] for index in range(group_count * fuse)]

    groups = []
    for offset in range(0, group_count * fuse, fuse):
        group = [snapshots[index] for index in order[offset:offset + fuse]]
        # One single-model flat snapshot per row, in Flex preload group order.
        starts = torch.stack([start for start, _ in group])
        targets = torch.stack([target for _, target in group[:active_count]])
        groups.append((starts, targets))
    print(f"Preloaded {group_count} expert groups of {fuse}; padded {padding} lanes by reusing experts")
    return groups, shape_list

def main(args):
    if args.model != 'ConvNet':
        raise ValueError('This baseline expects --model ConvNet')
    fuse_mask_list = args.fuse_mask_list 
    bwd_Fuse = sum(fuse_mask_list)
    args.Fuse = str(len(fuse_mask_list)) # 对于Flex fuse来说，只用fuse_mask_list控制即可
    if any(m not in (0, 1) for m in fuse_mask_list) or bwd_Fuse == 0 or fuse_mask_list != sorted(fuse_mask_list, reverse=True):
        raise ValueError("Expected a nonempty prefix of ones followed by optional zeros")
    if args.syn_steps < 1:
        raise ValueError("syn_steps must be positive")
    if args.AccTest and args.dataset != "Tiny":
        args.ipc = 10  # fixture is IPC=10; establish this before labels and batch_syn
    if (args.AccTest):
        set_random_seed(42)
        args.Iteration = 0

    prep_time = 0
    test_time = 0
    kernel_time = 0
    system_time = 0
    syn_time = 0
    bwd_time = 0
    param_dist_time = 0

    Fuse = int(args.Fuse)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    pre_start = time.time()

    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")
    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")
    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    eval_it_pool = []
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    im_res = im_size[0]
    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    if args.dsa:
        # args.epoch_eval_train = 1000
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
    args.distributed = False # 多卡情况目前不考虑
    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])
    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")
    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]
    ''' initialize the synthetic data '''
    label_syn = torch.tensor([np.ones(args.ipc,dtype=np.int_)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    if args.texture:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0]*args.canvas_size, im_size[1]*args.canvas_size), dtype=torch.float)
    else:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)
    syn_lr = torch.tensor(args.lr_teacher).to(args.device)
    if args.pix_init == 'real':
        # print('initialize synthetic data from random real images')
        if args.texture:
            for c in range(num_classes):
                for i in range(args.canvas_size):
                    for j in range(args.canvas_size):
                        image_syn.data[c * args.ipc:(c + 1) * args.ipc, :, i * im_size[0]:(i + 1) * im_size[0],
                        j * im_size[1]:(j + 1) * im_size[1]] = torch.cat(
                            [get_images(c, 1).detach().data for s in range(args.ipc)])
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print('initialize synthetic data from random noise')
    if(args.AccTest ):
        if( args.dataset == "Tiny"):
            image_syn = torch.load("./test_script/script/in_Tiny.pt")
        else:
        # elif( args.dataset == "CIFAR10" or args.dataset == "CIFAR100"):
            args.ipc = 10
            image_syn = torch.load("./test_script/script/in_ip10.pt")
    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr)    
    optimizer_img.zero_grad()
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    expert_files = []
    n = 0
    while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
        expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
        n += 1
    if args.max_files is not None:
        expert_files = expert_files[:args.max_files]
    if not expert_files:
        raise ValueError("No replay buffers selected at {}".format(expert_dir))

    expert_idx = 0
    if args.load_all:
        buffer, shape_list = preload_expert_groups(
            expert_files, args.max_experts, fuse_mask_list, args.expert_epochs
        )
    else:
        print("loading file {}".format(expert_files[0]))
        buffer = torch.load(expert_files[0])[:args.max_experts]
        if not buffer:
            raise ValueError("No expert trajectories selected in {}".format(expert_files[0]))
        shape_list = [p.shape for p in buffer[0][0]]
    student_net = get_network(args.model, channel, num_classes, im_size, dist=False, v_fuse=False).to(args.device)
    student_net.train()
    ce_one, grad_many = make_step(student_net, shape_list, criterion)
    print(f"Native vmap: mode={args.vmap_mode}, K={Fuse}, active={bwd_Fuse}; v_fuse is ignored")
    if (args.AccTest):
        warmup = 0
    else:
        warmup = 3


    torch.cuda.synchronize()

    args.Iteration += warmup
    pre_end = time.time()


    for it in range(0, args.Iteration+1):
        if it >= warmup:
            if (args.use_barrier):
                torch.cuda.synchronize()
            if it == warmup:
                torch.cuda.synchronize()  # exclude pending warmup kernels
                total_start = time.time()
            pre_end = time.time()
        if args.load_all:
            start_cpu, target_cpu = buffer[expert_idx]
            expert_idx = (expert_idx + 1) % len(buffer)
            initial = start_cpu.to(args.device, copy=True)
            target = target_cpu.to(args.device)
        else:
            trajectory = buffer[expert_idx]
            expert_idx += Fuse
            while expert_idx >= len(buffer):
                expert_idx -= len(buffer)
                buffer = torch.load(expert_files[0])[:args.max_experts]
            # Match Flex streaming: one expert supplies every lane this iteration.
            initial = flat(trajectory[0]).to(args.device).repeat(Fuse, 1)
            target = flat(trajectory[args.expert_epochs]).to(args.device).repeat(bwd_Fuse, 1)
        indices_chunks = []
        if it >= warmup:
            if args.use_barrier:
                torch.cuda.synchronize()
            syn_start = time.time()
        staged = args.vmap_mode == "staged" and bwd_Fuse < Fuse
        student_params = initial
        checkpoints, batches = [], []

        for step in range(args.syn_steps):
            if not indices_chunks:
                indices_chunks = list(torch.arange(len(image_syn)).split(args.batch_syn))
            idx = indices_chunks.pop()
            batches.append(idx)
            if staged:
                checkpoints.append(student_params[:bwd_Fuse].detach().clone())
            with torch.set_grad_enabled(not staged):
                x, this_y = image_syn[idx], label_syn[idx]
                grad, ces = grad_many(student_params, x, this_y)
                student_params = student_params - syn_lr * grad
                del x, grad
        if it >= warmup:
            if args.use_barrier and args.split_fwd_barrier:
                torch.cuda.synchronize()
            syn_end = time.time()
        denom = (initial[:bwd_Fuse] - target).square().sum() / bwd_Fuse
        diff = student_params[:bwd_Fuse] - target
        grand_loss = diff.square().sum() / denom
        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()
        if staged:
            adj = 2 * diff / denom
            dx, dlr = torch.zeros_like(image_syn), torch.zeros_like(syn_lr)
            for prev, idx in zip(reversed(checkpoints), reversed(batches)):
                prev.requires_grad_()
                x = image_syn[idx].detach().requires_grad_()
                g_active, _ = grad_many(prev, x, label_syn[idx])
                nxt = prev - syn_lr * g_active
                adj, gx, glr = torch.autograd.grad(nxt, (prev, x, syn_lr), adj)
                dx.index_add_(0, idx.to(dx.device), gx)
                dlr.add_(glr)
            image_syn.grad, syn_lr.grad = dx, dlr
        else:
            grand_loss.backward()
        if args.AccTest:
            expected = serial_reference(ce_one, initial, target, image_syn, label_syn, syn_lr, batches)
            for name, got, want in zip(("params", "CE", "loss", "image gradient", "lr gradient"),
                                       (student_params, ces, grand_loss, image_syn.grad, syn_lr.grad), expected):
                torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-5, msg=lambda msg: name + ": " + msg)
            print("AccTest passed (all tensors vs serial autograd)")
            print("--Celoss--", ces.sum().item(), "--GradLoss--", grand_loss.item(), "----GRAD-----", image_syn.grad.sum().item())
        optimizer_img.step()
        optimizer_lr.step()
        del ces, student_params, diff, grand_loss, checkpoints
        if staged:
            del prev, nxt, g_active, x, gx, glr, adj, dx, dlr
        if it >= warmup:
            if args.use_barrier:
                torch.cuda.synchronize()
            iter_end = time.time()
            if args.use_barrier and args.split_fwd_barrier:
                prep_time += (syn_start- pre_end) # 从iter开始一直到内层循环。包含了data loading以及system 操作。
                syn_time += (syn_end-syn_start) # 内层循环的时间
                bwd_time += (iter_end-syn_end) # 广义的backward 时间（还有一些数据准备）
            else:
                system_time += (syn_start- pre_end) # 从iter开始一直到内层循环。包含了data loading以及system 操作。
                kernel_time += (iter_end-syn_start) # 内层循环的时间，只包含了kernel以及一些系统操作。
    torch.cuda.synchronize()
    iter_end = time.time()
    print("------------TESTING FINISHED-------------")
    if (not args.use_barrier):
        print("total time: ", iter_end - total_start)
        print("kernel time( without barrier): ", kernel_time)
    elif (args.split_fwd_barrier):
        print("prepare time (", args.Iteration ,"): ", prep_time)
        print("syn_time     (", args.Iteration ,"): ", syn_time)
        print("backward_time(", args.Iteration ,"): ", bwd_time)
    else:
        print("system time  (", args.Iteration ,"): ", system_time)
        print("Kernel time  (", args.Iteration ,"): ", kernel_time)
    print("峰值cache使用:", torch.cuda.max_memory_reserved() / 1024**2, "MB") # 你的 Tensor 实际占用了多少显存（真实使用量）
    print("峰值tensor使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB") # PyTorch CUDA 内存缓存池占用的显存（包含已分配+缓存未释放的）



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument("--vmap-mode", choices=("whole", "staged"), default="staged") # 只有在flex 1-0s 的时候有意义。区别就是double bwd 和bwd2-1用几个fuse
    parser.add_argument(   '--fuse_mask_list',   type=int,   nargs='+',       required=True,        help='fuse mask list, e.g. 1 1 0')
    parser.add_argument('--Fuse', type=str, default="1", help='num of models being stacked')
    parser.add_argument('--v_fuse', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--use-barrier', dest='use_barrier', action=argparse.BooleanOptionalAction, default=False, help='use explicit cuda.synchronize timing')
    parser.add_argument('--split-fwd-barrier', action=argparse.BooleanOptionalAction, default=False, help='split timing of fwd & bwd')
    parser.add_argument('--AccTest', type=bool, default=False, help='num of models being stacked')

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

    parser.add_argument('--load_all', action='store_true', help="preload shuffled, packed expert groups into CPU RAM before the loop")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    args = parser.parse_args()

    main(args)# 7.18 sys overhgead还是比较多。按照gpt意见修改一下
