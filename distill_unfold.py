import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import wandb
import copy
import random
from reparam_module import ReparamModule
import torch.profiler

import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭自动优化，确保计算确定性
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 保证 CUDA 计算稳定（仅对 PyTorch 1.8+ 有效）

set_random_seed(42)
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

def main(args):

    pre_start = time.time()

    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

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

    wandb.init(sync_tensorboard=False,
               project="DatasetDistillation",
               job_type="CleanRepo",
               config=args,
               )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1


    # print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    # TODO: IMages_all 占用了大约600M（batch100）
    images_all = torch.cat(images_all, dim=0).to("cpu")
    # images_all = torch.cat(images_all, dim=0).to(args.device)
    # labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    # for c in range(num_classes):
    #     print('class c = %d: %d real images'%(c, len(indices_class[c])))

    # for ch in range(channel):
    #     print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

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
        print('initialize synthetic data from random real images')
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

    image_syn = torch.load("./script/in.pt")

    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    # optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5) # TODO: 关掉动量，保证梯度下降方向一致
    # optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr)
    optimizer_img.zero_grad()

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    print("Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        # random.shuffle(expert_files)  # TODO: 为了测精度把随机输入全去掉了
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        # random.shuffle(buffer) # TODO: 为了测精度把随机输入全去掉了

    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}
    student_net = get_network('ConvNet_unfold', channel, num_classes, im_size, dist=False).to(args.device)  # get a random model
    student_net = ReparamModule(student_net)
    if args.distributed:
        student_net = torch.nn.DataParallel(student_net)


    # TODO:  compile
    # student_net = torch.compile(student_net, mode="reduce-overhead")

    
    # syn_images = image_syn.detach().requires_grad_(True)
    # y_hat = label_syn.to(args.device).detach()
    # expert_trajectory = buffer[expert_idx]
    # starting_params = expert_trajectory[0]
    # target_params = expert_trajectory[args.expert_epochs]
    # target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)
    # student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
    # starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)
    # num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])
    # indices = torch.arange(len(syn_images))
    # indices_chunks = list(torch.split(indices, args.batch_syn))
    # these_indices = indices_chunks.pop()
    # x = syn_images[these_indices]
    # this_y = y_hat[these_indices]
    # forward_paramst = student_params[-1].detach().requires_grad_(True)
    # criteriont=nn.CrossEntropyLoss().to(args.device)
    # grad = student_net(x,target =this_y, criterion=criteriont, flat_param=forward_paramst)
    # param_loss = torch.tensor(0.0).to(args.device)
    # param_dist = torch.tensor(0.0).to(args.device)
    # param_loss += torch.nn.functional.mse_loss(forward_paramst+grad, target_params, reduction="sum")
    # param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
    # param_loss /= num_params
    # param_dist /= num_params
    # grand_loss = param_loss
    # optimizer_img.zero_grad()
    # optimizer_lr.zero_grad()
    # grand_loss.backward()
    # optimizer_img.step()
    # optimizer_lr.step()
    # print("当前显存使用:", torch.cuda.max_memory_reserved() / 1024**2, "MB")
    # print("峰值显存使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    # exit()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    pre_end = time.time()

    for it in range(0, args.Iteration+1):
        start = time.time()
        save_this_it = False
        student_net.train()
        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])
        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    # random.shuffle(expert_files)
                print("loading file {}".format(expert_files[file_idx]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                # random.shuffle(buffer)
        # start_epoch = np.random.randint(0, args.max_start_epoch)
        start_epoch = 0
        starting_params = expert_trajectory[start_epoch]
        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)
        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)
        syn_images = image_syn
        y_hat = label_syn.to(args.device)

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []

        # with torch.profiler.profile(
        # activities=[torch.profiler.ProfilerActivity.CUDA],profile_memory=True,record_shapes=True,with_stack=True) as prof:
        syn_start = time.time()
        for step in range(args.syn_steps):
            if not indices_chunks:
                # indices = torch.randperm(len(syn_images))
                indices = torch.arange(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))
            these_indices = indices_chunks.pop()
            x = syn_images[these_indices]
            this_y = y_hat[these_indices]
            if args.texture:
                x = torch.cat([torch.stack([torch.roll(im, (torch.randint(im_size[0]*args.canvas_size, (1,)), torch.randint(im_size[1]*args.canvas_size, (1,))), (1,2))[:,:im_size[0],:im_size[1]] for im in x]) for _ in range(args.canvas_samples)])
                this_y = torch.cat([this_y for _ in range(args.canvas_samples)])
            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)
            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
            # with torch.no_grad():
            grad = student_net(x,target =this_y, criterion=criterion, flat_param=forward_params)
            # TODO: 这里debug了一下。发现每次append的量也是requires_grad 的。所以就不知道哪里有可能导致计算图没连上
            student_params.append(student_params[-1] - syn_lr * grad)

        # print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))

        syn_end = time.time()
        # print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)
        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)
        param_loss /= num_params
        param_dist /= num_params
        param_loss /= param_dist
        grand_loss = param_loss
        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()

        # print("-------------LOSS-------------")
        # print(grand_loss.item())
        grand_loss.backward()

        print("-------------GRADX-------------")
        print( syn_images.grad.sum().item())

        optimizer_img.step()
        optimizer_lr.step()
        print("峰值cache使用:(nvidia-smi)", torch.cuda.max_memory_reserved() / 1024**2, "MB")
        print("峰值tensor使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB")

        iter_end = time.time()
        syn_time = syn_end-syn_start
        iter_time = iter_end-syn_start
        # print("--TIME---")
        # print("prepare time (", args.syn_steps ,"): ", syn_start- start)
        # print("syn_time     (", args.syn_steps ,"): ", syn_time)
        # print("backward_time(", args.syn_steps ,"): ", iter_time-syn_time)
        # print("sum time (", args.syn_steps ,"): ", iter_end- start)

        # wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
        #            "Start_Epoch": start_epoch})

        for _ in student_params:
            del _

    iter_end = time.time()
    print("------------FIN TIME-------------")
    print(iter_end - pre_end)


    wandb.finish()


if __name__ == '__main__':
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

    main(args)


