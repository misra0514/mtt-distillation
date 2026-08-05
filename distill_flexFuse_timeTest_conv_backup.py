# 5.24 因为conv需要修改写法了。所以保存一版旧版用来保证精度问题。

# 

# 2.24
# 从v2复制来的stable版本！ 正式更名成conv！ 
# TODO: NOTE 这个版本。或者前面的flex fuse 在 --fuse_mask_list 1  --> 11 的时候，grad都有问题。没有像Batched 一样成倍。
# 同时，这个版本没有做过优化，可能会有一些sys上冗余的计算。
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
import time 
import warnings
from reparam_module import ReparamModule

from networks.networks_stateless import  ConvBlock_double_bwd,ConvBlock_bwd2_1,ConvBlock_bwd1_2,conv3_double_bwd,conv3_bwd
from networks.networks_stateless_basicblock import linear_bwd, conv_bwd, insNormNRelu_bwd, \
linear_double_bwd, conv_double_bwd, insNormNRelu_double_bwd, avgPool_bwd, crossEntropy_bwd, \
    avgPool_double_bwd,bmm_bwd, linerFused_bwd, linearFused_double_bwd,crossEntropy_double_bwd

from utils_flex import build_global_group_mask, fuse_params_with_mask,split_half_snd_dim,recover_params,set_random_seed

# def compute_param_dist_fuse_mean(starting_params_flat, target_params_flat, shape_list, Fuse):
#     """
#     starting_params_flat: 已经 fuse 后、flatten 的 starting_params
#     target_params_flat:   已经 fuse 后、flatten 的 target_params
#     shape_list:           原始（未 fuse）参数 shape 列表
#     Fuse:                 分支数

#     return:
#         param_dist_raw_mean: 先对每个 fuse 的整套参数做 sum，再对 fuse 求平均
#         base_num_params:     原始单模型参数总数（未 fuse）
#     """
#     start_param_list = recover_params(starting_params_flat, shape_list, Fuse)
#     target_param_list = recover_params(target_params_flat, shape_list, Fuse)

#     fuse_dists = []
#     device = starting_params_flat.device
#     dtype = starting_params_flat.dtype

#     for f in range(Fuse):
#         dist_f = torch.zeros((), device=device, dtype=dtype)

#         for sp, tp, base_shape in zip(start_param_list, target_param_list, shape_list):
#             # 标量参数一般不会有，但为了稳一点还是兼容一下
#             if len(base_shape) == 0:
#                 dist_f = dist_f + F.mse_loss(sp, tp, reduction="sum")
#             else:
#                 B0 = base_shape[0]   # 原始未 fuse 的第 0 维
#                 sp_f = sp.narrow(0, f * B0, B0)
#                 tp_f = tp.narrow(0, f * B0, B0)
#                 dist_f = dist_f + F.mse_loss(sp_f, tp_f, reduction="sum")

#         fuse_dists.append(dist_f)

#     param_dist_raw_mean = torch.stack(fuse_dists).mean()

#     # 原始单模型参数个数（未 fuse）
#     base_num_params = sum(int(np.prod(s)) for s in shape_list)

#     return param_dist_raw_mean, base_num_params

def main(args):
    args.model = 'ConvNet' # conv file 目前写死比较好。
    fuse_mask_list = args.fuse_mask_list 
    bwd_Fuse = sum(fuse_mask_list)
    args.Fuse = str(len(fuse_mask_list)) # 对于Flex fuse来说，只用fuse_mask_list控制即可
    if (args.AccTest):
        set_random_seed(42)
        args.Iteration = 0

    prep_time = 0
    test_time = 0
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

    # print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

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

    # wandb.init(sync_tensorboard=False,
    #            project="DatasetDistillation",
    #            job_type="CleanRepo",
    #            config=args,
    #            )
    # args = type('', (), {})()
    # for key in wandb.config._items:
    #     setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    # args.distributed = torch.cuda.device_count() > 1
    args.distributed = False # 多卡情况目前不考虑
    


    # print('Hyper-parameters: \n', args.__dict__)
    # print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    # print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    # for c in range(num_classes):
    #     print('class c = %d: %d real images'%(c, len(indices_class[c])))

    # for ch in range(channel):
    #     print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    ''' initialize the synthetic data '''
    label_syn = torch.tensor([np.ones(args.ipc,dtype=np.int_)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    # TODO: 修改建议：
    # label_syn = torch.from_numpy(
    #     np.repeat(np.arange(num_classes), args.ipc)
    # ).to(dtype=torch.long, device=args.device)



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

    # torch.save(image_syn, "./script/in.pt")
    # exit()


    if(args.AccTest):
        args.ipc = 10
        image_syn = torch.load("./script/in_ip10.pt")
    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    # optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    # optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr)    
    optimizer_img.zero_grad()

    criterion = nn.CrossEntropyLoss().to(args.device)
    # print('%s training begins'%get_time())

    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    # print("Expert Dir: {}".format(expert_dir))

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
        # random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        # random.shuffle(buffer)

    best_acc = {m: 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}

    # TODO: Setups for L2 opt
    # import ctypes
    # curr_stm = torch.cuda.current_stream()
    # cuda = ctypes.CDLL("libcudart.so")
    # cuda.cudaStreamSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    # class cudaStreamAttrValue(ctypes.Structure):
    #     _fields_ = [("accessPolicyWindow", ctypes.c_int * 5)]  # 示例结构，需根据 CUDA 版本调整

    # attr = cudaStreamAttrValue()
    # attr.accessPolicyWindow[0] = 1  # 假设这里是 L2 Residency 配置
    # cuda.cudaStreamSetAttribute(curr_stm, 1, ctypes.byref(attr))
    # TODO: 这里目前遇到一点问题。因为输入还需要随机排序，取样等等操作。在syn开始之前很难确定数组的起始下标，估计需要改原来的代码
    # print(type(image_syn))
    # from StreamBind import bind
    # image_syn = image_syn.contiguous() 
    # indices = torch.randperm(len(image_syn))
    # indices_chunks = list(torch.split(indices, args.batch_syn))
    # these_indices = indices_chunks.pop()
    # x = image_syn[these_indices]
    # bind(0.2 ,0, x)


    # student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model
    student_net = get_network("Conv_Flexfuse_backup"+args.Fuse, channel, num_classes, im_size, dist=False, v_fuse=args.v_fuse).to(args.device)  

    student_net = ReparamModule(student_net)

    if args.distributed:
        student_net = torch.nn.DataParallel(student_net)
    if (args.AccTest):
        warmup = 0
    else:
        warmup = 3
    # warmup = 0

    args.Iteration += warmup

    pre_end = time.time()

    for it in range(0, args.Iteration+1):
        if it >= warmup:
            start = time.time()

        save_this_it = False

        # writer.add_scalar('Progress', it, it)
        # wandb.log({"Progress": it}, step=it)
        # ''' Evaluate synthetic data '''
        # if it in eval_it_pool:
        #     for model_eval in model_eval_pool:
        #         print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
        #         if args.dsa:
        #             print('DSA augmentation strategy: \n', args.dsa_strategy)
        #             print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
        #         else:
        #             print('DC augmentation parameters: \n', args.dc_aug_param)

        #         accs_test = []
        #         accs_train = []
        #         for it_eval in range(args.num_eval):
        #             net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model

        #             eval_labs = label_syn
        #             with torch.no_grad():
        #                 image_save = image_syn
        #             image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification

        #             args.lr_net = syn_lr.item()
        #             _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, texture=args.texture)
        #             accs_test.append(acc_test)
        #             accs_train.append(acc_train)
        #         accs_test = np.array(accs_test)
        #         accs_train = np.array(accs_train)
        #         acc_test_mean = np.mean(accs_test)
        #         acc_test_std = np.std(accs_test)
        #         if acc_test_mean > best_acc[model_eval]:
        #             best_acc[model_eval] = acc_test_mean
        #             best_std[model_eval] = acc_test_std
        #             save_this_it = True
        #         print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
        #         wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
        #         wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
        #         wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
        #         wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)


        # if it in eval_it_pool and (save_this_it or it % 1000 == 0):
        #     with torch.no_grad():
        #         image_save = image_syn.cuda()

        #         save_dir = os.path.join(".", "logged_files", args.dataset, wandb.run.name)

        #         if not os.path.exists(save_dir):
        #             os.makedirs(save_dir)

        #         torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
        #         torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_{}.pt".format(it)))

        #         if save_this_it:
        #             torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt".format(it)))
        #             torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_best.pt".format(it)))

        #         wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)

        #         if args.ipc < 50 or args.force_save:
        #             upsampled = image_save
        #             if args.dataset != "ImageNet":
        #                 upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
        #                 upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
        #             grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
        #             wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
        #             wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

        #             for clip_val in [2.5]:
        #                 std = torch.std(image_save)
        #                 mean = torch.mean(image_save)
        #                 upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
        #                 if args.dataset != "ImageNet":
        #                     upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
        #                     upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
        #                 grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
        #                 wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

        #             if args.zca:
        #                 image_save = image_save.to(args.device)
        #                 image_save = args.zca_trans.inverse_transform(image_save)
        #                 image_save.cpu()

        #                 torch.save(image_save.cpu(), os.path.join(save_dir, "images_zca_{}.pt".format(it)))

        #                 upsampled = image_save
        #                 if args.dataset != "ImageNet":
        #                     upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
        #                     upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
        #                 grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
        #                 wandb.log({"Reconstructed_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
        #                 wandb.log({'Reconstructed_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

        #                 for clip_val in [2.5]:
        #                     std = torch.std(image_save)
        #                     mean = torch.mean(image_save)
        #                     upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
        #                     if args.dataset != "ImageNet":
        #                         upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
        #                         upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
        #                     grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
        #                     wandb.log({"Clipped_Reconstructed_Images/std_{}".format(clip_val): wandb.Image(
        #                         torch.nan_to_num(grid.detach().cpu()))}, step=it)

        # wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)
        student_net.train()

        # 这里有两个改动：1 不用student_net.parameters() 这个是前向的，应该直接用load上来的size（而且是在fuse 之前）
        # 但是因为fuse 之前的starting_params 长度 不太好获取。反正就是一个值而已。在这里处理一下吧。。
        # num_params 就是一个param的值
        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())]) / (Fuse)
        # print(num_params)

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer): # expert_idx可能类似一个counter，全部读完之后再load
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files): 
                    file_idx = 0
                    # random.shuffle(expert_files)
                # print("loading file {}".format(expert_files[file_idx]))
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
        
        # TODO: 这里可能甚至是有点吃亏的。因为如果单纯算计算时间的话，这里可能并不应该算进去，这个是数据准备阶段的事情，而且没有做过优化，本来说不定可以快一点。

        for index, i in enumerate(target_params):
            if i.ndim  != 0 :
                # target_params[index] = torch.cat([i, i], dim=0)   
                target_params[index] = i.repeat((int(Fuse),) + (1,) * (i.ndim - 1))  
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        # TODO: 2 stu param 作为forward 的flat_param传入，要么在这里修改，要么重载forward
        # BUFFER:             1* 11* 14 * model Params
        # BUFFER 在每个Iter之中，有可能会重新load 新的参数进来。
        # expert_trajectory ： 11 * 14 * model Params
        # starting_params   ： 14 * model Params


        # # shape list 是一个的。后面可能需要expand
        shape_list = [p.shape for p in starting_params]

        mask_params = []
        Fuse = len(fuse_mask_list)    # 关键：Fuse = 分组数

        for index, p in enumerate(starting_params):
            if p.ndim != 0:

                # ===== 1）repeat（第 0 维复制 Fuse 次）=====
                B0 = p.shape[0]
                repeat_shape = (Fuse,) + (1,) * (p.ndim - 1)
                p_rep = p.repeat(repeat_shape)          # shape = [Fuse*B0, ...]
                starting_params[index] = p_rep

                # ===== 2）构造 mask（和 p_rep 同 shape），按 block 切 =====

                # row_mask.shape = [Fuse*B0]
                # fuse_mask_list = [1,1,0] → [True,True,False]
                row_mask = torch.tensor(fuse_mask_list, dtype=torch.bool, device=p.device)
                row_mask = row_mask.repeat_interleave(B0)
                # 现在 row_mask = [T,T,...B0 次, T,T,...B0 次, F,F,...B0 次]

                # 扩展成 p_rep 同 shape（广播）
                view_shape = (Fuse * B0,) + (1,) * (p_rep.ndim - 1)
                mask_param = row_mask.view(view_shape).expand_as(p_rep)

                mask_params.append(mask_param)

        # ===== flatten =====
        student_params = [
            torch.cat([pp.data.to(args.device).reshape(-1) for pp in starting_params], 0)
            .requires_grad_(True)
        ]
        mask = torch.cat([mm.reshape(-1) for mm in mask_params], 0)   # already bool
        del mask_params


        # mask_params = []
        # for index, p in enumerate(starting_params):
        #     if p.ndim != 0:
        #         # ===== 原来就有的：复制权重（不改动） =====
        #         repeat_shape = (int(Fuse),) + (1,) * (p.ndim - 1)
        #         starting_params[index] = p.repeat(repeat_shape)
        #         # ===== 新增：构造对应的 mask =====
        #         # 每次复制一块 mask，值为 fuse_mask_list[k]（0 or 1）
        #         mask_blocks = []
        #         for m in fuse_mask_list:
        #             # 用 full_like 保证 dtype / device 一致，后面再统一 to(args.device)
        #             mask_blocks.append(torch.full_like(p, fill_value=m))
        #         mask_param = torch.cat(mask_blocks, dim=0)   # 和 starting_params[index] 同 shape
        #         mask_params.append(mask_param.to(args.device))
        #     # else:
        #     #     # 标量参数：原代码不处理，这里给一个同 shape 的 mask（全部 1，或者你想要的值）
        #     #     mask_params.append(torch.ones_like(p).to(args.device))

        # # =====  student_params 完全不动 =====
        # student_params = [
        #     torch.cat([pp.data.to(args.device).reshape(-1) for pp in starting_params], 0)
        #         .requires_grad_(True)
        # ]
        # # ===== 新增：把 mask 也 flatten 成一维，和 student_params[0] 对齐 =====
        # mask = torch.cat([mm.reshape(-1).bool() for mm in mask_params], 0)
        # del mask_params

        # for index, i in enumerate(starting_params):
        #     # print(i)
        #     if i.ndim != 0 :
        #         # starting_params[index] = torch.cat([i, i], dim=0) 
        #         starting_params[index] = i.repeat((int(Fuse),) + (1,) * (i.ndim - 1))   
        # student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
        # mask = build_global_group_mask(student_params,fuse_mask_list)




        # student_params, mask = fuse_params_with_mask(starting_params, Fuse, [1,0])
        # student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params  for _ in range(int(Fuse))], 0).requires_grad_(True)]
        # student_params = [torch.cat([item.data.to(args.device).reshape(-1) for p in starting_params  for item in (p, p[0]+"1")], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)


        # with torch.no_grad():
        #     if Fuse != bwd_Fuse:
        #         starting_params_bwd = starting_params[mask]
        #         target_params_bwd = target_params[mask]
        #         init_params_bwd = student_params[0][mask]
        #     else:
        #         starting_params_bwd = starting_params
        #         target_params_bwd = target_params
        #         init_params_bwd = student_params[0]

        #     # 版本 A：保留原函数语义
        #     param_dist, base_num_params = compute_param_dist_fuse_mean(
        #         starting_params_bwd,
        #         target_params_bwd,
        #         shape_list,
        #         bwd_Fuse,
        #     )

        #     # 可选：bwd_Fuse 版本的初始权重也可以提前 recover
        #     (
        #         conv1_w_bwd, conv1_b_bwd, norm1_w_bwd, norm1_b_bwd,
        #         conv2_w_bwd, conv2_b_bwd, norm2_w_bwd, norm2_b_bwd,
        #         conv3_w_bwd, conv3_b_bwd, norm3_w_bwd, norm3_b_bwd,
        #         lin_w_bwd, _
        #     ) = recover_params(init_params_bwd, shape_list, bwd_Fuse)

        syn_images = image_syn

        y_hat = label_syn.to(args.device)

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []

        if it >= warmup:
            if args.use_barrier:
                torch.cuda.synchronize()
            syn_start = time.time()
        conv1_w, conv1_b, norm1_w, norm1_b, conv2_w, conv2_b, norm2_w, norm2_b, conv3_w, conv3_b, norm3_w, norm3_b, lin_w, _  =recover_params(student_params[0],shape_list, Fuse)

        for step in range(args.syn_steps):

            if not indices_chunks:
                indices = torch.arange(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()


            x = syn_images[these_indices]
            this_y = y_hat[these_indices]

            # if args.texture:
            #     x = torch.cat([torch.stack([torch.roll(im, (torch.randint(im_size[0]*args.canvas_size, (1,)), torch.randint(im_size[1]*args.canvas_size, (1,))), (1,2))[:,:im_size[0],:im_size[1]] for im in x]) for _ in range(args.canvas_samples)])
            #     this_y = torch.cat([this_y for _ in range(args.canvas_samples)])

            # if args.dsa and (not args.no_aug):
            #     x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
            # 因为group conv的原因，最开始应该在Channel 维度做cat
            x = x.repeat(1, int(Fuse), 1, 1).requires_grad_(True)
            this_y = this_y.repeat(int(Fuse))

            with torch.no_grad():

                x_conv1,x_norm1, x_pool1,x_conv2,x_norm2, x_pool2,x_conv3,x_norm3, x_pool3, x_lin, x_out  = student_net(x, flat_param=forward_params)
                x_out = x_out.view(-1,num_classes)

                ce_loss = criterion(x_out, this_y)
                ce_loss *= int(Fuse)
                # grad = torch.autograd.grad(ce_loss, student_params[-1], retain_graph=True)[0]
                # dx_norm1,dx_pool1,dx_norm2,dx_pool2,dx_norm3,dx_pool3,dx_out,grad = torch.autograd.grad(ce_loss, [x_norm1,x_pool1,x_norm2,x_pool2,x_norm3,x_pool3,x_out,student_params[-1]] ) # TODO: 可以一次做完的。
                # dx_out= torch.autograd.grad(ce_loss, x_out)[0] # 改成分段计算了。麻烦的点在于weight。需要全部手动改。而且会收到reparam影响
                dx_out = crossEntropy_bwd(x_out, this_y, Fuse)
                dx_lin, dlin_w, dlin_b = linerFused_bwd(x_lin, lin_w, grad_output=dx_out, Fuse=Fuse)
                dx_lin = dx_lin.reshape(-1, student_net.module.net_width * Fuse, 4,4)  # 4*4 可能需要灵活改
                x_lin,x_out,dx_out = split_half_snd_dim([x_lin,x_out,dx_out],fuse_mask_list)
                dx_conv3, dx_norm3, dx_pool3,  dconv3_w , dconv3_b ,dnorm3_w ,dnorm3_b = ConvBlock_bwd1_2(x_conv3, x_norm3, x_pool3, conv3_w, norm3_w, dx_lin, Fuse=Fuse, v_fuse=args.v_fuse)
                del dx_lin
                x_conv3, x_norm3, x_pool3, dx_norm3, dx_pool3 = split_half_snd_dim([x_conv3, x_norm3, x_pool3, dx_norm3, dx_pool3],fuse_mask_list)
                dx_conv2, dx_norm2, dx_pool2, dconv2_w , dconv2_b ,dnorm2_w ,dnorm2_b = ConvBlock_bwd1_2(x_conv2, x_norm2, x_pool2, conv2_w, norm2_w, dx_conv3, Fuse=Fuse, v_fuse=args.v_fuse)
                x_conv2, x_norm2, x_pool2, dx_norm2, dx_pool2 = split_half_snd_dim([x_conv2, x_norm2, x_pool2, dx_norm2, dx_pool2 ],fuse_mask_list)
                del dx_conv3
                # _, dx_norm1, dx_pool1, dconv1_w , dconv1_b ,dnorm1_w ,dnorm1_b = ConvBlock_bwd1_2(x_conv1, x_norm1, x_pool1, conv1_w, norm1_w, dx_conv2, Fuse=Fuse)
                dx_pool1 = avgPool_bwd( x_pool1, grad_output= dx_conv2 )
                del dx_conv2
                # dx_conv2 = split_half_snd_dim([dx_conv2], fuse_mask_list)[0]
                # dx_lin_d1.copy_(dx_lin_d1[:, :, ...].contiguous())
                dx_norm1, dnorm1_w, dnorm1_b = insNormNRelu_bwd(x_norm1, norm1_w, x_pool1, grad_output=dx_pool1,v_fuse=args.v_fuse)
                x_norm1 = split_half_snd_dim([x_norm1],fuse_mask_list)[0]
                x_pool1 = split_half_snd_dim([x_pool1 ],fuse_mask_list)[0]
                dx_pool1 = split_half_snd_dim([dx_pool1],fuse_mask_list)[0]
                # del dx_pool_d1
                _, dconv1_w, dconv1_b = conv_bwd(x_conv1, conv1_w, grad_output=dx_norm1, groups=Fuse)
                x_conv1 = split_half_snd_dim([x_conv1 ],fuse_mask_list)[0]
                dx_norm1 = split_half_snd_dim([dx_norm1 ],fuse_mask_list)[0]
                grad = [dconv1_w , dconv1_b ,dnorm1_w ,dnorm1_b, dconv2_w , dconv2_b ,dnorm2_w ,dnorm2_b,dconv3_w , dconv3_b ,dnorm3_w ,dnorm3_b,dlin_w, dlin_b]
            grad = torch.cat([mm.reshape(-1).detach().requires_grad_(True) for mm in grad], 0)   # already bool
            student_params.append(student_params[-1] - syn_lr *  grad)

        if it >= warmup:
            if args.use_barrier:
                torch.cuda.synchronize()
            syn_end = time.time()

        with torch.no_grad():
            # weight = student_params[0] # weight是原始参数，不加dw
            # if Fuse != bwd_Fuse:

            #     # TODO: 对weights 做了一些masking。难道是这里影响了性能？ 这里大概花了1s左右。其实直接把load时候变量留下来就好了。
            #     weight = weight[mask]
            #     student_params[-1] = student_params[-1][mask]
            #     starting_params = starting_params[mask]
            #     target_params = target_params[mask]
            #     conv1_w, conv1_b, norm1_w, norm1_b, conv2_w, conv2_b, norm2_w, norm2_b, conv3_w, conv3_b, norm3_w, norm3_b, lin_w, _  =recover_params(student_params[0][mask],shape_list, bwd_Fuse)
            if Fuse != bwd_Fuse:
                final_params_bwd = student_params[-1][mask]
                starting_params_bwd = starting_params[mask]
                target_params_bwd = target_params[mask]
                init_params_bwd = student_params[0][mask]
                student_params[-1] = final_params_bwd
                starting_params = starting_params_bwd
                target_params = target_params_bwd
                (
                    conv1_w, conv1_b, norm1_w, norm1_b,
                    conv2_w, conv2_b, norm2_w, norm2_b,
                    conv3_w, conv3_b, norm3_w, norm3_b,
                    lin_w, _
                ) = recover_params(init_params_bwd, shape_list, bwd_Fuse)
                
                
            param_loss = torch.tensor(0.0).to(args.device)
            param_dist = torch.tensor(0.0).to(args.device)

            param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum") # 好像是因为reduction的原因。。。。
            # param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
            # 不可以直接用param dist ！ 对每个层，分别对每个 fuse 计算 MSE
            # # TODO: 后面可以优化一下，直接在init的时候把param_dist 算好。反正只是一个数值
            # start_param_list = recover_params(starting_params, shape_list, Fuse)
            # target_param_list = recover_params(target_params, shape_list, Fuse)
            # fuse_losses = []
            # for sp, tp in zip(start_param_list, target_param_list):
            #     B0 = sp.shape[0] // Fuse
            #     for f in range(Fuse):
            #         sp_f = sp[f*B0:(f+1)*B0].reshape(-1)
            #         tp_f = tp[f*B0:(f+1)*B0].reshape(-1)
            #         fuse_losses.append(F.mse_loss(sp_f, tp_f, reduction="mean"))
            # param_dist = torch.stack(fuse_losses).mean()
            # print("v_fuse",args.v_fuse)
            param_dist_time_start = time.time()

            # param_dist_raw_mean, base_num_params = compute_param_dist_fuse_mean(
            #     starting_params,
            #     target_params,
            #     shape_list,
            #     bwd_Fuse
            # )
            param_dist_raw_mean = (
                starting_params - target_params
            ).square().sum() / bwd_Fuse
            param_dist_time_end = time.time()

            param_dist = param_dist + param_dist_raw_mean
            # diff0 = starting_params - target_params
            # param_dist = diff0.square().sum() / bwd_Fuse

            param_loss_list.append(param_loss)
            param_dist_list.append(param_dist)
            # param_loss /= num_params
            # param_dist /= num_params
            param_loss /= param_dist
            grand_loss = param_loss # 是为了抵消num_params变化带来的影响。但是flex fuse 之后num_params没有变化（还是Fuse）
            optimizer_img.zero_grad()
            optimizer_lr.zero_grad()

            ddx_conv = torch.zeros_like(x_conv1).cuda()
            ddw = 2*(student_params[-1]- target_params)/param_dist
            ddw *= (-syn_lr)
            # if args.use_barrier:
            #     torch.cuda.synchronize()
            #     temp_start = time.time()
            del grad, target_params, starting_params
            # for _ in student_params:
            #     del _
            ddconv1_w,ddconv1_b,ddnorm1_w,ddnorm1_b,ddconv2_w,ddconv2_b,ddnorm2_w,ddnorm2_b,ddconv3_w,ddconv3_b,ddnorm3_w,ddnorm3_b,ddlin_w,ddlin_b  =recover_params(ddw, shape_list,bwd_Fuse )
            # ddx_conv2, dxconv1_d2, _,dx_norm1_d2,_ = ConvBlock_double_bwd(x_conv1, x_norm1, x_pool1, dx_norm1, dx_pool1, \
            #                                                                 ddx_conv, conv1_w, norm1_w,ddconv1_w, ddconv1_b, ddnorm1_w, ddnorm1_b,bwd_Fuse  )
            # del ddx_conv,dx_norm1,dx_pool1
            ddx_norm, dxconv1_d2, _ = conv_double_bwd(ddx_conv, ddconv1_w, ddconv1_b, dx_norm1, conv1_w, x_conv1, groups_=bwd_Fuse )
            del ddx_conv,ddconv1_w,ddconv1_b, dx_norm1
            ddx_pool, dx_norm1_d2, _ = insNormNRelu_double_bwd(ddx_norm, ddnorm1_w, ddnorm1_b, dx_pool1, x_pool1, norm1_w, x_norm1, v_fuse=args.v_fuse)
            del ddx_norm,dx_pool1,ddnorm1_w,ddnorm1_b
            ddx_conv2 = avgPool_double_bwd(ddx_pool)    
            del ddx_pool

            del dconv1_w, dconv1_b, dnorm1_w, dnorm1_b
            del dconv2_w, dconv2_b, dnorm2_w, dnorm2_b
            del dconv3_w, dconv3_b, dnorm3_w, dnorm3_b
            del dlin_w, dlin_b

            ddx_conv3, dxconv2_d2, _,dx_norm2_d2,_ = ConvBlock_double_bwd(x_conv2, x_norm2, x_pool2, dx_norm2, dx_pool2, \
                                                                            ddx_conv2,conv2_w, norm2_w, ddconv2_w, ddconv2_b, ddnorm2_w, ddnorm2_b,bwd_Fuse, v_fuse=args.v_fuse )
            del ddx_conv2,dx_norm2,dx_pool2
            ddx_lin, dxconv3_d2, _,dx_norm3_d2,_ = ConvBlock_double_bwd(x_conv3, x_norm3, x_pool3, dx_norm3, dx_pool3, \
                                                                        ddx_conv3, conv3_w,norm3_w, ddconv3_w, ddconv3_b, ddnorm3_w, ddnorm3_b,bwd_Fuse, v_fuse=args.v_fuse )
            del ddx_conv3,dx_norm3,dx_pool3
            ddx_out, dx_lin_d2, _ = linearFused_double_bwd(x_lin,lin_w, dx_out, ddx_lin, ddlin_w ,ddlin_b ,bwd_Fuse)
            del dx_out, ddx_lin

            dx_out_d1 = crossEntropy_double_bwd(x_out, ddx_out, bwd_Fuse)
            del ddx_out,x_out

            dx_lin_d1, _, _ = linerFused_bwd(x_lin, lin_w, grad_output=dx_out_d1, Fuse=bwd_Fuse)
            dx_lin_d1 = dx_lin_d1.reshape(-1, student_net.module.net_width * bwd_Fuse, 4,4)  # 这里128是net_width， 但是distill里面好像没有这个变量。。
            dx_lin_d1 += dx_lin_d2 
            del x_lin,dx_out_d1,dx_lin_d2
            dx_conv3_d1 , _ ,_ ,_ ,_ = ConvBlock_bwd2_1(x_conv3, x_norm3, x_pool3,conv3_w, norm3_w, dx_lin_d1,dx_norm3_d2,dxconv3_d2 ,Fuse=bwd_Fuse, v_fuse=args.v_fuse )
            del dx_norm3_d2,dxconv3_d2, x_conv3, x_norm3,x_pool3,dx_lin_d1
            dx_conv2_d1 , _ ,_ ,_ ,_ = ConvBlock_bwd2_1(x_conv2, x_norm2, x_pool2, conv2_w, norm2_w, dx_conv3_d1,dx_norm2_d2,dxconv2_d2 ,Fuse=bwd_Fuse, v_fuse=args.v_fuse)
            del dx_norm2_d2,dxconv2_d2, x_conv2, x_norm2,x_pool2,dx_conv3_d1
            dx_conv1_d1 , _ ,_ ,_ ,_ = ConvBlock_bwd2_1(x_conv1, x_norm1, x_pool1,conv1_w, norm1_w, dx_conv2_d1,dx_norm1_d2,dxconv1_d2 ,Fuse=bwd_Fuse, v_fuse=args.v_fuse)
            del dx_norm1_d2,dxconv1_d2, x_conv1, x_norm1,x_pool1, dx_conv2_d1
            # ddw_output = [ddconv1_w,ddconv1_b,ddnorm1_w,ddnorm1_b,ddconv2_w,ddconv2_b,ddnorm2_w,ddnorm2_b,ddconv3_w,ddconv3_b,ddnorm3_w,ddnorm3_b,ddlin_w,ddlin_b ]
            # if args.use_barrier:
            #     torch.cuda.synchronize()
            # temp_end = time.time()

        if(args.AccTest):
            print("--Celoss--",ce_loss.item())
            print("--GradLoss--",grand_loss.item())
            print("----GRAD-----", dx_conv1_d1.sum().item()) 

        optimizer_img.step()
        optimizer_lr.step()
        if it >= warmup:
            if args.use_barrier:
                torch.cuda.synchronize()
            iter_end = time.time()
            prep_time += (syn_start- start) # 从iter开始一直到内层循环
            syn_time += (syn_end-syn_start) # 内层循环的时间
            bwd_time += (iter_end-syn_end) # 广义的backward 时间（还有一些数据准备）
            param_dist_time += (param_dist_time_end-param_dist_time_start) # 计算param_dist的时间

            # test_time +=(temp_end-temp_start)
        # wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
        #            "Start_Epoch": start_epoch})

        # for _ in student_params:
        #     del _

        # if it%10 == 0:
    #     #     print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))
    #     print("alloc", torch.cuda.memory_allocated() / 1024**2,
    #   "reserved", torch.cuda.memory_reserved() / 1024**2)


    iter_end = time.time()
    print("------------FIN TIME-------------")
    print("time sumation: ", iter_end - pre_end)
    print("prepare time (", args.syn_steps ,"): ", prep_time)
    print("syn_time     (", args.syn_steps ,"): ", syn_time)
    print("backward_time(", args.syn_steps ,"): ", bwd_time)
    print("param_dist_time(", args.syn_steps ,"): ", param_dist_time)
    # print("test_time (", args.syn_steps ,"): ", test_time)

    print("峰值cache使用:", torch.cuda.max_memory_reserved() / 1024**2, "MB") # 你的 Tensor 实际占用了多少显存（真实使用量）
    print("峰值tensor使用:", torch.cuda.max_memory_allocated() / 1024**2, "MB") # PyTorch CUDA 内存缓存池占用的显存（包含已分配+缓存未释放的）

    # wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument(   '--fuse_mask_list',   type=int,   nargs='+',       required=True,        help='fuse mask list, e.g. 1 1 0')

    parser.add_argument('--Fuse', type=str, default="1", help='num of models being stacked')
    parser.add_argument('--v_fuse', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--use-barrier', dest='use_barrier', action=argparse.BooleanOptionalAction, default=False, help='use explicit cuda.synchronize timing')

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


