#!/usr/bin/env bash
set -e

# Original DC
# WANDB_SILENT=true python main_optimized.py \
WANDB_SILENT=true python main_fused.py \
    --num_exp=1 \
    --method=DC \
    --dataset=CIFAR10 \
    --model=VGG11 \
    --ipc=10 \
    --Iteration=50 \
    --lr_img=0.1 \
    --lr_net=0.01 \
    --batch_real=256 \
    --batch_train=256 \
    --init=real \
    --data_path=/scratch/yguo25/files/mtt-distillation/dataset \
    --Fuse=2\
    --no-v_fuse \
    --no-use-barrier \
    # --AccTest=True \


# Fused DC, Fuse=2 checkpoints
# WANDB_SILENT=true python main_fused.py \
#     --method=DC \
#     --dataset=CIFAR10 \
#     --model=ConvNet \
#     --ipc=10 \
#     --Iteration=50 \
#     --lr_img=0.1 \
#     --lr_net=0.01 \
#     --batch_real=256 \
#     --batch_train=256 \
#     --init=real \
#     --data_path=/scratch/yguo25/files/mtt-distillation/dataset \
#     --Fuse=2 \
#     --v_fuse \
#     --no-use-barrier \
#     --AccTest=True

# 要分别测真实的 FWD/BWD GPU 时间时，把上面对应命令里的
# --no-use-barrier 改成 --use-barrier。
# 不做正确性验证、正常多轮测试时，去掉 --AccTest=True；时间和显存仍会输出。