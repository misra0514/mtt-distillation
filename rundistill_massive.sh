#!/bin/sh
# fixed 这次是对好精度之后测的。

echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH10 , BASECODE " >> testlogs_distill_fixed.txt

WANDB_SILENT=true python distill_original.py --dataset=CIFAR10 --pix_init=real --ipc=1  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=100 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=2 >> testlogs_distill_fixed.txt

WANDB_SILENT=true python distill_original.py --dataset=CIFAR10 --pix_init=real --ipc=1  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=100 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=2 >> testlogs_distill_fixed.txt


# BATCH10 , fuse 2 
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH10 , fuse 2 " >> testlogs_distill_fixed.txt
WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=1  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=100 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=2 >> testlogs_distill_fixed.txt

# BATCH10 , fuse 4
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH10 , fuse 4 " >> testlogs_distill_fixed.txt

WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=1  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=100 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=4 >> testlogs_distill_fixed.txt

# BATCH10 , fuse 6
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH10 , fuse 6 " >> testlogs_distill_fixed.txt

WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=1  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=100 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=6 >> testlogs_distill_fixed.txt

# BATCH10 , fuse 8
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH10 , fuse 8 " >> testlogs_distill_fixed.txt

WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=1  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=100 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=8 >> testlogs_distill_fixed.txt






# BATCH50 , fuse 2 
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH50 , BASECODE" >> testlogs_distill_fixed.txt

WANDB_SILENT=true python distill_original.py --dataset=CIFAR10 --pix_init=real --ipc=5  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=100 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=2 >> testlogs_distill_fixed.txt
WANDB_SILENT=true python distill_original.py --dataset=CIFAR10 --pix_init=real --ipc=5  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=100 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=2 >> testlogs_distill_fixed.txt

# BATCH50 , fuse 2 
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH50 , fuse 2 " >> testlogs_distill_fixed.txt

WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=5  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=100 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=2 >> testlogs_distill_fixed.txt

# BATCH50 , fuse 4
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH50 , fuse 4 " >> testlogs_distill_fixed.txt

WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=5  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=100 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=4 >> testlogs_distill_fixed.txt

# BATCH50 , fuse 6
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH50 , fuse 6 " >> testlogs_distill_fixed.txt
WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=5  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=100 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=6 >> testlogs_distill_fixed.txt

# BATCH50 , fuse 8
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH50 , fuse 8 " >> testlogs_distill_fixed.txt

WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=5  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=100 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=8 >> testlogs_distill_fixed.txt




echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH100 , BASE CODE " >> testlogs_distill_fixed.txt
WANDB_SILENT=true python distill_original.py --dataset=CIFAR10 --pix_init=real --ipc=10  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=100 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=2 >> testlogs_distill_fixed.txt
WANDB_SILENT=true python distill_original.py --dataset=CIFAR10 --pix_init=real --ipc=10  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=100 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=2 >> testlogs_distill_fixed.txt


# BATCH100 , fuse 2 
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH100 , fuse 2 " >> testlogs_distill_fixed.txt
WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=10  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=100 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=2 >> testlogs_distill_fixed.txt

# BATCH100 , fuse 4 
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH100 , fuse 4 " >> testlogs_distill_fixed.txt
WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=10  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=100 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=4 >> testlogs_distill_fixed.txt

# BATCH100 , fuse 6 
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH100 , fuse 6 " >> testlogs_distill_fixed.txt
WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=10  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=100 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=6 >> testlogs_distill_fixed.txt

# BATCH100 , fuse 8
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH100 , fuse 8 " >> testlogs_distill_fixed.txt
WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=10  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=100 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=8 >> testlogs_distill_fixed.txt








echo "==============================" >> testlogs_distill_fixed.txt
echo "==============================" >> testlogs_distill_fixed.txt
echo "==============================" >> testlogs_distill_fixed.txt
echo "==============================" >> testlogs_distill_fixed.txt


echo "==ITER 1000==" >> testlogs_distill_fixed.txt



echo "==============================" >> testlogs_distill_fixed.txt
echo "==============================" >> testlogs_distill_fixed.txt
echo "==============================" >> testlogs_distill_fixed.txt
echo "==============================" >> testlogs_distill_fixed.txt









#!/bin/sh
# fixed 这次是对好精度之后测的。

echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH10 , BASECODE " >> testlogs_distill_fixed.txt

WANDB_SILENT=true python distill_original.py --dataset=CIFAR10 --pix_init=real --ipc=1  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=1000 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=2 >> testlogs_distill_fixed.txt

WANDB_SILENT=true python distill_original.py --dataset=CIFAR10 --pix_init=real --ipc=1  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=1000 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=2 >> testlogs_distill_fixed.txt


# BATCH10 , fuse 2 
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH10 , fuse 2 " >> testlogs_distill_fixed.txt
WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=1  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=1000 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=2 >> testlogs_distill_fixed.txt

# BATCH10 , fuse 4
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH10 , fuse 4 " >> testlogs_distill_fixed.txt

WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=1  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=1000 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=4 >> testlogs_distill_fixed.txt

# BATCH10 , fuse 6
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH10 , fuse 6 " >> testlogs_distill_fixed.txt

WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=1  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=1000 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=6 >> testlogs_distill_fixed.txt

# BATCH10 , fuse 8
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH10 , fuse 8 " >> testlogs_distill_fixed.txt

WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=1  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=1000 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=8 >> testlogs_distill_fixed.txt






# BATCH50 , fuse 2 
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH50 , BASECODE" >> testlogs_distill_fixed.txt

WANDB_SILENT=true python distill_original.py --dataset=CIFAR10 --pix_init=real --ipc=5  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=1000 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=2 >> testlogs_distill_fixed.txt
WANDB_SILENT=true python distill_original.py --dataset=CIFAR10 --pix_init=real --ipc=5  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=1000 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=2 >> testlogs_distill_fixed.txt

# BATCH50 , fuse 2 
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH50 , fuse 2 " >> testlogs_distill_fixed.txt

WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=5  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=1000 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=2 >> testlogs_distill_fixed.txt

# BATCH50 , fuse 4
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH50 , fuse 4 " >> testlogs_distill_fixed.txt

WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=5  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=1000 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=4 >> testlogs_distill_fixed.txt

# BATCH50 , fuse 6
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH50 , fuse 6 " >> testlogs_distill_fixed.txt
WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=5  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=1000 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=6 >> testlogs_distill_fixed.txt

# BATCH50 , fuse 8
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH50 , fuse 8 " >> testlogs_distill_fixed.txt

WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=5  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=1000 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=8 >> testlogs_distill_fixed.txt




echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH100 , BASE CODE " >> testlogs_distill_fixed.txt
WANDB_SILENT=true python distill_original.py --dataset=CIFAR10 --pix_init=real --ipc=10  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=1000 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=2 >> testlogs_distill_fixed.txt
WANDB_SILENT=true python distill_original.py --dataset=CIFAR10 --pix_init=real --ipc=10  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=1000 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=2 >> testlogs_distill_fixed.txt


# BATCH100 , fuse 2 
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH100 , fuse 2 " >> testlogs_distill_fixed.txt
WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=10  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=1000 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=2 >> testlogs_distill_fixed.txt

# BATCH100 , fuse 4 
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH100 , fuse 4 " >> testlogs_distill_fixed.txt
WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=10  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=1000 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=4 >> testlogs_distill_fixed.txt

# BATCH100 , fuse 6 
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH100 , fuse 6 " >> testlogs_distill_fixed.txt
WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=10  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=1000 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=6 >> testlogs_distill_fixed.txt

# BATCH100 , fuse 8
echo "==============================" >> testlogs_distill_fixed.txt
echo "BATCH100 , fuse 8 " >> testlogs_distill_fixed.txt
WANDB_SILENT=true python distill_batched.py --dataset=CIFAR10 --pix_init=real --ipc=10  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=1000 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=8 >> testlogs_distill_fixed.txt