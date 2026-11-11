WANDB_SILENT=true python main_optimized.py  --fuse_mask_list 1 0 \
    --dataset=CIFAR10 --pix_init=real --ipc=10 \
    --syn_steps=1 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=50 \
    --detachNum=0  \
    --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=/scratch/yguo25/files/mtt-distillation/buffer  --data_path=/scratch/yguo25/files/mtt-distillation/dataset \
    --Fuse=1 \
    --model=ConvNet \
    --v_fuse \
    --AccTest=True \
    # --use-barrier \