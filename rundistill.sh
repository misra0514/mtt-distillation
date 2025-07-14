

WANDB_SILENT=true python distill_memProf.py --dataset=CIFAR100 --pix_init=real --ipc=1  \
    --syn_steps=1 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=0 \
    --detachNum=0 --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --no_aug=True \
    # --model=ConvNet_unfold \
    # --zca 