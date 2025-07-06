

WANDB_SILENT=true python distill_ckptManuel.py --dataset=CIFAR100 --pix_init=real --ipc=10  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=50 \
    --detachNum=0 --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --no_aug=True \
    --zca 