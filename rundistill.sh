

# WANDB_SILENT=true python /scratch/yguo25/files/mtt-original/distill_test.py --dataset=CIFAR10 --pix_init=real --ipc=1  \
WANDB_SILENT=true python distill_ckptManuel_tesla.py --dataset=CIFAR10 --pix_init=real --ipc=10  \
    --syn_steps=20  --Iteration=50 \
    --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --no_aug=True \
    --model=ViT \
    # --model=ResNet50 \
    # --max_experts=1 --expert_epochs=1 --max_start_epoch=1
    # --zca 

#  python distill_test.py --dataset=CIFAR10 --ipc=1 --syn_steps=20 --expert_epochs=3  \
#    --max_start_epoch=20 --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01    --eval_mode=C \ 
#    --buffer_path=../mtt-distillation/buffer --data_path=../mtt-distillation/dataset/ \ 
#    --model=ConvNet --Iteration=50