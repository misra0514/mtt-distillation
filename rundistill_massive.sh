
for i in 3 4 5 6 7 8 9 10 11 12 13 14 15
do
WANDB_SILENT=true python distill.py --dataset=CIFAR10 --pix_init=real --ipc=1  \
    --syn_steps=$i --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=0 \
    --detachNum=$(($i-1)) --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset
done

