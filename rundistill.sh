

# WANDB_SILENT=true python distill.py --dataset=CIFAR10 --pix_init=real --ipc=1  \
#     --syn_steps=14 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=40 \
#     --detachNum=0 --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
#     --buffer_path=./buffer --data_path=./dataset


python distill.py --dataset=CIFAR100 --ipc=1 --syn_steps=20 --expert_epochs=3 \
    --max_start_epoch=20 --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --trunkSize=4 --eval_mode=TM \
    --buffer_path=buffer --data_path=dataset > output_3.txt  

    # python distill.py --dataset=CIFAR100 --ipc=10 --syn_steps=20 --expert_epochs=3    --max_start_epoch=20 --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01    --eval_mode=M --buffer_path=buffer --data_path=dataset > output_2.txt  