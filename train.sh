python distill_compile_train2.py --dataset=CIFAR100 --ipc=1 --syn_steps=20 \
     --expert_epochs=3    --max_start_epoch=20 --zca --lr_img=1000 --lr_lr=1e-05 \
     --lr_teacher=0.01    --eval_mode=C --buffer_path=../mtt-distillation/buffer  \
     --data_path=../mtt-distillation/dataset/ --model=ConvNet