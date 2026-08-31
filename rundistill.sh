# distill_original

#    --model=ResNet50\
# WANDB_SILENT=true python distill_original_fwdtimetest.py  \
# WANDB_SILENT=true python distill_original_timeTest_prof1.py  \
# CUDA_VISIBLE_DEVICES=2
# WANDB_SILENT=true python distill_flexFuse_timeTest.py  --fuse_mask_list 1 0 \
# WANDB_SILENT=true python distill_flexFuse_timeTest_temp_prof.py  --fuse_mask_list 1 0 \
# WANDB_SILENT=true python distill_flexFuse_timeTest_v2_prof.py  --fuse_mask_list 1 0 \
# WANDB_SILENT=true python distill_ckpt.py \
# WANDB_SILENT=true python distill_flexFuse_timeTest_temp.py  --fuse_mask_list 1 0 \
# WANDB_SILENT=true python distill_flexFuse_timeTest_v2.py  --fuse_mask_list 1 0 \
# WANDB_SILENT=true python distill_original_timeTest.py  \
# WANDB_SILENT=true python distill_flexFuse_timeTest_conv.py  --fuse_mask_list 1 0\
# WANDB_SILENT=true python distill_ckpt_flex_conv_timetest.py --fuse_mask_list 1 1 \
# WANDB_SILENT=true python distill_batched_timeTest.py   \
# WANDB_SILENT=true python distill_flexFuse_timeTest_conv_backup.py  --fuse_mask_list 1 1 \
# WANDB_SILENT=true python distill_original_timeTest.py  \
# WANDB_SILENT=true python distill_original_fwdtimetest.py  \
# WANDB_SILENT=true python distill_flexFuse_timeTest_conv.py --fuse_mask_list 1 0\
# WANDB_SILENT=true python distill_original_timeTest.py  \
# WANDB_SILENT=true python distill_batched_timeTest.py \

# WANDB_SILENT=true python distill_memory_prof.py \
# WANDB_SILENT=true python distill_flexFuse_timeTest_resnet18.py  --fuse_mask_list 1 \
# WANDB_SILENT=true python distill_flexFuse_timeTest_ViT.py  --fuse_mask_list 1 0 \
WANDB_SILENT=true python distill_flexFuse_timeTest_conv_v2.py  --fuse_mask_list 1 0 \
    --dataset=CIFAR10 --pix_init=real --ipc=10 \
    --syn_steps=1 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=50 \
    --detachNum=0  \
    --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=/scratch/yguo25/files/mtt-distillation/buffer  --data_path=/scratch/yguo25/files/mtt-distillation/dataset \
    --Fuse=1 \
    --model=ConvNet \
    --use-barrier \
    # --AccTest=True \

    # --mem_profile --mem_snapshot_dir=./prof-conv3 \
    # --model=ViT \
    # --v_fuse \
    # --no-v_fuse \
 
    # --model=ConvNet \
    # --model=ResNet18 \
    # --zca \

