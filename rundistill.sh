# distill_original

# WANDB_SILENT=true python distill_original.py --dataset=CIFAR10 --pix_init=real --ipc=1  \
WANDB_SILENT=true python distill_batched_timeTest.py --dataset=CIFAR10 --pix_init=real --ipc=10  \
    --syn_steps=20 --max_experts=1 --expert_epochs=1 --max_start_epoch=1 --Iteration=100 \
    --detachNum=0  \
    --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 \
    --buffer_path=./buffer --data_path=./dataset \
    --Fuse=2



# celoss: 2.3739895820617676
# ------GRAD----
# 5.969873484445998e-10
# celoss: 2.4085354804992676
# ------GRAD----
# -3.793591218936854e-09


# celoss: 2.374037981033325
# ------GRAD----
# 2.9172131377208643e-10
# celoss: 2.408501148223877
# ------GRAD----
# -1.9240715687374177e-09


# 2.3739895820617676 vs 1.7816638946533203，但是fuse4？？