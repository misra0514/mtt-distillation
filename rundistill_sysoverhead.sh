#!/usr/bin/env bash
# Run ConvNet flex_mask_list 1 0 + --v_fuse
# IPC = 1 / 10 / 100, each repeated 3 times.
#
# Run with:
#   bash run_conv_10_three_repeats_systest.sh
#
# All output goes to:
#   systest.txt

set -u
set -o pipefail

LOG_FILE="systest.txt"
: > "${LOG_FILE}"

IPC_LIST=(1 10 100)
REPEATS=3

# SCRIPT="distill_flexFuse_timeTest_conv_sys.py"
SCRIPT="distill_flexFuse_timeTest_conv_backup.py"

DATASET="CIFAR10"
PIX_INIT="real"
SYN_STEPS=1
MAX_EXPERTS=1
EXPERT_EPOCHS=1
MAX_START_EPOCH=1
ITERATION=350
DETACH_NUM=0

LR_IMG=1000
LR_LR=1e-05
LR_TEACHER=0.01

BUFFER_PATH="/scratch/yguo25/files/mtt-distillation/buffer"
DATA_PATH="/scratch/yguo25/files/mtt-distillation/dataset"

MODEL="ConvNet"
FUSE=1

clean_output() {
  sed -u \
    -e '/Warning:/d' \
    -e '/warnings.warn/d' \
    -e '/FutureWarning/d' \
    -e '/UserWarning/d' \
    -e '/^[[:space:]]*[0-9]\+%|/d' \
    -e '/^[[:space:]]*100%|/d' \
    -e '/^[[:space:]]*[0-9]\+it \[/d' \
    -e '/it\/s]/d' \
    -e '/it\/s/d'
}

run_one() {
  local ipc="$1"
  local repeat_id="$2"

  local tag="IPC=${ipc} | REPEAT=${repeat_id}/${REPEATS} | CASE=flex_mask_10_vfuse"

  echo "[RUN] ${tag}"

  {
    echo "=============================="
    echo "$(date '+%Y-%m-%d %H:%M:%S') | ${tag}"
    echo "CMD: WANDB_SILENT=true PYTHONWARNINGS=ignore TQDM_DISABLE=1 python ${SCRIPT} --fuse_mask_list 1 0 --dataset=${DATASET} --pix_init=${PIX_INIT} --ipc=${ipc} --syn_steps=${SYN_STEPS} --max_experts=${MAX_EXPERTS} --expert_epochs=${EXPERT_EPOCHS} --max_start_epoch=${MAX_START_EPOCH} --Iteration=${ITERATION} --detachNum=${DETACH_NUM} --lr_img=${LR_IMG} --lr_lr=${LR_LR} --lr_teacher=${LR_TEACHER} --buffer_path=${BUFFER_PATH} --data_path=${DATA_PATH} --model=${MODEL} --Fuse=${FUSE} --v_fuse"
    echo "------------------------------"
  } >> "${LOG_FILE}"

  WANDB_SILENT=true \
  PYTHONWARNINGS=ignore \
  TQDM_DISABLE=1 \
  PYTHONUNBUFFERED=1 \
  python "${SCRIPT}" \
    --fuse_mask_list 1 0 \
    --dataset="${DATASET}" \
    --pix_init="${PIX_INIT}" \
    --ipc="${ipc}" \
    --syn_steps="${SYN_STEPS}" \
    --max_experts="${MAX_EXPERTS}" \
    --expert_epochs="${EXPERT_EPOCHS}" \
    --max_start_epoch="${MAX_START_EPOCH}" \
    --Iteration="${ITERATION}" \
    --detachNum="${DETACH_NUM}" \
    --lr_img="${LR_IMG}" \
    --lr_lr="${LR_LR}" \
    --lr_teacher="${LR_TEACHER}" \
    --buffer_path="${BUFFER_PATH}" \
    --data_path="${DATA_PATH}" \
    --model="${MODEL}" \
    --Fuse="${FUSE}" \
    --v_fuse \
    2>&1 | clean_output >> "${LOG_FILE}"

  local exit_code=${PIPESTATUS[0]}

  {
    echo "------------------------------"
    echo "EXIT_CODE: ${exit_code}"
    if [ "${exit_code}" -ne 0 ]; then
      echo "ERROR: command failed with EXIT_CODE=${exit_code}"
    fi
    echo ""
  } >> "${LOG_FILE}"

  if [ "${exit_code}" -ne 0 ]; then
    echo "[ERROR] ${tag} failed with EXIT_CODE=${exit_code}"
  fi
}

{
  echo "============================================================"
  echo "START_TIME: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "SCRIPT: ${SCRIPT}"
  echo "IPC_LIST: ${IPC_LIST[*]}"
  echo "REPEATS: ${REPEATS}"
  echo "LOG_FILE: ${LOG_FILE}"
  echo "============================================================"
  echo ""
} >> "${LOG_FILE}"

for ipc in "${IPC_LIST[@]}"; do
  for repeat_id in $(seq 1 "${REPEATS}"); do
    run_one "${ipc}" "${repeat_id}"
  done
done

{
  echo "============================================================"
  echo "END_TIME: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"
} >> "${LOG_FILE}"

echo "[DONE] Results written to ${LOG_FILE}"
