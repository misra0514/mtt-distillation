#!/usr/bin/env bash
# Test --fuse_mask_list 1 1 0 0 for ConvNet / ViT / ResNet18.
#
# 老规矩：
#   - 每个 case 跑 1 次 --use-barrier
#   - 再跑 2 次不带 --use-barrier
#   - fwd/bwd 后面用 barrier 那次
#   - time sumation 后面用 no_barrier 两次平均
#
# Iteration / IPC 规则：
#   - ConvNet:   IPC 1 10 100,       Iteration=350
#   - ViT:       IPC 1 10 100 1000,  IPC>=100 用 Iteration=50，否则 350
#   - ResNet18:  IPC 1 10 100 1000,  IPC>=100 用 Iteration=50，否则 350
#
# Run with bash, not sh:
#   bash run_flex_mask_1100_all_models.sh

set -u
set -o pipefail

LOG_DIR="${LOG_DIR:-testlog524}"
mkdir -p "${LOG_DIR}"

# All models write into the same log file.
LOG_FILE="${LOG_FILE:-${LOG_DIR}/flex_mask_1100_all_models.txt}"

DATASET="${DATASET:-CIFAR10}"
PIX_INIT="${PIX_INIT:-real}"
BUFFER_PATH="${BUFFER_PATH:-/scratch/yguo25/files/mtt-distillation/buffer}"
DATA_PATH="${DATA_PATH:-/scratch/yguo25/files/mtt-distillation/dataset}"

DEFAULT_ITERATION="${DEFAULT_ITERATION:-350}"
FAST_ITERATION="${FAST_ITERATION:-50}"

MAX_EXPERTS="${MAX_EXPERTS:-1}"
EXPERT_EPOCHS="${EXPERT_EPOCHS:-1}"
MAX_START_EPOCH="${MAX_START_EPOCH:-1}"
DETACH_NUM="${DETACH_NUM:-0}"

LR_IMG="${LR_IMG:-1000}"
LR_LR="${LR_LR:-1e-05}"
LR_TEACHER="${LR_TEACHER:-0.01}"

FUSE="${FUSE:-1}"
FUSE_MASK=(1 1 0 0)
CASE_NAME="flex_mask_1100_vfuse"

CONV_FLEX_FILE="${CONV_FLEX_FILE:-distill_flexFuse_timeTest_conv.py}"
VIT_FLEX_FILE="${VIT_FLEX_FILE:-distill_flexFuse_timeTest_ViT.py}"
RESNET_FLEX_FILE="${RESNET_FLEX_FILE:-distill_flexFuse_timeTest_resnet18.py}"

CONV_IPC_LIST=(1 10 100)
VIT_IPC_LIST=(1 10 100 1000)
RESNET_IPC_LIST=(1 10 100 1000)

SYN_STEPS=1

# Overwrite old log by default, same as previous sweep scripts.
: > "${LOG_FILE}"

iteration_for_model_ipc() {
  local model="$1"
  local ipc="$2"

  if [ "${model}" = "ConvNet" ]; then
    echo "${DEFAULT_ITERATION}"
    return
  fi

  # ViT / ResNet18: IPC 100 and 1000 use 50.
  if [ "${ipc}" -ge 100 ]; then
    echo "${FAST_ITERATION}"
  else
    echo "${DEFAULT_ITERATION}"
  fi
}

clean_output() {
  sed -u \
    -e '/Warning:/d' \
    -e '/warnings.warn/d' \
    -e '/FutureWarning/d' \
    -e '/UserWarning/d' \
    -e '/torch.cuda.amp.custom_fwd/d' \
    -e '/Triggered internally at/d' \
    -e '/SECURITY.md#untrusted-models/d' \
    -e '/weights_only=False/d' \
    -e '/Please consider converting the list/d' \
    -e '/Please open an issue/d' \
    -e '/^[[:space:]]*[0-9]\+%|/d' \
    -e '/^[[:space:]]*100%|/d' \
    -e '/^[[:space:]]*[0-9]\+it \[/d' \
    -e '/it\/s]/d' \
    -e '/it\/s/d'
}

write_global_header() {
  {
    echo "============================================================"
    echo "FLEX MASK 1 1 0 0 SWEEP START"
    echo "START_TIME: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "MODELS: ConvNet ViT ResNet18"
    echo "CASE: ${CASE_NAME}"
    echo "FUSE_MASK_LIST: ${FUSE_MASK[*]}"
    echo "V_FUSE: --v_fuse"
    echo "SYN_STEPS: ${SYN_STEPS}"
    echo "CONV_IPC_LIST: ${CONV_IPC_LIST[*]}"
    echo "VIT_IPC_LIST: ${VIT_IPC_LIST[*]}"
    echo "RESNET_IPC_LIST: ${RESNET_IPC_LIST[*]}"
    echo "DEFAULT_ITERATION: ${DEFAULT_ITERATION}"
    echo "FAST_ITERATION: ${FAST_ITERATION}"
    echo "USE_BARRIER_REPEATS: 1"
    echo "NO_BARRIER_REPEATS: 2"
    echo "ITERATION RULE: ConvNet always 350; ViT/ResNet18 use 50 when IPC>=100, otherwise 350."
    echo "LOG_FILE: ${LOG_FILE}"
    echo "NOTE: this script only runs distill_flexFuse_timeTest_* with --fuse_mask_list 1 1 0 0 --v_fuse."
    echo "============================================================"
    echo ""
  } >> "${LOG_FILE}"
}

write_global_footer() {
  {
    echo ""
    echo "============================================================"
    echo "FLEX MASK 1 1 0 0 SWEEP END"
    echo "END_TIME: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
  } >> "${LOG_FILE}"
}

run_one() {
  local model_label="$1"
  local model_arg="$2"
  local flex_file="$3"
  local ipc="$4"
  local barrier_mode="$5"
  local repeat_id="$6"
  local repeat_total="$7"

  local iteration
  iteration="$(iteration_for_model_ipc "${model_arg}" "${ipc}")"

  local barrier_args=()
  if [ "${barrier_mode}" = "use_barrier" ]; then
    barrier_args=(--use-barrier)
  fi

  local args=(
    --fuse_mask_list "${FUSE_MASK[@]}"
    --v_fuse
    "${barrier_args[@]}"
    --dataset="${DATASET}"
    --pix_init="${PIX_INIT}"
    --ipc="${ipc}"
    --syn_steps="${SYN_STEPS}"
    --max_experts="${MAX_EXPERTS}"
    --expert_epochs="${EXPERT_EPOCHS}"
    --max_start_epoch="${MAX_START_EPOCH}"
    --Iteration="${iteration}"
    --detachNum="${DETACH_NUM}"
    --lr_img="${LR_IMG}"
    --lr_lr="${LR_LR}"
    --lr_teacher="${LR_TEACHER}"
    --buffer_path="${BUFFER_PATH}"
    --data_path="${DATA_PATH}"
    --model="${model_arg}"
    --Fuse="${FUSE}"
  )

  local tag="MODEL=${model_label} | IPC=${ipc} | SYN_STEPS=${SYN_STEPS} | ITERATION=${iteration} | CASE=${CASE_NAME} | BARRIER=${barrier_mode}"

  echo "[RUN] ${tag} | REPEAT=${repeat_id}/${repeat_total}"

  {
    echo "=============================="
    echo "$(date '+%Y-%m-%d %H:%M:%S') | ${tag} | REPEAT=${repeat_id}/${repeat_total}"
    echo "CMD: WANDB_SILENT=true PYTHONWARNINGS=ignore TQDM_DISABLE=1 python ${flex_file} ${args[*]}"
    echo "------------------------------"
  } >> "${LOG_FILE}"

  WANDB_SILENT=true \
  PYTHONWARNINGS=ignore \
  TQDM_DISABLE=1 \
  PYTHONUNBUFFERED=1 \
  python "${flex_file}" "${args[@]}" 2>&1 | clean_output >> "${LOG_FILE}"

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
    echo "[ERROR] ${tag} | REPEAT=${repeat_id}/${repeat_total} | EXIT_CODE=${exit_code}"
  fi

  # Always continue the sweep even if one run fails.
  return 0
}

run_case_for_ipc() {
  local model_label="$1"
  local model_arg="$2"
  local flex_file="$3"
  local ipc="$4"

  # One synchronized/barrier run.
  run_one "${model_label}" "${model_arg}" "${flex_file}" "${ipc}" "use_barrier" "1" "1"

  # Two normal/no-barrier runs.
  run_one "${model_label}" "${model_arg}" "${flex_file}" "${ipc}" "no_barrier" "1" "2"
  run_one "${model_label}" "${model_arg}" "${flex_file}" "${ipc}" "no_barrier" "2" "2"
}

run_model_block() {
  local model_label="$1"
  local model_arg="$2"
  local flex_file="$3"
  shift 3
  local ipc_list=( "$@" )

  {
    echo "============================================================"
    echo "MODEL BLOCK START: ${model_label}"
    echo "MODEL ARG: ${model_arg}"
    echo "FLEX FILE: ${flex_file}"
    echo "IPC_LIST: ${ipc_list[*]}"
    echo "SYN_STEPS: ${SYN_STEPS}"
    echo "CASE: ${CASE_NAME}"
    echo "============================================================"
    echo ""
  } >> "${LOG_FILE}"

  local ipc
  for ipc in "${ipc_list[@]}"; do
    run_case_for_ipc "${model_label}" "${model_arg}" "${flex_file}" "${ipc}"
  done

  {
    echo "============================================================"
    echo "MODEL BLOCK END: ${model_label}"
    echo "END_TIME: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo ""
  } >> "${LOG_FILE}"
}

write_global_header

run_model_block "ConvNet"  "ConvNet"  "${CONV_FLEX_FILE}"   "${CONV_IPC_LIST[@]}"
run_model_block "ViT"      "ViT"      "${VIT_FLEX_FILE}"    "${VIT_IPC_LIST[@]}"
run_model_block "ResNet18" "ResNet18" "${RESNET_FLEX_FILE}" "${RESNET_IPC_LIST[@]}"

write_global_footer

echo "[DONE] flex_mask_1100 sweep finished."
echo "Log file: ${LOG_FILE}"