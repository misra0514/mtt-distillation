#!/usr/bin/env bash
# Massive distill time-test sweep, v3.
#
# Logs are split by model:
#   testlog524/conv.txt
#   testlog524/resnet18.txt
#   testlog524/vit.txt
#
# For each model / ipc / syn_steps, run:
#   original
#   fuse_mask_list 1   + --no-v_fuse
#   fuse_mask_list 1   + --v_fuse
#   fuse_mask_list 1 1 + --v_fuse
#   fuse_mask_list 1 0 + --v_fuse
#
# Each test is repeated twice consecutively.
#
# Run with bash, not sh:
#   bash rundistill_massive_524_v3.sh

set -u
set -o pipefail

LOG_DIR="${LOG_DIR:-testlog524}"
mkdir -p "${LOG_DIR}"

CONV_LOG="${LOG_DIR}/conv.txt"
RESNET_LOG="${LOG_DIR}/resnet18.txt"
VIT_LOG="${LOG_DIR}/vit.txt"

DATASET="CIFAR10"
PIX_INIT="real"
BUFFER_PATH="/scratch/yguo25/files/mtt-distillation/buffer"
DATA_PATH="/scratch/yguo25/files/mtt-distillation/dataset"

ITERATION=350
MAX_EXPERTS=1
EXPERT_EPOCHS=1
MAX_START_EPOCH=1
DETACH_NUM=0

LR_IMG=1000
LR_LR=1e-05
LR_TEACHER=0.01

FUSE=1
REPEATS=2

ORIGINAL_FILE="distill_original_timeTest.py"

CONV_FLEX_FILE="distill_flexFuse_timeTest_conv_backup.py"
RESNET_FLEX_FILE="distill_flexFuse_timeTest_resnet18.py"
VIT_FLEX_FILE="distill_flexFuse_timeTest_ViT.py"

IPC_LIST=(1 10 100)
SYN_STEPS_LIST=(1 20)

# Clean old logs at the start.
: > "${CONV_LOG}"
: > "${RESNET_LOG}"
: > "${VIT_LOG}"

write_header() {
  local log_file="$1"
  local model_label="$2"
  local model_arg="$3"
  local flex_file="$4"

  {
    echo "============================================================"
    echo "TIME TEST SWEEP START"
    echo "START_TIME: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "MODEL BLOCK: ${model_label}"
    echo "MODEL ARG:   ${model_arg}"
    echo "FLEX FILE:   ${flex_file}"
    echo "ORDER:       original -> mask1_no_vfuse -> mask1_vfuse -> mask11_vfuse -> mask10_vfuse"
    echo "IPC_LIST:    ${IPC_LIST[*]}"
    echo "SYN_STEPS:   ${SYN_STEPS_LIST[*]}"
    echo "ITERATION:   ${ITERATION}"
    echo "REPEATS:     ${REPEATS}"
    echo "LOG_FILE:    ${log_file}"
    echo "NOTE:        no plain tests; --AccTest=True is removed."
    echo "============================================================"
    echo ""
  } >> "${log_file}"
}

write_footer() {
  local log_file="$1"

  {
    echo ""
    echo "============================================================"
    echo "TIME TEST SWEEP END"
    echo "END_TIME: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
  } >> "${log_file}"
}

common_args() {
  local ipc="$1"
  local syn_steps="$2"
  local model="$3"

  echo \
    --dataset="${DATASET}" \
    --pix_init="${PIX_INIT}" \
    --ipc="${ipc}" \
    --syn_steps="${syn_steps}" \
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
    --model="${model}" \
    --Fuse="${FUSE}"
}

# Filter out warnings and tqdm/progress-bar lines from the log.
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

run_cmd_once() {
  local log_file="$1"
  local tag="$2"
  local repeat_id="$3"
  shift 3

  echo "[RUN] ${tag} | repeat ${repeat_id}/${REPEATS}"

  {
    echo "=============================="
    echo "$(date '+%Y-%m-%d %H:%M:%S') | ${tag} | REPEAT=${repeat_id}/${REPEATS}"
    echo "CMD: WANDB_SILENT=true PYTHONWARNINGS=ignore TQDM_DISABLE=1 $*"
    echo "------------------------------"
  } >> "${log_file}"

  WANDB_SILENT=true \
  PYTHONWARNINGS=ignore \
  TQDM_DISABLE=1 \
  PYTHONUNBUFFERED=1 \
  "$@" 2>&1 | clean_output >> "${log_file}"

  local exit_code=${PIPESTATUS[0]}

  {
    echo "------------------------------"
    echo "EXIT_CODE: ${exit_code}"
    if [ "${exit_code}" -ne 0 ]; then
      echo "ERROR: command failed with EXIT_CODE=${exit_code}"
    fi
    echo ""
  } >> "${log_file}"

  if [ "${exit_code}" -ne 0 ]; then
    echo "[ERROR] command failed with EXIT_CODE=${exit_code}: ${tag} | repeat ${repeat_id}/${REPEATS}"
  fi

  return 0
}

run_cmd() {
  local log_file="$1"
  local tag="$2"
  shift 2

  local r
  for r in $(seq 1 "${REPEATS}"); do
    run_cmd_once "${log_file}" "${tag}" "${r}" "$@"
  done
}

run_original() {
  local log_file="$1"
  local model_label="$2"
  local model_arg="$3"
  local ipc="$4"
  local syn_steps="$5"

  # shellcheck disable=SC2207
  local args=( $(common_args "${ipc}" "${syn_steps}" "${model_arg}") )

  run_cmd \
    "${log_file}" \
    "MODEL=${model_label} | IPC=${ipc} | SYN_STEPS=${syn_steps} | CASE=original" \
    python "${ORIGINAL_FILE}" "${args[@]}"
}

run_flex_case() {
  local log_file="$1"
  local model_label="$2"
  local model_arg="$3"
  local flex_file="$4"
  local ipc="$5"
  local syn_steps="$6"
  local mask_string="$7"
  local vfuse_arg="$8"

  read -r -a mask_args <<< "${mask_string}"
  local mask_tag="${mask_string// /}"
  local vfuse_tag

  if [ "${vfuse_arg}" = "--v_fuse" ]; then
    vfuse_tag="vfuse"
  else
    vfuse_tag="no_vfuse"
  fi

  # shellcheck disable=SC2207
  local args=( $(common_args "${ipc}" "${syn_steps}" "${model_arg}") )

  run_cmd \
    "${log_file}" \
    "MODEL=${model_label} | IPC=${ipc} | SYN_STEPS=${syn_steps} | CASE=flex_mask_${mask_tag}_${vfuse_tag}" \
    python "${flex_file}" \
      --fuse_mask_list "${mask_args[@]}" \
      "${vfuse_arg}" \
      "${args[@]}"
}

# ============================================================
# ConvNet tests
# ============================================================
write_header "${CONV_LOG}" "ConvNet" "ConvNet" "${CONV_FLEX_FILE}"

for ipc in "${IPC_LIST[@]}"; do
  for syn_steps in "${SYN_STEPS_LIST[@]}"; do
    run_original  "${CONV_LOG}" "ConvNet" "ConvNet" "${ipc}" "${syn_steps}"

    run_flex_case "${CONV_LOG}" "ConvNet" "ConvNet" "${CONV_FLEX_FILE}" "${ipc}" "${syn_steps}" "1"   "--no-v_fuse"
    run_flex_case "${CONV_LOG}" "ConvNet" "ConvNet" "${CONV_FLEX_FILE}" "${ipc}" "${syn_steps}" "1"   "--v_fuse"
    run_flex_case "${CONV_LOG}" "ConvNet" "ConvNet" "${CONV_FLEX_FILE}" "${ipc}" "${syn_steps}" "1 1" "--v_fuse"
    run_flex_case "${CONV_LOG}" "ConvNet" "ConvNet" "${CONV_FLEX_FILE}" "${ipc}" "${syn_steps}" "1 0" "--v_fuse"
  done
done

write_footer "${CONV_LOG}"

# ============================================================
# ViT tests
# ============================================================
write_header "${VIT_LOG}" "ViT" "ViT" "${VIT_FLEX_FILE}"

for ipc in "${IPC_LIST[@]}"; do
  for syn_steps in "${SYN_STEPS_LIST[@]}"; do
    run_original  "${VIT_LOG}" "ViT" "ViT" "${ipc}" "${syn_steps}"

    run_flex_case "${VIT_LOG}" "ViT" "ViT" "${VIT_FLEX_FILE}" "${ipc}" "${syn_steps}" "1"   "--no-v_fuse"
    run_flex_case "${VIT_LOG}" "ViT" "ViT" "${VIT_FLEX_FILE}" "${ipc}" "${syn_steps}" "1"   "--v_fuse"
    run_flex_case "${VIT_LOG}" "ViT" "ViT" "${VIT_FLEX_FILE}" "${ipc}" "${syn_steps}" "1 1" "--v_fuse"
    run_flex_case "${VIT_LOG}" "ViT" "ViT" "${VIT_FLEX_FILE}" "${ipc}" "${syn_steps}" "1 0" "--v_fuse"
  done
done

write_footer "${VIT_LOG}"

# ============================================================
# ResNet18 tests
# ============================================================
write_header "${RESNET_LOG}" "ResNet18" "ResNet18" "${RESNET_FLEX_FILE}"

for ipc in "${IPC_LIST[@]}"; do
  for syn_steps in "${SYN_STEPS_LIST[@]}"; do
    run_original  "${RESNET_LOG}" "ResNet18" "ResNet18" "${ipc}" "${syn_steps}"

    run_flex_case "${RESNET_LOG}" "ResNet18" "ResNet18" "${RESNET_FLEX_FILE}" "${ipc}" "${syn_steps}" "1"   "--no-v_fuse"
    run_flex_case "${RESNET_LOG}" "ResNet18" "ResNet18" "${RESNET_FLEX_FILE}" "${ipc}" "${syn_steps}" "1"   "--v_fuse"
    run_flex_case "${RESNET_LOG}" "ResNet18" "ResNet18" "${RESNET_FLEX_FILE}" "${ipc}" "${syn_steps}" "1 1" "--v_fuse"
    run_flex_case "${RESNET_LOG}" "ResNet18" "ResNet18" "${RESNET_FLEX_FILE}" "${ipc}" "${syn_steps}" "1 0" "--v_fuse"
  done
done

write_footer "${RESNET_LOG}"

echo "[DONE] All commands finished."
echo "ConvNet log:  ${CONV_LOG}"
echo "ViT log:      ${VIT_LOG}"
echo "ResNet18 log: ${RESNET_LOG}"
