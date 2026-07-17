#!/usr/bin/env bash
# Massive distill time-test sweep, v5.
#
# Logs are split by model:
#   testlog524/conv.txt
#   testlog524/vit.txt
#   testlog524/resnet18.txt
#
# Important changes in v5:
#   1. Original still runs syn_steps = 1 and 20.
#   2. Normal flex timetest scripts only run syn_steps = 1.
#   3. ConvNet additionally uses distill_ckpt_flex_conv_timetest.py to cover syn_steps = 20.
#   4. No plain tests.
#   5. No --AccTest=True.
#   6. Each case runs once with --use-barrier and twice without --use-barrier.
#   7. ViT/ResNet18 use --Iteration=50 when IPC>=100; ConvNet stays at 350.
#
# Per model / IPC, order is:
#   original syn1
#   normal flex syn1: mask1_no_vfuse, mask1_vfuse, mask11_vfuse, mask10_vfuse
#   original syn20
#   conv only: ckpt flex syn20: mask1_vfuse, mask11_vfuse, mask10_vfuse
#
# Run with bash, not sh:
#   bash rundistill_massive_524_v5_barrier_iter50.sh

set -u
set -o pipefail

LOG_DIR="${LOG_DIR:-testlog524}"
mkdir -p "${LOG_DIR}"

CONV_LOG="${LOG_DIR}/conv.txt"
VIT_LOG="${LOG_DIR}/vit.txt"
RESNET_LOG="${LOG_DIR}/resnet18.txt"

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
USE_BARRIER_REPEATS=1
NO_BARRIER_REPEATS=2

ORIGINAL_FILE="distill_original_timeTest.py"

CONV_FLEX_FILE="distill_flexFuse_timeTest_conv_backup.py" # BACKUP!! 
VIT_FLEX_FILE="distill_flexFuse_timeTest_ViT.py"
RESNET_FLEX_FILE="distill_flexFuse_timeTest_resnet18.py"

# Conv-only checkpoint flex script for syn_steps=20.
CONV_CKPT_FLEX_FILE="distill_ckpt_flex_conv_timetest.py"

# ConvNet keeps the old IPC sweep.
CONV_IPC_LIST=(1 10 100)

# ViT and ResNet18 include IPC=1000; for IPC>=100 they use --Iteration=50.
VIT_IPC_LIST=(1 10 100 1000)
RESNET_IPC_LIST=(1 10 100 1000)

ORIGINAL_SYN_STEPS_LIST=(1 20)
NORMAL_FLEX_SYN_STEPS=1
CKPT_FLEX_SYN_STEPS=20

# For normal flex syn_steps=1.
# Cases:
#   1   + --no-v_fuse
#   1   + --v_fuse
#   1 1 + --v_fuse
#   1 0 + --v_fuse

# For conv ckpt syn_steps=20.
# Cases:
#   1   + --v_fuse
#   1 1 + --v_fuse
#   1 0 + --v_fuse

# Clean old logs at the start.
: > "${CONV_LOG}"
: > "${VIT_LOG}"
: > "${RESNET_LOG}"

write_header() {
  local log_file="$1"
  local model_label="$2"
  local model_arg="$3"
  local flex_file="$4"
  local ipc_list_text="$5"
  local extra_note="${6:-}"

  {
    echo "============================================================"
    echo "TIME TEST SWEEP START"
    echo "START_TIME: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "MODEL BLOCK: ${model_label}"
    echo "MODEL ARG:   ${model_arg}"
    echo "FLEX FILE:   ${flex_file}"
    echo "IPC_LIST:    ${ipc_list_text}"
    echo "ORIGINAL_SYN_STEPS: ${ORIGINAL_SYN_STEPS_LIST[*]}"
    echo "NORMAL_FLEX_SYN_STEPS: ${NORMAL_FLEX_SYN_STEPS}"
    echo "DEFAULT_ITERATION: ${ITERATION}"
    echo "USE_BARRIER_REPEATS: ${USE_BARRIER_REPEATS}"
    echo "NO_BARRIER_REPEATS:  ${NO_BARRIER_REPEATS}"
    echo "LOG_FILE:    ${log_file}"
    echo "NOTE:        no plain tests; --AccTest=True is removed; each case runs once with --use-barrier and twice without it."
    if [ -n "${extra_note}" ]; then
      echo "EXTRA:       ${extra_note}"
    fi
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
  local iteration="${4:-${ITERATION}}"

  echo \
    --dataset="${DATASET}" \
    --pix_init="${PIX_INIT}" \
    --ipc="${ipc}" \
    --syn_steps="${syn_steps}" \
    --max_experts="${MAX_EXPERTS}" \
    --expert_epochs="${EXPERT_EPOCHS}" \
    --max_start_epoch="${MAX_START_EPOCH}" \
    --Iteration="${iteration}" \
    --detachNum="${DETACH_NUM}" \
    --lr_img="${LR_IMG}" \
    --lr_lr="${LR_LR}" \
    --lr_teacher="${LR_TEACHER}" \
    --buffer_path="${BUFFER_PATH}" \
    --data_path="${DATA_PATH}" \
    --model="${model}" \
    --Fuse="${FUSE}"
}

# ViT/ResNet18 special rule:
# For large IPC settings, run fewer iterations.
# ConvNet always keeps the default ITERATION.
iteration_for_model_ipc() {
  local model_label="$1"
  local ipc="$2"

  if { [ "${model_label}" = "ViT" ] || [ "${model_label}" = "ResNet18" ]; } && [ "${ipc}" -ge 100 ]; then
    echo "50"
  else
    echo "${ITERATION}"
  fi
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
  local barrier_mode="$3"
  local repeat_id="$4"
  local repeat_total="$5"
  shift 5

  local barrier_args=()
  if [ "${barrier_mode}" = "use_barrier" ]; then
    barrier_args=(--use-barrier)
  fi

  echo "[RUN] ${tag} | BARRIER=${barrier_mode} | repeat ${repeat_id}/${repeat_total}"

  {
    echo "=============================="
    echo "$(date '+%Y-%m-%d %H:%M:%S') | ${tag} | BARRIER=${barrier_mode} | REPEAT=${repeat_id}/${repeat_total}"
    echo "CMD: WANDB_SILENT=true PYTHONWARNINGS=ignore TQDM_DISABLE=1 $* ${barrier_args[*]}"
    echo "------------------------------"
  } >> "${log_file}"

  WANDB_SILENT=true \
  PYTHONWARNINGS=ignore \
  TQDM_DISABLE=1 \
  PYTHONUNBUFFERED=1 \
  "$@" "${barrier_args[@]}" 2>&1 | clean_output >> "${log_file}"

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
    echo "[ERROR] command failed with EXIT_CODE=${exit_code}: ${tag} | BARRIER=${barrier_mode} | repeat ${repeat_id}/${repeat_total}"
  fi

  return 0
}

run_cmd() {
  local log_file="$1"
  local tag="$2"
  shift 2

  local r

  # Run once with --use-barrier.
  for r in $(seq 1 "${USE_BARRIER_REPEATS}"); do
    run_cmd_once "${log_file}" "${tag}" "use_barrier" "${r}" "${USE_BARRIER_REPEATS}" "$@"
  done

  # Then run twice without --use-barrier.
  for r in $(seq 1 "${NO_BARRIER_REPEATS}"); do
    run_cmd_once "${log_file}" "${tag}" "no_barrier" "${r}" "${NO_BARRIER_REPEATS}" "$@"
  done
}

run_original() {
  local log_file="$1"
  local model_label="$2"
  local model_arg="$3"
  local ipc="$4"
  local syn_steps="$5"
  local iteration="${6:-${ITERATION}}"

  # shellcheck disable=SC2207
  local args=( $(common_args "${ipc}" "${syn_steps}" "${model_arg}" "${iteration}") )

  run_cmd \
    "${log_file}" \
    "MODEL=${model_label} | IPC=${ipc} | SYN_STEPS=${syn_steps} | ITERATION=${iteration} | CASE=original" \
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
  local case_prefix="${9:-flex}"
  local iteration="${10:-${ITERATION}}"

  read -r -a mask_args <<< "${mask_string}"
  local mask_tag="${mask_string// /}"
  local vfuse_tag

  if [ "${vfuse_arg}" = "--v_fuse" ]; then
    vfuse_tag="vfuse"
  else
    vfuse_tag="no_vfuse"
  fi

  # shellcheck disable=SC2207
  local args=( $(common_args "${ipc}" "${syn_steps}" "${model_arg}" "${iteration}") )

  run_cmd \
    "${log_file}" \
    "MODEL=${model_label} | IPC=${ipc} | SYN_STEPS=${syn_steps} | ITERATION=${iteration} | CASE=${case_prefix}_mask_${mask_tag}_${vfuse_tag}" \
    python "${flex_file}" \
      --fuse_mask_list "${mask_args[@]}" \
      "${vfuse_arg}" \
      "${args[@]}"
}

# # ============================================================
# # ConvNet tests
# # ============================================================
# write_header \
#   "${CONV_LOG}" \
#   "ConvNet" \
#   "ConvNet" \
#   "${CONV_FLEX_FILE}" \
#   "${CONV_IPC_LIST[*]}" \
#   "normal flex only uses syn_steps=1; ckpt flex file covers conv syn_steps=20. CKPT_FILE=${CONV_CKPT_FLEX_FILE}; ConvNet keeps --Iteration=${ITERATION} for all IPCs."

# for ipc in "${CONV_IPC_LIST[@]}"; do
#   # original syn_steps=1
#   run_original "${CONV_LOG}" "ConvNet" "ConvNet" "${ipc}" "1"

#   # normal flex syn_steps=1 only
#   run_flex_case "${CONV_LOG}" "ConvNet" "ConvNet" "${CONV_FLEX_FILE}" "${ipc}" "${NORMAL_FLEX_SYN_STEPS}" "1"   "--no-v_fuse" "flex"
#   run_flex_case "${CONV_LOG}" "ConvNet" "ConvNet" "${CONV_FLEX_FILE}" "${ipc}" "${NORMAL_FLEX_SYN_STEPS}" "1"   "--v_fuse"    "flex"
#   run_flex_case "${CONV_LOG}" "ConvNet" "ConvNet" "${CONV_FLEX_FILE}" "${ipc}" "${NORMAL_FLEX_SYN_STEPS}" "1 1" "--v_fuse"    "flex"
#   run_flex_case "${CONV_LOG}" "ConvNet" "ConvNet" "${CONV_FLEX_FILE}" "${ipc}" "${NORMAL_FLEX_SYN_STEPS}" "1 0" "--v_fuse"    "flex"

#   # original syn_steps=20
#   run_original "${CONV_LOG}" "ConvNet" "ConvNet" "${ipc}" "20"

#   # ckpt flex syn_steps=20 only, to fill the removed normal-flex syn_steps=20 cases.
#   run_flex_case "${CONV_LOG}" "ConvNet" "ConvNet" "${CONV_CKPT_FLEX_FILE}" "${ipc}" "${CKPT_FLEX_SYN_STEPS}" "1"   "--v_fuse" "ckpt_flex"
#   run_flex_case "${CONV_LOG}" "ConvNet" "ConvNet" "${CONV_CKPT_FLEX_FILE}" "${ipc}" "${CKPT_FLEX_SYN_STEPS}" "1 1" "--v_fuse" "ckpt_flex"
#   run_flex_case "${CONV_LOG}" "ConvNet" "ConvNet" "${CONV_CKPT_FLEX_FILE}" "${ipc}" "${CKPT_FLEX_SYN_STEPS}" "1 0" "--v_fuse" "ckpt_flex"
# done

# write_footer "${CONV_LOG}"

# ============================================================
# ViT tests
# ============================================================
write_header \
  "${VIT_LOG}" \
  "ViT" \
  "ViT" \
  "${VIT_FLEX_FILE}" \
  "${VIT_IPC_LIST[*]}" \
  "normal flex only uses syn_steps=1; no flex syn_steps=20 for ViT; ViT uses --Iteration=50 when --ipc>=100."

for ipc in "${VIT_IPC_LIST[@]}"; do
  vit_iteration="$(iteration_for_model_ipc "ViT" "${ipc}")"

  # original syn_steps=1
  run_original "${VIT_LOG}" "ViT" "ViT" "${ipc}" "1" "${vit_iteration}"

  # normal flex syn_steps=1 only
  run_flex_case "${VIT_LOG}" "ViT" "ViT" "${VIT_FLEX_FILE}" "${ipc}" "${NORMAL_FLEX_SYN_STEPS}" "1"   "--no-v_fuse" "flex" "${vit_iteration}"
  run_flex_case "${VIT_LOG}" "ViT" "ViT" "${VIT_FLEX_FILE}" "${ipc}" "${NORMAL_FLEX_SYN_STEPS}" "1"   "--v_fuse"    "flex" "${vit_iteration}"
  run_flex_case "${VIT_LOG}" "ViT" "ViT" "${VIT_FLEX_FILE}" "${ipc}" "${NORMAL_FLEX_SYN_STEPS}" "1 1" "--v_fuse"    "flex" "${vit_iteration}"
  run_flex_case "${VIT_LOG}" "ViT" "ViT" "${VIT_FLEX_FILE}" "${ipc}" "${NORMAL_FLEX_SYN_STEPS}" "1 0" "--v_fuse"    "flex" "${vit_iteration}"

  # original syn_steps=20 still runs
  run_original "${VIT_LOG}" "ViT" "ViT" "${ipc}" "20" "${vit_iteration}"
done

write_footer "${VIT_LOG}"

# ============================================================
# ResNet18 tests
# ============================================================
write_header \
  "${RESNET_LOG}" \
  "ResNet18" \
  "ResNet18" \
  "${RESNET_FLEX_FILE}" \
  "${RESNET_IPC_LIST[*]}" \
  "normal flex only uses syn_steps=1; no flex syn_steps=20 for ResNet18; ResNet18 uses --Iteration=50 when --ipc>=100."

for ipc in "${RESNET_IPC_LIST[@]}"; do
  resnet_iteration="$(iteration_for_model_ipc "ResNet18" "${ipc}")"

  # original syn_steps=1
  run_original "${RESNET_LOG}" "ResNet18" "ResNet18" "${ipc}" "1" "${resnet_iteration}"

  # normal flex syn_steps=1 only
  run_flex_case "${RESNET_LOG}" "ResNet18" "ResNet18" "${RESNET_FLEX_FILE}" "${ipc}" "${NORMAL_FLEX_SYN_STEPS}" "1"   "--no-v_fuse" "flex" "${resnet_iteration}"
  run_flex_case "${RESNET_LOG}" "ResNet18" "ResNet18" "${RESNET_FLEX_FILE}" "${ipc}" "${NORMAL_FLEX_SYN_STEPS}" "1"   "--v_fuse"    "flex" "${resnet_iteration}"
  run_flex_case "${RESNET_LOG}" "ResNet18" "ResNet18" "${RESNET_FLEX_FILE}" "${ipc}" "${NORMAL_FLEX_SYN_STEPS}" "1 1" "--v_fuse"    "flex" "${resnet_iteration}"
  run_flex_case "${RESNET_LOG}" "ResNet18" "ResNet18" "${RESNET_FLEX_FILE}" "${ipc}" "${NORMAL_FLEX_SYN_STEPS}" "1 0" "--v_fuse"    "flex" "${resnet_iteration}"

  # original syn_steps=20 still runs
  run_original "${RESNET_LOG}" "ResNet18" "ResNet18" "${ipc}" "20" "${resnet_iteration}"
done

write_footer "${RESNET_LOG}"

echo "[DONE] All commands finished."
echo "ConvNet log:  ${CONV_LOG}"
echo "ViT log:      ${VIT_LOG}"
echo "ResNet18 log: ${RESNET_LOG}"