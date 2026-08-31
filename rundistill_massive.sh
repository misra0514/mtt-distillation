#!/usr/bin/env bash
# Massive distill time-test sweep, v7.
#
# v6 changes:
#   1) normal cases run twice with --use-barrier and twice without --use-barrier.
#   2) add distill_original_fwdtimetest.py for every model/ipc/original syn_steps.
#      It runs once only and never receives --use-barrier.
#   3) remove all ckpt flex tests.
#   4) ConvNet/ViT/ResNet18 all run.
#   5) ViT uses --Iteration=50 when IPC>=50; ResNet18 uses --Iteration=50 when IPC>=10; ConvNet stays at 350.
#   6) all distill flex cases use --v_fuse only.
#   7) flex masks are 1, 1 1, 1 0, and 1 1 0 0.
#
# Run:
#   bash rundistill_massive_524_v7_vfuse_masks_1_11_10_1100.sh

set -u
set -o pipefail

LOG_DIR="${LOG_DIR:-testlog84_vit}"
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
USE_BARRIER_REPEATS=2
NO_BARRIER_REPEATS=2
FWDTEST_REPEATS=1

ORIGINAL_FILE="distill_original_timeTest.py"
ORIGINAL_FWD_FILE="distill_original_fwdtimetest.py"

CONV_FLEX_FILE="distill_flexFuse_timeTest_conv.py"
VIT_FLEX_FILE="distill_flexFuse_timeTest_ViT.py"
RESNET_FLEX_FILE="distill_flexFuse_timeTest_resnet18.py"

# All three models now use the same IPC sweep.
CONV_IPC_LIST=(1 10 50 100)
VIT_IPC_LIST=(1 10 50 100)
RESNET_IPC_LIST=(1 10 50 100)

ORIGINAL_SYN_STEPS_LIST=(1 20)
NORMAL_FLEX_SYN_STEPS=1

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
    echo "ORIGINAL FILE: ${ORIGINAL_FILE}"
    echo "ORIGINAL FWDTEST FILE: ${ORIGINAL_FWD_FILE}"
    echo "IPC_LIST:    ${ipc_list_text}"
    echo "ORIGINAL_SYN_STEPS: ${ORIGINAL_SYN_STEPS_LIST[*]}"
    echo "NORMAL_FLEX_SYN_STEPS: ${NORMAL_FLEX_SYN_STEPS}"
    echo "DEFAULT_ITERATION: ${ITERATION}"
    echo "USE_BARRIER_REPEATS: ${USE_BARRIER_REPEATS}"
    echo "NO_BARRIER_REPEATS:  ${NO_BARRIER_REPEATS}"
    echo "FWDTEST_REPEATS: ${FWDTEST_REPEATS}"
    echo "LOG_FILE:    ${log_file}"
    echo "NOTE: no ckpt flex tests; --AccTest=True is removed; normal cases run twice with --use-barrier and twice without it; all flex cases use --v_fuse."
    echo "FWDTEST: ${ORIGINAL_FWD_FILE} runs once only and never receives --use-barrier."
    if [ -n "${extra_note}" ]; then
      echo "EXTRA: ${extra_note}"
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

iteration_for_model_ipc() {
  local model_label="$1"
  local ipc="$2"

  if [ "${model_label}" = "ViT" ] && [ "${ipc}" -ge 50 ]; then
    echo "50"
  elif [ "${model_label}" = "ResNet18" ] && [ "${ipc}" -ge 10 ]; then
    echo "50"
  else
    echo "${ITERATION}"
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

  for r in $(seq 1 "${USE_BARRIER_REPEATS}"); do
    run_cmd_once "${log_file}" "${tag}" "use_barrier" "${r}" "${USE_BARRIER_REPEATS}" "$@"
  done

  for r in $(seq 1 "${NO_BARRIER_REPEATS}"); do
    run_cmd_once "${log_file}" "${tag}" "no_barrier" "${r}" "${NO_BARRIER_REPEATS}" "$@"
  done
}

run_fwdtest_once() {
  local log_file="$1"
  local tag="$2"
  shift 2

  local barrier_mode="no_barrier"
  local repeat_id="1"
  local repeat_total="1"

  echo "[RUN] ${tag} | FWDTEST | no --use-barrier | repeat ${repeat_id}/${repeat_total}"

  {
    echo "=============================="
    echo "$(date '+%Y-%m-%d %H:%M:%S') | ${tag} | BARRIER=${barrier_mode} | REPEAT=${repeat_id}/${repeat_total}"
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
    echo "[ERROR] command failed with EXIT_CODE=${exit_code}: ${tag} | FWDTEST"
  fi

  return 0
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

run_original_fwdtest() {
  local log_file="$1"
  local model_label="$2"
  local model_arg="$3"
  local ipc="$4"
  local syn_steps="$5"
  local iteration="${6:-${ITERATION}}"

  # shellcheck disable=SC2207
  local args=( $(common_args "${ipc}" "${syn_steps}" "${model_arg}" "${iteration}") )

  run_fwdtest_once \
    "${log_file}" \
    "MODEL=${model_label} | IPC=${ipc} | SYN_STEPS=${syn_steps} | ITERATION=${iteration} | CASE=original_fwdtest" \
    python "${ORIGINAL_FWD_FILE}" "${args[@]}"
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

run_model_block() {
  local log_file="$1"
  local model_label="$2"
  local model_arg="$3"
  local flex_file="$4"
  shift 4
  local ipc_list=( "$@" )

  local ipc
  local model_iteration

  for ipc in "${ipc_list[@]}"; do
    model_iteration="$(iteration_for_model_ipc "${model_label}" "${ipc}")"

    run_original "${log_file}" "${model_label}" "${model_arg}" "${ipc}" "1" "${model_iteration}"
    run_original_fwdtest "${log_file}" "${model_label}" "${model_arg}" "${ipc}" "1" "${model_iteration}"

    # Distill flex: v_fuse is always ON.
    # Run exactly these four masks:
    #   1
    #   1 1
    #   1 0
    #   1 1 0 0
    run_flex_case "${log_file}" "${model_label}" "${model_arg}" "${flex_file}" "${ipc}" "${NORMAL_FLEX_SYN_STEPS}" "1"       "--v_fuse" "flex" "${model_iteration}"
    run_flex_case "${log_file}" "${model_label}" "${model_arg}" "${flex_file}" "${ipc}" "${NORMAL_FLEX_SYN_STEPS}" "1 1"     "--v_fuse" "flex" "${model_iteration}"
    run_flex_case "${log_file}" "${model_label}" "${model_arg}" "${flex_file}" "${ipc}" "${NORMAL_FLEX_SYN_STEPS}" "1 0"     "--v_fuse" "flex" "${model_iteration}"
    run_flex_case "${log_file}" "${model_label}" "${model_arg}" "${flex_file}" "${ipc}" "${NORMAL_FLEX_SYN_STEPS}" "1 1 0 0" "--v_fuse" "flex" "${model_iteration}"

    run_original "${log_file}" "${model_label}" "${model_arg}" "${ipc}" "20" "${model_iteration}"
    run_original_fwdtest "${log_file}" "${model_label}" "${model_arg}" "${ipc}" "20" "${model_iteration}"
  done
}

# ============================================================
# ConvNet tests
# ============================================================
write_header \
  "${CONV_LOG}" \
  "ConvNet" \
  "ConvNet" \
  "${CONV_FLEX_FILE}" \
  "${CONV_IPC_LIST[*]}" \
  "normal flex only uses syn_steps=1; all flex uses --v_fuse; masks are 1 / 1 1 / 1 0 / 1 1 0 0; no ckpt flex; ConvNet keeps --Iteration=${ITERATION} for all IPCs."

run_model_block "${CONV_LOG}" "ConvNet" "ConvNet" "${CONV_FLEX_FILE}" "${CONV_IPC_LIST[@]}"
write_footer "${CONV_LOG}"

# # ============================================================
# # ViT tests
# # ============================================================
# write_header \
#   "${VIT_LOG}" \
#   "ViT" \
#   "ViT" \
#   "${VIT_FLEX_FILE}" \
#   "${VIT_IPC_LIST[*]}" \
#   "normal flex only uses syn_steps=1; all flex uses --v_fuse; masks are 1 / 1 1 / 1 0 / 1 1 0 0; no flex syn_steps=20 for ViT; ViT uses --Iteration=50 when --ipc>=50."

# run_model_block "${VIT_LOG}" "ViT" "ViT" "${VIT_FLEX_FILE}" "${VIT_IPC_LIST[@]}"
# write_footer "${VIT_LOG}"

# # ============================================================
# # ResNet18 tests
# # ============================================================
# write_header \
#   "${RESNET_LOG}" \
#   "ResNet18" \
#   "ResNet18" \
#   "${RESNET_FLEX_FILE}" \
#   "${RESNET_IPC_LIST[*]}" \
#   "normal flex only uses syn_steps=1; all flex uses --v_fuse; masks are 1 / 1 1 / 1 0 / 1 1 0 0; no flex syn_steps=20 for ResNet18; ResNet18 uses --Iteration=50 when --ipc>=10."

# run_model_block "${RESNET_LOG}" "ResNet18" "ResNet18" "${RESNET_FLEX_FILE}" "${RESNET_IPC_LIST[@]}"
# write_footer "${RESNET_LOG}"

# echo "[DONE] All commands finished."
# echo "ConvNet log:  ${CONV_LOG}"
# echo "ViT log:      ${VIT_LOG}"
# echo "ResNet18 log: ${RESNET_LOG}"