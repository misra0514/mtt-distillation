#!/usr/bin/env bash
# Massive distill time-test sweep, v9.
# v9: add VGG11 as a selectable model; use the SAME sweep configuration as ResNet18.
# Existing ConvNet/ViT/ResNet18 test cases and command-line arguments are unchanged.
# Only selected model logs are truncated or written.
#
# Examples:
#   bash rundistill_massive_v9_select_model_date_logdir.sh --model vgg
#   bash rundistill_massive_v9_select_model_date_logdir.sh --models vgg,resnet
#   bash rundistill_massive_v9_select_model_date_logdir.sh --model all

set -u
set -o pipefail

usage() {
  cat <<'HELP'
Usage:
  bash rundistill_massive_v9_select_model_date_logdir.sh [--model conv|vit|resnet|vgg|all]
  bash rundistill_massive_v9_select_model_date_logdir.sh [--models conv,vit,resnet,vgg]

Examples:
  bash rundistill_massive_v9_select_model_date_logdir.sh --model vgg
  bash rundistill_massive_v9_select_model_date_logdir.sh --models vgg,resnet
  bash rundistill_massive_v9_select_model_date_logdir.sh --model all
  LOG_DIR=testlog_custom bash rundistill_massive_v9_select_model_date_logdir.sh --model vgg

Default:
  --model all
  LOG_DIR=testlogYYYYMMDD
HELP
}

MODEL_SELECTION="${MODEL_SELECTION:-all}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --model|--models|-m)
      if [ "$#" -lt 2 ]; then
        echo "[ERROR] $1 requires an argument: conv, vit, resnet, vgg, or all" >&2
        usage
        exit 1
      fi
      MODEL_SELECTION="$2"
      shift 2
      ;;
    --model=*|--models=*)
      MODEL_SELECTION="${1#*=}"
      shift
      ;;
    conv|ConvNet|convnet|vit|ViT|resnet|ResNet18|resnet18|vgg|VGG|vgg11|VGG11|all)
      MODEL_SELECTION="$1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

normalize_model_selection() {
  local raw="${1//,/ }"
  local item
  local normalized=""

  for item in ${raw}; do
    case "${item,,}" in
      all)
        echo "ConvNet ViT ResNet18 VGG11"
        return 0
        ;;
      conv|convnet)
        normalized="${normalized} ConvNet"
        ;;
      vit)
        normalized="${normalized} ViT"
        ;;
      resnet|resnet18)
        normalized="${normalized} ResNet18"
        ;;
      vgg|vgg11)
        normalized="${normalized} VGG11"
        ;;
      "")
        ;;
      *)
        echo "[ERROR] unknown model selection: ${item}" >&2
        usage >&2
        exit 1
        ;;
    esac
  done

  if [ -z "${normalized}" ]; then
    echo "[ERROR] empty model selection" >&2
    usage >&2
    exit 1
  fi

  echo "${normalized# }"
}

SELECTED_MODELS="$(normalize_model_selection "${MODEL_SELECTION}")"
DATE_TAG="$(date '+%Y%m%d')"
LOG_DIR="${LOG_DIR:-testlog${DATE_TAG}}"
mkdir -p "${LOG_DIR}"

CONV_LOG="${LOG_DIR}/conv.txt"
VIT_LOG="${LOG_DIR}/vit.txt"
RESNET_LOG="${LOG_DIR}/resnet18.txt"
VGG_LOG="${LOG_DIR}/vgg11.txt"

model_selected() {
  local target="$1"
  case " ${SELECTED_MODELS} " in
    *" ${target} "*) return 0 ;;
    *) return 1 ;;
  esac
}

# DATASET="CIFAR10"
DATASET="Tiny"
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
VGG_FLEX_FILE="distill_flexFuse_timeTest_vgg.py"

CONV_IPC_LIST=(1 10 50 100)
VIT_IPC_LIST=(1 10 50 100)
RESNET_IPC_LIST=(1 10 50 100)
VGG_IPC_LIST=(1 10 50 100)

ORIGINAL_SYN_STEPS_LIST=(1 20)
NORMAL_FLEX_SYN_STEPS=1

write_header() {
  local log_file="$1"
  local model_label="$2"
  local model_arg="$3"
  local flex_file="$4"
  local ipc_list_text="$5"
  local extra_note="${6:-}"

  # Truncate only the log file for the selected model being run.
  : > "${log_file}"

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
  elif { [ "${model_label}" = "ResNet18" ] || [ "${model_label}" = "VGG11" ]; } && [ "${ipc}" -ge 10 ]; then
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

    # Distill flex: v_fuse is always ON; run exactly the same four masks.
    run_flex_case "${log_file}" "${model_label}" "${model_arg}" "${flex_file}" "${ipc}" "${NORMAL_FLEX_SYN_STEPS}" "1"       "--v_fuse" "flex" "${model_iteration}"
    run_flex_case "${log_file}" "${model_label}" "${model_arg}" "${flex_file}" "${ipc}" "${NORMAL_FLEX_SYN_STEPS}" "1 1"     "--v_fuse" "flex" "${model_iteration}"
    run_flex_case "${log_file}" "${model_label}" "${model_arg}" "${flex_file}" "${ipc}" "${NORMAL_FLEX_SYN_STEPS}" "1 0"     "--v_fuse" "flex" "${model_iteration}"
    run_flex_case "${log_file}" "${model_label}" "${model_arg}" "${flex_file}" "${ipc}" "${NORMAL_FLEX_SYN_STEPS}" "1 1 0 0" "--v_fuse" "flex" "${model_iteration}"

    run_original "${log_file}" "${model_label}" "${model_arg}" "${ipc}" "20" "${model_iteration}"
    run_original_fwdtest "${log_file}" "${model_label}" "${model_arg}" "${ipc}" "20" "${model_iteration}"
  done
}

run_convnet_tests() {
  write_header \
    "${CONV_LOG}" \
    "ConvNet" \
    "ConvNet" \
    "${CONV_FLEX_FILE}" \
    "${CONV_IPC_LIST[*]}" \
    "normal flex only uses syn_steps=1; all flex uses --v_fuse; masks are 1 / 1 1 / 1 0 / 1 1 0 0; no ckpt flex; ConvNet keeps --Iteration=${ITERATION} for all IPCs."

  run_model_block "${CONV_LOG}" "ConvNet" "ConvNet" "${CONV_FLEX_FILE}" "${CONV_IPC_LIST[@]}"
  write_footer "${CONV_LOG}"
}

run_vit_tests() {
  write_header \
    "${VIT_LOG}" \
    "ViT" \
    "ViT" \
    "${VIT_FLEX_FILE}" \
    "${VIT_IPC_LIST[*]}" \
    "normal flex only uses syn_steps=1; all flex uses --v_fuse; masks are 1 / 1 1 / 1 0 / 1 1 0 0; no flex syn_steps=20 for ViT; ViT uses --Iteration=50 when --ipc>=50."

  run_model_block "${VIT_LOG}" "ViT" "ViT" "${VIT_FLEX_FILE}" "${VIT_IPC_LIST[@]}"
  write_footer "${VIT_LOG}"
}

run_resnet18_tests() {
  write_header \
    "${RESNET_LOG}" \
    "ResNet18" \
    "ResNet18" \
    "${RESNET_FLEX_FILE}" \
    "${RESNET_IPC_LIST[*]}" \
    "normal flex only uses syn_steps=1; all flex uses --v_fuse; masks are 1 / 1 1 / 1 0 / 1 1 0 0; no flex syn_steps=20 for ResNet18; ResNet18 uses --Iteration=50 when --ipc>=10."

  run_model_block "${RESNET_LOG}" "ResNet18" "ResNet18" "${RESNET_FLEX_FILE}" "${RESNET_IPC_LIST[@]}"
  write_footer "${RESNET_LOG}"
}

run_vgg11_tests() {
  write_header \
    "${VGG_LOG}" \
    "VGG11" \
    "VGG11" \
    "${VGG_FLEX_FILE}" \
    "${VGG_IPC_LIST[*]}" \
    "same settings as ResNet18: flex syn_steps=1, --v_fuse, masks 1 / 1 1 / 1 0 / 1 1 0 0; original syn_steps=1/20; --Iteration=50 when --ipc>=10."

  run_model_block "${VGG_LOG}" "VGG11" "VGG11" "${VGG_FLEX_FILE}" "${VGG_IPC_LIST[@]}"
  write_footer "${VGG_LOG}"
}

echo "[INFO] LOG_DIR=${LOG_DIR}"
echo "[INFO] SELECTED_MODELS=${SELECTED_MODELS}"

if model_selected "ConvNet"; then
  run_convnet_tests
else
  echo "[SKIP] ConvNet"
fi

if model_selected "ViT"; then
  run_vit_tests
else
  echo "[SKIP] ViT"
fi

if model_selected "ResNet18"; then
  run_resnet18_tests
else
  echo "[SKIP] ResNet18"
fi

if model_selected "VGG11"; then
  run_vgg11_tests
else
  echo "[SKIP] VGG11"
fi

echo "[DONE] Selected commands finished."
if model_selected "ConvNet"; then
  echo "ConvNet log:  ${CONV_LOG}"
fi
if model_selected "ViT"; then
  echo "ViT log:      ${VIT_LOG}"
fi
if model_selected "ResNet18"; then
  echo "ResNet18 log: ${RESNET_LOG}"
fi
if model_selected "VGG11"; then
  echo "VGG11 log:    ${VGG_LOG}"
fi
