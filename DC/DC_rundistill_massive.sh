#!/usr/bin/env bash
# DC FlexFuse time-test sweep.
# Select models with --model/--models and dataset with --dataset.
# Default: all four models, CIFAR10 + Tiny.
# batch_real: CIFAR10=1 / 10 / 50 / 100; Tiny=1 / 10 / 50.
# Original: main_optimized.py. Fused: Fuse 1 / 2 / 3.
# Every configuration runs twice.
# Logs are written under testlog_dc/<dataset>/.

set -u
set -o pipefail

usage() {
  cat <<'HELP'
Usage:
  bash rundistill_dc_select_model_dataset.sh [--model conv|vit|resnet|vgg|all] [--dataset cifar10|tiny|all]
  bash rundistill_dc_select_model_dataset.sh [--models conv,vit,resnet,vgg] [--dataset cifar10|tiny|all]

Examples:
  bash rundistill_dc_select_model_dataset.sh --model conv --dataset cifar10
  bash rundistill_dc_select_model_dataset.sh --models vit,resnet --dataset tiny
  bash rundistill_dc_select_model_dataset.sh --model all --dataset all

Default:
  --model all
  --dataset all
  LOG_DIR=testlog_dc
HELP
}

MODEL_SELECTION="${MODEL_SELECTION:-all}"
DATASET_SELECTION="${DATASET_SELECTION:-all}"

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
    --dataset|-d)
      if [ "$#" -lt 2 ]; then
        echo "[ERROR] $1 requires an argument: cifar10, tiny, or all" >&2
        usage
        exit 1
      fi
      DATASET_SELECTION="$2"
      shift 2
      ;;
    --dataset=*)
      DATASET_SELECTION="${1#*=}"
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

normalize_dataset_selection() {
  case "${1,,}" in
    cifar10|cifar)
      echo "CIFAR10"
      ;;
    tiny|tinyimagenet|tiny-imagenet)
      echo "Tiny"
      ;;
    all)
      echo "CIFAR10 Tiny"
      ;;
    *)
      echo "[ERROR] unknown dataset selection: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
}

SELECTED_MODELS="$(normalize_model_selection "${MODEL_SELECTION}")"
SELECTED_DATASETS="$(normalize_dataset_selection "${DATASET_SELECTION}")"

MAIN_FILE="${MAIN_FILE:-main_fused.py}"
ORIGINAL_FILE="${ORIGINAL_FILE:-main_optimized.py}"
LOG_DIR="${LOG_DIR:-testlog_dc}"
DATA_PATH="${DATA_PATH:-/scratch/yguo25/files/mtt-distillation/dataset}"
SAVE_PATH="${SAVE_PATH:-${LOG_DIR}/results}"

CIFAR10_BATCH_REAL_LIST=(1 10 50 100)
TINY_BATCH_REAL_LIST=(1 10 50)
read -r -a FUSE_LIST <<< "${FUSE_VALUES:-1 2 3}"
REPEATS="${REPEATS:-2}"

IPC="${IPC:-10}"
LR_IMG="${LR_IMG:-0.1}"
LR_NET="${LR_NET:-0.01}"
BATCH_TRAIN="${BATCH_TRAIN:-256}"
INIT="${INIT:-noise}"
DIS_METRIC="${DIS_METRIC:-ours}"
ITERATION="${ITERATION:-50}"

mkdir -p "${LOG_DIR}" "${SAVE_PATH}"

model_selected() {
  local target="$1"
  case " ${SELECTED_MODELS} " in
    *" ${target} "*) return 0 ;;
    *) return 1 ;;
  esac
}

model_tag() {
  case "$1" in
    ConvNet) echo "conv" ;;
    ViT) echo "vit" ;;
    ResNet18) echo "resnet18" ;;
    VGG11) echo "vgg11" ;;
    *) echo "$1" | tr '[:upper:]' '[:lower:]' ;;
  esac
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

write_header() {
  local log_file="$1"
  local dataset="$2"
  local model="$3"

  : > "${log_file}"

  {
    echo "============================================================"
    echo "DC TIME TEST SWEEP START"
    echo "START_TIME: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "DATASET: ${dataset}"
    echo "MODEL: ${model}"
    echo "ORIGINAL_FILE: ${ORIGINAL_FILE}"
    echo "MAIN_FILE: ${MAIN_FILE}"
    if [ "${dataset}" = "Tiny" ]; then
      echo "BATCH_REAL_LIST: ${TINY_BATCH_REAL_LIST[*]}"
    else
      echo "BATCH_REAL_LIST: ${CIFAR10_BATCH_REAL_LIST[*]}"
    fi
    echo "FUSE_LIST: ${FUSE_LIST[*]}"
    echo "REPEATS: ${REPEATS}"
    echo "IPC: ${IPC}"
    echo "ITERATION: ${ITERATION}"
    echo "USE_BARRIER: True"
    echo "ACC_TEST: False"
    echo "V_FUSE: True"
    echo "DATA_PATH: ${DATA_PATH}"
    echo "SAVE_PATH: ${SAVE_PATH}"
    echo "LOG_FILE: ${log_file}"
    echo "============================================================"
    echo ""
  } >> "${log_file}"
}

write_footer() {
  local log_file="$1"

  {
    echo ""
    echo "============================================================"
    echo "DC TIME TEST SWEEP END"
    echo "END_TIME: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
  } >> "${log_file}"
}

run_original_once() {
  local dataset="$1"
  local model="$2"
  local batch_real="$3"
  local repeat_id="$4"
  local log_file="$5"

  local tag="DATASET=${dataset} | MODEL=${model} | CASE=original | BATCH_REAL=${batch_real}"
  local cmd=(
    python "${ORIGINAL_FILE}"
    --method=DC
    --dataset="${dataset}"
    --model="${model}"
    --ipc="${IPC}"
    --lr_img="${LR_IMG}"
    --lr_net="${LR_NET}"
    --batch_real="${batch_real}"
    --batch_train="${BATCH_TRAIN}"
    --init="${INIT}"
    --data_path="${DATA_PATH}"
    --save_path="${SAVE_PATH}"
    --dis_metric="${DIS_METRIC}"
    --Iteration="${ITERATION}"
    --use-barrier
    --skip_eval
  )

  echo "[RUN] ${tag} | repeat ${repeat_id}/${REPEATS}"

  {
    echo "=============================="
    echo "$(date '+%Y-%m-%d %H:%M:%S') | ${tag} | REPEAT=${repeat_id}/${REPEATS}"
    printf 'CMD:'
    printf ' %q' "${cmd[@]}"
    echo ""
    echo "------------------------------"
  } >> "${log_file}"

  WANDB_SILENT=true \
  PYTHONWARNINGS=ignore \
  TQDM_DISABLE=1 \
  PYTHONUNBUFFERED=1 \
  "${cmd[@]}" 2>&1 | clean_output >> "${log_file}"

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
    echo "[ERROR] ${tag} | repeat ${repeat_id}/${REPEATS} | EXIT_CODE=${exit_code}"
  fi

  return 0
}

run_once() {
  local dataset="$1"
  local model="$2"
  local fuse="$3"
  local batch_real="$4"
  local repeat_id="$5"
  local log_file="$6"

  local tag="DATASET=${dataset} | MODEL=${model} | FUSE=${fuse} | BATCH_REAL=${batch_real}"
  local cmd=(
    python "${MAIN_FILE}"
    --method=DC
    --dataset="${dataset}"
    --model="${model}"
    --ipc="${IPC}"
    --lr_img="${LR_IMG}"
    --lr_net="${LR_NET}"
    --batch_real="${batch_real}"
    --batch_train="${BATCH_TRAIN}"
    --init="${INIT}"
    --data_path="${DATA_PATH}"
    --save_path="${SAVE_PATH}"
    --dis_metric="${DIS_METRIC}"
    --Iteration="${ITERATION}"
    --Fuse="${fuse}"
    --v_fuse
    --use-barrier
    --skip_eval
  )

  echo "[RUN] ${tag} | repeat ${repeat_id}/${REPEATS}"

  {
    echo "=============================="
    echo "$(date '+%Y-%m-%d %H:%M:%S') | ${tag} | REPEAT=${repeat_id}/${REPEATS}"
    printf 'CMD:'
    printf ' %q' "${cmd[@]}"
    echo ""
    echo "------------------------------"
  } >> "${log_file}"

  WANDB_SILENT=true \
  PYTHONWARNINGS=ignore \
  TQDM_DISABLE=1 \
  PYTHONUNBUFFERED=1 \
  "${cmd[@]}" 2>&1 | clean_output >> "${log_file}"

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
    echo "[ERROR] ${tag} | repeat ${repeat_id}/${REPEATS} | EXIT_CODE=${exit_code}"
  fi

  return 0
}

run_model_tests() {
  local dataset="$1"
  local model="$2"
  local dataset_log_dir="$3"
  local log_file="${dataset_log_dir}/$(model_tag "${model}").txt"
  local fuse
  local batch_real
  local repeat_id
  local batch_real_list=()

  if [ "${dataset}" = "Tiny" ]; then
    batch_real_list=( "${TINY_BATCH_REAL_LIST[@]}" )
  else
    batch_real_list=( "${CIFAR10_BATCH_REAL_LIST[@]}" )
  fi

  write_header "${log_file}" "${dataset}" "${model}"

  for batch_real in "${batch_real_list[@]}"; do
    for repeat_id in $(seq 1 "${REPEATS}"); do
      run_original_once "${dataset}" "${model}" "${batch_real}" "${repeat_id}" "${log_file}"
    done
    for fuse in "${FUSE_LIST[@]}"; do
      for repeat_id in $(seq 1 "${REPEATS}"); do
        run_once "${dataset}" "${model}" "${fuse}" "${batch_real}" "${repeat_id}" "${log_file}"
      done
    done
  done

  write_footer "${log_file}"
  echo "${model} log: ${log_file}"
}

run_dataset_tests() {
  local dataset="$1"
  local dataset_log_dir="${LOG_DIR}/${dataset}"
  mkdir -p "${dataset_log_dir}"

  echo "[INFO] DATASET=${dataset}"

  if model_selected "ConvNet"; then
    run_model_tests "${dataset}" "ConvNet" "${dataset_log_dir}"
  else
    echo "[SKIP] ConvNet"
  fi

  if model_selected "ViT"; then
    run_model_tests "${dataset}" "ViT" "${dataset_log_dir}"
  else
    echo "[SKIP] ViT"
  fi

  if model_selected "ResNet18"; then
    run_model_tests "${dataset}" "ResNet18" "${dataset_log_dir}"
  else
    echo "[SKIP] ResNet18"
  fi

  if model_selected "VGG11"; then
    run_model_tests "${dataset}" "VGG11" "${dataset_log_dir}"
  else
    echo "[SKIP] VGG11"
  fi
}

echo "[INFO] LOG_DIR=${LOG_DIR}"
echo "[INFO] SELECTED_MODELS=${SELECTED_MODELS}"
echo "[INFO] SELECTED_DATASETS=${SELECTED_DATASETS}"

for dataset in ${SELECTED_DATASETS}; do
  run_dataset_tests "${dataset}"
done

echo "[DONE] DC timing sweep finished."
echo "[DONE] Logs: ${LOG_DIR}/"
