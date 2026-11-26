#!/usr/bin/env bash
# Massive distill time-test sweep, v17 (kernel-only; dataset-specific IPC; ImageNet added).
# ONE timing set per case: no --load_all, with --use-barrier.
# Kernel set repeats twice. Original -> Flex masks (1 / 11 / 10, v_fuse on/off) -> FWD time test.
# Default ConvNet only; other model blocks use unified Flex implementation when selected.
# ConvNet: bf=Conv Python file, ff=DataLayout Python file (neither receives --datalayout).
# ViT/ResNet18/VGG11: unified Python file, explicitly pass --datalayout=b_f or --datalayout=f_f.
# All cases use syn_steps=1; never test syn_steps=20.
# No checkpoint tests.
# IPC by dataset: CIFAR10/ImageNet = 1,10,50,100; Tiny = 1,5,10.
#
# Examples:
#   bash rundistill_massive_v17_tiny_imagenet_ipc.sh --model vgg --dataset cifar10
#   bash rundistill_massive_v17_tiny_imagenet_ipc.sh --models vgg,resnet --dataset tiny
#   bash rundistill_massive_v17_tiny_imagenet_ipc.sh --model conv --dataset imagenet --layout bf
#   bash rundistill_massive_flex_only.sh --model conv --dataset tiny --layout ff --flex-only
#   bash rundistill_massive_v17_tiny_imagenet_ipc.sh --model conv --dataset cifar10 --layout both
#   bash rundistill_massive_v17_tiny_imagenet_ipc.sh --model all --dataset all

set -u
set -o pipefail

usage() {
  cat <<'HELP'
Usage:
  bash rundistill_massive_v17_tiny_imagenet_ipc.sh [--model conv|vit|resnet|vgg|all] [--dataset cifar10|tiny|imagenet|all]
  bash rundistill_massive_flex_only.sh [--models conv,vit,resnet,vgg] [--dataset cifar10|tiny|imagenet|all] [--layout bf|ff|both] [--flex-only]

Examples:
  bash rundistill_massive_v17_tiny_imagenet_ipc.sh --model vgg --dataset cifar10
  bash rundistill_massive_v17_tiny_imagenet_ipc.sh --models vgg,resnet --dataset tiny
  bash rundistill_massive_v17_tiny_imagenet_ipc.sh --model conv --dataset imagenet --layout bf
  bash rundistill_massive_flex_only.sh --model conv --dataset tiny --layout ff --flex-only
  bash rundistill_massive_v17_tiny_imagenet_ipc.sh --model conv --dataset cifar10 --layout both
  bash rundistill_massive_v17_tiny_imagenet_ipc.sh --model all --dataset all
  bash rundistill_massive_v17_tiny_imagenet_ipc.sh --model conv --dataset tiny --layout both --name trialA
  LOG_DIR=testlog_custom bash rundistill_massive_v17_tiny_imagenet_ipc.sh --model vgg --dataset all

Default:
  --model conv
  --dataset cifar10
  --layout bf (all models; ConvNet selects a file, other models pass --datalayout to unified)
  --flex-only: skip Original and Original FWD; run only Flex cases (default: off)
  --name LABEL (alias: --log-name; optional)
  LOG_DIR=testlog_massive_[LABEL_]YYYYMMDD_HHMMSS (override with LOG_DIR=...)
HELP
}

MODEL_SELECTION="${MODEL_SELECTION:-conv}"
DATASET_SELECTION="${DATASET_SELECTION:-cifar10}"
LAYOUT_SELECTION="${LAYOUT_SELECTION:-bf}"
LOG_NAME="${LOG_NAME:-}"
FLEX_ONLY=0

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
        echo "[ERROR] $1 requires an argument: cifar10, tiny, imagenet, or all" >&2
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
    --layout|--layouts)
      if [ "$#" -lt 2 ]; then
        echo "[ERROR] $1 requires bf, ff, or both" >&2
        exit 1
      fi
      LAYOUT_SELECTION="$2"
      shift 2
      ;;
    --layout=*|--layouts=*)
      LAYOUT_SELECTION="${1#*=}"
      shift
      ;;
    --flex-only)
      FLEX_ONLY=1
      shift
      ;;
    --name|--log-name)
      if [ "$#" -lt 2 ] || [[ -z "$2" || "$2" == --* ]]; then
        echo '[ERROR] --name requires a nonempty label' >&2
        exit 2
      fi
      LOG_NAME="$2"
      shift 2
      ;;
    --name=*|--log-name=*)
      LOG_NAME="${1#*=}"
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
    imagenet|imagenet1k|imagenet-1k)
      echo "ImageNet"
      ;;
    all)
      echo "CIFAR10 Tiny ImageNet"
      ;;
    *)
      echo "[ERROR] unknown dataset selection: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
}

if [[ -n "$LOG_NAME" && ! "$LOG_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo '[ERROR] --name must start with a letter/number and contain only letters, numbers, dots, underscores, or hyphens' >&2
  exit 2
fi

SELECTED_MODELS="$(normalize_model_selection "${MODEL_SELECTION}")"
SELECTED_DATASETS="$(normalize_dataset_selection "${DATASET_SELECTION}")"

case "${LAYOUT_SELECTION,,}" in
  bf|batch-first|batch_first) LAYOUT_SELECTION="bf" ;;
  ff|fuse-first|fuse_first) LAYOUT_SELECTION="ff" ;;
  both) LAYOUT_SELECTION="both" ;;
  *) echo "[ERROR] --layout must be bf, ff, or both (got: $LAYOUT_SELECTION)" >&2; exit 1 ;;
esac


RUN_TIME="$(date '+%Y%m%d_%H%M%S')"
LOG_SUFFIX="${LOG_NAME:+${LOG_NAME}_}"
LOG_DIR="${LOG_DIR:-testlog_massive_${LOG_SUFFIX}${RUN_TIME}}"

mkdir -p "${LOG_DIR}"

model_selected() {
  local target="$1"
  case " ${SELECTED_MODELS} " in
    *" ${target} "*) return 0 ;;
    *) return 1 ;;
  esac
}

DATASET=""
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
KERNEL_SET_REPEATS=2
# All cases (including the FWD test) run only the kernel timing set; no ckpt cases.

ORIGINAL_FILE="distill_original_timeTest.py"
ORIGINAL_FWD_FILE="distill_original_fwdtimetest.py"
# IMPORTANT: both Original and Original FWDTEST must accept --use-barrier.

# Keep the ConvNet Flex implementation unchanged.
CONV_FLEX_FILE="distill_flexFuse_timeTest_conv.py"   # Change only if your actual ConvNet file has another name.
# Only FF ConvNet runs use this file. Neither ConvNet script accepts --datalayout.
CONV_DATALAYOUT_FLEX_FILE="distill_flexFuse_timeTest_conv_datalayout.py"
# Unified only: accepts --datalayout=b_f or --datalayout=f_f.

# All three non-ConvNet models share one Flex implementation; --model selects architecture.
UNIFIED_FLEX_FILE="distill_flexFuse_timeTest_unified.py"

# Default IPC sweep for CIFAR10 and ImageNet; all models share the dataset-specific sweep.
# ResNet18/VGG11 use Iteration=50 at all IPC.
CONV_IPC_LIST=(1 10 50 100)
VIT_IPC_LIST=(1 10 50 100)
RESNET_IPC_LIST=(1 10 50 1
00)
VGG_IPC_LIST=(1 10 50 100)

# Single syn_steps setting applies to Original, Flex, and FWDTEST alike.
SYN_STEPS=1

write_header() {
  local log_file="$1" model_label="$2" model_arg="$3" flex_file="$4" ipc_list_text="$5"
  local extra_note="${6:-}"
  : > "$log_file"
  cat >> "$log_file" <<EOF
============================================================
TIME TEST SWEEP START
START_TIME: $(date '+%Y-%m-%d %H:%M:%S')
MODEL BLOCK: ${model_label}
MODEL ARG: ${model_arg}
FLEX FILE: ${flex_file}
ORIGINAL FILE: ${ORIGINAL_FILE}
ORIGINAL FWDTEST FILE: ${ORIGINAL_FWD_FILE}
IPC_LIST: ${ipc_list_text}
SYN_STEPS: ${SYN_STEPS} (all cases)
DEFAULT_ITERATION: ${ITERATION}
SET_KERNEL_ONLY: --use-barrier, NO --load_all, repeats=${KERNEL_SET_REPEATS}
FLEX_ONLY: ${FLEX_ONLY}
FLEX_MASKS: 1 / 1 1 / 1 0; VFUSE: --v_fuse / --no-v_fuse
RUN_SELECTION: $(if [[ "$FLEX_ONLY" -eq 1 ]]; then echo "Flex only (Original and FWD skipped)"; else echo "Original -> Flex -> Original FWDTEST"; fi)
LOG_FILE: ${log_file}
CSV_FILE: ${SUMMARY_CSV}
NOTE: no ckpt or total-time cases; kernel-only timing for Original / Flex / FWDTEST.
EXTRA: ${extra_note}
============================================================

EOF
}

write_footer() {
  local log_file="$1"
  {
    echo ""
    echo "============================================================"
    echo "TIME TEST SWEEP END"
    echo "END_TIME: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
  } >> "$log_file"
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
  elif [ "${model_label}" = "ResNet18" ] || [ "${model_label}" = "VGG11" ]; then
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

# The log is authoritative; metric extraction is BEST EFFORT, not an exit condition.
# Prefer synchronized "Kernel time"; alternatively add syn_time + backward_time.
extract_metric() {
  local kind="$1" file="$2"
  awk -v kind="$kind" '
    function number(v) { return v ~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/ }
    {
      low=tolower($0); val=$NF
      if (!number(val) && tolower(val)=="mb" && NF>1) val=$(NF-1)
      if (!number(val)) next
      if (kind=="kernel" && low ~ /^[[:space:]]*kernel[[:space:]_]*time/ && low !~ /without[[:space:]]+barrier/) kernel=val
      if (low ~ /^[[:space:]]*syn_time/) syn=val
      if (low ~ /^[[:space:]]*backward_time/) bwd=val
      if (kind=="cache" && (low ~ /峰值cache使用/ || low ~ /peak.cache/)) cache=val
      if (kind=="tensor" && (low ~ /峰值tensor使用/ || low ~ /peak.tensor/)) tensor=val
    }
    END {
      if (kind=="kernel") {
        if (kernel!="") print kernel
        else if (syn!="" && bwd!="") printf "%.9f\n", syn+bwd
      }
      else if (kind=="cache" && cache!="") print cache
      else if (kind=="tensor" && tensor!="") print tensor
    }
  ' "$file"
}

mean_of_two() {
  if [[ -n "$1" && -n "$2" ]]; then
    awk -v a="$1" -v b="$2" 'BEGIN {printf "%.9f", (a+b)/2}'
  fi
}

# Globals filled by run_cmd_once (no subshell, so run_set can inspect them).
RUN_EXIT=0
RUN_PRIMARY=""
RUN_WALL=""
RUN_CACHE=""
RUN_TENSOR=""

run_cmd_once() {
  local log_file="$1" tag="$2" mode="$3" repeat_id="$4" repeat_total="$5"
  shift 5
  local mode_args=(--use-barrier)  # ALWAYS; never pass --load_all
  local metric_kind="kernel"

  local tmp_output
  tmp_output="$(mktemp "${log_file}.run.XXXXXX")"
  echo "[RUN] ${tag} | SET=${mode} | repeat ${repeat_id}/${repeat_total}"
  {
    echo "=============================="
    echo "$(date '+%Y-%m-%d %H:%M:%S') | ${tag} | SET=${mode} | REPEAT=${repeat_id}/${repeat_total}"
    echo "LOAD_ALL=0 BARRIER=1"
    printf 'CMD: WANDB_SILENT=true PYTHONWARNINGS=ignore TQDM_DISABLE=1 '
    printf '%q ' "$@" "${mode_args[@]}"
    printf '\n------------------------------\n'
  } >> "$log_file"

  local started ended
  started="$(date +%s%N)"
  WANDB_SILENT=true PYTHONWARNINGS=ignore TQDM_DISABLE=1 PYTHONUNBUFFERED=1 \
    "$@" "${mode_args[@]}" 2>&1 | clean_output | tee -a "$log_file" > "$tmp_output"
  local statuses=("${PIPESTATUS[@]}")
  ended="$(date +%s%N)"
  RUN_EXIT="${statuses[0]}"
  RUN_WALL="$(awk -v a="$started" -v b="$ended" 'BEGIN {printf "%.6f", (b-a)/1000000000}')"
  RUN_PRIMARY="$(extract_metric "$metric_kind" "$tmp_output")"
  RUN_CACHE="$(extract_metric cache "$tmp_output")"
  RUN_TENSOR="$(extract_metric tensor "$tmp_output")"
  rm -f "$tmp_output"

  {
    echo "EXIT_CODE: ${RUN_EXIT}"
    echo "PROCESS_WALL_SECONDS: ${RUN_WALL} (includes Python startup, data initialization and preload)"
    echo "PYTHON_${metric_kind^^}_SECONDS: ${RUN_PRIMARY:-N/A}"
    echo "PEAK_CACHE_MB: ${RUN_CACHE:-N/A}"
    echo "PEAK_TENSOR_MB: ${RUN_TENSOR:-N/A}"
    [[ "$RUN_EXIT" -eq 0 ]] || echo "ERROR: command failed with EXIT_CODE=${RUN_EXIT}"
    echo ""
  } >> "$log_file"
  if [[ "$RUN_EXIT" -ne 0 ]]; then
    echo "[ERROR] exit=${RUN_EXIT}: ${tag} | SET=${mode} | repeat ${repeat_id}/${repeat_total}"
  fi
  return 0  # Never mistake a change in printed fields for a failed experiment.
}

# One timing SET, exactly two repeats, summarize both in the TXT and CSV.
run_set() {
  local log_file="$1" tag="$2" mode="$3" model="$4" ipc="$5" steps="$6" iteration="$7" case_name="$8"
  shift 8
  local r successful=0 metric_1="" metric_2="" wall_1="" wall_2="" cache_1="" cache_2="" tensor_1="" tensor_2=""
  local repeats="$KERNEL_SET_REPEATS" metric_name="synchronized_kernel_time_s"
  for ((r=1; r<=repeats; r++)); do
    run_cmd_once "$log_file" "$tag" "$mode" "$r" "$repeats" "$@"
    if [[ "$RUN_EXIT" -eq 0 ]]; then
      ((successful+=1))
      if [[ "$r" -eq 1 ]]; then
        metric_1="$RUN_PRIMARY"; wall_1="$RUN_WALL"; cache_1="$RUN_CACHE"; tensor_1="$RUN_TENSOR"
      elif [[ "$r" -eq 2 ]]; then
        metric_2="$RUN_PRIMARY"; wall_2="$RUN_WALL"; cache_2="$RUN_CACHE"; tensor_2="$RUN_TENSOR"
      fi
    fi
  done
  local mean_metric="" mean_wall="" mean_cache="" mean_tensor=""
  # No partial mean: both repetitions must succeed and expose the field.
  if [[ "$successful" -eq 2 && "$repeats" -eq 2 ]]; then
    mean_metric="$(mean_of_two "$metric_1" "$metric_2")"
    mean_wall="$(mean_of_two "$wall_1" "$wall_2")"
    mean_cache="$(mean_of_two "$cache_1" "$cache_2")"
    mean_tensor="$(mean_of_two "$tensor_1" "$tensor_2")"
  fi
  {
    echo "[MEAN] ${tag} | SET=${mode} | SUCCEEDED=${successful}/${repeats} | ${metric_name}=${mean_metric:-N/A} | process_wall_s=${mean_wall:-N/A} | peak_cache_mb=${mean_cache:-N/A} | peak_tensor_mb=${mean_tensor:-N/A}"
    echo ""
  } >> "$log_file"
  echo "[MEAN] ${tag} | ${mode}: Python=${mean_metric:-N/A} s | process_wall=${mean_wall:-N/A} s (${successful}/${repeats} successful runs)"
  # All fields are simple tags with no commas; no synthetic numbers for missing metrics.
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$DATASET" "$model" "$ipc" "$steps" "$iteration" "$case_name" "$mode" "$repeats" "$successful" \
    "$metric_name" "$metric_1" "$metric_2" "$mean_metric" "$mean_wall" "$mean_cache" "$mean_tensor" "$wall_1" "$wall_2" \
    >> "$SUMMARY_CSV"
}

run_cmd() {
  local log_file="$1" tag="$2" model="$3" ipc="$4" steps="$5" iteration="$6" case_name="$7"
  shift 7
  run_set "$log_file" "$tag" kernel "$model" "$ipc" "$steps" "$iteration" "$case_name" "$@"
}

run_original() {
  local log_file="$1" model_label="$2" model_arg="$3" ipc="$4" syn_steps="$5"
  local iteration="${6:-${ITERATION}}"
  local args=( $(common_args "$ipc" "$syn_steps" "$model_arg" "$iteration") )
  local tag="MODEL=${model_label} | IPC=${ipc} | SYN_STEPS=${syn_steps} | ITERATION=${iteration} | CASE=original"
  run_cmd "$log_file" "$tag" "$model_label" "$ipc" "$syn_steps" "$iteration" original \
    python "$ORIGINAL_FILE" "${args[@]}"
}

run_original_fwdtest() {
  local log_file="$1" model_label="$2" model_arg="$3" ipc="$4" syn_steps="$5"
  local iteration="${6:-${ITERATION}}"
  local args=( $(common_args "$ipc" "$syn_steps" "$model_arg" "$iteration") )
  local tag="MODEL=${model_label} | IPC=${ipc} | SYN_STEPS=${syn_steps} | ITERATION=${iteration} | CASE=original_fwdtest"
  run_cmd "$log_file" "$tag" "$model_label" "$ipc" "$syn_steps" "$iteration" original_fwdtest \
    python "$ORIGINAL_FWD_FILE" "${args[@]}"
}

run_flex_case() {
  local log_file="$1" model_label="$2" model_arg="$3" flex_file="$4" ipc="$5" syn_steps="$6"
  local mask_string="$7" vfuse_arg="$8" case_prefix="${9:-flex}" iteration="${10:-${ITERATION}}"
  local mask_args=()
  read -r -a mask_args <<< "$mask_string"
  local mask_tag="${mask_string// /}" vfuse_tag="no_vfuse"
  [[ "$vfuse_arg" == "--v_fuse" ]] && vfuse_tag="vfuse"
  # Optional final argument after iteration: the actual Python datalayout flag.
  # Empty for ConvNet; unified receives explicit b_f/f_f.
  local datalayout_arg="${11:-}"
  local case_name="${case_prefix}_mask_${mask_tag}_${vfuse_tag}"
  local tag="MODEL=${model_label} | IPC=${ipc} | SYN_STEPS=${syn_steps} | ITERATION=${iteration} | CASE=${case_name}"
  local args=( $(common_args "$ipc" "$syn_steps" "$model_arg" "$iteration") )
  local layout_args=()
  if [[ -n "$datalayout_arg" ]]; then layout_args+=("$datalayout_arg"); fi
  run_cmd "$log_file" "$tag" "$model_label" "$ipc" "$syn_steps" "$iteration" "$case_name" \
    python "$flex_file" --fuse_mask_list "${mask_args[@]}" "$vfuse_arg" "${layout_args[@]}" "${args[@]}"
}

run_model_block() {
  local log_file="$1" model_label="$2" model_arg="$3" flex_file="$4"
  shift 4
  local ipc_list=( "$@" )
  local ipc iteration syn_steps mask vfuse
  for ipc in "${ipc_list[@]}"; do
    iteration="$(iteration_for_model_ipc "$model_label" "$ipc")"
    # 1. Original: syn_steps=1 only.
    if [[ "$FLEX_ONLY" -eq 0 ]]; then
      run_original "$log_file" "$model_label" "$model_arg" "$ipc" "$SYN_STEPS" "$iteration"
    fi
    # 2. Flex: apply --layout to ALL models.
    # Original and FWD baselines remain once per IPC (not duplicated per layout).
    local layout actual_flex actual_extra case_prefix
    local layouts=(bf)
    case "$LAYOUT_SELECTION" in
      ff) layouts=(ff) ;;
      both) layouts=(bf ff) ;;
    esac
    for layout in "${layouts[@]}"; do
      actual_flex="$flex_file"
      actual_extra=""
      case_prefix="flex_${layout}"
      if [[ "$model_label" == "ConvNet" ]]; then
        # ConvNet selects the Python file; neither file has --datalayout.
        if [[ "$layout" == "ff" ]]; then
          actual_flex="$CONV_DATALAYOUT_FLEX_FILE"
        fi
      else
        # Unified implementation has an explicit --datalayout parser option.
        if [[ "$layout" == "ff" ]]; then
          actual_extra="--datalayout=f_f"
        else
          actual_extra="--datalayout=b_f"
        fi
      fi
      for mask in "1" "1 1" "1 0"; do
        for vfuse in "--v_fuse" "--no-v_fuse"; do
          run_flex_case "$log_file" "$model_label" "$model_arg" "$actual_flex" "$ipc" \
            "$SYN_STEPS" "$mask" "$vfuse" "$case_prefix" "$iteration" "$actual_extra"
        done
      done
    done
    # 3. Original forward-only test: syn_steps=1 only.
    if [[ "$FLEX_ONLY" -eq 0 ]]; then
      run_original_fwdtest "$log_file" "$model_label" "$model_arg" "$ipc" "$SYN_STEPS" "$iteration"
    fi
  done
}

run_convnet_tests() {
  SUMMARY_CSV="${CONV_LOG%.txt}_summary.csv"
  printf 'dataset,model,ipc,syn_steps,iteration,case,mode,repeats,success_count,metric_name,run1_s,run2_s,mean_s,process_wall_mean_s,peak_cache_mean_mb,peak_tensor_mean_mb,process_wall_run1_s,process_wall_run2_s\n' > "$SUMMARY_CSV"
  write_header "$CONV_LOG" "ConvNet" "ConvNet" "$CONV_FLEX_FILE" "${CONV_IPC_LIST[*]}" \
    "ConvNet layout=${LAYOUT_SELECTION}; BF=${CONV_FLEX_FILE}; FF=${CONV_DATALAYOUT_FLEX_FILE}; neither passes --datalayout; Original and FWD once per IPC unless --flex-only."
  run_model_block "$CONV_LOG" "ConvNet" "ConvNet" "$CONV_FLEX_FILE" "${CONV_IPC_LIST[@]}"
  write_footer "$CONV_LOG"
}

run_vit_tests() {
  SUMMARY_CSV="${VIT_LOG%.txt}_summary.csv"
  printf 'dataset,model,ipc,syn_steps,iteration,case,mode,repeats,success_count,metric_name,run1_s,run2_s,mean_s,process_wall_mean_s,peak_cache_mean_mb,peak_tensor_mean_mb,process_wall_run1_s,process_wall_run2_s\n' > "$SUMMARY_CSV"
  write_header "$VIT_LOG" "ViT" "ViT" "$UNIFIED_FLEX_FILE" "${VIT_IPC_LIST[*]}" \
    "Unified Flex via --model=ViT; layout=${LAYOUT_SELECTION} (explicit --datalayout=b_f/f_f); --Iteration=50 for IPC >= 50."
  run_model_block "$VIT_LOG" "ViT" "ViT" "$UNIFIED_FLEX_FILE" "${VIT_IPC_LIST[@]}"
  write_footer "$VIT_LOG"
}

run_resnet18_tests() {
  SUMMARY_CSV="${RESNET_LOG%.txt}_summary.csv"
  printf 'dataset,model,ipc,syn_steps,iteration,case,mode,repeats,success_count,metric_name,run1_s,run2_s,mean_s,process_wall_mean_s,peak_cache_mean_mb,peak_tensor_mean_mb,process_wall_run1_s,process_wall_run2_s\n' > "$SUMMARY_CSV"
  write_header "$RESNET_LOG" "ResNet18" "ResNet18" "$UNIFIED_FLEX_FILE" "${RESNET_IPC_LIST[*]}" \
    "Unified Flex via --model=ResNet18; layout=${LAYOUT_SELECTION} (explicit --datalayout=b_f/f_f); --Iteration=50 for all IPC (${RESNET_IPC_LIST[*]})."
  run_model_block "$RESNET_LOG" "ResNet18" "ResNet18" "$UNIFIED_FLEX_FILE" "${RESNET_IPC_LIST[@]}"
  write_footer "$RESNET_LOG"
}

run_vgg11_tests() {
  SUMMARY_CSV="${VGG_LOG%.txt}_summary.csv"
  printf 'dataset,model,ipc,syn_steps,iteration,case,mode,repeats,success_count,metric_name,run1_s,run2_s,mean_s,process_wall_mean_s,peak_cache_mean_mb,peak_tensor_mean_mb,process_wall_run1_s,process_wall_run2_s\n' > "$SUMMARY_CSV"
  write_header "$VGG_LOG" "VGG11" "VGG11" "$UNIFIED_FLEX_FILE" "${VGG_IPC_LIST[*]}" \
    "Unified Flex via --model=VGG11; layout=${LAYOUT_SELECTION} (explicit --datalayout=b_f/f_f); --Iteration=50 for all IPC (${VGG_IPC_LIST[*]})."
  run_model_block "$VGG_LOG" "VGG11" "VGG11" "$UNIFIED_FLEX_FILE" "${VGG_IPC_LIST[@]}"
  write_footer "$VGG_LOG"
}

run_dataset_tests() {
  DATASET="$1"
  # Important: reset for each dataset (including --dataset all).
  # ImageNet retains the original IPC sweep; only Tiny has a reduced sweep.
  local dataset_ipcs=(1 10 50 100)
  if [[ "$DATASET" == "Tiny" || "$DATASET" == "ImageNet" ]]; then
    dataset_ipcs=(1 5 10)
  fi
  CONV_IPC_LIST=("${dataset_ipcs[@]}")
  VIT_IPC_LIST=("${dataset_ipcs[@]}")
  RESNET_IPC_LIST=("${dataset_ipcs[@]}")
  VGG_IPC_LIST=("${dataset_ipcs[@]}")

  local dataset_log_dir="${LOG_DIR}/${DATASET}"
  mkdir -p "${dataset_log_dir}"
  CONV_LOG="${dataset_log_dir}/conv.txt"
  # Each model has an independent summary CSV; set before each model block.
  VIT_LOG="${dataset_log_dir}/vit.txt"
  RESNET_LOG="${dataset_log_dir}/resnet18.txt"
  VGG_LOG="${dataset_log_dir}/vgg11.txt"

  echo "[INFO] DATASET=${DATASET} | IPC_LIST=${dataset_ipcs[*]} | LAYOUT=${LAYOUT_SELECTION}"
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
}

echo "[INFO] LOG_DIR=${LOG_DIR} LOG_NAME=${LOG_NAME:-<none>}"
echo "[INFO] SELECTED_MODELS=${SELECTED_MODELS}"
echo "[INFO] SELECTED_DATASETS=${SELECTED_DATASETS}"
echo "[INFO] FLEX_ONLY=${FLEX_ONLY} (1 skips Original and Original FWD)"
echo "[INFO] LAYOUT=${LAYOUT_SELECTION} (ConvNet: BF=${CONV_FLEX_FILE}, FF=${CONV_DATALAYOUT_FLEX_FILE}; unified: --datalayout=b_f/f_f)"

for dataset in ${SELECTED_DATASETS}; do
  run_dataset_tests "${dataset}"
done

echo "[DONE] Selected commands finished."
