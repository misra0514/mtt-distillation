#!/usr/bin/env bash
# ConvNet / CIFAR10 FlexFuse scalability sweep; stops each family at CUDA OOM.
# Mask family A: [1], [1 1], [1 1 1], ...
# Mask family B: [1 0], [1 1 0 0], [1 1 1 0 0 0], ...
#
# Python exit code ONLY determines SUCCESS; stdout metrics are OPTIONAL.
# Missing/renamed print fields NEVER turn a successful run into an error.
# Full stdout/stderr is always saved in the TXT log; CSV is best-effort.
# Other Python failures (non-OOM nonzero exit) stop the sweep.

set -u
set -o pipefail

# ================== MANUAL SETTINGS ==================
# Set values here; --family can override MASK_FAMILY from the command line.
BARRIER_MODE="on"         # "both": run with and without --use-barrier; "on": only with; "off": only without
LOAD_ALL=0                  # 1 adds --load_all; 0 omits it
SPLIT_FWD_BARRIER=0         # 1 adds --split-fwd-barrier; 0 omits it
MAX_FUSE=0                  # 0: continue until CUDA OOM; N > 0: stop at K=N
MASK_FAMILY="both"          # "1-1": all ones; "1-0": K ones then K zeros; "both": both families
LOG_NAME=""                 # Optional user suffix; implementation is always included in output dir
START_K=1                    # First K to test (inclusive); CLI --start-k overrides
FLEX_IMPL="layout"           # layout or conv; CLI --impl overrides; --file selects a custom script
IMPL_EXPLICIT=0             # Track explicit --impl so stale FLEX_FILE env cannot override it
CUSTOM_FLEX_FILE=""          # Optional manual override of actual script path (CLI --file preferred)

# Other experiment settings. Edit here as needed.
DATASET=CIFAR10
MODEL=ConvNet
IPC=1
SYN_STEPS=1
ITERATION=50
PIX_INIT=real
MAX_EXPERTS=1
EXPERT_EPOCHS=1
MAX_START_EPOCH=1
DETACH_NUM=0
LR_IMG=1000
LR_LR=1e-05
LR_TEACHER=0.01
FUSE=1
# --v_fuse is always enabled; --AccTest is never passed.
# =====================================================

# CLI overrides: --start-k N, --impl conv|layout, --file /path/to/script.py,
# --family 1-1|1-0|both, --name LABEL, and --max-fuse N.
# A selected script changes ONLY the Python filename, not Python arguments.
while (( $# > 0 )); do
  case "$1" in
    --start-k|--start)
      if (( $# < 2 )); then echo '[ERROR] --start-k requires an integer >= 1' >&2; exit 2; fi
      START_K="$2"; shift 2 ;;
    --start-k=*|--start=*)
      START_K="${1#*=}"; shift ;;
    --impl|--implementation)
      if (( $# < 2 )); then echo '[ERROR] --impl requires conv or layout' >&2; exit 2; fi
      FLEX_IMPL="$2"; IMPL_EXPLICIT=1; shift 2 ;;
    --impl=*|--implementation=*)
      FLEX_IMPL="${1#*=}"; IMPL_EXPLICIT=1; shift ;;
    --file|--flex-file)
      if (( $# < 2 )) || [[ -z "$2" ]]; then echo '[ERROR] --file requires a Python file path' >&2; exit 2; fi
      CUSTOM_FLEX_FILE="$2"; shift 2 ;;
    --file=*|--flex-file=*)
      CUSTOM_FLEX_FILE="${1#*=}"; shift ;;
    --max-fuse)
      if (( $# < 2 )); then echo '[ERROR] --max-fuse requires a nonnegative integer' >&2; exit 2; fi
      MAX_FUSE="$2"; shift 2 ;;
    --max-fuse=*)
      MAX_FUSE="${1#*=}"; shift ;;
    --family)
      if (( $# < 2 )); then
        echo '[ERROR] --family requires 1-1, 1-0, or both' >&2
        exit 2
      fi
      MASK_FAMILY="$2"
      shift 2 ;;
    --family=*)
      MASK_FAMILY="${1#--family=}"
      shift ;;
    --name|--log-name)
      if (( $# < 2 )) || [[ -z "$2" ]]; then
        echo '[ERROR] --name requires a nonempty label' >&2
        exit 2
      fi
      LOG_NAME="$2"
      shift 2 ;;
    --name=*|--log-name=*)
      LOG_NAME="${1#*=}"
      shift ;;
    -h|--help)
      printf 'Usage: bash %s [--start-k N] [--impl conv|layout] [--file PYTHON.py] [--max-fuse N] [--family 1-1|1-0|both] [--name LABEL]\n' "${0##*/}"
      printf '  --start-k N: start sweep at K=N (default 1, inclusive)\n'
      printf '  --impl: conv selects *_conv.py; layout selects *_conv_datalayout.py (default layout)\n'
      printf '  --file: exact custom .py file; overrides --impl and FLEX_FILE env (if set)\n'
      printf '  Selection precedence: --file > explicit --impl > FLEX_FILE env > default layout\n'
      printf '  --max-fuse N: last K, 0 means stop on OOM (default 0)\n'
      printf '  1-1: masks [1], [1 1], [1 1 1], ...\n'
      printf '  1-0: masks [1 0], [1 1 0 0], [1 1 1 0 0 0], ...\n'
      printf '  both: test both families (default)\n'
      printf '  --name: optional output directory label (alias: --log-name)\n'
      printf '  Output: testlog_scalability_IMPL[_LABEL]_YYYYMMDD_HHMMSS/\n'
      exit 0 ;;
    *)
      printf '[ERROR] Unrecognized option: %s\n' "$1" >&2
      exit 2 ;;
  esac
done

# Prevent paths or shell-special characters from entering the directory name.
# Empty is the default, giving a clean name without a doubled underscore.
if [[ -n "$LOG_NAME" && ! "$LOG_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo '[ERROR] --name must contain only letters, numbers, dots, underscores, or hyphens and start with a letter/number' >&2
  exit 2
fi

case "$MASK_FAMILY" in
  1-1) FAMILIES=(all_ones) ;;
  1-0) FAMILIES=(ones_zeros) ;;
  both) FAMILIES=(all_ones ones_zeros) ;;
  *) echo '[ERROR] MASK_FAMILY / --family must be 1-1, 1-0, or both' >&2; exit 2 ;;
esac

# Check configuration values, NOT the Python program's output.
case "$BARRIER_MODE" in
  both|on|off) ;;
  *) echo "[ERROR] BARRIER_MODE must be both, on, or off" >&2; exit 2 ;;
esac
if [[ ! "$START_K" =~ ^[1-9][0-9]*$ ]]; then
  echo '[ERROR] START_K / --start-k must be a positive integer' >&2; exit 2
fi
case "$FLEX_IMPL" in
  conv|layout) ;;
  *) echo '[ERROR] --impl must be conv or layout' >&2; exit 2 ;;
esac
if [[ ! "$MAX_FUSE" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo '[ERROR] MAX_FUSE must be a nonnegative integer' >&2; exit 2
fi
for setting in LOAD_ALL SPLIT_FWD_BARRIER; do
  value="${!setting}"
  if [[ "$value" != 0 && "$value" != 1 ]]; then
    echo "[ERROR] $setting must be 0 or 1" >&2; exit 2
  fi
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# Selection precedence: --file > explicit --impl > FLEX_FILE env > default layout.
# This is the EXACT Python program used for every run and printed into CMD.
if [[ -n "$CUSTOM_FLEX_FILE" ]]; then
  SELECTED_FILE="$CUSTOM_FLEX_FILE"
  FILE_SOURCE="--file"
elif (( IMPL_EXPLICIT == 1 )); then
  if [[ "$FLEX_IMPL" == conv ]]; then
    SELECTED_FILE="${SCRIPT_DIR}/distill_flexFuse_timeTest_conv.py"
  else
    SELECTED_FILE="${SCRIPT_DIR}/distill_flexFuse_timeTest_conv_datalayout.py"
  fi
  FILE_SOURCE="explicit --impl"
elif [[ -n "${FLEX_FILE:-}" ]]; then
  SELECTED_FILE="$FLEX_FILE"
  FILE_SOURCE="FLEX_FILE environment"
else
  if [[ "$FLEX_IMPL" == conv ]]; then
    SELECTED_FILE="${SCRIPT_DIR}/distill_flexFuse_timeTest_conv.py"
  else
    SELECTED_FILE="${SCRIPT_DIR}/distill_flexFuse_timeTest_conv_datalayout.py"
  fi
  FILE_SOURCE="default --impl"
fi
# Resolve relative filenames relative to the script directory, never the caller cwd.
if [[ "$SELECTED_FILE" != /* ]]; then
  SELECTED_FILE="${SCRIPT_DIR}/${SELECTED_FILE}"
fi
if [[ ! -f "$SELECTED_FILE" ]]; then
  echo "[ERROR] Python file not found: $SELECTED_FILE" >&2
  exit 2
fi
FLEX_FILE="$(cd -- "$(dirname -- "$SELECTED_FILE")" && pwd -P)/$(basename -- "$SELECTED_FILE")"
# Reflect actual selected Python filename even if --impl and external FLEX_FILE conflict.
case "$(basename -- "$FLEX_FILE")" in
  distill_flexFuse_timeTest_conv.py) ACTUAL_IMPL="conv" ;;
  distill_flexFuse_timeTest_conv_datalayout.py) ACTUAL_IMPL="layout" ;;
  *) ACTUAL_IMPL="custom" ;;
esac
if [[ "$ACTUAL_IMPL" != custom && "$ACTUAL_IMPL" != "$FLEX_IMPL" && "$FILE_SOURCE" != "explicit --impl" ]]; then
  echo "[INFO] Python file override ($FILE_SOURCE): actual implementation=$ACTUAL_IMPL, --impl/default=$FLEX_IMPL" >&2
fi
if command -v sha256sum >/dev/null 2>&1; then
  FLEX_SHA256="$(sha256sum -- "$FLEX_FILE" | awk '{print $1}')"
else
  FLEX_SHA256="<sha256sum unavailable>"
fi
if (( MAX_FUSE > 0 )) && (( START_K > MAX_FUSE )); then
  echo "[ERROR] --start-k=$START_K exceeds --max-fuse=$MAX_FUSE" >&2
  exit 2
fi
# One timestamp for the output folder and its files; LOG_DIR can still override.
RUN_TIME="$(date '+%Y%m%d_%H%M%S')"
LOG_SUFFIX="${LOG_NAME:+_${LOG_NAME}}"
LOG_DIR="${LOG_DIR:-testlog_scalability_${ACTUAL_IMPL}${LOG_SUFFIX}_${RUN_TIME}}"
BUFFER_PATH="${BUFFER_PATH:-/scratch/yguo25/files/mtt-distillation/buffer}"
DATA_PATH="${DATA_PATH:-/scratch/yguo25/files/mtt-distillation/dataset}"
mkdir -p -- "$LOG_DIR"
RUN_TAG="${RUN_TIME}_$$"
LOG_FILE="${LOG_DIR}/conv_fuse_until_oom_${RUN_TAG}.txt"
CSV_FILE="${LOG_DIR}/conv_fuse_until_oom_${RUN_TAG}.csv"

COMMON_ARGS=(
  "--dataset=${DATASET}"
  "--pix_init=${PIX_INIT}"
  "--ipc=${IPC}"
  "--syn_steps=${SYN_STEPS}"
  "--max_experts=${MAX_EXPERTS}"
  "--expert_epochs=${EXPERT_EPOCHS}"
  "--max_start_epoch=${MAX_START_EPOCH}"
  "--Iteration=${ITERATION}"
  "--detachNum=${DETACH_NUM}"
  "--lr_img=${LR_IMG}"
  "--lr_lr=${LR_LR}"
  "--lr_teacher=${LR_TEACHER}"
  "--buffer_path=${BUFFER_PATH}"
  "--data_path=${DATA_PATH}"
  "--model=${MODEL}"
  "--Fuse=${FUSE}"
)

if [[ "$BARRIER_MODE" == both ]]; then
  MODES=(use_barrier no_barrier)
elif [[ "$BARRIER_MODE" == on ]]; then
  MODES=(use_barrier)
else
  MODES=(no_barrier)
fi

{
  printf '============================================================\n'
  printf 'CONV FLEX MASK UNTIL OOM SWEEP START\n'
  printf 'START_TIME: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  printf 'DATASET: %s\nMODEL: %s\nIPC: %s\nSYN_STEPS: %s\nITERATION: %s\n' \
    "$DATASET" "$MODEL" "$IPC" "$SYN_STEPS" "$ITERATION"
  printf 'FLEX_FILE: %s\n' "$FLEX_FILE"
  printf 'FILE_SOURCE: %s\nACTUAL_IMPL: %s\nFLEX_SHA256: %s\n' "$FILE_SOURCE" "$ACTUAL_IMPL" "$FLEX_SHA256"
  printf 'START_K: %s\n' "$START_K"
  printf 'MAX_FUSE: %s (0=unlimited)\n' "$MAX_FUSE"
  printf 'MASK_FAMILY: %s\n' "$MASK_FAMILY"
  printf 'LOG_NAME: %s\n' "${LOG_NAME:-<none>}"
  printf 'BARRIER_MODE: %s\n' "$BARRIER_MODE"
  printf 'LOAD_ALL: %s\n' "$LOAD_ALL"
  printf 'SPLIT_FWD_BARRIER: %s\n' "$SPLIT_FWD_BARRIER"
  printf 'MODE_ORDER: %s\n' "${MODES[*]}"
  printf 'V_FUSE: True\nACC_TEST: omitted\n'
  printf 'FAMILY_ORDER: %s\n' "${FAMILIES[*]}"
  printf 'CSV_POLICY: optional best-effort extraction; missing fields never fail a run\n'
  printf 'LOG_FILE: %s\nCSV_FILE: %s\n' "$LOG_FILE" "$CSV_FILE"
  printf '============================================================\n\n'
} > "$LOG_FILE"
printf 'timestamp,family,num_ones,mask,barrier,status,exit_code,system_s,kernel_s,total_s,peak_cache_mb,peak_tensor_mb,prepare_s,syn_s,backward_s,kernel_no_barrier_s,load_all,split_fwd_barrier\n' > "$CSV_FILE"

echo "[INFO] LOG_FILE=$LOG_FILE"
echo "[INFO] CSV_FILE=$CSV_FILE"
echo "[INFO] ACTUAL_IMPL=$ACTUAL_IMPL FILE_SOURCE=$FILE_SOURCE FLEX_FILE=$FLEX_FILE"
echo "[INFO] START_K=$START_K FLEX_SHA256=$FLEX_SHA256"
echo "[INFO] BARRIER_MODE=$BARRIER_MODE LOAD_ALL=$LOAD_ALL SPLIT_FWD_BARRIER=$SPLIT_FWD_BARRIER"
echo "[INFO] IPC=$IPC ITERATION=$ITERATION MAX_FUSE=$MAX_FUSE MASK_FAMILY=$MASK_FAMILY"

RUN_STATUS=''
CURRENT_TMP=''
cleanup() {
  if [[ -n "$CURRENT_TMP" && -f "$CURRENT_TMP" ]]; then
    rm -f -- "$CURRENT_TMP"
  fi
}
trap cleanup EXIT

# Optional CSV extraction. It is NEVER checked for success, and the TXT log
# keeps the raw output even if Python changes any print labels or formatting.
# For repeated values, take the last matching line as the original script did.
last_field() {
  local field="$1" file="$2"
  awk -v target="$field" '
    {
      pos=index($0, ":")
      if (!pos) next
      label=tolower(substr($0, 1, pos-1))
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", label)
      ok=0
      if      (target == "system")    ok=(label ~ /^system time/)
      else if (target == "kernel")    ok=(label ~ /^kernel time/ && label !~ /without barrier/)
      else if (target == "total")     ok=(label ~ /^total time/)
      else if (target == "prepare")   ok=(label ~ /^prepare time/)
      else if (target == "syn")       ok=(label ~ /^syn_time/)
      else if (target == "backward")  ok=(label ~ /^backward_time/)
      else if (target == "kernel_nb") ok=(label ~ /^kernel time/ && label ~ /without barrier/)
      else if (target == "cache")     ok=(label ~ /^峰值cache使用/)
      else if (target == "tensor")    ok=(label ~ /^峰值tensor使用/)
      if (!ok) next
      raw=substr($0, pos+1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", raw)
      split(raw, pieces, /[[:space:]]+/)
      if (pieces[1] ~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/)
        last=pieces[1]
    }
    END { if (last != "") printf "%s", last }
  ' "$file"
}

run_mode() {
  local family="$1" count="$2" mask_tag="$3" barrier_mode="$4"
  shift 4
  local mask=( "$@" )
  local extra=()
  if [[ "$barrier_mode" == use_barrier ]]; then extra+=(--use-barrier); fi
  if [[ "$LOAD_ALL" == 1 ]]; then extra+=(--load_all); fi
  if [[ "$SPLIT_FWD_BARRIER" == 1 ]]; then extra+=(--split-fwd-barrier); fi

  local command=(
    python "$FLEX_FILE"
    --fuse_mask_list "${mask[@]}"
    --v_fuse
    "${COMMON_ARGS[@]}"
    "${extra[@]}"
  )
  local timestamp rc status system kernel total cache tensor prepare syn backward kernel_nb
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[RUN] FAMILY=$family K=$count MASK=$mask_tag BARRIER=$barrier_mode LOAD_ALL=$LOAD_ALL SPLIT=$SPLIT_FWD_BARRIER"
  {
    printf '==============================\n'
    printf '%s | MODEL=%s | IPC=%s | SYN_STEPS=%s | ITERATION=%s | FAMILY=%s | K=%s | MASK=%s | BARRIER=%s | LOAD_ALL=%s | SPLIT_FWD_BARRIER=%s\n' \
      "$timestamp" "$MODEL" "$IPC" "$SYN_STEPS" "$ITERATION" "$family" "$count" "$mask_tag" "$barrier_mode" "$LOAD_ALL" "$SPLIT_FWD_BARRIER"
    printf 'RUN_FILE: %s | ACTUAL_IMPL=%s | SHA256=%s\n' "$FLEX_FILE" "$ACTUAL_IMPL" "$FLEX_SHA256"
    printf 'CMD: WANDB_SILENT=true PYTHONWARNINGS=ignore TQDM_DISABLE=1 '
    printf '%q ' "${command[@]}"
    printf '\n------------------------------\n'
  } >> "$LOG_FILE"

  CURRENT_TMP="$(mktemp "${LOG_DIR}/.conv_fuse_mask_run.XXXXXXXX")"
  WANDB_SILENT=true PYTHONWARNINGS=ignore TQDM_DISABLE=1 PYTHONUNBUFFERED=1 \
    "${command[@]}" 2>&1 | tee "$CURRENT_TMP" >> "$LOG_FILE"
  local pipeline_status=( "${PIPESTATUS[@]}" )
  rc="${pipeline_status[0]}"
  if (( pipeline_status[1] != 0 )); then
    echo "[ERROR] tee could not write temporary output (exit=${pipeline_status[1]})" >&2
    stop_fatal "Logging failed for $family K=$count"
  fi

  # Only the Python process outcome matters. No field is required.
  if (( rc == 0 )); then
    status='SUCCESS'
  elif grep -Eqi \
    'CUDA out of memory|OutOfMemoryError|cudaErrorMemoryAllocation|CUDA error: out of memory|CUDNN_STATUS_ALLOC_FAILED|CUBLAS_STATUS_ALLOC_FAILED|HIP out of memory' \
    "$CURRENT_TMP"; then
    status='OOM'
  else
    status='ERROR'
  fi

  # All of these may be empty. A missing print is never an ERROR.
  system="$(last_field system "$CURRENT_TMP")"
  kernel="$(last_field kernel "$CURRENT_TMP")"
  total="$(last_field total "$CURRENT_TMP")"
  cache="$(last_field cache "$CURRENT_TMP")"
  tensor="$(last_field tensor "$CURRENT_TMP")"
  prepare="$(last_field prepare "$CURRENT_TMP")"
  syn="$(last_field syn "$CURRENT_TMP")"
  backward="$(last_field backward "$CURRENT_TMP")"
  kernel_nb="$(last_field kernel_nb "$CURRENT_TMP")"

  {
    printf '%s\n' '------------------------------'
    printf 'EXIT_CODE: %s\n' "$rc"
    printf 'RUN_STATUS: %s\n' "$status"
    printf 'RESULT: system_s=%s kernel_s=%s total_s=%s peak_cache_mb=%s peak_tensor_mb=%s prepare_s=%s syn_s=%s backward_s=%s kernel_no_barrier_s=%s\n\n' \
      "$system" "$kernel" "$total" "$cache" "$tensor" "$prepare" "$syn" "$backward" "$kernel_nb"
  } >> "$LOG_FILE"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$timestamp" "$family" "$count" "$mask_tag" "$barrier_mode" "$status" "$rc" \
    "$system" "$kernel" "$total" "$cache" "$tensor" "$prepare" "$syn" "$backward" "$kernel_nb" "$LOAD_ALL" "$SPLIT_FWD_BARRIER" >> "$CSV_FILE"
  echo "[$status] MASK=$mask_tag BARRIER=$barrier_mode EXIT_CODE=$rc system=${system:-NA} kernel=${kernel:-NA} total=${total:-NA}"

  cleanup
  CURRENT_TMP=''
  RUN_STATUS="$status"
}

make_mask() {
  local family="$1" k="$2" i
  MASK=()
  MASK_TAG=''
  for ((i=0; i<k; i++)); do MASK+=(1); MASK_TAG+='1'; done
  if [[ "$family" == ones_zeros ]]; then
    for ((i=0; i<k; i++)); do MASK+=(0); MASK_TAG+='0'; done
  fi
}

stop_fatal() {
  local msg="$1"
  printf '[ERROR] %s\n' "$msg" >&2
  printf 'FATAL: %s\n' "$msg" >> "$LOG_FILE"
  printf 'SWEEP_END_STATUS: ERROR\nEND_TIME: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
  exit 1
}

for family in "${FAMILIES[@]}"; do
  echo "[INFO] Start family: $family"
  printf 'FAMILY_START: %s\n' "$family" >> "$LOG_FILE"
  k="$START_K"
  while :; do
    if (( MAX_FUSE > 0 && k > MAX_FUSE )); then
      echo "[INFO] $family: reached --max-fuse=$MAX_FUSE (not necessarily OOM)"
      printf 'FAMILY_END: %s | REASON=MAX_FUSE | K=%s\n' "$family" "$MAX_FUSE" >> "$LOG_FILE"
      break
    fi

    make_mask "$family" "$k"
    for mode in "${MODES[@]}"; do
      run_mode "$family" "$k" "$MASK_TAG" "$mode" "${MASK[@]}"
      case "$RUN_STATUS" in
        OOM)
          echo "[INFO] $family: stopped on OOM for MASK=$MASK_TAG in $mode"
          printf 'FAMILY_END: %s | REASON=OOM | MASK=%s | BARRIER=%s\n' "$family" "$MASK_TAG" "$mode" >> "$LOG_FILE"
          break 2 ;;
        SUCCESS) ;;
        *) stop_fatal "Python failed at $family K=$k $mode (exit != 0); see $LOG_FILE" ;;
      esac
    done

    (( k += 1 ))
  done
  echo "[INFO] Finished family: $family"
done

{
  printf '============================================================\n'
  printf 'CONV FLEX MASK UNTIL OOM SWEEP END\n'
  printf 'SWEEP_END_STATUS: COMPLETE\n'
  printf 'END_TIME: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  printf '============================================================\n'
} >> "$LOG_FILE"
echo "[DONE] $LOG_FILE"
echo "[DONE] $CSV_FILE"
