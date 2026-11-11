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
# Edit the following values in THIS script; no command-line options needed.
BARRIER_MODE="on"         # "both": run with and without --use-barrier; "on": only with; "off": only without
LOAD_ALL=0                  # 1 adds --load_all; 0 omits it
SPLIT_FWD_BARRIER=0         # 1 adds --split-fwd-barrier; 0 omits it
MAX_FUSE=0                  # 0: continue until CUDA OOM; N > 0: stop at K=N

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

# Check configuration values, NOT the Python program's output.
case "$BARRIER_MODE" in
  both|on|off) ;;
  *) echo "[ERROR] BARRIER_MODE must be both, on, or off" >&2; exit 2 ;;
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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FLEX_FILE="${FLEX_FILE:-${SCRIPT_DIR}/distill_flexFuse_timeTest_conv_preload.py}"
LOG_DIR="${LOG_DIR:-testlog$(date '+%Y%m%d')}"
BUFFER_PATH="${BUFFER_PATH:-/scratch/yguo25/files/mtt-distillation/buffer}"
DATA_PATH="${DATA_PATH:-/scratch/yguo25/files/mtt-distillation/dataset}"
mkdir -p -- "$LOG_DIR"
RUN_TAG="$(date '+%Y%m%d_%H%M%S')_$$"
LOG_FILE="${LOG_DIR}/conv_fuse_until_oom_${RUN_TAG}.txt"
CSV_FILE="${LOG_DIR}/conv_fuse_until_oom_${RUN_TAG}.csv"

if [[ ! -f "$FLEX_FILE" ]]; then
  echo "[ERROR] Python file not found: $FLEX_FILE" >&2
  echo '        Put the Python file next to this script or set FLEX_FILE=/path/to/file.py' >&2
  exit 2
fi

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
  printf 'MAX_FUSE: %s (0=unlimited)\n' "$MAX_FUSE"
  printf 'BARRIER_MODE: %s\n' "$BARRIER_MODE"
  printf 'LOAD_ALL: %s\n' "$LOAD_ALL"
  printf 'SPLIT_FWD_BARRIER: %s\n' "$SPLIT_FWD_BARRIER"
  printf 'MODE_ORDER: %s\n' "${MODES[*]}"
  printf 'V_FUSE: True\nACC_TEST: omitted\nFAMILY_ORDER: all_ones then ones_zeros\n'
  printf 'CSV_POLICY: optional best-effort extraction; missing fields never fail a run\n'
  printf 'LOG_FILE: %s\nCSV_FILE: %s\n' "$LOG_FILE" "$CSV_FILE"
  printf '============================================================\n\n'
} > "$LOG_FILE"
printf 'timestamp,family,num_ones,mask,barrier,status,exit_code,system_s,kernel_s,total_s,peak_cache_mb,peak_tensor_mb,prepare_s,syn_s,backward_s,kernel_no_barrier_s,load_all,split_fwd_barrier\n' > "$CSV_FILE"

echo "[INFO] LOG_FILE=$LOG_FILE"
echo "[INFO] CSV_FILE=$CSV_FILE"
echo "[INFO] FLEX_FILE=$FLEX_FILE"
echo "[INFO] BARRIER_MODE=$BARRIER_MODE LOAD_ALL=$LOAD_ALL SPLIT_FWD_BARRIER=$SPLIT_FWD_BARRIER"
echo "[INFO] IPC=$IPC ITERATION=$ITERATION MAX_FUSE=$MAX_FUSE"

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
    printf 'CMD: WANDB_SILENT=true PYTHONWARNINGS=ignore TQDM_DISABLE=1 '
    printf '%q ' "${command[@]}"
    printf '\n------------------------------\n'
  } >> "$LOG_FILE"

  CURRENT_TMP="$(mktemp "${LOG_DIR}/.conv_fuse_mask_run.XXXXXXXX")"
  WANDB_SILENT=true PYTHONWARNINGS=ignore TQDM_DISABLE=1 PYTHONUNBUFFERED=1 \
    "${command[@]}" 2>&1 | tee "$CURRENT_TMP" >> "$LOG_FILE"
  rc="${PIPESTATUS[0]}"

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

for family in all_ones ones_zeros; do
  echo "[INFO] Start family: $family"
  printf 'FAMILY_START: %s\n' "$family" >> "$LOG_FILE"
  k=1
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
