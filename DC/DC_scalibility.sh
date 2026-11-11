#!/usr/bin/env bash
# DC / ConvNet / CIFAR10 scalability sweep for main_optimalFuse.py.
# Fuse = 2, 4, 6, 8, ... (ONLY positive even values).
# Stop when CUDA OOM is reported, or when MAX_FUSE is reached.
# Only Python's exit status determines success. Timing fields are optional.
# Keep full stdout/stderr in TXT; extract best-effort metrics to CSV.
# A non-OOM Python failure stops the sweep so errors aren't mistaken for OOM.

set -u
set -o pipefail

# ====================== MANUAL SETTINGS ======================
# Edit settings here, then simply run: bash run_optimalFuse_DC_scalability.sh
BARRIER_MODE="on"          # "on" / "off" / "both" (the given command uses --use-barrier)
MAX_FUSE=0                 # 0 = keep increasing until CUDA OOM; otherwise an EVEN maximum
START_FUSE=2               # positive EVEN number; fixed increment of +2
V_FUSE=1                   # 1 = --v_fuse, 0 = no --v_fuse

# Compatibility toggles from the old sweep. Keep 0 unless main_optimalFuse.py
# actually implements the corresponding command-line option.
LOAD_ALL=0                 # 1 = add --load_all; 0 = omit
SPLIT_FWD_BARRIER=0        # 1 = add --split-fwd-barrier; 0 = omit

# Exactly the non-Fuse settings supplied in the DC command:
NUM_EXP=1
METHOD=DC
DATASET=CIFAR10
MODEL=ConvNet
IPC=10
ITERATION=50
LR_IMG=0.1
LR_NET=0.01
BATCH_REAL=256
BATCH_TRAIN=256
INIT=real
DATA_PATH="/scratch/yguo25/files/mtt-distillation/dataset"
# --AccTest is NOT passed.
# =============================================================

# Check script configuration, never assert on Python's printed metrics.
case "$BARRIER_MODE" in
  on|off|both) ;;
  *) echo '[ERROR] BARRIER_MODE must be on/off/both' >&2; exit 2 ;;
esac
for key in V_FUSE LOAD_ALL SPLIT_FWD_BARRIER; do
  value="${!key}"
  if [[ "$value" != 0 && "$value" != 1 ]]; then
    echo "[ERROR] $key must be 0 or 1" >&2
    exit 2
  fi
done
if [[ ! "$START_FUSE" =~ ^[1-9][0-9]*$ ]] || (( START_FUSE % 2 != 0 )); then
  echo '[ERROR] START_FUSE must be a positive even integer' >&2
  exit 2
fi
if [[ ! "$MAX_FUSE" =~ ^(0|[1-9][0-9]*)$ ]] || (( MAX_FUSE > 0 && MAX_FUSE % 2 != 0 )); then
  echo '[ERROR] MAX_FUSE must be 0 or a positive even integer' >&2
  exit 2
fi
if (( MAX_FUSE > 0 && MAX_FUSE < START_FUSE )); then
  echo '[ERROR] MAX_FUSE must be >= START_FUSE (or 0 for unlimited)' >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MAIN_FILE="${MAIN_FILE:-${SCRIPT_DIR}/main_optimalFuse.py}"
LOG_DIR="${LOG_DIR:-testlog$(date '+%Y%m%d')}"
mkdir -p -- "$LOG_DIR"
RUN_TAG="$(date '+%Y%m%d_%H%M%S')_$$"
LOG_FILE="${LOG_DIR}/dc_optimalFuse_scalability_${RUN_TAG}.txt"
CSV_FILE="${LOG_DIR}/dc_optimalFuse_scalability_${RUN_TAG}.csv"

if [[ ! -f "$MAIN_FILE" ]]; then
  echo "[ERROR] Python file not found: $MAIN_FILE" >&2
  echo '[HINT] Put this bash script next to main_optimalFuse.py, or set MAIN_FILE=/path/to/main_optimalFuse.py' >&2
  exit 2
fi

COMMON_ARGS=(
  "--num_exp=${NUM_EXP}"
  "--method=${METHOD}"
  "--dataset=${DATASET}"
  "--model=${MODEL}"
  "--ipc=${IPC}"
  "--Iteration=${ITERATION}"
  "--lr_img=${LR_IMG}"
  "--lr_net=${LR_NET}"
  "--batch_real=${BATCH_REAL}"
  "--batch_train=${BATCH_TRAIN}"
  "--init=${INIT}"
  "--data_path=${DATA_PATH}"
)

case "$BARRIER_MODE" in
  both) MODES=(use_barrier no_barrier) ;;
  on)   MODES=(use_barrier) ;;
  off)  MODES=(no_barrier) ;;
esac

{
  printf '============================================================\n'
  printf 'DC OPTIMAL FUSE SCALABILITY SWEEP START\n'
  printf 'START_TIME: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  printf 'MAIN_FILE: %s\n' "$MAIN_FILE"
  printf 'METHOD: %s\nDATASET: %s\nMODEL: %s\nNUM_EXP: %s\nIPC: %s\nITERATION: %s\n' \
    "$METHOD" "$DATASET" "$MODEL" "$NUM_EXP" "$IPC" "$ITERATION"
  printf 'LR_IMG: %s\nLR_NET: %s\nBATCH_REAL: %s\nBATCH_TRAIN: %s\nINIT: %s\nDATA_PATH: %s\n' \
    "$LR_IMG" "$LR_NET" "$BATCH_REAL" "$BATCH_TRAIN" "$INIT" "$DATA_PATH"
  printf 'START_FUSE: %s\nFUSE_STEP: 2\nMAX_FUSE: %s (0=until OOM)\n' "$START_FUSE" "$MAX_FUSE"
  printf 'BARRIER_MODE: %s\nMODE_ORDER: %s\nV_FUSE: %s\nLOAD_ALL: %s\nSPLIT_FWD_BARRIER: %s\n' \
    "$BARRIER_MODE" "${MODES[*]}" "$V_FUSE" "$LOAD_ALL" "$SPLIT_FWD_BARRIER"
  printf 'ACC_TEST: omitted\n'
  printf 'CSV_POLICY: timing fields optional; missing output never fails a successful process\n'
  printf 'LOG_FILE: %s\nCSV_FILE: %s\n' "$LOG_FILE" "$CSV_FILE"
  printf '============================================================\n\n'
} > "$LOG_FILE"

printf 'timestamp,fuse,barrier,status,exit_code,system_s,kernel_s,total_s,prepare_s,syn_s,backward_s,kernel_no_barrier_s,peak_cache_mb,peak_tensor_mb,v_fuse,load_all,split_fwd_barrier\n' > "$CSV_FILE"

echo "[INFO] MAIN_FILE=$MAIN_FILE"
echo "[INFO] LOG_FILE=$LOG_FILE"
echo "[INFO] CSV_FILE=$CSV_FILE"
echo "[INFO] START_FUSE=$START_FUSE STEP=2 MAX_FUSE=$MAX_FUSE BARRIER_MODE=$BARRIER_MODE"

CURRENT_TMP=''
cleanup() {
  if [[ -n "$CURRENT_TMP" && -f "$CURRENT_TMP" ]]; then
    rm -f -- "$CURRENT_TMP"
  fi
}
trap cleanup EXIT

# Optional metric extraction: last matching numeric field in this invocation.
# Unrecognized labels are ignored, not considered a test failure.
last_field() {
  local field="$1" file="$2"
  awk -v target="$field" '
    {
      pos=index($0, ":")
      if (!pos) next
      label=tolower(substr($0,1,pos-1))
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", label)
      ok=0
      if      (target=="system")    ok=(label ~ /^system time/)
      else if (target=="kernel")    ok=(label ~ /^kernel time/ && label !~ /without barrier/)
      else if (target=="total")     ok=(label ~ /^total time/)
      else if (target=="prepare")   ok=(label ~ /^prepare time/)
      else if (target=="syn")       ok=(label ~ /^syn_time/)
      else if (target=="backward")  ok=(label ~ /^backward_time/)
      else if (target=="kernel_nb") ok=(label ~ /^kernel time/ && label ~ /without barrier/)
      else if (target=="cache")     ok=(label ~ /^峰值cache使用/ || label ~ /^peak cache/)
      else if (target=="tensor")    ok=(label ~ /^峰值tensor使用/ || label ~ /^peak tensor/)
      if (!ok) next
      raw=substr($0,pos+1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", raw)
      split(raw,pieces,/[[:space:]]+/)
      if (pieces[1] ~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/)
        last=pieces[1]
    }
    END {if(last != "") printf "%s", last}
  ' "$file"
}

RUN_STATUS=''
run_mode() {
  local fuse="$1" barrier_mode="$2"
  local extra=()
  if [[ "$V_FUSE" == 1 ]]; then extra+=(--v_fuse); fi
  if [[ "$barrier_mode" == use_barrier ]]; then extra+=(--use-barrier); fi
  if [[ "$LOAD_ALL" == 1 ]]; then extra+=(--load_all); fi
  if [[ "$SPLIT_FWD_BARRIER" == 1 ]]; then extra+=(--split-fwd-barrier); fi
  local command=(python "$MAIN_FILE" "${COMMON_ARGS[@]}" "--Fuse=${fuse}" "${extra[@]}")
  local timestamp rc status system kernel total prepare syn backward kernel_nb cache tensor
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

  echo "[RUN] FUSE=$fuse BARRIER=$barrier_mode V_FUSE=$V_FUSE LOAD_ALL=$LOAD_ALL SPLIT=$SPLIT_FWD_BARRIER"
  {
    printf '==============================\n'
    printf '%s | METHOD=%s | MODEL=%s | DATASET=%s | IPC=%s | ITERATION=%s | FUSE=%s | BARRIER=%s | V_FUSE=%s | LOAD_ALL=%s | SPLIT_FWD_BARRIER=%s\n' \
      "$timestamp" "$METHOD" "$MODEL" "$DATASET" "$IPC" "$ITERATION" "$fuse" "$barrier_mode" "$V_FUSE" "$LOAD_ALL" "$SPLIT_FWD_BARRIER"
    printf 'CMD: WANDB_SILENT=true PYTHONWARNINGS=ignore TQDM_DISABLE=1 '
    printf '%q ' "${command[@]}"
    printf '\n------------------------------\n'
  } >> "$LOG_FILE"

  CURRENT_TMP="$(mktemp "${LOG_DIR}/.dc_optimalFuse_run.XXXXXXXX")"
  WANDB_SILENT=true PYTHONWARNINGS=ignore TQDM_DISABLE=1 PYTHONUNBUFFERED=1 \
    "${command[@]}" 2>&1 | tee "$CURRENT_TMP" >> "$LOG_FILE"
  rc="${PIPESTATUS[0]}"

  if (( rc == 0 )); then
    status=SUCCESS
  elif grep -Eqi \
    'CUDA out of memory|OutOfMemoryError|cudaErrorMemoryAllocation|CUDA error: out of memory|CUDNN_STATUS_ALLOC_FAILED|CUBLAS_STATUS_ALLOC_FAILED|HIP out of memory' \
    "$CURRENT_TMP"; then
    status=OOM
  else
    status=ERROR
  fi

  # All metrics are optional. Never judge success by whether they printed.
  system="$(last_field system "$CURRENT_TMP")"
  kernel="$(last_field kernel "$CURRENT_TMP")"
  total="$(last_field total "$CURRENT_TMP")"
  prepare="$(last_field prepare "$CURRENT_TMP")"
  syn="$(last_field syn "$CURRENT_TMP")"
  backward="$(last_field backward "$CURRENT_TMP")"
  kernel_nb="$(last_field kernel_nb "$CURRENT_TMP")"
  cache="$(last_field cache "$CURRENT_TMP")"
  tensor="$(last_field tensor "$CURRENT_TMP")"

  {
    printf '%s\n' '------------------------------'
    printf 'EXIT_CODE: %s\nRUN_STATUS: %s\n' "$rc" "$status"
    printf 'RESULT: system_s=%s kernel_s=%s total_s=%s prepare_s=%s syn_s=%s backward_s=%s kernel_no_barrier_s=%s peak_cache_mb=%s peak_tensor_mb=%s\n\n' \
      "$system" "$kernel" "$total" "$prepare" "$syn" "$backward" "$kernel_nb" "$cache" "$tensor"
  } >> "$LOG_FILE"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$timestamp" "$fuse" "$barrier_mode" "$status" "$rc" \
    "$system" "$kernel" "$total" "$prepare" "$syn" "$backward" "$kernel_nb" "$cache" "$tensor" \
    "$V_FUSE" "$LOAD_ALL" "$SPLIT_FWD_BARRIER" >> "$CSV_FILE"
  echo "[$status] FUSE=$fuse BARRIER=$barrier_mode EXIT_CODE=$rc kernel=${kernel:-NA} total=${total:-NA} cache=${cache:-NA}"

  cleanup
  CURRENT_TMP=''
  RUN_STATUS="$status"
}

finish_sweep() {
  local reason="$1" fuse="$2" mode="$3"
  {
    printf 'SWEEP_END_REASON: %s\nEND_FUSE: %s\nEND_MODE: %s\n' "$reason" "$fuse" "$mode"
    printf 'SWEEP_END_STATUS: COMPLETE\nEND_TIME: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    printf '============================================================\n'
  } >> "$LOG_FILE"
  echo "[DONE] REASON=$reason LAST_FUSE=$fuse LOG_FILE=$LOG_FILE"
  echo "[DONE] CSV_FILE=$CSV_FILE"
}

fuse="$START_FUSE"
while :; do
  if (( MAX_FUSE > 0 && fuse > MAX_FUSE )); then
    finish_sweep MAX_FUSE "$(( fuse - 2 ))" NA
    exit 0
  fi

  for mode in "${MODES[@]}"; do
    run_mode "$fuse" "$mode"
    case "$RUN_STATUS" in
      SUCCESS) ;;
      OOM)
        finish_sweep OOM "$fuse" "$mode"
        exit 0
        ;;
      *)
        printf 'FATAL: Non-OOM Python error at FUSE=%s MODE=%s\nSWEEP_END_STATUS: ERROR\n' "$fuse" "$mode" >> "$LOG_FILE"
        echo "[ERROR] Non-OOM Python failure at FUSE=$fuse MODE=$mode; see $LOG_FILE" >&2
        exit 1
        ;;
    esac
  done
  (( fuse += 2 ))
done
