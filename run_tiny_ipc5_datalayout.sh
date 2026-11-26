#!/usr/bin/env bash
# Tiny ImageNet / IPC=5 / ConvNet: Original + DataLayout Flex + FWD
# Same Python argument list as the original v17 sweep; only the Flex Python filename differs.
# All runs: --use-barrier, no --load_all, syn_steps=1, two repeats.
set -euo pipefail

cd "$(dirname "$0")"
DATA_PATH="${DATA_PATH:-/scratch/yguo25/files/mtt-distillation/dataset}"
BUFFER_PATH="${BUFFER_PATH:-/scratch/yguo25/files/mtt-distillation/buffer}"
ORIGINAL="${ORIGINAL:-distill_original_timeTest.py}"
FLEX="${FLEX:-distill_flexFuse_timeTest_conv_datalayout.py}"
FWD="${FWD:-distill_original_fwdtimetest.py}"
REPEATS="${REPEATS:-2}"
ITERATION="${ITERATION:-350}"
LOG_DIR="${LOG_DIR:-testlog_tiny_ipc5_datalayout_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

COMMON=(--dataset=Tiny --model=ConvNet --ipc=5 --syn_steps=1
  --pix_init=real --max_experts=1 --expert_epochs=1 --max_start_epoch=1
  --Iteration="$ITERATION" --detachNum=0 --lr_img=1000 --lr_lr=1e-05
  --lr_teacher=0.01 --Fuse=1 --data_path="$DATA_PATH" --buffer_path="$BUFFER_PATH"
  --use-barrier)

for file in "$ORIGINAL" "$FLEX" "$FWD"; do
  [[ -f "$file" ]] || { echo "[ERROR] Missing $file" >&2; exit 1; }
done

# Handles: Kernel time  ( 353 ): value; and FWD's syn_time + backward_time.
extract_kernel() {
  awk '
    function first_number(s, arr, n, i) {
      n=split(s, arr, /[[:space:]]+/)
      for (i=1;i<=n;i++) if (arr[i] ~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/) return arr[i]
      return ""
    }
    {
      low=tolower($0)
      pos=index($0, ":")
      if (!pos) next
      label=tolower(substr($0,1,pos-1))
      value=first_number(substr($0,pos+1))
      if (value=="") next
      if (label ~ /^[[:space:]]*kernel[[:space:]_]*time/ && label !~ /without[[:space:]]+barrier/) kernel=value
      if (label ~ /^[[:space:]]*syn_time/) syn=value
      if (label ~ /^[[:space:]]*backward_time/) bwd=value
    }
    END {
      if (kernel!="") print kernel
      else if (syn!="" && bwd!="") printf "%.9f\n",syn+bwd
    }' "$1"
}
extract_memory() {
  local field="$1" file="$2"
  awk -v field="$field" '
    {
      pos=index($0, ":")
      if (!pos) next
      label=tolower(substr($0,1,pos-1))
      if (field=="cache" && label !~ /峰值cache使用/) next
      if (field=="tensor" && label !~ /峰值tensor使用/) next
      value=substr($0,pos+1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      split(value,a,/[[:space:]]+/)
      if (a[1] ~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/) last=a[1]
    }
    END { if (last!="") print last }' "$file"
}

echo 'case,repeat,exit_code,kernel_s,peak_cache_mb,peak_tensor_mb' > "$LOG_DIR/results.csv"
run_case() {
  local name="$1" file="$2"; shift 2
  local repeat log rc kernel cache tensor
  for ((repeat=1; repeat<=REPEATS; repeat++)); do
    log="$LOG_DIR/${name}_r${repeat}.txt"
    echo "[RUN] $name repeat $repeat/$REPEATS"
    printf 'CMD: python %q ' "$file" > "$log"
    printf '%q ' "${COMMON[@]}" "$@" >> "$log"
    printf '\n' >> "$log"
    rc=0
    WANDB_SILENT=true PYTHONWARNINGS=ignore TQDM_DISABLE=1 \
      python "$file" "${COMMON[@]}" "$@" >> "$log" 2>&1 || rc=$?
    kernel="$(extract_kernel "$log")"
    cache="$(extract_memory cache "$log")"
    tensor="$(extract_memory tensor "$log")"
    printf '%s,%s,%s,%s,%s,%s\n' "$name" "$repeat" "$rc" "$kernel" "$cache" "$tensor" >> "$LOG_DIR/results.csv"
    echo "[RESULT] exit=$rc kernel=${kernel:-NA}s cache=${cache:-NA}MB tensor=${tensor:-NA}MB"
    if ((rc != 0)); then
      echo "[ERROR] $name failed; see $log (continuing, as in the v17 sweep)" >&2
    fi
  done
}

run_case original "$ORIGINAL"
for mask in '1' '1 1' '1 0'; do
  read -r -a mask_args <<< "$mask"
  tag="${mask// /}"
  run_case "layout_${tag}_vfuse" "$FLEX" --fuse_mask_list "${mask_args[@]}" --v_fuse
  run_case "layout_${tag}_no_vfuse" "$FLEX" --fuse_mask_list "${mask_args[@]}" --no-v_fuse
done
run_case original_fwd "$FWD"

# Full two-run means only; failures/missing measurements are left blank.
awk -F, 'BEGIN { OFS=","; print "case,success_count,mean_kernel_s,mean_peak_cache_mb,mean_peak_tensor_mb" }
  NR>1 {
    key=$1; if (!(key in seen)) { keys[++n]=key; seen[key]=1 }
    if ($3==0 && $4!="" && $5!="" && $6!="") {
      count[key]++; kernel[key]+=$4; cache[key]+=$5; tensor[key]+=$6
    }
  }
  END {
    for (i=1;i<=n;i++) {
      k=keys[i]
      if (count[k]==2) printf "%s,%d,%.9f,%.9f,%.9f\n",k,count[k],kernel[k]/2,cache[k]/2,tensor[k]/2
      else print k,count[k]+0,"","",""
    }
  }' "$LOG_DIR/results.csv" > "$LOG_DIR/means.csv"

echo "[DONE] Logs: $LOG_DIR"
echo "[DONE] CSV: $LOG_DIR/results.csv"
echo "[DONE] Means: $LOG_DIR/means.csv"
