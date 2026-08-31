#!/usr/bin/env bash
# Distill-only timing sweep for testing split/layout changes.
#
# Default behavior:
#   - ConvNet only
#   - CIFAR10
#   - IPC: 1 10 50 100
#   - syn_steps=1
#   - optional original distill baseline
#   - flex masks: 1 / 1 1 / 1 0 / 1 1 0 0
#   - --v_fuse always enabled
#   - each case: 2x --use-barrier + 2x without barrier
#   - NO fwdtest / NO ckpt / NO AccTest
#
# Typical usage:
#   bash rundistill_only_split_contig_test.sh
#
# Test a modified flex file without editing this script:
#   FLEX_FILE=distill_flexFuse_timeTest_conv_fake_contig.py \
#   CASE_NAME=fake_contig \
#   bash rundistill_only_split_contig_test.sh
#
# Skip original baseline and only test flex cases:
#   RUN_ORIGINAL=0 bash rundistill_only_split_contig_test.sh
#
# Only test split-related masks:
#   MASKS="1 0|1 1 0 0" bash rundistill_only_split_contig_test.sh

set -u
set -o pipefail

# ------------------------------------------------------------
# User-overridable settings
# ------------------------------------------------------------
MODEL="${MODEL:-ConvNet}"
DATASET="${DATASET:-CIFAR10}"
PIX_INIT="${PIX_INIT:-real}"

BUFFER_PATH="${BUFFER_PATH:-/scratch/yguo25/files/mtt-distillation/buffer}"
DATA_PATH="${DATA_PATH:-/scratch/yguo25/files/mtt-distillation/dataset}"

ITERATION="${ITERATION:-350}"
MAX_EXPERTS="${MAX_EXPERTS:-1}"
EXPERT_EPOCHS="${EXPERT_EPOCHS:-1}"
MAX_START_EPOCH="${MAX_START_EPOCH:-1}"
DETACH_NUM="${DETACH_NUM:-0}"

LR_IMG="${LR_IMG:-1000}"
LR_LR="${LR_LR:-1e-05}"
LR_TEACHER="${LR_TEACHER:-0.01}"

FUSE="${FUSE:-1}"
SYN_STEPS="${SYN_STEPS:-1}"

USE_BARRIER_REPEATS="${USE_BARRIER_REPEATS:-2}"
NO_BARRIER_REPEATS="${NO_BARRIER_REPEATS:-2}"

RUN_ORIGINAL="${RUN_ORIGINAL:-1}"

ORIGINAL_FILE="${ORIGINAL_FILE:-distill_original_timeTest.py}"
FLEX_FILE="${FLEX_FILE:-distill_flexFuse_timeTest_conv_v2.py}"

# Pipe-separated masks because each mask itself contains spaces.
# Control masks 1 and 1 1 are useful for checking whether the slowdown is
# specifically caused by the split/layout path.
MASKS="${MASKS:-1|1 1|1 0|1 1 0 0}"

IPC_LIST_STR="${IPC_LIST:-1 10 50 100}"
read -r -a IPC_LIST_ARR <<< "${IPC_LIST_STR}"

CASE_NAME="${CASE_NAME:-default}"
LOG_DIR="${LOG_DIR:-testlog_split_contig}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${MODEL}_${CASE_NAME}.txt}"

: > "${LOG_FILE}"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
iteration_for_model_ipc() {
    local model="$1"
    local ipc="$2"

    if [ "${model}" = "ViT" ] && [ "${ipc}" -ge 50 ]; then
        echo "50"
    elif [ "${model}" = "ResNet18" ] && [ "${ipc}" -ge 10 ]; then
        echo "50"
    else
        echo "${ITERATION}"
    fi
}

common_args() {
    local ipc="$1"
    local iteration="$2"

    echo \
        --dataset="${DATASET}" \
        --pix_init="${PIX_INIT}" \
        --ipc="${ipc}" \
        --syn_steps="${SYN_STEPS}" \
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
        --model="${MODEL}" \
        --Fuse="${FUSE}"
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

run_once() {
    local tag="$1"
    local barrier_mode="$2"
    local repeat_id="$3"
    local repeat_total="$4"
    shift 4

    local barrier_args=()
    if [ "${barrier_mode}" = "use_barrier" ]; then
        barrier_args=(--use-barrier)
    fi

    echo "[RUN] ${tag} | ${barrier_mode} | ${repeat_id}/${repeat_total}"

    {
        echo "============================================================"
        echo "$(date '+%Y-%m-%d %H:%M:%S')"
        echo "${tag}"
        echo "BARRIER=${barrier_mode} | REPEAT=${repeat_id}/${repeat_total}"
        echo "CMD: WANDB_SILENT=true PYTHONWARNINGS=ignore TQDM_DISABLE=1 $* ${barrier_args[*]}"
        echo "------------------------------------------------------------"
    } >> "${LOG_FILE}"

    WANDB_SILENT=true \
    PYTHONWARNINGS=ignore \
    TQDM_DISABLE=1 \
    PYTHONUNBUFFERED=1 \
    "$@" "${barrier_args[@]}" 2>&1 | clean_output >> "${LOG_FILE}"

    local exit_code=${PIPESTATUS[0]}

    {
        echo "------------------------------------------------------------"
        echo "EXIT_CODE: ${exit_code}"
        echo ""
    } >> "${LOG_FILE}"

    if [ "${exit_code}" -ne 0 ]; then
        echo "[ERROR] ${tag} failed with EXIT_CODE=${exit_code}"
    fi

    # Keep the sweep running even if one case fails.
    return 0
}

run_case() {
    local tag="$1"
    shift

    local r
    for r in $(seq 1 "${USE_BARRIER_REPEATS}"); do
        run_once "${tag}" "use_barrier" "${r}" "${USE_BARRIER_REPEATS}" "$@"
    done

    for r in $(seq 1 "${NO_BARRIER_REPEATS}"); do
        run_once "${tag}" "no_barrier" "${r}" "${NO_BARRIER_REPEATS}" "$@"
    done
}

run_original() {
    local ipc="$1"
    local iteration="$2"

    # shellcheck disable=SC2207
    local args=( $(common_args "${ipc}" "${iteration}") )

    run_case \
        "MODEL=${MODEL} | IPC=${ipc} | SYN_STEPS=${SYN_STEPS} | ITERATION=${iteration} | CASE=original_distill" \
        python "${ORIGINAL_FILE}" "${args[@]}"
}

run_flex() {
    local ipc="$1"
    local iteration="$2"
    local mask_string="$3"

    read -r -a mask_args <<< "${mask_string}"
    local mask_tag="${mask_string// /}"

    # shellcheck disable=SC2207
    local args=( $(common_args "${ipc}" "${iteration}") )

    run_case \
        "MODEL=${MODEL} | IPC=${ipc} | SYN_STEPS=${SYN_STEPS} | ITERATION=${iteration} | CASE=${CASE_NAME}_mask_${mask_tag}_vfuse" \
        python "${FLEX_FILE}" \
            --fuse_mask_list "${mask_args[@]}" \
            --v_fuse \
            "${args[@]}"
}

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
{
    echo "============================================================"
    echo "DISTILL-ONLY SPLIT/CONTIGUITY TEST"
    echo "START_TIME: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "MODEL: ${MODEL}"
    echo "DATASET: ${DATASET}"
    echo "IPC_LIST: ${IPC_LIST_ARR[*]}"
    echo "SYN_STEPS: ${SYN_STEPS}"
    echo "DEFAULT_ITERATION: ${ITERATION}"
    echo "ORIGINAL_FILE: ${ORIGINAL_FILE}"
    echo "FLEX_FILE: ${FLEX_FILE}"
    echo "CASE_NAME: ${CASE_NAME}"
    echo "MASKS: ${MASKS}"
    echo "RUN_ORIGINAL: ${RUN_ORIGINAL}"
    echo "USE_BARRIER_REPEATS: ${USE_BARRIER_REPEATS}"
    echo "NO_BARRIER_REPEATS: ${NO_BARRIER_REPEATS}"
    echo "NOTE: distill only; no fwdtest; no ckpt; no AccTest."
    echo "============================================================"
    echo ""
} >> "${LOG_FILE}"

# ------------------------------------------------------------
# Sweep
# ------------------------------------------------------------
IFS='|' read -r -a MASK_ARR <<< "${MASKS}"

for ipc in "${IPC_LIST_ARR[@]}"; do
    model_iteration="$(iteration_for_model_ipc "${MODEL}" "${ipc}")"

    if [ "${RUN_ORIGINAL}" -eq 1 ]; then
        run_original "${ipc}" "${model_iteration}"
    fi

    for mask in "${MASK_ARR[@]}"; do
        run_flex "${ipc}" "${model_iteration}" "${mask}"
    done
done

{
    echo "============================================================"
    echo "DISTILL-ONLY TEST END"
    echo "END_TIME: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
} >> "${LOG_FILE}"

echo "[DONE] Distill-only tests finished."
echo "Log: ${LOG_FILE}"
