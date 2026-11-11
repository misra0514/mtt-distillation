#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash profile_gpu_bandwidth.sh REPORT_PREFIX -- COMMAND [ARG ...]

Collect GPU hardware metrics and CUDA kernel/copy traces for the whole run.
REPORT_PREFIX is the output path without the .nsys-rep suffix.

Environment variables:
  GPU_METRICS_DEVICE     Nsight Systems GPU index (default: 0)
  GPU_METRICS_SET        Optional metric set alias (e.g. ga10x for RTX A5000)
  GPU_METRICS_FREQUENCY  Sampling frequency in Hz (default: 10000)
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if [[ $# -lt 3 || ${2:-} != -- ]]; then
    usage >&2
    exit 2
fi

report_prefix=$1
shift 2

if ! command -v nsys >/dev/null; then
    echo "Nsight Systems CLI (nsys) is required on the GPU machine." >&2
    exit 127
fi

nsys_help=$(nsys profile --help)
case "$nsys_help" in
    *--gpu-metrics-devices*) device_option=--gpu-metrics-devices ;;
    *--gpu-metrics-device*)  device_option=--gpu-metrics-device ;;
    *)
        echo "This nsys installation does not expose GPU metric sampling." >&2
        exit 2
        ;;
esac

metrics_options=(
    "${device_option}=${GPU_METRICS_DEVICE:-0}"
    "--gpu-metrics-frequency=${GPU_METRICS_FREQUENCY:-10000}"
)
if [[ -n ${GPU_METRICS_SET:-} ]]; then
    metrics_options+=("--gpu-metrics-set=$GPU_METRICS_SET")
fi

mkdir -p -- "$(dirname -- "$report_prefix")"
exec nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    "${metrics_options[@]}" --output="$report_prefix" "$@"
