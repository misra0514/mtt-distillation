#!/usr/bin/env bash
set -euo pipefail
# SCAN=fuse: fixed IPC, sweep FUSES. SCAN=ipc: fixed FUSE, double IPC.
SCAN=${SCAN:-fuse}
IPC=${IPC:-1}
FUSE=${FUSE:-1}
FUSES=${FUSES:-"1 2 4 8 16 32 64"}
IPC_START=${IPC_START:-10}
IPC_MAX=${IPC_MAX:-640}
WARMUP=${WARMUP:-10}
REPEATS=${REPEATS:-20}
CONV_WIDTH=${CONV_WIDTH:-128}
V_FUSE=${V_FUSE:-true}
REPO=${REPO:-$PWD}
PYTHON_BIN=${PYTHON_BIN:-python}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
read -r -a FUSE_VALUES <<< "$FUSES"
export PYTHONWARNINGS=ignore
"$PYTHON_BIN" "$SCRIPT_DIR/conv_kernel_scalability_v3.py" \
  --scan "$SCAN" --repo "$REPO" --ipc "$IPC" --fuse "$FUSE" \
  --fuses "${FUSE_VALUES[@]}" --ipc-start "$IPC_START" --ipc-max "$IPC_MAX" \
  --warmup "$WARMUP" --repeats "$REPEATS" \
  --conv-width "$CONV_WIDTH" --v-fuse "$V_FUSE" "$@"
