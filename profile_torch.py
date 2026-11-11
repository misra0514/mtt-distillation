"""Capture CUDA memory history or a kernel trace from a Python script."""
"""
USAGE:
python profile_torch.py --mode memory --out-dir profiles/conv \
  -- distill_flexFuse_timeTest_conv.py \
  --fuse_mask_list 1 0 --v_fuse --ipc 10 --syn_steps 1 \
  --Iteration 0 --AccTest=True





"""

import argparse
import runpy
import sys
from pathlib import Path


def run_script(script, script_args):
    original_argv = sys.argv
    original_path = sys.path[:]
    try:
        sys.argv = [str(script), *script_args]
        sys.path[0] = str(script.parent)
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        usage="%(prog)s --mode {memory,kernel} [options] -- script.py [script arguments]",
    )
    parser.add_argument("--mode", choices=("memory", "kernel"), required=True)
    parser.add_argument("--out-dir", type=Path, help="default: profiles/<script name>")
    parser.add_argument("--max-entries", type=int, default=100_000,
                        help="maximum allocator events retained in memory mode")
    parser.add_argument("script", type=Path, help="Python entry point to profile")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    script = args.script.resolve()
    if not script.is_file():
        parser.error(f"script not found: {script}")
    if args.max_entries < 1:
        parser.error("--max-entries must be positive")

    import torch

    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment")

    out_dir = (args.out_dir or Path("profiles") / script.stem).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "memory":
        snapshot_path = out_dir / "memory_snapshot.pickle"
        torch.cuda.memory._record_memory_history(stacks="python", max_entries=args.max_entries)
        try:
            run_script(script, args.script_args)
        finally:
            try:
                torch.cuda.memory._dump_snapshot(str(snapshot_path))
                print(f"CUDA memory snapshot: {snapshot_path}")
            finally:
                torch.cuda.memory._record_memory_history(enabled=None)
        return

    trace_path = out_dir / "kernel_trace.json"
    profiler = None
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=True,
        ) as profiler:
            run_script(script, args.script_args)
    finally:
        if profiler is not None:
            profiler.export_chrome_trace(str(trace_path))
            print(f"CUDA kernel trace: {trace_path}")


if __name__ == "__main__":
    main()
