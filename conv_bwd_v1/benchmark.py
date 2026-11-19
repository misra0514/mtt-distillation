"""Dense3/stride1 family: v5 vs v6 dW; one benchmark, two machine parts."""
import argparse
import contextlib
import hashlib
import json
import math
import platform
import random
import statistics as stats
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
import triton
import conv_bwd as kernel


class Tee:
    def __init__(self,console,log):
        self.console,self.log=console,log

    def write(self,text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()

def aten(x,w,d,st,pad,dil,groups):
    return torch.ops.aten.convolution_backward.default(
        d,x,w,[w.size(0)],st,pad,dil,False,[0,0],groups,[True,True,True])

def compare(a,b,relative=4e-4):
    # Bound transient memory even for B=1000. Accumulate error norms in FP64.
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch {a.shape} / {b.shape}")
    # Only flatten a contiguous slice, important for native grouped references.
    limit = 1 << 20
    metrics = torch.zeros(4,dtype=torch.float64,device=a.device)
    def visit(aa,bb):
        if aa.numel() > limit:
            axis = max(range(aa.ndim),key=lambda k: aa.shape[k])
            step = max(1,aa.shape[axis]*limit//aa.numel())
            for start in range(0,aa.shape[axis],step):
                n = min(step,aa.shape[axis]-start)
                visit(aa.narrow(axis,start,n),bb.narrow(axis,start,n))
            return
        u = aa.double()
        v = bb.double()
        delta = (u-v).abs()
        metrics[0].add_(delta.square().sum())
        metrics[1].add_(v.square().sum())
        metrics[2].copy_(torch.maximum(metrics[2],delta.max()))
        metrics[3].copy_(torch.maximum(metrics[3],v.abs().max()))
    visit(a,b)
    e,n,m,peak = metrics.tolist()
    rel_l2 = math.sqrt(e)/max(math.sqrt(n),1e-12)
    if not (rel_l2 <= relative and m <= 2e-4+2e-3*peak):
        raise AssertionError(f"rel_l2={rel_l2:g}, max_abs={m:g}, ref_peak={peak:g}")
    return rel_l2

def original(x,w,d,st,pad,dil,groups):
    args = (st,pad,dil,groups)
    return (torch.nn.grad.conv2d_input(x.shape,w,d,*args),
            torch.nn.grad.conv2d_weight(x,w.shape,d,*args),d.sum((0,2,3)))

def pack(t):
    g,b,c,h,w = t.shape
    return t.transpose(0,1).reshape(b,g*c,h,w).contiguous()


FAMILY = "dense3_stride1"


def model_sources():
    # The supplied ConvNet: RGB input, width=128, depth=3, pad=1,
    # and 2x2/stride2 pooling after each convolution. Exclude its RGB
    # first layer from this dense-hidden-layer family; deduplicate shapes.
    sources = {}
    for dataset, image_size in (("cifar10", 32), ("tinyimagenet", 64)):
        c, h = 3, image_size
        for layer in range(1, 4):
            if c == 128:
                sources.setdefault(h, []).append(f"{dataset}/conv{layer}")
            c, h = 128, h // 2
    return {h: sources[h] for h in sorted(sources)}


SOURCES = model_sources()


def case_matrix(spatial, groups, part):
    batches = (1, 10) if part == "all" else (1,) if part == "1" else (10,)
    return [(h, g, b) for h in spatial for b in batches for g in groups]


def make_inputs(h, g, b, seed):
    torch.manual_seed(seed)
    x = torch.randn(g, b, 128, h, h, device="cuda")
    w = torch.randn(g, 128, 128, 3, 3, device="cuda")
    d = torch.randn_like(x)
    if seed == 17 and g > 1:
        d[-1].zero_()
    return x, w, d


def unpack(outputs, g):
    dx, dw, db = outputs
    return (dx.unflatten(1, (g, dx.shape[1] // g)).transpose(0, 1).contiguous(),
            dw.unflatten(0, (g, dw.shape[0] // g)).contiguous(),
            db.view(g, -1).contiguous())


def torch_group_first(x, w, d, use_aten):
    g = x.shape[0]
    xp, wp, dp = pack(x), w.flatten(0, 1), pack(d)
    fn = aten if use_aten else original
    return unpack(fn(xp, wp, dp, (1, 1), (1, 1), (1, 1), g), g)


def check_outputs(actual, target, case, seed, provider):
    errors = {}
    for name, aa, bb in zip(("grad_x", "grad_weight", "grad_bias"), actual, target):
        try:
            errors[name] = compare(aa, bb)
        except AssertionError as exc:
            raise AssertionError(f"{case} seed={seed} {provider} {name}: {exc}") from exc
    print("CORRECTNESS", json.dumps(dict(**case, seed=seed, provider=provider,
                                         rel_l2=errors)), flush=True)


def validate(x, w, d, case, seed):
    g = x.shape[0]
    target = torch_group_first(x, w, d, False)
    for name, fn in (("v5_row", kernel.conv_bwd_v5),
                     ("v6_implicit", kernel.conv_bwd)):
        check_outputs(fn(x, w, d), target, case, seed, name)
    # Validate the additional full-call baselines, not only the Triton outputs.
    if seed == 0:
        xp, wp, dp = pack(x), w.flatten(0, 1), pack(d)
        native = aten(xp, wp, dp, (1, 1), (1, 1), (1, 1), g)
        check_outputs(unpack(native, g), target, case, seed, "aten_grouped")
        serial = [aten(x[j], w[j], d[j], (1, 1), (1, 1), (1, 1), 1)
                  for j in range(g)]
        stacked = tuple(torch.stack([z[i] for z in serial]) for i in range(3))
        check_outputs(stacked, target, case, seed, "aten_serial")
        del native, serial, stacked
        # Exercise the public copy path and the groups=G alias.
        xx, ww, dd = (t.transpose(-1, -2).contiguous().transpose(-1, -2)
                      for t in (x, w, d))
        check_outputs(kernel.conv_bwd(xx, ww, dd), target, case, seed, "v6_noncontiguous")
        check_outputs(kernel.conv_bwd(x, w, d, groups=g), target, case, seed, "v6_groups_alias")


def once(fn, n):
    torch.cuda.synchronize()
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start = time.perf_counter()
    begin.record()
    for _ in range(n):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / n, 1000 * (time.perf_counter() - start) / n


def paired(funcs, a):
    counts, peak = {}, {}
    samples = {name: [] for name in funcs}
    for name, fn in funcs.items():
        for _ in range(a.warmup):
            fn()
        torch.cuda.synchronize()
        duration = once(fn, 2)[0]
        # Compile/autotune first, then sustain activity before measuring.
        remaining = a.warmup_ms
        while remaining > 0:
            n = min(a.iters, max(1, math.ceil(min(remaining, 50) / max(duration, 0.001))))
            elapsed, _ = once(fn, n)
            remaining -= max(elapsed * n, 0.001)
        duration = once(fn, 2)[0]
        counts[name] = min(a.iters, max(2, math.ceil(a.target_ms / max(duration, 0.001))))
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn()
        torch.cuda.synchronize()
        peak[name] = (torch.cuda.max_memory_allocated() - base) / 2**20
    rng = random.Random(131)
    for _ in range(a.rounds):
        names = list(funcs)
        rng.shuffle(names)
        for name in names:
            samples[name].append(once(funcs[name], counts[name]))
    results = {}
    for name, pairs in samples.items():
        ev, wall = map(list, zip(*pairs))
        cv = lambda values: 100 * stats.stdev(values) / max(stats.mean(values), 1e-9)
        results[name] = dict(event_ms=stats.median(ev), wall_ms=stats.median(wall),
            event_mean_ms=stats.mean(ev), wall_mean_ms=stats.mean(wall),
            event_cv_pct=cv(ev), wall_cv_pct=cv(wall),
            event_rounds_ms=ev, wall_rounds_ms=wall, iterations_per_round=counts[name],
            peak_extra_MiB=peak[name], stable=cv(ev) <= 5.)
    return results


def emit_stage(stage, funcs, a, case, workspace_mib, launch_count):
    results = paired(funcs, a)
    torch_names = [name for name in results if name.startswith(("torch", "aten"))]
    best = min(torch_names, key=lambda name: results[name]["event_ms"]) if torch_names else None
    baseline = "v5_row" if "v5_row" in results else None
    for name, measured in results.items():
        row = dict(**case, stage=stage, provider=name, **measured)
        if best:
            row.update(reference=best,
                vs_best_torch=results[best]["event_ms"] / measured["event_ms"],
                comparison_stable=measured["stable"] and results[best]["stable"])
        if name == "v6_implicit" and baseline:
            row["speedup_vs_v5"] = results[baseline]["event_ms"] / measured["event_ms"]
            row["v5_comparison_stable"] = measured["stable"] and results[baseline]["stable"]
        if name.startswith(("v5", "v6", "common")):
            row["gx_workspace_MiB"] = workspace_mib if stage != "grad_weight_bias" else 0.
            row["kernel_launches"] = launch_count[name] if isinstance(launch_count, dict) else launch_count
        row["event_us_per_group"] = measured["event_ms"] * 1000 / case["G"]
        print("STAGE_RESULT", json.dumps(row), flush=True)


def profile_kernels(funcs, case):
    # Optional trace-derived kernel durations/names. No hardware counters or
    # trace files are required; these diagnostic times are not benchmark rows.
    for name, fn in funcs.items():
        fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        kernels = {}
        for event in prof.events():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                duration = getattr(event, "device_time_total", None)
                if duration is None:
                    duration = event.cuda_time_total
                item = kernels.setdefault(event.name, dict(calls=0, total_us=0.))
                item["calls"] += 1
                item["total_us"] += duration
        print("PROFILE_KERNELS", json.dumps(dict(**case, provider=name,
              scope="one profiled full call; diagnostic only", kernels=kernels)), flush=True)


def run_case(h, g, b, a):
    case = dict(family=FAMILY, H=h, W=h, C=128, O=128, G=g, B=b,
                sources=SOURCES[h], part=a.part)
    print("CASE", json.dumps(dict(**case, x_shape=[g,b,128,h,h],
          weight_shape=[g,128,128,3,3], grad_output_shape=[g,b,128,h,h],
          stride=[1,1], padding=[1,1], dilation=[1,1],
          gw_reduction=b*h*h, gw_v5_tile=[32,32,16],
          gw_v6_tile=[64,32,16], gw_v5_ctas=48*g, gw_v6_ctas=72*g,
          v6_bias="independent reduction", v6_bias_ctas=128*g)), flush=True)
    for seed in (0, 17, 123):
        x, w, d = make_inputs(h, g, b, seed)
        validate(x, w, d, case, seed)
    xp, wp, dp = pack(x), w.flatten(0,1), pack(d)
    dx = torch.empty_like(x)
    old_dw, new_dw = torch.empty_like(w), torch.empty_like(w)
    old_db, new_db = x.new_empty((g,128)), x.new_empty((g,128))
    workspace = kernel.grad_x_workspace(d, dx)

    def gx():
        kernel.launch_grad_x(d, w, dx, workspace)
        return dx

    def old_gw():
        kernel.launch_grad_weight_bias_v5(x, d, old_dw, old_db)
        return old_dw, old_db

    def new_gw():
        kernel.launch_grad_weight_bias(x, d, new_dw, new_db)
        return new_dw, new_db

    def aten_subset(mask):
        return torch.ops.aten.convolution_backward.default(
            dp, xp, wp, [g*128], [1,1], [1,1], [1,1], False, [0,0], g, mask)

    def torch_gw():
        return torch.nn.grad.conv2d_weight(xp, wp.shape, dp, 1,1,1,g), dp.sum((0,2,3))

    gx()
    old_gw()
    new_gw()
    target = torch_group_first(x, w, d, False)
    check_outputs((dx, old_dw, old_db), target, case, 123, "v5_preallocated_stages")
    check_outputs((dx, new_dw, new_db), target, case, 123, "v6_preallocated_stages")
    del target
    chunks = math.ceil(g / workspace.shape[0])
    gx_launches = 2 * b * chunks
    workspace_mib = workspace.numel() * workspace.element_size() / 2**20
    print("WORKSPACE", json.dumps(dict(**case, budget_MiB=kernel.WORKSPACE_LIMIT_BYTES/2**20,
          allocated_MiB=workspace_mib, groups_per_chunk=workspace.shape[0],
          chunks=chunks, gx_kernel_launches=gx_launches, gw_workspace_MiB=0.)), flush=True)
    full = {
        "v5_row": lambda: kernel.conv_bwd_v5(x,w,d),
        "v6_implicit": lambda: kernel.conv_bwd(x,w,d),
        "torch_given": lambda: original(xp,wp,dp,1,1,1,g),
        "aten_grouped": lambda: aten(xp,wp,dp,(1,1),(1,1),(1,1),g),
        "aten_serial": lambda: [aten(x[j],w[j],d[j],(1,1),(1,1),(1,1),1) for j in range(g)],
    }
    # Match v5's strict native-layout baseline, with packing outside timing.
    full_launches = {"v5_row":gx_launches+1, "v6_implicit":gx_launches+2}
    emit_stage("full_api", full, a, case, workspace_mib, full_launches)
    # Also compare exactly the requested group-first inputs and outputs;
    # Torch packing and contiguous group-first dX are included here.
    emit_stage("group_first_api", {
        "v5_row": full["v5_row"], "v6_implicit": full["v6_implicit"],
        "torch_given_layout": lambda: torch_group_first(x,w,d,False),
        "aten_grouped_layout": lambda: torch_group_first(x,w,d,True),
    }, a, case, workspace_mib, full_launches)
    emit_stage("grad_x", {
        "common_v5_gemm_fold": gx,
        "torch_given": lambda: torch.nn.grad.conv2d_input(xp.shape,wp,dp,1,1,1,g),
        "aten_grouped": lambda: aten_subset([True,False,False]),
    }, a, case, workspace_mib, gx_launches)
    emit_stage("grad_weight_bias", {
        "v5_row": old_gw, "v6_implicit": new_gw,
        "torch_given": torch_gw, "aten_grouped": lambda: aten_subset([False,True,True]),
    }, a, case, workspace_mib, {"v5_row":1, "v6_implicit":2})

    # Diagnostic dX components: ONE batch, first G chunk, as in v5.
    chunk = workspace.shape[0]
    dd, ww, out = d[:chunk], w[:chunk], dx[:chunk]
    def gemm():
        kernel.launch_gemm_one_batch(dd, ww, workspace, 0)
        return workspace
    def fold():
        kernel.launch_fold_one_batch(workspace, out, 0)
        return out
    def weight_only():
        kernel.launch_grad_weight(x, d, new_dw)
        return new_dw
    def bias_only():
        kernel.launch_grad_bias(d, new_db)
        return new_db
    gemm()
    for name, measured in paired({"gemm":gemm, "fold":fold,
                                 "v6_dw":weight_only, "v6_db":bias_only}, a).items():
        scope = dict(scope="all batches, all groups", measured_groups=g)
        if name in ("gemm", "fold"):
            scope = dict(scope="one batch, first group chunk only", batch_index=0,
                         measured_groups=chunk)
        print("INTERNAL_RESULT", json.dumps(dict(**case, provider=name,
              **scope, **measured)), flush=True)
    if a.profile:
        try:
            profile_kernels({name:full[name] for name in ("v5_row", "v6_implicit", "aten_grouped")}, case)
        except Exception as exc:
            # Optional profiler availability must not discard the sweep.
            print("PROFILE_UNAVAILABLE", json.dumps(dict(**case,
                  error=f"{type(exc).__name__}: {exc}")), flush=True)
    print("CASE_FINISHED", json.dumps(case), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["1","2","all"], default="all",
                        help="1: B=1; 2: B=10; all: both; family is unchanged")
    parser.add_argument("--spatial", nargs="+", type=int, choices=[8,16,32], default=[8,16,32])
    parser.add_argument("--groups", nargs="+", type=int, default=[1,2,4,8,16,32,64])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--warmup-ms", type=float, default=100.)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--target-ms", type=float, default=50.)
    parser.add_argument("--profile", action="store_true", help="print CUDA kernel names/times to the same txt")
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    if (min(a.groups) < 1 or max(a.groups) > 65535 or a.warmup < 1 or
        a.warmup_ms < 0 or a.rounds < 2 or a.iters < 2 or a.target_ms <= 0 or
        not math.isfinite(a.warmup_ms + a.target_ms)):
        parser.error("invalid groups or timing parameters")
    if len(set(a.groups)) != len(a.groups) or len(set(a.spatial)) != len(a.spatial):
        parser.error("duplicate groups or spatial sizes")
    a.output = a.output or Path(f"dense3_s1_v6_part{a.part}.txt")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8", buffering=1) as f:
        with contextlib.redirect_stdout(Tee(sys.stdout,f)), contextlib.redirect_stderr(Tee(sys.stderr,f)):
            try:
                print("START", datetime.now().astimezone().isoformat(), flush=True)
                print("ARGS", json.dumps(vars(a),default=str), flush=True)
                print("SOURCE_SHA256", json.dumps({name:hashlib.sha256(
                      Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ("benchmark.py","conv_bwd.py")}), flush=True)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cudnn.benchmark = True
                props = torch.cuda.get_device_properties(torch.cuda.current_device())
                print("ENV", json.dumps(dict(host=platform.node(),torch=torch.__version__,
                      triton=triton.__version__,cuda=torch.version.cuda,
                      cudnn=torch.backends.cudnn.version(),gpu=props.name,
                      sms=props.multi_processor_count,total_memory_MiB=props.total_memory/2**20,
                      device=torch.cuda.current_device(),precision="FP32 IEEE")), flush=True)
                matrix = case_matrix(a.spatial,a.groups,a.part)
                print("FAMILY dense3_stride1; C=O=128; kernel=3x3; stride=padding=dilation=1",flush=True)
                print("MODEL_SPEC ConvNet depth=3 width=128 input_channels=3; "
                      "pooling=2x2 stride2 after each conv; CIFAR10=32x32 TinyImageNet=64x64. "
                      "RGB conv1 is outside this family.",flush=True)
                print("MODEL_MAPPING", json.dumps(SOURCES), flush=True)
                print("CASES", json.dumps(matrix), flush=True)
                print("ALGORITHM v6: one dW accumulator 64x32x16, implicit patch loads, separate dB reduction. "
                      "v5: three-tap dW 32x32x16. Both use identical v5 dX GEMM+fold. "
                      "The v5 H=8/32 paths generalize its H=16 implementation. No selector or split-K.",flush=True)
                print("TIMING event_ms/wall_ms are round medians; means and all rounds are also logged. "
                      "Full calls include allocations; Triton stages preallocate outputs/workspace. "
                      "full_api Torch uses prepacked native tensors; group_first_api includes packing "
                      "and contiguous group-first outputs. INTERNAL_RESULT gemm/fold measure one batch/chunk; "
                      "v6_dw/v6_db measure all batches/groups. "
                      "Isolated medians need not sum to full_api. CV>5% marks stable=false.",flush=True)
                with torch.no_grad():
                    for h,g,b in matrix:
                        print("CASE_BEGIN",h,g,b,flush=True)
                        run_case(h,g,b,a)
                print("FINISHED",datetime.now().astimezone().isoformat(),flush=True)
            except Exception:
                traceback.print_exc()
                raise SystemExit(1)


if __name__ == "__main__":
    main()

