"""Correctness-gated FP32 forward comparison on Conv3 and ResNet18 shapes."""
import argparse
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import platform
import random
import statistics
import time
from functools import partial
from pathlib import Path
from typing import Union
from zlib import crc32

import torch
import torch.nn.functional as F
import triton

from group_first_conv import _conv, _conv_batch_tile, grouped_conv2d, pair

REVISION = "fwd-v2-batch-tile-resume-1"
RESUME_REVISIONS = ("fwd-v2-batch-tile-1", REVISION)
ATOL, RTOL, REL_L2_LIMIT = 0.002, 0.0005, 2e-4
FP64_MAX_COORDS = 4096
DATASETS = {"cifar10": 32, "tinyimagenet": 64, "imagenet10": 128}
CORE_PAIRS = [(1, 1), (1, 16), (1, 64), (10, 1), (10, 4), (10, 64), (100, 1), (100, 4)]
FULL_PAIRS = [(1, 1), (1, 4), (1, 16), (1, 64), (10, 1), (10, 4), (10, 16), (10, 64),
              (100, 1), (100, 2), (100, 4), (1000, 1), (1000, 2), (1000, 4)]
SpatialArg = Union[int, tuple[int, int]]


@dataclass(frozen=True)
class Shape:
    C: int
    O: int
    H: int
    W: int
    R: int = 3
    S: int = 3
    stride: SpatialArg = 1
    padding: SpatialArg = 1
    dilation: SpatialArg = 1
    bias: bool = False

    def output_hw(self):
        sh, sw = pair(self.stride)
        ph, pw = pair(self.padding)
        dh, dw = pair(self.dilation)
        p = (self.H + 2*ph - dh*(self.R-1) - 1)//sh + 1
        q = (self.W + 2*pw - dw*(self.S-1) - 1)//sw + 1
        return p, q

    def conv_args(self):
        return self.stride, self.padding, self.dilation


def resnet_layers(size, small_stem):
    def layer(c, o, h, r=3, stride=1, pad=1):
        return Shape(c, o, h, h, r, r, stride, pad)

    stem = layer(3, 64, size) if small_stem else layer(3, 64, size, 7, 2, 3)
    yield "stem", stem
    h = size if small_stem else (stem.output_hw()[0]+1)//2
    c = 64
    for stage, o in enumerate((64, 128, 256, 512), 1):
        for block in (0, 1):
            name = f"layer{stage}.{block}"
            if stage > 1 and block == 0:
                yield name+".downsample.0", layer(c, o, h, 1, 2, 0)
                down = layer(c, o, h, 3, 2, 1)
                yield name+".conv1", down
                h = down.output_hw()[0]
            else:
                yield name+".conv1", layer(c, o, h)
            yield name+".conv2", layer(o, o, h)
            c = o


def model_cases(suites, datasets):
    # Equal convolutions, including the bias setting, share one timing.
    cases = {}
    for suite in suites:
        for dataset in datasets:
            size = DATASETS[dataset]
            if suite == "conv3":
                layers = [
                    (f"conv{i+1}", Shape(c, 128, size//2**i, size//2**i, bias=True))
                    for i, c in enumerate((3, 128, 128))
                ]
            else:
                layers = resnet_layers(size, small_stem=dataset == "cifar10")
            for name, shape in layers:
                cases.setdefault(shape, []).append(f"{suite}/{dataset}/{name}")
    return cases


def batch_candidate(shape, b):
    p, q = shape.output_hw()
    return b > 1 and p*q <= 64


def shard_tasks(tasks, parts):
    def cost(task):
        shape, _, b, g = task
        p, q = shape.output_hw()
        flops = 2*b*g*shape.C*shape.O*shape.R*shape.S*p*q
        traffic = b*g*(shape.C*shape.H*shape.W + shape.O*p*q)
        return 1 + 0.55*batch_candidate(shape, b) + flops/25e9 + traffic/30e6

    bins = [[] for _ in range(parts)]
    loads = [0.0]*parts
    for index in sorted(range(len(tasks)), key=lambda i: -cost(tasks[i])):
        shard = min(range(parts), key=loads.__getitem__)
        bins[shard].append(index)
        loads[shard] += cost(tasks[index])
    return [[tasks[i] for i in sorted(indices)] for indices in bins], loads


def pack_input(x, g):
    n, c, h, w = x.shape
    b = n//g
    return x.view(g, b, c, h, w).transpose(0, 1).reshape(b, g*c, h, w).contiguous()


def unpack_output(y, g):
    b, go, p, q = y.shape
    o = go//g
    return y.view(b, g, o, p, q).transpose(0, 1).contiguous().view(g*b, o, p, q)


def torch_layer(x, weight, bias, shape, g):
    packed = pack_input(x, g)
    w = weight.flatten(0, 1)
    bias = None if bias is None else bias.flatten()
    y = F.conv2d(packed, w, bias, *shape.conv_args(), groups=g)
    return unpack_output(y, g)


def triton_layer(x, weight, bias, shape, g, algorithm):
    return grouped_conv2d(x, weight, bias, *shape.conv_args(), groups=g,
                          precision="ieee", algorithm=algorithm)


def inputs(shape, b, g, seed):
    torch.manual_seed(seed)
    options = dict(device="cuda", dtype=torch.float32)
    x = torch.randn(g*b, shape.C, shape.H, shape.W, **options)
    w = torch.randn(g, shape.O, shape.C, shape.R, shape.S, **options)
    bias = torch.randn(g, shape.O, **options) if shape.bias else None
    return x, w, bias


class ValidationError(AssertionError):
    def __init__(self, label, metrics):
        self.metrics = metrics
        super().__init__(f"{label}: {metrics}")


def fp64_at_indices(x, weight, bias, shape, indices):
    """Recompute sparse output coordinates with direct FP64 cross-correlation."""
    p, q = shape.output_hw()
    sh, sw = pair(shape.stride)
    ph, pw = pair(shape.padding)
    dh, dw = pair(shape.dilation)
    b = x.shape[0]//weight.shape[0]
    channel = torch.arange(shape.C, device=x.device)
    tap = torch.arange(shape.R*shape.S, device=x.device)
    block = max(1, min(128, (1 << 18)//(shape.C*shape.R*shape.S)))
    results = []
    for flat in indices.split(block):
        image = flat//(shape.O*p*q)
        group = image//b
        oc = flat//(p*q)%shape.O
        ih = (flat//q%p*sh-ph)[:, None] + tap//shape.S*dh
        iw = (flat%q*sw-pw)[:, None] + tap%shape.S*dw
        valid = (ih >= 0) & (ih < shape.H) & (iw >= 0) & (iw < shape.W)
        patch = x[image[:, None, None], channel[None, :, None],
                  ih.clamp(0, shape.H-1)[:, None, :], iw.clamp(0, shape.W-1)[:, None, :]]
        patch = patch.masked_fill(~valid[:, None, :], 0).double()
        filters = weight[group, oc].reshape(-1, shape.C, shape.R*shape.S).double()
        values = (patch*filters).sum(dim=(1, 2))
        if bias is not None:
            values += bias[group, oc].double()
        results.append(values)
    return torch.cat(results)


def compare(actual, reference, label, *, fp64_reference=None):
    if actual.shape != reference.shape or not actual.is_contiguous():
        raise ValidationError(label, dict(reason="invalid_shape_or_layout",
                                         actual_shape=list(actual.shape),
                                         reference_shape=list(reference.shape),
                                         contiguous=actual.is_contiguous()))
    a, r = actual.view(-1), reference.view(-1)
    ds = ns = max_abs = 0.0
    mismatches = 0
    positions = []
    offset = 0
    for ax, rx in zip(a.split(1 << 20), r.split(1 << 20)):
        ax, rx = ax.double(), rx.double()
        diff = (ax-rx).abs()
        failed = ~(diff <= ATOL + RTOL*rx.abs())
        chunk = torch.stack((diff.square().sum(), rx.square().sum(), diff.max(),
                             failed.sum().double()))
        de, re, ma, bad = chunk.tolist()
        ds += de
        ns += re
        max_abs = max(max_abs, ma)
        mismatches += int(bad)
        if bad and fp64_reference is not None and mismatches <= FP64_MAX_COORDS:
            positions.append(failed.nonzero(as_tuple=True)[0] + offset)
        offset += ax.numel()
    rel_l2 = math.sqrt(ds/max(ns, 1e-60))
    metrics = dict(rel_l2=rel_l2 if math.isfinite(rel_l2) else None,
                   max_abs=max_abs if math.isfinite(max_abs) else None,
                   mismatches=mismatches)
    if not math.isfinite(rel_l2) or rel_l2 > REL_L2_LIMIT:
        raise ValidationError(label, dict(metrics, reason="relative_l2"))
    if not mismatches:
        return dict(metrics, validation="pointwise_pass")
    if fp64_reference is None or mismatches > FP64_MAX_COORDS:
        raise ValidationError(label, dict(metrics, reason="pointwise_mismatch"))

    indices = torch.cat(positions)
    truth = fp64_reference(indices)
    diff = (a[indices].double()-truth).abs()
    torch_diff = (r[indices].double()-truth).abs()
    limit = ATOL + RTOL*truth.abs()
    metrics.update(fp64_checked=indices.numel(),
                   fp64_mismatches=int((~(diff <= limit)).sum().item()),
                   fp64_torch_mismatches=int((~(torch_diff <= limit)).sum().item()),
                   fp64_max_abs=diff.max().item(),
                   fp64_torch_max_abs=torch_diff.max().item())
    if metrics["fp64_mismatches"]:
        raise ValidationError(label, dict(metrics, reason="fp64_mismatch"))
    return dict(metrics, validation="fp64_resolved")


def stats(values):
    mean = statistics.mean(values)
    return dict(median=statistics.median(values), mean=mean, minimum=min(values),
                maximum=max(values),
                cv_pct=100*statistics.stdev(values)/mean if len(values) > 1 and mean else 0.0)


def timed(fn, count):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall = time.perf_counter()
    begin.record()
    for _ in range(count):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end)/count, (time.perf_counter()-wall)*1000/count


def measure(functions, args):
    counts = {}
    for label, fn in functions.items():
        event, _ = timed(fn, args.warmup)
        remaining = args.warmup_ms
        while remaining > 0:
            count = min(args.iters, max(1, math.ceil(remaining/max(event, .001))))
            event, _ = timed(fn, count)
            remaining -= max(event*count, .001)
        counts[label] = min(args.iters, max(1, math.ceil(args.target_ms/max(event, .001))))
    events, walls = {key: [] for key in functions}, {key: [] for key in functions}
    rng = random.Random(913)
    for _ in range(args.rounds):
        order = list(functions)
        rng.shuffle(order)
        for label in order:
            event, wall = timed(functions[label], counts[label])
            events[label].append(event)
            walls[label].append(wall)

    result = {}
    for label, fn in functions.items():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        value = fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()-base
        del value
        result[label] = dict(iterations=counts[label], event_ms=stats(events[label]),
                             wall_ms=stats(walls[label]), peak_extra_bytes=peak,
                             event_rounds=events[label], wall_rounds=walls[label])
    return result


def tile_plan(kernel, shape, b, g, merge_batch):
    p, q = shape.output_hw()
    config = kernel.best_config
    meta = config.kwargs
    bm, bn, bk = (meta[key] for key in ("TM", "TN", "TK"))
    positions, slices = (b*p*q, g) if merge_batch else (p*q, b*g)
    ntiles = triton.cdiv(positions, bn)
    rs = shape.R*shape.S
    padded_k = triton.cdiv(shape.C, bk)*bk*rs if meta["RS_LOOP"] else triton.cdiv(shape.C*rs, bk)*bk
    return dict(TM=bm, TN=bn, TK=bk, RS_LOOP=meta["RS_LOOP"], num_warps=config.num_warps,
                num_stages=config.num_stages, ctas=triton.cdiv(shape.O, bm)*ntiles*slices,
                n_utilization=positions/(ntiles*bn), k_utilization=shape.C*rs/padded_k)


EDGE_CASES = [
    (Shape(5, 7, 9, 11, 3, 2, (2, 1), (2, 1), (2, 1), True), 3, 2),
    (Shape(9, 17, 5, 7, 1, 1, 2, 0), 5, 3),
    (Shape(3, 19, 7, 9, 7, 7, 2, 3, bias=True), 5, 2),
    (Shape(17, 35, 2, 2, bias=True), 17, 2),
]


def validate_edge(index):
    shape, b, g = EDGE_CASES[index]
    seed = 21240 + index
    x, w, bias = inputs(shape, b, g, seed)
    ref = torch.cat([
        F.conv2d(x[i*b:(i+1)*b].double(), w[i].double(),
                 None if bias is None else bias[i].double(), *shape.conv_args())
        for i in range(g)
    ])
    got = {"torch_layer": compare(torch_layer(x, w, bias, shape, g), ref, "torch_layer")}
    for algorithm in ("image_tile", "batch_tile"):
        actual = triton_layer(x, w, bias, shape, g, algorithm)
        got[algorithm] = compare(actual, ref, algorithm)
    return dict(**asdict(shape), B=b, G=g, seed=seed, validation=got, status="ok")


def run_case(shape, b, g, seed, args):
    p, q = shape.output_hw()
    largest = max(b*g*shape.C*shape.H*shape.W, g*shape.O*shape.C*shape.R*shape.S,
                  b*g*shape.O*p*q)
    if b*g > 65535 or largest >= 2**31:
        return dict(P=p, Q=q, status="skipped_size", largest_tensor_numel=largest,
                    error="Outside the image_tile int32 indexing or grid.y limits")
    x, w, bias = inputs(shape, b, g, seed)
    xp = pack_input(x, g)
    wp, bp = w.flatten(0, 1), None if bias is None else bias.flatten()
    functions = {
        "torch_native": partial(F.conv2d, xp, wp, bp, *shape.conv_args(), groups=g),
        "torch_layer": partial(torch_layer, x, w, bias, shape, g),
        "image_tile": partial(triton_layer, x, w, bias, shape, g, "image_tile"),
    }
    if batch_candidate(shape, b):
        functions["batch_tile"] = partial(triton_layer, x, w, bias, shape, g, "batch_tile")
    native = functions["torch_native"]()
    ref = unpack_output(native, g)
    if tuple(ref.shape) != (b*g, shape.O, p, q):
        raise AssertionError(f"unexpected native output shape {tuple(ref.shape)}")
    copy_in = xp.data_ptr() != x.data_ptr()
    copy_out = ref.data_ptr() != native.data_ptr()
    validation, plan = {}, {}
    valid = {"torch_native": functions["torch_native"]}
    fp64 = partial(fp64_at_indices, x, w, bias, shape)
    for name, fn in functions.items():
        if name == "torch_native":
            continue
        try:
            validation[name] = compare(fn(), ref, name, fp64_reference=fp64)
        except ValidationError as exc:
            validation[name] = dict(exc.metrics, validation="failed")
            continue
        valid[name] = fn
        if name in ("image_tile", "batch_tile"):
            merge_batch = name == "batch_tile"
            tuner = _conv_batch_tile if merge_batch else _conv
            plan[name] = tile_plan(tuner, shape, b, g, merge_batch)
    del ref, native
    full = measure(valid, args)
    layout = dict(input_copy=copy_in, output_copy=copy_out,
                  input_bytes=x.numel()*x.element_size() if copy_in else 0,
                  output_bytes=b*g*shape.O*p*q*x.element_size() if copy_out else 0)
    if args.layout_stages and (copy_in or copy_out):
        stages = {}
        if copy_in:
            stages["pack_input"] = partial(pack_input, x, g)
        if copy_out:
            stages["unpack_output"] = partial(unpack_output, functions["torch_native"](), g)
        layout["stages"] = measure(stages, args)
    times = {name: value["event_ms"]["median"] for name, value in full.items()}
    speedups = {name: {base: times[base]/times[name] for base in ("torch_native", "torch_layer")
                       if base in times}
                for name in plan}
    if "batch_tile" in plan and "image_tile" in plan:
        speedups["batch_tile"]["image_tile"] = times["image_tile"]/times["batch_tile"]
    status = "ok" if len(valid) == len(functions) else "validation_error"
    return dict(P=p, Q=q, status=status, validation=validation, triton_plan=plan,
                full=full, layout=layout, event_speedups=speedups)


def bg_pair(value):
    try:
        b, g = map(int, value.lower().split("x"))
        if b > 0 and g > 0:
            return b, g
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("use BxG, e.g. 10x64")


def task_key(record):
    names = [field.name for field in fields(Shape)] + ["B", "G"]
    return tuple(tuple(record[k]) if isinstance(record[k], list) else record[k] for k in names)


def read_completed(paths, assigned, args, kernel_sha):
    expected = {task_key(dict(asdict(shape), B=b, G=g)): names
                for shape, names, b, g in assigned}
    resumed = set()
    prior_envs = []
    compared_args = ("mode", "suite", "dataset", "sweep", "pairs", "filter", "parts", "part",
                     "warmup", "warmup_ms", "rounds", "iters", "target_ms", "layout_stages")
    options = json.loads(json.dumps({k: getattr(args, k) for k in compared_args}))
    for path in paths:
        prior = None
        with path.open(encoding="utf-8-sig") as log:
            for line_num, line in enumerate(log, 1):
                tag, _, payload = line.partition(" ")
                if tag not in ("ENV", "RESULT"):
                    continue
                try:
                    record = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_num}: invalid {tag} JSON") from exc
                if tag == "ENV":
                    if (prior is not None or record["revision"] not in RESUME_REVISIONS
                        or record["sha256"]["group_first_conv.py"] != kernel_sha
                        or any(record["args"][k] != v for k, v in options.items())):
                        raise ValueError(f"{path}:{line_num}: benchmark options or kernel differ")
                    prior = record
                    prior_envs.append(record)
                else:
                    if prior is None:
                        raise ValueError(f"{path}:{line_num}: RESULT appears before ENV")
                    key = task_key(record)
                    if key not in expected or record["sources"] != expected[key]:
                        raise ValueError(f"{path}:{line_num}: result is outside this shard")
                    if record["status"] == "ok":
                        resumed.add(key)
        if prior is None:
            raise ValueError(f"{path}: missing ENV")
    return resumed, prior_envs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", nargs="?", choices=("bench", "correctness"), default="bench")
    p.add_argument("--suite", nargs="+", choices=("conv3", "resnet18"), default=["conv3", "resnet18"])
    p.add_argument("--dataset", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    p.add_argument("--sweep", choices=("core", "full"), default="core")
    p.add_argument("--pairs", type=bg_pair, nargs="+", help="Override sweep pairs, e.g. 10x4 100x4")
    p.add_argument("--filter", default="", help="Keep shapes with a matching model/dataset/layer path")
    p.add_argument("--parts", type=int, default=1)
    p.add_argument("--part", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--warmup-ms", type=float, default=50.0)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--target-ms", type=float, default=25.0)
    p.add_argument("--layout-stages", action="store_true")
    p.add_argument("--resume-from", type=Path, action="append", default=[],
                   help="Skip successful records from a compatible prior log for this shard")
    p.add_argument("--list-tasks", action="store_true", help="Print shard contents without CUDA")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if not 1 <= a.part <= a.parts or min(a.warmup, a.rounds, a.iters) <= 0:
        p.error("require 1 <= part <= parts and positive warmup, rounds, iters")
    if a.warmup_ms < 0 or a.target_ms <= 0:
        p.error("require nonnegative warmup-ms and positive target-ms")
    if a.resume_from and a.mode != "bench":
        p.error("--resume-from is for bench mode")
    return a


@torch.no_grad()
def main():
    args = parse_args()
    pairs = args.pairs or (FULL_PAIRS if args.sweep == "full" else CORE_PAIRS)
    cases = model_cases(args.suite, args.dataset)
    tasks = [(shape, names, b, g) for shape, names in cases.items()
             if any(args.filter in name for name in names) for b, g in dict.fromkeys(pairs)]
    shards, loads = shard_tasks(tasks, args.parts)
    assigned = shards[args.part-1]
    source = Path(__file__).resolve().parent
    hashes = {fn: hashlib.sha256((source/fn).read_bytes()).hexdigest()
              for fn in ("benchmark.py", "group_first_conv.py")}
    path = args.output or Path(f"fwd_v2_resume_{args.mode}_part{args.part}.txt")
    if any(path.resolve() == prior.resolve() for prior in args.resume_from):
        raise SystemExit("--output must differ from resumed logs")
    try:
        resumed, prior_envs = read_completed(args.resume_from, assigned, args, hashes["group_first_conv.py"])
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
    selected = [(shape, names, b, g) for shape, names, b, g in assigned
                if task_key(dict(asdict(shape), B=b, G=g)) not in resumed]
    shard = dict(index=args.part, count=args.parts, all_tasks=len(tasks),
                 assigned_tasks=len(assigned), resumed_ok=len(resumed), selected_tasks=len(selected),
                 estimated_loads=[round(v, 2) for v in loads])
    if args.list_tasks:
        print(json.dumps(shard))
        for shape, names, b, g in selected:
            print(json.dumps(dict(**asdict(shape), sources=names, B=b, G=g,
                                  batch_tile=batch_candidate(shape, b))))
        return
    if args.mode == "bench" and not assigned:
        raise SystemExit("This shard has no matching cases")
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    env = dict(revision=REVISION, host=platform.node(), gpu=torch.cuda.get_device_name(),
               capability=list(torch.cuda.get_device_capability()), torch=torch.__version__,
               triton=triton.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
               sm_count=properties.multi_processor_count, total_memory_bytes=properties.total_memory,
               cudnn_benchmark=True, tf32=False, sha256=hashes, args=vars(args), shard=shard)
    compared_env = ("gpu", "capability", "torch", "triton", "cuda", "cudnn", "tf32", "cudnn_benchmark")
    for prior in prior_envs:
        mismatched = [k for k in compared_env if prior[k] != env[k]]
        if mismatched:
            raise SystemExit(f"Prior log environment differs in {', '.join(mismatched)}")
    with path.open("w", buffering=1) as log:
        def emit(tag, record):
            line = f"{tag} {json.dumps(record, default=str, allow_nan=False)}"
            print(line, flush=True)
            print(line, file=log)

        emit("ENV", env)
        if args.mode == "correctness":
            edges = range(len(EDGE_CASES))
        else:
            edges = range(args.part-1, len(EDGE_CASES), args.parts)
        for index in edges:
            emit("CORRECTNESS", validate_edge(index))
        if args.mode == "correctness":
            return
        counts = {"ok": 0, "skipped_oom": 0, "skipped_size": 0, "validation_error": 0}
        for shape, names, b, g in selected:
            seed = crc32(repr((shape, b, g)).encode())
            case = dict(**asdict(shape), sources=names, B=b, G=g, seed=seed)
            print(f"Testing {names[0]} B={b} G={g}", flush=True)
            try:
                result = run_case(shape, b, g, seed, args)
            except torch.cuda.OutOfMemoryError as exc:
                result = dict(status="skipped_oom", error=str(exc))
            except Exception as exc:
                emit("RESULT", dict(case, status="error", error=repr(exc)))
                raise
            if result["status"] == "skipped_oom":
                torch.cuda.empty_cache()
            counts[result["status"]] += 1
            emit("RESULT", dict(case, **result))
        emit("SUMMARY", dict(counts, previously_ok=len(resumed)))
        if counts["validation_error"]:
            raise SystemExit("Some providers failed validation. See RESULT records.")


if __name__ == "__main__":
    main()
