# python - <<'PY'
import gc
import math
import torch
import torch.nn.functional as F

import test_script.stateless_conv_double_bwd.conv_double_bwd as kernel
# from benchmark import make_args
from test_script.stateless_conv_double_bwd.benchmark import make_args, check

torch.manual_seed(123)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

ATOL = 0.003
RTOL = 0.002
REL_L2_LIMIT = 2e-4
CHUNK = 4 * 1024 * 1024


def check_tensor(actual, expected, label):
    a = actual.reshape(-1)
    b = expected.reshape(-1)

    diff_sq = 0.0
    ref_sq = 0.0
    max_abs = 0.0
    mismatches = 0

    for start in range(0, a.numel(), CHUNK):
        aa = a[start:start + CHUNK]
        bb = b[start:start + CHUNK]
        diff = aa.double() - bb.double()

        diff_sq += diff.square().sum().item()
        ref_sq += bb.double().square().sum().item()
        max_abs = max(max_abs, diff.abs().max().item())
        mismatches += (~torch.isclose(
            aa, bb, atol=ATOL, rtol=RTOL
        )).sum().item()

    rel_l2 = math.sqrt(diff_sq / max(ref_sq, 1e-60))
    print(
        f"  {label}: max_abs={max_abs:.6g}, "
        f"rel_l2={rel_l2:.6g}, mismatches={mismatches}"
    )

    assert mismatches == 0, (label, mismatches, max_abs, rel_l2)
    assert rel_l2 <= REL_L2_LIMIT, (label, rel_l2)


@torch.no_grad()
def validate(g):
    shape = (g, 1000, 3, 64, 128, 128, 7, 7)
    args = make_args(
        shape, torch.float32, "cuda",
        st=(2, 2), pad=(3, 3), dil=(1, 1),
        opts=(True, True, False),
    )

    ggI, ggW, _, gO, weight, x = args[:6]

    signature = kernel.convolution_double_backward_signature(
        *args,
        precision="ieee",
        workspace_limit_mb=128,
        compute=(True, True, True),
    )
    algorithm, source = kernel._select_algorithm_cached(signature)

    print(f"\nB=1000 G={g}: algorithm={algorithm}, source={source}")
    assert algorithm == "stem7_gi_ggo", (algorithm, source)

    # ggO
    ggO = kernel.convolution_double_backward(
        *args,
        precision="ieee",
        workspace_limit_mb=128,
        algorithm=algorithm,
        compute=(True, False, False),
    )[0]

    for group in range(g):
        ref = F.conv2d(
            ggI[group], weight[group],
            stride=2, padding=3,
        )
        term = F.conv2d(
            x[group], ggW[group],
            stride=2, padding=3,
        )
        ref.add_(term)
        del term

        check_tensor(ggO[group], ref, f"G{group}.ggO")
        del ref

    del ggO
    gc.collect()
    torch.cuda.empty_cache()

    # gI
    gI = kernel.convolution_double_backward(
        *args,
        precision="ieee",
        workspace_limit_mb=128,
        algorithm=algorithm,
        compute=(False, True, False),
    )[1]

    for group in range(g):
        ref = torch.nn.grad.conv2d_input(
            x[group].shape,
            ggW[group],
            gO[group],
            stride=2,
            padding=3,
        )
        check_tensor(gI[group], ref, f"G{group}.gI")
        del ref

    del gI
    gc.collect()
    torch.cuda.empty_cache()

    # gW
    gW = kernel.convolution_double_backward(
        *args,
        precision="ieee",
        workspace_limit_mb=128,
        algorithm=algorithm,
        compute=(False, False, True),
    )[2]

    for group in range(g):
        ref = torch.nn.grad.conv2d_weight(
            ggI[group],
            weight[group].shape,
            gO[group],
            stride=2,
            padding=3,
        )
        # check_tensor(gW[group], ref, f"G{group}.gW")
        group_args = tuple(
            None if t is None else t[group:group + 1]
            for t in args[:6]
        ) + tuple(args[6:])

        metrics = check(
            (None, None, gW[group:group + 1]),
            (None, None, ref.unsqueeze(0)),
            torch.float32,
            "ieee",
            args=group_args,
        )
        print(f"  G{group}.gW:", metrics)
        del ref

    del gW, args, ggI, ggW, gO, weight, x
    gc.collect()
    torch.cuda.empty_cache()
    print(f"B=1000 G={g}: PASS")


validate(2)
validate(4)
print("\nV34 previously skipped correctness cases: ALL PASS")
# PY