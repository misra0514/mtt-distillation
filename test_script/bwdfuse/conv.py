#!/usr/bin/env python3
"""Compare serial and mixed Conv_Flexfuse input gradients for two MTT jobs.

Copy the updated Conv_Flexfuse class into networks/networks_Fuse.py, then
run this script from the project root. The model and its kernels are
imported from the user's project.
"""

import argparse
import math

import torch
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from networks.networks_Fuse import Conv_Flexfuse

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--net-width", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--v-fuse", choices=["both", "true", "false"], default="both")
    parser.add_argument("--lr-teacher", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=3e-3)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rel-l2-tol", type=float, default=3e-3)
    args = parser.parse_args()
    if args.batch_size < 1 or args.net_width < 1 or args.num_classes < 2:
        parser.error("batch-size and net-width must be positive, and num-classes must be at least 2")
    if not math.isfinite(args.lr_teacher) or args.lr_teacher <= 0:
        parser.error("lr-teacher must be finite and positive")
    if any(not math.isfinite(value) or value < 0 for value in (args.rtol, args.atol, args.rel_l2_tol)):
        parser.error("tolerances must be finite and nonnegative")
    return args


def ordered_parameters(weights):
    params = []
    for block in weights["blocks"]:
        params.extend(block[key] for key in ("convw", "convb", "normw", "normb"))
    params.extend(weights["head"][key] for key in ("linw", "linb"))
    return params


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(item) for item in value]
    if value is None:
        return None
    raise TypeError(f"Unexpected state type {type(value)}")


class Checks:
    def __init__(self, args):
        self.args = args
        self.count = 0
        self.worst_rel_l2 = 0.0
        self.worst_name = ""

    def compare(self, name, actual, expected):
        if isinstance(expected, dict):
            if actual.keys() != expected.keys():
                raise AssertionError(f"{name}: dictionary keys differ")
            for key in expected:
                self.compare(f"{name}.{key}", actual[key], expected[key])
            return
        if isinstance(expected, (list, tuple)):
            if len(actual) != len(expected):
                raise AssertionError(f"{name}: state lengths differ")
            for index, (got, want) in enumerate(zip(actual, expected)):
                self.compare(f"{name}[{index}]", got, want)
            return
        if expected is None:
            if actual is not None:
                raise AssertionError(f"{name}: expected None")
            return
        got = actual.detach().cpu().double()
        want = expected.detach().cpu().double()
        if got.shape != want.shape:
            raise AssertionError(f"{name}: shape {tuple(got.shape)} != {tuple(want.shape)}")
        if not torch.isfinite(got).all() or not torch.isfinite(want).all():
            raise AssertionError(f"{name}: NaN or Inf")
        diff = got - want
        diff_norm = torch.linalg.vector_norm(diff).item()
        ref_norm = torch.linalg.vector_norm(want).item()
        rel_l2 = diff_norm / ref_norm if ref_norm else (0.0 if diff_norm == 0 else float("inf"))
        max_abs = diff.abs().max().item() if diff.numel() else 0.0
        try:
            torch.testing.assert_close(got, want, rtol=self.args.rtol, atol=self.args.atol)
        except AssertionError as error:
            raise AssertionError(f"{name}: max_abs={max_abs:.6e}, rel_l2={rel_l2:.6e}\n{error}") from error
        # Absolute tolerance alone could let a zero image gradient pass.
        if name.endswith("grad_image") and (ref_norm == 0 or rel_l2 > self.args.rel_l2_tol):
            raise AssertionError(f"{name}: ref_norm={ref_norm:.6e}, rel_l2={rel_l2:.6e}")
        self.count += 1
        if rel_l2 > self.worst_rel_l2:
            self.worst_rel_l2, self.worst_name = rel_l2, name
        if name.endswith("grad_image"):
            print(f"  {name}: ref_norm={ref_norm:.6e}, max_abs={max_abs:.6e}, rel_l2={rel_l2:.6e}")


def mtt_dd_seeds(weights, gradients, target_parameters, lr):
    """d L_mtt / d g for one inner step theta1 = theta0 - lr * g.

    L_mtt = ||theta1 - theta_target||^2 / ||theta0 - theta_target||^2.
    Feeding these seeds into double backward yields d L_mtt / d images.
    Each tensor follows collect_ordered_grads / pack_recovered_dd order.
    """
    params = ordered_parameters(weights)
    if len(params) != len(gradients) or len(params) != len(target_parameters):
        raise AssertionError("MTT parameter, gradient, and target counts differ")
    distance = sum((param - target).square().sum() for param, target in zip(params, target_parameters))
    if not torch.isfinite(distance) or distance.item() <= 0:
        raise AssertionError("MTT target distance must be finite and positive")
    seeds = []
    for param, grad, target in zip(params, gradients, target_parameters):
        if grad.shape != param.shape:
            raise AssertionError(f"Parameter gradient shape {grad.shape} != {param.shape}")
        seeds.append((-2.0 * lr / distance) * (param - lr * grad - target))
    return seeds


def serial_reference(net, images, labels, weights, targets, lr):
    tape = net(images.clone())[1]
    d_head, d_acts, d_weights, gradients = net.run_first_bwd(
        tape, labels, Fuse=1, weights=weights)
    seeds = mtt_dd_seeds(weights, gradients, targets, lr)
    # The full backward consumes its tape and activation states.
    expected = snapshot({
        "tape": tape, "d_head": d_head, "d_acts": d_acts,
        "d_weights": d_weights, "gradients": gradients, "dd_seeds": seeds,
    })
    grad_image = net.run_double_bwd_full(
        tape, d_acts, Fuse=1, weights=weights,
        d_bwd_state=d_head, dd_tensors_all=seeds)
    if grad_image.shape != images.shape:
        raise AssertionError(f"Serial image gradient shape {grad_image.shape} != {images.shape}")
    expected["grad_image"] = snapshot(grad_image)
    return expected


def concatenate_packs(first, second, is_weights=False):
    # Packs are in the chosen lane order. Conv activation channels use dim=1.
    # Logits are Fuse-major on dim=0, as are all packed model parameters.
    def concatenate(left, right, axis):
        if left is None and right is None:
            return None
        return torch.cat((left, right), dim=axis)

    blocks = [
        {key: concatenate(left[key], right[key], 0 if is_weights else 1) for key in left}
        for left, right in zip(first["blocks"], second["blocks"])
    ]
    head = {
        key: concatenate(first["head"][key], second["head"][key],
                         0 if is_weights or key == "x_out" else 1)
        for key in first["head"]
    }
    return {"blocks": blocks, "head": head}


def mixed_schedule(net_a, net_b, images_a, images_b, labels_a, labels_b,
                   weights_a, weights_b, targets_a, targets_b, mask, references, args):
    checks = Checks(args)
    reference_a, reference_b = references

    # A runs through forward-order double backward, with bwd2-1 pending.
    tape_a = net_a(images_a.clone())[1]
    head_a, acts_a, _, gradients_a = net_a.run_first_bwd(
        tape_a, labels_a, Fuse=1, weights=weights_a)
    checks.compare("A.first_gradients", gradients_a, reference_a["gradients"])
    seeds_a = mtt_dd_seeds(weights_a, gradients_a, targets_a, args.lr_teacher)
    state_a = net_a.run_double_bwd(
        tape_a, acts_a, Fuse=1, weights=weights_a,
        d_bwd_state=head_a, dd_tensors_all=seeds_a)
    del gradients_a, seeds_a, head_a, acts_a

    # B has run only forward. Use different inputs, labels, AND weights.
    tape_b = net_b(images_b.clone())[1]
    if mask == [1, 0]:
        fused_tape = concatenate_packs(tape_b, tape_a)
        fused_weights = concatenate_packs(weights_b, weights_a, is_weights=True)
    else:
        fused_tape = concatenate_packs(tape_a, tape_b)
        fused_weights = concatenate_packs(weights_a, weights_b, is_weights=True)
    del tape_a, tape_b

    # One Fuse=2 traversal performs B's bwd1 and A's bwd2-1 together.
    head_b, acts_b, d_weights_b, gradients_b, grad_image_a = net_b.run_fused_bwd(
        fused_tape, labels_b, state_a, mask, weights=fused_weights)
    del fused_weights
    if state_a:
        raise AssertionError("A's bwd2-1 state was not consumed")
    checks.compare("A.grad_image", grad_image_a, reference_a["grad_image"])
    checks.compare("B.tape", fused_tape, reference_b["tape"])
    checks.compare("B.d_head", head_b, reference_b["d_head"])
    checks.compare("B.d_acts", acts_b, reference_b["d_acts"])
    checks.compare("B.d_weights", d_weights_b, reference_b["d_weights"])
    checks.compare("B.first_gradients", gradients_b, reference_b["gradients"])

    # Recompute the MTT seeds from B's actual mixed first-bwd gradients.
    seeds_b = mtt_dd_seeds(weights_b, gradients_b, targets_b, args.lr_teacher)
    checks.compare("B.dd_seeds", seeds_b, reference_b["dd_seeds"])
    grad_image_b = net_b.run_double_bwd_full(
        fused_tape, acts_b, Fuse=1, weights=weights_b,
        d_bwd_state=head_b, dd_tensors_all=seeds_b)
    checks.compare("B.grad_image", grad_image_b, reference_b["grad_image"])
    print(f"  PASS: {checks.count} tensor checks, worst rel_l2={checks.worst_rel_l2:.6e} ({checks.worst_name})")


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script in the project CUDA environment")
    model_class = Conv_Flexfuse
    for method in ("run_first_bwd", "run_double_bwd", "run_double_bwd_full", "run_fused_bwd"):
        if not hasattr(model_class, method):
            raise RuntimeError(f"networks.networks_Fuse.Conv_Flexfuse is missing {method}; update that class first")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    vfuse_values = [True, False] if args.v_fuse == "both" else [args.v_fuse == "true"]
    print(f"Model=networks.networks_Fuse.Conv_Flexfuse, device={device}, batch={args.batch_size}, width={args.net_width}")
    print("Manual derivatives with grad mode disabled; TF32 disabled.")
    cases = 0
    with torch.no_grad():
        for vfuse in vfuse_values:
            for seed in args.seeds:
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                kwargs = dict(channel=3, num_classes=args.num_classes, net_width=args.net_width,
                              Fuse=1, v_fuse=vfuse)
                net_a = model_class(**kwargs).to(device=device, dtype=torch.float32).eval()
                net_b = model_class(**kwargs).to(device=device, dtype=torch.float32).eval()
                weights_a, weights_b = net_a.get_weight_pack(), net_b.get_weight_pack()
                images_a = torch.randn(args.batch_size, 3, 32, 32, device=device)
                images_b = torch.randn_like(images_a) + 0.2
                labels_a = torch.randint(args.num_classes, (args.batch_size,), device=device)
                labels_b = (labels_a + 1) % args.num_classes
                targets_a = [param + 0.05 * torch.randn_like(param) for param in ordered_parameters(weights_a)]
                targets_b = [param + 0.05 * torch.randn_like(param) for param in ordered_parameters(weights_b)]
                references = (
                    serial_reference(net_a, images_a, labels_a, weights_a, targets_a, args.lr_teacher),
                    serial_reference(net_b, images_b, labels_b, weights_b, targets_b, args.lr_teacher),
                )
                for mask in ([1, 0], [0, 1]):
                    print(f"seed={seed}, v_fuse={vfuse}, mask={mask}")
                    mixed_schedule(net_a, net_b, images_a, images_b, labels_a, labels_b,
                                   weights_a, weights_b, targets_a, targets_b, mask, references, args)
                    cases += 1
                del net_a, net_b, weights_a, weights_b, references, targets_a, targets_b
    print(f"ALL PASSED: {cases} mixed schedules; both A and B image gradients match serial MTT gradients.")


if __name__ == "__main__":
    main()
