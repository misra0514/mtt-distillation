"""Real replay-buffer VGG11 manual first + double-bwd vs independent ordinary VGG autograd.

Place the VGG11_Flexfuse class from VGG11_Flexfuse_double_bwd.py in
networks/networks_Fuse.py and run from the repository root:
python script_fuse_vgg/test_vgg11_manual_double_bwd.py --fuse_mask_list 1 0

The model's methods use only the project's manual backward primitives.
Autograd is used ONLY for the independent reference in this test.
"""
import argparse
import os
import sys
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from networks.networks import VGG11
from networks.networks_Fuse import VGG11_Flexfuse


def select_param(p, mask):
    """Select branch rows in ONE parameter tensor; preserve parameter-major order."""
    fuse = len(mask)
    rows = p.shape[0] // fuse
    ids = [i for i, active in enumerate(mask) if active]
    return p.reshape(fuse, rows, *p.shape[1:])[ids].reshape(len(ids) * rows, *p.shape[1:])


def metrics(name, value, ref, rtol, atol, failures, *, check_relative=False):
    a, b = value.detach().double(), ref.detach().double()
    if a.shape != b.shape:
        raise AssertionError(f'{name}: shape {tuple(a.shape)} vs {tuple(b.shape)}')
    max_abs = (a-b).abs().max().item()
    rel_l2 = (a-b).norm().item() / max(b.norm().item(), 1e-30)
    passed = torch.allclose(a, b, rtol=rtol, atol=atol) and (
        not check_relative or rel_l2 <= rtol)
    print(f'{name:<25} max_abs={max_abs:.4e} rel_L2={rel_l2:.4e} '
          f'{"PASS" if passed else "FAIL"}')
    if not passed:
        failures.append(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--buffer', default='/scratch/yguo25/files/mtt-distillation/buffer/CIFAR10_NO_ZCA/VGG11/replay_buffer_0.pt')
    p.add_argument('--expert-index', type=int, default=0)
    p.add_argument('--epoch', type=int, default=0)
    p.add_argument('--fuse_mask_list', '--fuse-mask-list', type=int, nargs='+', default=[1, 1])
    p.add_argument('--image-pt', default=None)
    p.add_argument('--image-index', type=int, default=0)
    p.add_argument('--label', type=int, default=0)
    p.add_argument('--syn-lr', type=float, default=0.01)
    p.add_argument('--target-value', type=float, default=0.0)
    p.add_argument('--rtol', type=float, default=3e-3)
    p.add_argument('--atol', type=float, default=1e-6)
    p.add_argument('--image-atol', type=float, default=1e-10)
    p.add_argument('--seed', type=int, default=7)
    args = p.parse_args()

    mask = args.fuse_mask_list
    if not mask or any(m not in (0, 1) for m in mask) or sum(mask) == 0:
        p.error('A nonempty 0/1 mask with at least one active branch is required')
    ids = [i for i, m in enumerate(mask) if m]
    if ids != list(range(ids[0], ids[-1]+1)):
        p.error('Active branches must be contiguous')
    fuse, bwd_fuse = len(mask), sum(mask)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device} mask={mask} full Fuse={fuse} bwd Fuse={bwd_fuse}')

    replay = torch.load(args.buffer, map_location='cpu', weights_only=False)  # trusted local buffer
    snapshot = replay[args.expert_index][args.epoch]
    ordinary = VGG11(3, 10).to(device).train()
    fused = VGG11_Flexfuse(3, 10, Fuse=fuse).to(device).train()
    orig_params, fused_params = list(ordinary.parameters()), list(fused.parameters())
    assert len(snapshot) == len(orig_params) == len(fused_params) == 34
    with torch.no_grad():
        for i, (src, po, pf) in enumerate(zip(snapshot, orig_params, fused_params)):
            src = src.to(device=device, dtype=po.dtype)
            repeated = src.repeat((fuse,) + (1,) * (src.ndim - 1))
            assert po.shape == src.shape and pf.shape == repeated.shape, (i, po.shape, pf.shape, src.shape)
            po.copy_(src)
            pf.copy_(repeated)
    del replay, snapshot

    if args.image_pt is None:
        image = torch.randn(1, 3, 32, 32, device=device)
    else:
        saved = torch.load(args.image_pt, map_location='cpu', weights_only=True)
        image = saved[args.image_index:args.image_index+1].to(device)
        assert tuple(image.shape) == (1, 3, 32, 32)
    label = torch.tensor([args.label], dtype=torch.long, device=device)

    # Independent autograd reference uses the ordinary (UNFUSED) VGG11.
    x_ref = image.detach().clone().requires_grad_(True)
    lr_ref = torch.tensor(args.syn_lr, device=device, dtype=x_ref.dtype, requires_grad=True)
    logits_ref = ordinary(x_ref)
    ce_ref = F.cross_entropy(logits_ref, label)
    orig_grads = torch.autograd.grad(ce_ref, orig_params, create_graph=True)
    next_ref = torch.cat([(param - lr_ref*g).flatten() for param, g in zip(orig_params, orig_grads)])
    outer_ref = (next_ref - args.target_value).square().mean()
    gx_ref, glr_ref = torch.autograd.grad(outer_ref, (x_ref, lr_ref))
    logits_ref, ce_ref, outer_ref, gx_ref, glr_ref = (
        t.detach().clone() for t in (logits_ref, ce_ref, outer_ref, gx_ref, glr_ref))
    first_ref = [g.detach().clone() for g in orig_grads]
    del ordinary, orig_params, orig_grads, next_ref, x_ref, lr_ref
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Production-style manual path: forward + first-bwd + outer seed + double-bwd.
    with torch.no_grad():
        x_fused = image.repeat(1, fuse, 1, 1)
        labels = label.repeat(fuse)
        logits, tape = fused(x_fused, return_tape=True)
        d_head, d_acts, _, first_grads = fused.run_first_bwd(
            tape, labels, Fuse=fuse,
            fuse_mask_list=mask if bwd_fuse != fuse else None)
        assert len(first_grads) == 34
        ce_manual = F.cross_entropy(logits, labels)
        for branch in range(fuse):
            assert logits[branch:branch+1].shape == logits_ref.shape

        # Match the test's OUTER MSE(mean): active next parameters vs fixed scalar.
        next_params = [w - args.syn_lr*g for w, g in zip(fused_params, first_grads)]
        active_next = torch.cat([select_param(p, mask).flatten() for p in next_params])
        outer_manual = (active_next - args.target_value).square().mean()
        ddw_flat = (-args.syn_lr) * 2*(active_next - args.target_value)/active_next.numel()
        active_shapes = [select_param(p, mask).shape for p in fused_params]
        dd_tensors = [chunk.reshape(sh) for chunk, sh in zip(
            torch.split(ddw_flat, [s.numel() for s in active_shapes]), active_shapes)]
        dd_head, dd_weights = fused.pack_recovered_dd(dd_tensors, tape=tape)
        weights = None
        if bwd_fuse != fuse:
            weights = fused.pack_recovered_weights([select_param(w, mask) for w in fused_params])
        grad_full_channels = fused.run_double_bwd(
            tape=tape, d_activates_list=d_acts, dd_weights_list=dd_weights,
            d_head_tensors=d_head, dd_head_tensors=dd_head,
            Fuse=bwd_fuse, weights=weights)
        # Original image was repeated in CHANNEL; sum all active branch gradients.
        gx_manual = grad_full_channels.reshape(1, bwd_fuse, 3, 32, 32).sum(dim=1)
        active_grad = torch.cat([select_param(g, mask).flatten() for g in first_grads])
        glr_manual = -(2*(active_next-args.target_value)/active_next.numel()*active_grad).sum()

    failures = []
    for branch in range(fuse):
        metrics(f'logits branch {branch}', logits[branch:branch+1], logits_ref,
                args.rtol, args.atol, failures)
    metrics('CE loss', ce_manual, ce_ref, args.rtol, args.atol, failures)
    first_errors = []
    for name_idx, (actual, reference) in enumerate(zip(first_grads, first_ref)):
        for branch in range(fuse):
            branch_grad = actual.reshape(fuse, reference.shape[0], *reference.shape[1:])[branch]
            if not torch.allclose(branch_grad, reference, rtol=args.rtol, atol=args.atol):
                first_errors.append(f'{name_idx}:branch{branch}')
    print('first parameter grad (34 tensors):', 'PASS' if not first_errors else 'FAIL '+', '.join(first_errors))
    failures.extend(first_errors)
    metrics('outer MSE loss', outer_manual, outer_ref, args.rtol, args.atol, failures)
    metrics('image double-bwd grad', gx_manual, gx_ref, args.rtol, args.image_atol,
            failures, check_relative=True)
    metrics('synthetic LR grad', glr_manual, glr_ref, args.rtol, args.image_atol, failures)
    print('image grad sums: original=', gx_ref.sum().item(), 'manual=', gx_manual.sum().item())
    if failures:
        raise AssertionError('Manual VGG mismatch: '+', '.join(failures))
    print('PASS: real buffer, manual full-Fuse first-bwd + active double-bwd vs original VGG autograd')


if __name__ == '__main__':
    main()
