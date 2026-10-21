"""Compare original VGG11 and VGG11_Flexfuse with ONE REAL expert snapshot.

This tests PyTorch autograd's end-to-end MTT VJP, not the manual VGG double-bwd.
Run from the repository root (or put this script in your repo's test/ directory).
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

# Allow execution from a subdirectory while keeping the project imports intact.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from networks.networks import VGG11
from networks.networks_Fuse import VGG11_Flexfuse
from reparam_module import ReparamModule


def output_only(value):
    return value[0] if isinstance(value, tuple) else value


def load_expert_snapshot(path, expert_index, epoch, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Buffer file not found: {path}')
    # Replay buffers normally contain lists of tensor snapshots rather than a
    # pure state_dict. Only load buffers you trust (weights_only=False unpickles).
    replay = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(replay, (list, tuple)) or not replay:
        raise ValueError('Expected buffer = [expert_trajectory, ...]')
    if not 0 <= expert_index < len(replay):
        raise IndexError(f'expert-index {expert_index} not in [0, {len(replay)})')
    trajectory = replay[expert_index]
    if not isinstance(trajectory, (list, tuple)) or not 0 <= epoch < len(trajectory):
        raise IndexError(f'epoch {epoch} invalid for selected trajectory')
    snapshot = trajectory[epoch]
    if not isinstance(snapshot, (list, tuple)) or not snapshot or not all(torch.is_tensor(p) for p in snapshot):
        raise ValueError('Expected a snapshot = [parameter_tensor, ...]')
    return [p.detach().to(device) for p in snapshot]


def install_and_check_parameters(original, fused, snapshot, fuse):
    original_params = list(original.named_parameters())
    fused_params = list(fused.named_parameters())
    print(f'Parameter tensor count: buffer={len(snapshot)}, original={len(original_params)}, fused={len(fused_params)}')
    if not (len(original_params) == len(fused_params) == len(snapshot)):
        raise AssertionError('Parameter tensor count mismatch: check cfg_vgg/norm/classifier')

    # Check modules without parameters too: pool/activation choices must agree.
    orig_features = list(original.features)
    fused_features = list(fused.features)
    if len(orig_features) != len(fused_features):
        raise AssertionError('features module count mismatch')
    for i, (a, b) in enumerate(zip(orig_features, fused_features)):
        if type(a) is not type(b):
            raise AssertionError(f'features[{i}]: {type(a).__name__} != {type(b).__name__}')
        if isinstance(a, torch.nn.Conv2d):
            if a.groups != 1 or b.groups != fuse:
                raise AssertionError(f'features[{i}]: incorrect conv groups')
            for key in ('kernel_size', 'stride', 'padding', 'dilation'):
                if getattr(a, key) != getattr(b, key):
                    raise AssertionError(f'features[{i}].{key} mismatch')
        if isinstance(a, torch.nn.GroupNorm):
            if b.num_channels != a.num_channels * fuse or b.num_groups != a.num_groups * fuse:
                raise AssertionError(f'features[{i}]: GroupNorm channels/groups mismatch')
            if a.eps != b.eps:
                raise AssertionError(f'features[{i}]: GroupNorm eps mismatch')
        if isinstance(a, (torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
            for key in ('kernel_size', 'stride', 'padding', 'ceil_mode'):
                if getattr(a, key) != getattr(b, key):
                    raise AssertionError(f'features[{i}].{key} mismatch')

    with torch.no_grad():
        for i, (p, (oname, op), (fname, fp)) in enumerate(zip(snapshot, original_params, fused_params)):
            expected_fused = (p.shape[0] * fuse, *p.shape[1:]) if p.ndim else ()
            if tuple(op.shape) != tuple(p.shape) or tuple(fp.shape) != expected_fused:
                raise AssertionError(
                    f'[{i:02d}] buffer={tuple(p.shape)}, original {oname}={tuple(op.shape)}, '
                    f'fused {fname}={tuple(fp.shape)}, expected fused={expected_fused}'
                )
            # Copy by POSITION, exactly like the existing ResNet comparison.
            # Convs: grouped output filters; GN: affine channels; Linear: class rows.
            op.copy_(p)
            fp.copy_(p.repeat((fuse,) + (1,) * (p.ndim - 1)))
            if not torch.equal(op, p):
                raise AssertionError(f'Original copy failed at index {i}')
            if not torch.equal(fp, p.repeat((fuse,) + (1,) * (p.ndim - 1))):
                raise AssertionError(f'Fused copy failed at index {i}')
            print(f'[{i:02d}] {oname:<35} {str(tuple(op.shape)):<20} -> {fname:<35} {tuple(fp.shape)}')


def selected_flat(flat, original_shapes, mask):
    """Parameter-major, then branch-major; matches .repeat(Fuse, ...) per tensor."""
    fuse = len(mask)
    offset = 0
    chunks = []
    for shape in original_shapes:
        n = shape.numel()
        for f, is_active in enumerate(mask):
            if is_active:
                chunks.append(flat[offset + f*n: offset + (f+1)*n])
        offset += fuse*n
    if offset != flat.numel():
        raise AssertionError('Flat parameter layout mismatch')
    return torch.cat(chunks)


def relative_l2(a, b):
    a = a.detach().double().reshape(-1)
    b = b.detach().double().reshape(-1)
    return (a-b).norm().item() / max(b.norm().item(), 1e-30)


def report(name, actual, reference, rtol, atol):
    a = actual.detach().cpu()
    b = reference.detach().cpu()
    if a.shape != b.shape:
        raise AssertionError(f'{name}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}')
    delta = (a.double()-b.double()).abs()
    print(f'{name:<21} ref={b.reshape(-1)[:1].item(): .9e} '
          f'fused={a.reshape(-1)[:1].item(): .9e} '
          f'max_abs={delta.max().item():.3e} rel_L2={relative_l2(a, b):.3e}')
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol, msg=f'Mismatch at {name}')


def run_one(model, image, label, lr_value, target_value, mask, fuse, device):
    """One synthetic SGD step + outer MSE + VJP to the SAME unfused image."""
    wrapped = ReparamModule(model).to(device)
    wrapped.train()
    original_shapes = [torch.Size((p.shape[0] // fuse, *p.shape[1:])) for p in model.parameters()]
    flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).requires_grad_(True)
    x = image.detach().clone().requires_grad_(True)
    syn_lr = torch.tensor(lr_value, device=device, dtype=x.dtype, requires_grad=True)

    # Fuse input along C, matching distill's repeat(1, Fuse, 1, 1).
    x_for_net = x.repeat(1, fuse, 1, 1)
    labels = torch.tensor([label], dtype=torch.long, device=device).repeat(fuse)
    # Same plain-vs-ReparamModule check as in the supplied ResNet test.
    with torch.no_grad():
        plain_y = output_only(model(x_for_net))
        reparam_y = output_only(wrapped(x_for_net, flat_param=flat))
        torch.testing.assert_close(plain_y, reparam_y, rtol=1e-6, atol=1e-6,
                                   msg=f'Plain vs ReparamModule differs (Fuse={fuse})')
        print(f'plain vs ReparamModule (Fuse={fuse}) max abs: '
              f'{(plain_y - reparam_y).abs().max().item():.3e}')
    logits = output_only(wrapped(x_for_net, flat_param=flat))
    expected_shape = (fuse, model.num_classes) if hasattr(model, 'num_classes') else (fuse, logits.shape[-1])
    if tuple(logits.shape) != expected_shape:
        raise AssertionError(f'logits shape {tuple(logits.shape)}, expected {expected_shape}')

    # Multiply by Fuse so each branch sees the SAME CE/parameter gradient
    # as a standalone original network, instead of being divided by Fuse.
    ce_mean = F.cross_entropy(logits, labels)
    ce_for_backward = ce_mean * fuse
    (grad_flat,) = torch.autograd.grad(ce_for_backward, flat, create_graph=True)
    flat_next = flat - syn_lr * grad_flat

    if fuse == 1:
        active_next = flat_next
        active_grad = grad_flat
    else:
        active_next = selected_flat(flat_next, original_shapes, mask)
        active_grad = selected_flat(grad_flat, original_shapes, mask)

    # Mean reduction makes baseline and multiple identical active branches
    # comparable. A SUM would scale the outer gradient by active branch count.
    fixed_target = torch.full_like(active_next, target_value)
    outer_loss = F.mse_loss(active_next, fixed_target, reduction='mean')
    grad_image, grad_lr = torch.autograd.grad(outer_loss, (x, syn_lr))
    return {
        'logits': logits.detach().cpu(),
        'ce': ce_mean.detach().cpu(),
        'outer': outer_loss.detach().cpu(),
        'image_grad': grad_image.detach().cpu(),
        'lr_grad': grad_lr.detach().cpu(),
        'first_grad': active_grad.detach().cpu(),
    }


def main():
    p = argparse.ArgumentParser(description='Real-buffer VGG11 original vs FlexFuse: CE/MSE/image double-bwd')
    p.add_argument('--buffer', required=True, help='Actual VGG11 replay_buffer_0.pt path')
    p.add_argument('--expert-index', type=int, default=0)
    p.add_argument('--epoch', type=int, default=0)
    p.add_argument('--channel', type=int, default=3)
    p.add_argument('--num-classes', type=int, default=10)
    p.add_argument('--image-size', type=int, default=32)
    p.add_argument('--label', type=int, default=0)
    p.add_argument('--image-pt', default=None, help='Optional saved synthetic-image tensor, e.g. ./test_script/script/in_ip10.pt')
    p.add_argument('--image-index', type=int, default=0)
    p.add_argument('--fuse-mask-list', '--fuse_mask_list', type=int, nargs='+', default=[1, 0], dest='fuse_mask_list')
    p.add_argument('--syn-lr', type=float, default=0.01)
    p.add_argument('--target-value', type=float, default=0.0, help='Fixed scalar MSE target')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--rtol', type=float, default=2e-3)
    p.add_argument('--atol', type=float, default=1e-6)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = p.parse_args()
    mask = args.fuse_mask_list
    if not mask or any(v not in (0, 1) for v in mask) or sum(mask) < 1:
        p.error('--fuse_mask_list must be a nonempty 0/1 mask with at least one active branch')
    if not 0 <= args.label < args.num_classes:
        p.error('--label must be within the class range')
    fuse = len(mask)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable')
    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else
                          ('cpu' if args.device == 'auto' else args.device))
    print(f'device={device}, mask={mask}, active={sum(mask)}, CE scaling=Fuse, MSE reduction=mean')
    print(f'buffer={args.buffer}, expert={args.expert_index}, epoch={args.epoch}')
    snapshot = load_expert_snapshot(args.buffer, args.expert_index, args.epoch, device)
    ordinary = VGG11(args.channel, args.num_classes).to(device).train()
    fused = VGG11_Flexfuse(args.channel, args.num_classes, Fuse=fuse).to(device).train()
    install_and_check_parameters(ordinary, fused, snapshot, fuse)
    base_shapes = [p.shape for p in ordinary.parameters()]
    del snapshot

    # Identical original synthetic image for BOTH calls; no optimizer.step().
    if args.image_pt is None:
        image = torch.randn(1, args.channel, args.image_size, args.image_size, device=device)
    else:
        saved = torch.load(args.image_pt, map_location='cpu', weights_only=True)
        if not torch.is_tensor(saved) or saved.ndim != 4 or not 0 <= args.image_index < saved.size(0):
            raise ValueError('--image-pt must contain an [N,C,H,W] tensor and a valid --image-index')
        image = saved[args.image_index:args.image_index + 1].detach().to(device)
        if tuple(image.shape[1:]) != (args.channel, args.image_size, args.image_size):
            raise ValueError(f'Selected image has shape {tuple(image.shape)}, expected [1,{args.channel},{args.image_size},{args.image_size}]')
    original = run_one(ordinary, image, args.label, args.syn_lr, args.target_value,
                       mask=[1], fuse=1, device=device)
    del ordinary
    if device.type == 'cuda':
        torch.cuda.empty_cache()  # not needed for correctness
    fusion = run_one(fused, image, args.label, args.syn_lr, args.target_value,
                     mask=mask, fuse=fuse, device=device)
    del fused

    print('\n=== Result: original vs fused ===')
    # Each fused branch has its own logits. Check every one, even masked-out branches,
    # since first-bwd MUST run over all branches.
    for branch in range(fuse):
        report(f'logits branch {branch}', fusion['logits'][branch:branch+1], original['logits'], args.rtol, args.atol)
    report('CE loss', fusion['ce'], original['ce'], args.rtol, args.atol)
    # Repeat EACH reference gradient tensor along its first dimension to match
    # the parameter-major / branch-major flattened layout.
    orig_grad = original['first_grad']
    grad_per_tensor = torch.split(orig_grad, [shape.numel() for shape in base_shapes])
    expected_first_grad = torch.cat([g.repeat(sum(mask)) for g in grad_per_tensor])
    report('first parameter grad', fusion['first_grad'], expected_first_grad, args.rtol, args.atol)
    report('outer MSE loss', fusion['outer'], original['outer'], args.rtol, args.atol)
    report('image gradient', fusion['image_grad'], original['image_grad'], args.rtol, args.atol)
    # A 1e-6 absolute tolerance can hide errors when image gradients are tiny.
    # Also require a small relative error over the full image-gradient vector.
    image_relative_error = relative_l2(fusion['image_grad'], original['image_grad'])
    if image_relative_error > args.rtol:
        raise AssertionError(f'image gradient relative L2 error {image_relative_error:.3e} > {args.rtol:.3e}')
    report('synthetic LR grad', fusion['lr_grad'], original['lr_grad'], args.rtol, args.atol)
    print('image grad sums: original=', original['image_grad'].sum().item(),
          'fused=', fusion['image_grad'].sum().item())
    print('PASS: loaded buffer -> original/FlexFuse, CE, parameter MSE, and image double-bwd VJP')


if __name__ == '__main__':
    main()
