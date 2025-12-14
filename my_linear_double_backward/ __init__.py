import torch
import my_linear_double_backward  # load extension to register op

def linear_double_backward(*args):
    return torch.ops.autograd.linear_double_backward(*args)

__all__ = ["linear_double_backward"]
