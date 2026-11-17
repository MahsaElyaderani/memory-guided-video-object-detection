from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def profile_conv_linear(module, forward):
    total, handles = [0], []
    modes = [(child, child.training) for child in module.modules()]
    def hook(child, inputs, output):
        if isinstance(child, nn.Conv2d):
            total[0] += output.numel() * (child.in_channels // child.groups) * child.kernel_size[0] * child.kernel_size[1]
        elif isinstance(child, nn.Linear):
            total[0] += output.numel() * child.in_features
    module.eval()
    try:
        for child in module.modules():
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                handles.append(child.register_forward_hook(hook))
        forward()
    finally:
        for handle in handles:
            handle.remove()
        for child, mode in modes:
            child.training = mode
    return total[0]


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
