# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...kernel_backend import KernelBackend
from ..swiglu import swiglu_cute


def fused_swiglu_torch(
    x: torch.Tensor, gate_weight: torch.Tensor, up_weight: torch.Tensor, down_weight: torch.Tensor
) -> torch.Tensor:
    up = F.linear(x, up_weight)
    gate = F.linear(x, gate_weight)

    output = swiglu_cute(
        gate=gate, up=up, kernel_backend_forward=KernelBackend.torch, kernel_backend_backward=KernelBackend.torch
    )

    output = F.linear(output, down_weight)

    return output
