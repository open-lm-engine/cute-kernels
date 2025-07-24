# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ..swiglu import swiglu_torch


def fused_swiglu_torch(
    x: torch.Tensor, gate_weight: torch.Tensor, up_weight: torch.Tensor, down_weight: torch.Tensor
) -> torch.Tensor:
    up = F.linear(x, up_weight)
    gate = F.linear(x, gate_weight)

    output = swiglu_torch(gate=gate, up=up)
    output = F.linear(output, down_weight)

    return output
