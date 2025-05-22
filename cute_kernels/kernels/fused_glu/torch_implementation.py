# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ..swiglu import swiglu_torch


def fused_swiglu_torch(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_bias: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    down_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    up = F.linear(x, up_weight, up_bias)
    gate = F.linear(x, gate_weight, gate_bias)

    output = swiglu_torch(gate=gate, up=up)
    output = F.linear(output, down_weight, down_bias)

    return output
