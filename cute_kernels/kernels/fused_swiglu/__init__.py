# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from .torch_implementation import fused_swiglu_torch
from .triton_implementation import fused_swiglu_triton


class _FusedSwiglu_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, up_weight: torch.Tensor, gate_weight: torch.Tensor, down_weight: torch.Tensor
    ) -> torch.Tensor:
        output = torch.zeros_like(x)
        fused_swiglu_triton(x=x, gate_weight=gate_weight, up_weight=up_weight, down_weight=down_weight, output=output)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None]: ...


def fused_swiglu_cute(
    x: torch.Tensor, up_weight: torch.Tensor, gate_weight: torch.Tensor, down_weight: torch.Tensor
) -> torch.Tensor:
    return _FusedSwiglu_Cute.apply(x, up_weight, gate_weight, down_weight)
