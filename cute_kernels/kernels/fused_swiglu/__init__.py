# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from .torch_implementation import fused_swiglu_torch
from .triton_implementation import fused_swiglu_triton


class _FusedSwiglu_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        memory_efficient: bool,
    ) -> torch.Tensor:
        intermediate_state = torch.empty(x.size(0), up_weight.size(0))
        if memory_efficient:
            pass

        output = torch.zeros_like(x, dtype=torch.float32 if x.dtype == torch.bfloat16 else x.dtype)

        fused_swiglu_triton(x=x, gate_weight=gate_weight, up_weight=up_weight, down_weight=down_weight, output=output)

        ctx.save_for_backward(up_weight, gate_weight, down_weight)

        return output.type_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None]:
        up_weight, gate_weight, down_weight = ctx.saved_tensors
        return grad_output, up_weight, gate_weight, down_weight, None


def fused_swiglu_cute(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    memory_efficient: bool = False,
) -> torch.Tensor:
    return _FusedSwiglu_Cute.apply(x, gate_weight, up_weight, down_weight, memory_efficient)
