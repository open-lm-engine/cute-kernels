# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...utils import ensure_contiguous
from .torch_implementation import fused_swiglu_torch
from .triton_implementation import fused_swiglu_backward_triton, fused_swiglu_forward_triton


class _FusedSwiglu_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        memory_efficient: bool,
    ) -> torch.Tensor:
        output = torch.zeros_like(x, dtype=torch.float32 if x.dtype == torch.bfloat16 else x.dtype)
        gate = None if memory_efficient else torch.empty(x.size(0), up_weight.size(0))
        up = None if memory_efficient else torch.empty(x.size(0), up_weight.size(0))

        fused_swiglu_forward_triton(
            x=x,
            gate_weight=gate_weight,
            up_weight=up_weight,
            down_weight=down_weight,
            gate=gate,
            up=up,
            output=output,
            memory_efficient=memory_efficient,
        )

        ctx.save_for_backward(up_weight, gate_weight, down_weight, gate, up)

        output = output.type_as(x)

        return output

    @staticmethod
    @ensure_contiguous
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
