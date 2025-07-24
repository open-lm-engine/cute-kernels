# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...kernel_backend import KernelBackend
from ...utils import ensure_contiguous
from ..swiglu import swiglu_cute
from .triton_implementation import fused_swiglu_forward_triton


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
        dtype = torch.float32 if x.dtype == torch.bfloat16 else x.dtype

        output = torch.zeros_like(x, dtype=dtype)
        gate = None if memory_efficient else torch.empty(x.size(0), up_weight.size(0), device=x.device, dtype=dtype)
        up = None if memory_efficient else torch.empty(x.size(0), up_weight.size(0), device=x.device, dtype=dtype)

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
        ctx.memory_efficient = memory_efficient

        output = output.type_as(x)

        return output


def fused_swiglu_cute(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    memory_efficient: bool = False,
    *,
    kernel_backend_forward: KernelBackend = KernelBackend.triton,
    kernel_backend_backward: KernelBackend = KernelBackend.triton,
) -> torch.Tensor:
    if kernel_backend_forward == KernelBackend.triton:
        assert kernel_backend_backward == KernelBackend.triton
        x = _FusedSwiglu_Cute.apply(x, gate_weight, up_weight, down_weight, memory_efficient)
    elif kernel_backend_forward == KernelBackend.torch:
        assert kernel_backend_backward == KernelBackend.torch
        assert not memory_efficient

        up = F.linear(x, up_weight)
        gate = F.linear(x, gate_weight)

        x = swiglu_cute(
            gate=gate,
            up=up,
            kernel_backend_forward=kernel_backend_forward,
            kernel_backend_backward=kernel_backend_backward,
        )

        x = F.linear(x, down_weight)

    return x
