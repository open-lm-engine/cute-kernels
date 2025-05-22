# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from .torch_implementation import fused_swiglu_torch


class _FusedSwiglu_Cute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        up_weight: torch.Tensor,
        gate_weight: torch.Tensor,
        down_weight: torch.Tensor,
        up_bias: torch.Tensor | None,
        gate_bias: torch.Tensor | None,
        down_bias: torch.Tensor | None,
    ) -> torch.Tensor: ...

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None]: ...


def fused_swiglu_cute(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_bias: torch.Tensor | None,
    gate_bias: torch.Tensor | None,
    down_bias: torch.Tensor | None,
) -> torch.Tensor: ...
