# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....jit import cpp_jit


_FORWARD_KERNEL_NAME = "topk_softmax_forward_cuda"


@custom_op(f"{LIBRARY_NAME}::{_FORWARD_KERNEL_NAME}", mutates_args={"output"})
@cpp_jit()
def topk_softmax_forward_cuda(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
) -> None: ...
