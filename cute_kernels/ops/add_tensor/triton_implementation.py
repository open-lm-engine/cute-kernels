# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ...constants import LIBRARY_NAME
from ...math import ceil_divide


@triton.jit
def _add_tensor(x_ptr, y_ptr, output_ptr, indices, mask):
    x = tl.load(x_ptr + indices, mask=mask)
    y = tl.load(y_ptr + indices, mask=mask)
    tl.store(output_ptr + indices, x + y, mask=mask)


@triton.jit
def add_tensor_triton_kernel(x_ptr, y_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    BLOCK_ID = tl.program_id(axis=0)
    NUM_BLOCKS = tl.num_programs(axis=0)

    indices = BLOCK_ID * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if BLOCK_ID < NUM_BLOCKS - 1:
        _add_tensor(x_ptr=x_ptr, y_ptr=y_ptr, output_ptr=output_ptr, indices=indices, mask=None)
    else:
        _add_tensor(x_ptr=x_ptr, y_ptr=y_ptr, output_ptr=output_ptr, indices=indices, mask=indices < N)


@custom_op(f"{LIBRARY_NAME}::add_tensor_triton", mutates_args={"output"})
def add_tensor_triton(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor) -> None:
    N = x.numel()
    BLOCK_SIZE = 4096
    NUM_WARPS = 32

    with torch.device(x.device):
        add_tensor_triton_kernel[ceil_divide(N, BLOCK_SIZE),](
            x_ptr=x, y_ptr=y, output_ptr=output, N=N, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS
        )
