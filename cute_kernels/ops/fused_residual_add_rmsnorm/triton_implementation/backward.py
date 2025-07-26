# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ....constants import LIBRARY_NAME, MAX_TRITON_BLOCK_SIZE
from ....math import ceil_divide, get_next_power_of_2
from ....utils import get_num_elements_and_hidden_size, get_sm_count


@triton.jit
def fused_residual_add_rmsnorm_backward_triton_kernel(
    added_x_residual_ptr,
    weight_ptr,
    output_grad_ptr,
    added_x_residual_grad_ptr,
    x_grad_ptr,
    residual_grad_ptr,
    weight_grad_ptr,
    eps,
    multiplier,
    rmsnorm_denominator_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    num_elements_per_program = tl.cdiv(B, num_programs)

    indices_h = tl.arange(0, BLOCK_SIZE_H)
    mask_h = indices_h < H

    program_start = pid * num_elements_per_program
    program_end = min(program_start + num_elements_per_program, B)
    num_elements_in_current_program = program_end - program_start

    num_loops = tl.cdiv(num_elements_in_current_program, BLOCK_SIZE_B)

    x_dtype = added_x_residual_ptr.dtype.element_ty

    if weight_ptr is not None:
        weight = tl.load(weight_ptr + indices_h, mask=mask_h)[None, :]
        weight_grad = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)

    for i in range(num_loops):
        indices_b = program_start + i * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        indices_bh = indices_b[:, None] * H + indices_h[None, :]

        mask_b = indices_b < program_end
        mask_bh = mask_b[:, None] & mask_h[None, :]

        added_x_residual = tl.load(added_x_residual_ptr + indices_bh, mask=mask_bh).to(tl.float32)

        if rmsnorm_denominator_ptr is None:
            squared_sum = tl.sum(added_x_residual * added_x_residual, axis=1)
            inverse_rms = tl.rsqrt(squared_sum / H + eps)
        else:
            inverse_rms = tl.load(rmsnorm_denominator_ptr + indices_b, mask=mask_b)

        output_grad = tl.load(output_grad_ptr + indices_bh, mask=mask_bh)

        output_grad_weight = output_grad
        if weight_ptr is not None:
            output_grad_weight *= weight

        output_grad_weight = output_grad_weight.to(tl.float32)

        x_grad = inverse_rms[:, None] * output_grad_weight
        x_grad -= (
            (1 / H)
            * inverse_rms[:, None]
            * inverse_rms[:, None]
            * inverse_rms[:, None]
            * added_x_residual
            * tl.sum(output_grad_weight * added_x_residual, axis=1, keep_dims=True)
        )

        added_x_residual_grad = tl.load(added_x_residual_grad_ptr + indices_bh, mask=mask_bh)
        x_grad += added_x_residual_grad

        tl.store(residual_grad_ptr + indices_bh, x_grad, mask=mask_bh)

        if multiplier is not None:
            x_grad *= multiplier

        tl.store(x_grad_ptr + indices_bh, x_grad, mask=mask_bh)

        if weight_ptr is not None:
            weight_grad += tl.sum(output_grad * (added_x_residual * inverse_rms[:, None]).to(x_dtype), axis=0)

    if weight_ptr is not None:
        tl.atomic_add(weight_grad_ptr + indices_h, weight_grad, mask=mask_h, sem="relaxed")


@custom_op(
    f"{LIBRARY_NAME}::fused_residual_add_rmsnorm_backward_triton",
    mutates_args={"x_grad", "residual_grad", "weight_grad"},
)
def fused_residual_add_rmsnorm_backward_triton(
    added_x_residual: torch.Tensor,
    weight: torch.Tensor | None,
    output_grad: torch.Tensor,
    added_x_residual_grad: torch.Tensor,
    rmsnorm_denominator: torch.Tensor | None,
    x_grad: torch.Tensor,
    residual_grad: torch.Tensor,
    weight_grad: torch.Tensor | None,
    eps: float,
    multiplier: float | None,
) -> None:
    B, H = get_num_elements_and_hidden_size(added_x_residual)

    BLOCK_SIZE_B = 1
    BLOCK_SIZE_H = get_next_power_of_2(H)
    assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE
    NUM_WARPS = 8

    sm_count = get_sm_count(added_x_residual.device)
    num_programs = min(sm_count, ceil_divide(B, BLOCK_SIZE_B))

    with torch.device(added_x_residual.device):
        fused_residual_add_rmsnorm_backward_triton_kernel[num_programs,](
            added_x_residual_ptr=added_x_residual,
            weight_ptr=weight,
            output_grad_ptr=output_grad,
            added_x_residual_grad_ptr=added_x_residual_grad,
            x_grad_ptr=x_grad,
            residual_grad_ptr=residual_grad,
            weight_grad_ptr=weight_grad,
            eps=eps,
            multiplier=multiplier,
            rmsnorm_denominator_ptr=rmsnorm_denominator,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            num_warps=NUM_WARPS,
        )
