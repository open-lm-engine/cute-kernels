# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************


import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_powers_of_2
from ....triton_math import matmul, sigmoid
from ....utils import get_num_elements_and_hidden_size


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for BLOCK_SIZE_B in get_powers_of_2(64, 64):
        for BLOCK_SIZE_H_UP in get_powers_of_2(16, 256):
            for BLOCK_SIZE_H_DOWN in get_powers_of_2(16, 256):
                for NUM_STAGES_UP in get_powers_of_2(1, 4):
                    for NUM_STAGES_DOWN in get_powers_of_2(1, 4):
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_SIZE_B": BLOCK_SIZE_B,
                                    "BLOCK_SIZE_H_UP": BLOCK_SIZE_H_UP,
                                    "BLOCK_SIZE_H_DOWN": BLOCK_SIZE_H_DOWN,
                                    "NUM_STAGES_UP": NUM_STAGES_UP,
                                    "NUM_STAGES_DOWN": NUM_STAGES_DOWN,
                                },
                                num_warps=8,
                            )
                        )

    return configs


@triton.autotune(configs=_get_autotune_configs(), key=["MEMORY_EFFICIENT"], reset_to_zero=["y_ptr"])
@triton.jit
def fused_swiglu_forward_triton_kernel(
    x_ptr,
    Wg_ptr,
    Wu_ptr,
    Wd_ptr,
    y_ptr,
    g_ptr,
    u_ptr,
    B,
    H,
    I,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H_UP: tl.constexpr,
    BLOCK_SIZE_H_DOWN: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
    MEMORY_EFFICIENT: tl.constexpr,
    NUM_STAGES_UP: tl.constexpr,
    NUM_STAGES_DOWN: tl.constexpr,
):
    BLOCK_ID = tl.program_id(axis=0)

    indices_b = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_i = tl.arange(0, BLOCK_SIZE_I)

    mask_b = indices_b < B
    mask_i = indices_i < I

    u = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    g = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)

    for h in tl.range(tl.cdiv(H, BLOCK_SIZE_H_UP), num_stages=NUM_STAGES_UP):
        indices_h = h * BLOCK_SIZE_H_UP + tl.arange(0, BLOCK_SIZE_H_UP)
        mask_h = indices_h < H

        indices = indices_b[:, None] * H + indices_h[None, :]
        mask = mask_b[:, None] & mask_h[None, :]
        x = tl.load(x_ptr + indices, mask=mask)

        indices = indices_i[:, None] * H + indices_h[None, :]
        mask = mask_i[:, None] & mask_h[None, :]

        Wu = tl.load(Wu_ptr + indices, mask=mask)
        u = matmul(x, Wu.T, u, output_dtype=u.dtype)

        Wg = tl.load(Wg_ptr + indices, mask=mask)
        g = matmul(x, Wg.T, g, output_dtype=g.dtype)

    if not MEMORY_EFFICIENT:
        indices_bi = indices_b[:, None] * I + indices_i[None, :]
        mask_bi = mask_b[:, None] & mask_i[None, :]

        tl.store(g_ptr + indices_bi, g, mask=mask_bi)
        tl.store(u_ptr + indices_bi, u, mask=mask_bi)

    z = u * g * sigmoid(g)
    z = z.to(x_ptr.dtype.element_ty)

    for h in tl.range(tl.cdiv(H, BLOCK_SIZE_H_DOWN), num_stages=NUM_STAGES_DOWN):
        indices_h = h * BLOCK_SIZE_H_DOWN + tl.arange(0, BLOCK_SIZE_H_DOWN)
        mask_h = indices_h < H

        indices = indices_h[:, None] * I + indices_i[None, :]
        mask = mask_h[:, None] & mask_i[None, :]

        Wd = tl.load(Wd_ptr + indices, mask=mask)
        y = matmul(z, Wd.T, None, output_dtype=z.dtype)

        mask = mask_b[:, None] & mask_h[None, :]
        indices = indices_b[:, None] * H + indices_h[None, :]

        tl.store(y_ptr + indices, y, mask=mask)


@custom_op(f"{LIBRARY_NAME}::fused_swiglu_forward_triton", mutates_args={"gate", "up", "output"})
def fused_swiglu_forward_triton(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate: torch.Tensor | None,
    up: torch.Tensor | None,
    output: torch.Tensor,
    memory_efficient: bool,
) -> None:
    B, H = get_num_elements_and_hidden_size(x)
    I = down_weight.size(1)

    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]),)

    with torch.device(x.device):
        fused_swiglu_forward_triton_kernel[GRID](
            x_ptr=x,
            Wg_ptr=gate_weight,
            Wu_ptr=up_weight,
            Wd_ptr=down_weight,
            y_ptr=output,
            g_ptr=gate,
            u_ptr=up,
            B=B,
            H=H,
            I=I,
            BLOCK_SIZE_I=I,
            MEMORY_EFFICIENT=memory_efficient,
        )
