# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************


import triton
import triton.language as tl

from ...triton_math import sigmoid


@triton.jit
def fused_swiglu_triton_kernel(
    x_ptr,
    Wu_ptr,
    Wg_ptr,
    Wd_ptr,
    y_ptr,
    B,
    H,
    I,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
):
    BLOCK_ID = tl.program_id(axis=0)
    NUM_BLOCKS_I = tl.cdiv(I, BLOCK_SIZE_I)

    BLOCK_ID_B = BLOCK_ID // NUM_BLOCKS_I
    BLOCK_ID_I = BLOCK_ID % NUM_BLOCKS_I

    indices_b = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_i = BLOCK_ID_I * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)

    mask_b = indices_b < B
    mask_i = indices_i < I

    zu = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    zg = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)

    for h in range(tl.cdiv(H, BLOCK_SIZE_H)):
        indices_h = h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
        mask_h = indices_h < H

        mask_x = mask_b[:, None] & mask_h[None, :]
        indices_x = indices_b[:, None] * H + indices_h[None, :]
        x = tl.load(x_ptr + indices_x, mask=mask_x)

        mask_W = mask_i[:, None] & mask_h[None, :]
        indices_W = indices_i[:, None] * H + indices_h[None, :]

        Wu = tl.load(Wu_ptr + indices_W, mask=mask_W)
        zu = tl.dot(x, Wu.T, zu)

        Wg = tl.load(Wg_ptr + indices_W, mask=mask_W)
        zg = tl.dot(x, Wg.T, zg)

    z = zu * zg * sigmoid(zg)
    z = z.to(x_ptr.dtype.element_ty)

    for h in range(tl.cdiv(H, BLOCK_SIZE_H)):
        mask_W = mask_h[:, None] & mask_i[None, :]
        indices_W = indices_h[:, None] * I + indices_i[None, :]

        Wd = tl.load(Wd_ptr + indices_W, mask=mask_W)
        y = tl.dot(z, Wd.T)

        indices = indices_b[:, None] * H + indices_h[None, :]
        tl.atomic_add(y_ptr + indices, y, mask=mask_b[:, None] & mask_h[None, :])
