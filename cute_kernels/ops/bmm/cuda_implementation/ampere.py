# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


_DTYPE_MAP = {torch.float32: cute.Float32, torch.float16: cute.Float16, torch.bfloat16: cute.BFloat16}


class AmpereGemm:
    def __init__(
        self,
        A_dtype: type[cutlass.Numeric],
        B_dtype: type[cutlass.Numeric],
        C_dtype: type[cutlass.Numeric],
        D_dtype: type[cutlass.Numeric],
        cta_tiler: tuple[int, int, int] = (128, 128, 32),
        atom_layout: tuple[int, int, int] = (2, 2, 1),
        mma_instruction_shape: tuple[int, int, int] = (16, 8, 16),
        num_stages: int = 3,
    ) -> AmpereGemm:
        self.A_dtype = A_dtype
        self.B_dtype = B_dtype
        self.C_dtype = C_dtype
        self.D_dtype = D_dtype

        self.cta_tiler = cta_tiler
        self.atom_layout = atom_layout
        self.mma_instruction_shape = mma_instruction_shape
        self.num_stages = num_stages

        self.num_warps = atom_layout[0] * atom_layout[1] * atom_layout[2]
        self.num_threads = self.num_warps * 32

        assert atom_layout[2] == 1

        for i in range(3):
            assert cta_tiler[i] % (atom_layout[i] * mma_instruction_shape[i]) == 0

        assert self.num_stages >= 3

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mD: cute.Tensor) -> None:
        return
        # A_major = Layo


def ampere_gemm(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> None:
    A_dtype = _DTYPE_MAP[A.dtype]
    B_dtype = _DTYPE_MAP[B.dtype]
    C_dtype = _DTYPE_MAP[C.dtype]
    D_dtype = _DTYPE_MAP[D.dtype]

    A = from_dlpack(A, assumed_align=16)
    B = from_dlpack(B, assumed_align=16)
    C = from_dlpack(C, assumed_align=16)
    D = from_dlpack(D, assumed_align=16)

    A = A.mark_layout_dynamic(leading_dim=2)
    B = B.mark_layout_dynamic(leading_dim=2)
    C = C.mark_layout_dynamic(leading_dim=2)
    D = D.mark_layout_dynamic(leading_dim=2)

    A = A.mark_compact_shape_dynamic(mode=2, stride_order=(0, 1, 2), divisibility=128 // A_dtype.width)
    B = B.mark_compact_shape_dynamic(mode=2, stride_order=(0, 1, 2), divisibility=128 // B_dtype.width)
    C = C.mark_compact_shape_dynamic(mode=2, stride_order=(0, 1, 2), divisibility=128 // C_dtype.width)
    D = D.mark_compact_shape_dynamic(mode=2, stride_order=(0, 1, 2), divisibility=128 // D_dtype.width)

    print(A)
