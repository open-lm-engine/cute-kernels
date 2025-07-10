# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum


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
        A_major = LayoutEnum.ROW_MAJOR
        B_major = LayoutEnum.ROW_MAJOR
        C_major = LayoutEnum.ROW_MAJOR
        D_major = LayoutEnum.ROW_MAJOR

        copy_bits = 128

        sA_layout = self._make_shared_memory_layout_AB(
            self.A_dtype, A_major, copy_bits, (self.cta_tiler[0], self.cta_tiler[2], self.num_stages)
        )

        sB_layout = self._make_shared_memory_layout_AB(
            self.B_dtype, B_major, copy_bits, (self.cta_tiler[1], self.cta_tiler[2], self.num_stages)
        )

        sC_layout = self._make_smem_layout_C(self.C_dtype, C_major, copy_bits, (self.cta_tiler[0], self.cta_tiler[1]))
        sD_layout = self._make_smem_layout_C(self.D_dtype, D_major, copy_bits, (self.cta_tiler[0], self.cta_tiler[1]))

        # Shared memory allocated for operations with A, B will be
        # overwritten for operations on C. This is to improve performance
        # by reducing the size of shared memory requested by each block
        smem_size = max(
            cute.size_in_bytes(mC.element_type, sC_layout),
            cute.size_in_bytes(mA.element_type, sA_layout) + cute.size_in_bytes(mB.element_type, sB_layout),
        )

    def _make_shared_memory_layout_AB(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = smem_tiler[1] if major_mode == LayoutEnum.ROW_MAJOR else smem_tiler[0]
        major_mode_size = min(64, major_mode_size)

        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)

        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )

        layout_atom = cute.make_composed_layout(cute.make_swizzle(swizzle_bits, 3, 3), 0, layout_atom_outer)

        layout = cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))

        return layout

    def _make_smem_layout_C(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = smem_tiler[1] if major_mode == LayoutEnum.ROW_MAJOR else smem_tiler[0]

        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)

        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )

        if major_mode == LayoutEnum.ROW_MAJOR:
            layout_atom = cute.make_composed_layout(cute.make_swizzle(swizzle_bits, 3, 4), 0, layout_atom_outer)
        else:
            # Due to the thread layout of the mma, remove swizzle in C to
            # prevent shared memory fragments owned by an single thread from
            # holding swizzles
            layout_atom = cute.make_composed_layout(cute.make_swizzle(0, 3, 4), 0, layout_atom_outer)

        layout = cute.tile_to_shape(layout_atom, smem_tiler, (0, 1))

        return layout


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

    gemm = AmpereGemm(A_dtype=A_dtype, B_dtype=B_dtype, C_dtype=C_dtype, D_dtype=D_dtype)
    cute.compile(gemm, A, B, C, D)

    gemm(A, B, C, D)
