# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


_DTYPE_MAP = {torch.float32: cute.Float32, torch.float16: cute.Float16, torch.bfloat16: cute.BFloat16}


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
