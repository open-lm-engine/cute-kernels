# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from cutlass.cute.runtime import from_dlpack


def ampere_gemm(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> None:
    A = from_dlpack(A, assumed_align=16)
    B = from_dlpack(B, assumed_align=16)
    C = from_dlpack(C, assumed_align=16)
    D = from_dlpack(D, assumed_align=16)

    print(A)
