# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import KernelBackend, bmm_cute, set_seed

from ..test_commons import TestCommons


_SEED = 42


class BMMTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [False, True],  # is_A_transposed
            [False, True],  # is_B_transposed
            [False, True],  # has_C
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [bmm_cute, torch.compile(bmm_cute, fullgraph=True)],  # function
        )
    )
    def test_bmm(
        self,
        size: tuple[int],
        is_A_transposed: bool,
        is_B_transposed: bool,
        has_C: bool,
        device: torch.device,
        dtype: torch.dtype,
        function: Callable,
    ) -> None:
        set_seed(_SEED)

        std = 0.02

        L = 7
        M = 417

        A = (
            torch.randn(
                (L, size[0], M) if is_A_transposed else (L, M, size[0]),
                device=device,
                dtype=dtype,
                requires_grad=False,
            )
            * std
        )
        B = (
            torch.randn(
                (L, size[1], size[0]) if is_B_transposed else (L, *size),
                device=device,
                dtype=dtype,
                requires_grad=False,
            )
            * std
        )
        C = torch.randn(L, M, size[1], device=device, dtype=dtype, requires_grad=False) * std if has_C else None

        alpha = 0.3
        beta = 0.7 if has_C else 0

        output_kernel = function(
            A=A,
            B=B,
            C=C,
            is_A_transposed=is_A_transposed,
            is_B_transposed=is_B_transposed,
            alpha=alpha,
            beta=beta,
        )
        output_expected = bmm_cute(
            A=A,
            B=B,
            C=C,
            alpha=alpha,
            beta=beta,
            is_A_transposed=is_A_transposed,
            is_B_transposed=is_B_transposed,
            kernel_backend=KernelBackend.torch,
        )

        self.assert_equal_tensors(
            output_kernel,
            output_expected,
            False,
            atol_float32=7e-5,
            rtol_float32=1e-4,
            atol_float16=1e-4,
            rtol_float16=5e-3,
            atol_bfloat16=1e-3,
            rtol_bfloat16=7e-3,
        )
