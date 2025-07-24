# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import KernelBackend, p_norm_cute, set_seed

from ..test_commons import TestCommons


_EPSILON = 1e-5
_SEED = 42


def _get_sizes() -> list[tuple]:
    sizes = []
    for size in TestCommons.get_1d_tensor_sizes():
        sizes.append((size,))
        sizes.append((400, size))
    return sizes


class P_NormTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            _get_sizes(),  # size
            [torch.device("cuda")],  # device
            [torch.float32, torch.float16],  # dtype
            [True, False],  # memory_efficient
            [True, False],  # has_weight
            [2],  # p
            [p_norm_cute, torch.compile(p_norm_cute, fullgraph=True)],  # function
        )
    )
    def test_p_norm(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        memory_efficient: bool,
        has_weight: bool,
        p: int | str,
        function: Callable,
    ) -> None:
        set_seed(_SEED)

        if isinstance(size, int):
            size = (size,)

        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)

        if has_weight:
            weight_kernel, weight_expected = self.get_random_duplicated_tensors(size[-1], device=device, dtype=dtype)
        else:
            weight_kernel = None
            weight_expected = None

        z_kernel = function(x=x_kernel, p=p, weight=weight_kernel, eps=_EPSILON, memory_efficient=memory_efficient)
        z_expected = p_norm_cute(
            x=x_expected, p=p, weight=weight_expected, eps=_EPSILON, kernel_backend=KernelBackend.torch
        )

        z_kernel.sum().backward()
        z_expected.sum().backward()

        self.assert_equal_tensors(z_kernel, z_expected, False, atol_float16=1.6e-2, rtol_float16=0)

        self.assert_equal_tensors(
            x_kernel.grad,
            x_expected.grad,
            False,
            atol_float32=3.1e-5,
            rtol_float32=0,
            atol_float16=6.3e-2,
            rtol_float16=0,
        )

        if has_weight:
            self.assert_equal_tensors(
                weight_kernel.grad,
                weight_expected.grad,
                False,
                atol_float32=6.5e-5,
                rtol_float32=0,
                atol_float16=8e-3,
                rtol_float16=0,
            )
