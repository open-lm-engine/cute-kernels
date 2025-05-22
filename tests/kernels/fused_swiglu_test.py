# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import CutoTuneParameter, KernelBackend, fused_swiglu_cute, fused_swiglu_torch

from ..test_commons import TestCommons


class FusedSwiGLUTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [fused_swiglu_cute],  # , torch.compile(fused_swiglu_cute, fullgraph=True)],  # function
        )
    )
    def test_swiglu(self, size: tuple[int], device: torch.device, dtype: torch.dtype, function: Callable) -> None:
        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)

        gate_weight_kernel, gate_weight_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        up_weight_kernel, up_weight_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        down_weight_kernel, down_weight_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)

        z_kernel = function(
            x=x_kernel,
            gate_weight=gate_weight_kernel,
            up_weight=up_weight_kernel,
            down_weight=down_weight_kernel,
        )

        z_expected = fused_swiglu_torch(
            x=x_expected,
            gate_weight=gate_weight_expected,
            up_weight=up_weight_expected,
            down_weight=down_weight_expected,
        )

        # z_kernel.mean().backward()
        # z_expected.mean().backward()

        self.assert_equal_tensors(z_kernel, z_expected, False, atol_float32=5.5e-6, rtol_float32=0)
        # self.assert_equal_tensors(x_kernel.grad, x_expected.grad, False, atol_float32=5e-6, rtol_float32=0)
        # self.assert_equal_tensors(y_kernel.grad, y_expected.grad, False, atol_float32=5e-6, rtol_float32=0)
