# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import fused_swiglu_cute, fused_swiglu_torch, set_seed

from ..test_commons import TestCommons


_SEED = 42


class FusedSwiGLUTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [False, True],  # memory_efficient
            [fused_swiglu_cute],  # , torch.compile(fused_swiglu_cute, fullgraph=True)],  # function
        )
    )
    def test_swiglu(
        self, size: tuple[int], device: torch.device, dtype: torch.dtype, memory_efficient: bool, function: Callable
    ) -> None:
        set_seed(_SEED)

        std = 0.02
        intermediate_size = 4107

        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype, std=std)

        gate_weight_kernel, gate_weight_expected = self.get_random_duplicated_tensors(
            (intermediate_size, size[1]), device=device, dtype=dtype, std=std
        )
        up_weight_kernel, up_weight_expected = self.get_random_duplicated_tensors(
            (intermediate_size, size[1]), device=device, dtype=dtype, std=std
        )
        down_weight_kernel, down_weight_expected = self.get_random_duplicated_tensors(
            (size[1], intermediate_size), device=device, dtype=dtype, std=std
        )

        z_kernel = function(
            x=x_kernel,
            gate_weight=gate_weight_kernel,
            up_weight=up_weight_kernel,
            down_weight=down_weight_kernel,
            memory_efficient=memory_efficient,
        )

        z_expected = fused_swiglu_torch(
            x=x_expected,
            gate_weight=gate_weight_expected,
            up_weight=up_weight_expected,
            down_weight=down_weight_expected,
        )

        # z_kernel.mean().backward()
        # z_expected.mean().backward()

        self.assert_equal_tensors(
            z_kernel,
            z_expected,
            False,
            atol_float32=3e-5,
            rtol_float32=0,
            atol_float16=6e-5,
            rtol_float16=0,
            atol_bfloat16=6.2e-5,
            rtol_bfloat16=0,
        )
        # self.assert_equal_tensors(x_kernel.grad, x_expected.grad, False, atol_float32=5e-6, rtol_float32=0)
        # self.assert_equal_tensors(y_kernel.grad, y_expected.grad, False, atol_float32=5e-6, rtol_float32=0)
