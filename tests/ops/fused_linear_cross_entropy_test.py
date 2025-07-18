# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import random
from typing import Callable

import torch
from parameterized import parameterized

from cute_kernels import KernelBackend, fused_linear_cross_entropy_cute, set_seed

from ..test_commons import TestCommons


_SEED = 42


class FusedLinearCrossEntropyTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            [torch.float32, torch.bfloat16],  # dtype
            [None, 0.7],  # logits_multiplier
            [
                fused_linear_cross_entropy_cute,
                torch.compile(fused_linear_cross_entropy_cute, fullgraph=True),
            ],  # function
        )
    )
    def test_fused_linear_cross_entropy(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        logits_multiplier: float | None,
        function: Callable,
    ) -> None:
        set_seed(_SEED)

        if isinstance(size, int):
            size = (size,)

        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype, std=0.02)

        vocab_size = random.randint(max(100, size[0] - 100), size[0] + 100)
        weight_kernel, weight_expected = self.get_random_duplicated_tensors(
            (vocab_size, size[1]), device=device, dtype=dtype, std=2e-3
        )

        labels = torch.randint(0, vocab_size, (x_kernel.size(0),), device=x_kernel.device)

        loss_kernel = function(x=x_kernel, weight=weight_kernel, labels=labels, logits_multiplier=logits_multiplier)
        loss_expected = fused_linear_cross_entropy_cute(
            x=x_expected,
            weight=weight_expected,
            labels=labels,
            logits_multiplier=logits_multiplier,
            kernel_backend=KernelBackend.torch,
        )

        loss_kernel.backward()
        loss_expected.backward()

        self.assert_equal_tensors(loss_kernel, loss_expected, False, atol_float32=3.2e-4, rtol_float32=0)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, False)
        self.assert_equal_tensors(weight_kernel.grad, weight_expected.grad, False)
