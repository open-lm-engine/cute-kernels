# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
import torch._dynamo.config
from parameterized import parameterized

from cute_kernels import KernelBackend, pack_sequence_cute, unpack_sequence_cute

from ..test_commons import TestCommons


class PackSequenceTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [(7, 1000, 12, 14)],  # size
            [[0, 70, 170, 295, 393, 412, 515, 691]],  # cu_seqlens
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [False, True],  # use_output_shape
            ["left", "right"],  # padding_side
            [KernelBackend.cuda, KernelBackend.triton],  # kernel_backend
            [pack_sequence_cute, torch.compile(pack_sequence_cute, fullgraph=True)],  # function
        )
    )
    def test_pack_sequence(
        self,
        size: tuple[int],
        cu_seqlens: list[int],
        device: torch.device,
        dtype: torch.dtype,
        use_output_shape: bool,
        padding_side: str,
        kernel_backend: KernelBackend,
        function: Callable,
    ) -> None:
        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.uint32)

        output_shape = (cu_seqlens[-1].item(), *size[2:]) if use_output_shape else None

        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            z_kernel = function(
                x_kernel,
                cu_seqlens=cu_seqlens,
                output_shape=output_shape,
                padding_side=padding_side,
                kernel_backend_forward=kernel_backend,
                kernel_backend_backward=kernel_backend,
            )

        z_expected = pack_sequence_cute(
            x_expected,
            cu_seqlens=cu_seqlens.to(torch.int),
            output_shape=output_shape,
            padding_side=padding_side,
            kernel_backend_forward=KernelBackend.torch,
            kernel_backend_backward=KernelBackend.torch,
        )

        z_expected.sum().backward()
        z_kernel.sum().backward()

        self.assert_equal_tensors(z_kernel, z_expected, True)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, True)

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [(691, 12, 14)],  # size
            [[0, 70, 170, 295, 393, 412, 515, 691]],  # cu_seqlens
            [(7, 1000, 12, 14)],  # output_shape
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            ["left", "right"],  # padding_side
            [KernelBackend.cuda, KernelBackend.triton],  # kernel_backend
            [unpack_sequence_cute, torch.compile(unpack_sequence_cute, fullgraph=True)],  # function
        )
    )
    def test_unpack_sequence(
        self,
        size: tuple[int],
        cu_seqlens: list[int],
        output_shape: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        padding_side: str,
        kernel_backend: KernelBackend,
        function: Callable,
    ) -> None:
        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.uint32)

        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            z_kernel = function(
                x_kernel,
                cu_seqlens=cu_seqlens,
                output_shape=output_shape,
                padding_side=padding_side,
                kernel_backend_forward=kernel_backend,
                kernel_backend_backward=kernel_backend,
            )

        z_expected = unpack_sequence_cute(
            x_expected,
            cu_seqlens=cu_seqlens.to(torch.int),
            output_shape=output_shape,
            padding_side=padding_side,
            kernel_backend_forward=KernelBackend.torch,
            kernel_backend_backward=KernelBackend.torch,
        )

        z_expected.sum().backward()
        z_kernel.sum().backward()

        self.assert_equal_tensors(z_kernel, z_expected, True)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, True)
