# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from parameterized import parameterized

from cute_kernels import KernelBackend, MoE, set_seed
from cute_kernels.modules.memory import EmbeddingBag

from ..test_commons import TestCommons


_SEED = 42


class MoETest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            TestCommons.get_dtypes(),
            [8192],  # num_embeddings
            [2048],  # embedding_dim
            [KernelBackend.triton],  # kernel_backend
            [True, False],  # is_compiling
        )
    )
    def test_moe(
        self,
        device: torch.device,
        dtype: torch.dtype,
        num_embeddings: int,
        embedding_dim: int,
        kernel_backend: KernelBackend,
        is_compiling: bool,
    ) -> None:
        set_seed(_SEED)

        with torch.device(device):
            model = EmbeddingBag(num_embeddings=num_embeddings, embedding_dim=embedding_dim).to(dtype=dtype)

        model_kernel = model
        model_torch = model

        if is_compiling:
            model_kernel = torch.compile(model_kernel, fullgraph=True)

        x_torch = torch.randn(7, hidden_size, device=device, dtype=dtype, requires_grad=True)
        x_kernel = x_torch.clone().detach().requires_grad_()

        y_torch = moe_torch(x_torch, kernel_backend=KernelBackend.torch)[0]
        y_kernel = moe_kernel(x_kernel, kernel_backend=kernel_backend)[0]

        self.assert_equal_tensors(
            y_kernel,
            y_torch,
            False,
            atol_float32=6e-3,
            rtol_float32=0,
            atol_float16=4e-3,
            rtol_float16=0,
            atol_bfloat16=2.35e-2,
            rtol_bfloat16=0,
        )

        y_torch.sum().backward()
        weight_torch_grads = self.collect_gradients_from_module_and_zero_grads(moe)

        y_kernel.sum().backward()
        weight_kernel_grads = self.collect_gradients_from_module_and_zero_grads(moe)

        self.assert_equal_tensors(
            x_kernel.grad,
            x_torch.grad,
            False,
            atol_float32=6.5e-3,
            rtol_float32=0,
            atol_float16=4e-3,
            rtol_float16=0,
            atol_bfloat16=4e-2,
            rtol_bfloat16=0,
        )

        for weight_name in weight_torch_grads:
            if "gate" in weight_name:
                continue

            self.assert_equal_tensors(
                weight_kernel_grads[weight_name],
                weight_torch_grads[weight_name],
                False,
                atol_float32=3e-2,
                rtol_float32=0,
                atol_float16=4e-3,
                rtol_float16=0,
                atol_bfloat16=4e-2,
                rtol_bfloat16=0,
            )
