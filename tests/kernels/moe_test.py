# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from parameterized import parameterized

from cute_kernels import KernelBackend, MoE_Cute, set_seed

from ..test_commons import TestCommons


_SEED = 42


class MoETest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            TestCommons.get_dtypes(),
            [2, 4, 6, 8],  # num_experts
            [2, 4],  # num_experts_per_tok
            [2048],  # hidden_size
            [8192],  # intermediate_size
            [True, False],  # is_glu
            [KernelBackend.triton],  # kernel_backend
            [True, False],  # is_compiling
        )
        + TestCommons.make_args_matrix(
            [torch.device("cuda")],
            TestCommons.get_dtypes(),
            [128],  # num_experts
            [8],  # num_experts_per_tok
            [576],  # hidden_size
            [256],  # intermediate_size
            [True, False],  # is_glu
            [KernelBackend.triton],  # kernel_backend
            [True, False],  # is_compiling
        )
        + TestCommons.make_args_matrix(
            [torch.device("cuda")],
            [torch.bfloat16],
            [2, 4, 6, 8],  # num_experts
            [2, 4],  # num_experts_per_tok
            [2048],  # hidden_size
            [8192],  # intermediate_size
            [True, False],  # is_glu
            [KernelBackend.cuda],  # kernel_backend
            [False],  # is_compiling
        )
        + TestCommons.make_args_matrix(
            [torch.device("cuda")],
            [torch.bfloat16],
            [128],  # num_experts
            [8],  # num_experts_per_tok
            [576],  # hidden_size
            [256],  # intermediate_size
            [True, False],  # is_glu
            [KernelBackend.cuda],  # kernel_backend
            [False],  # is_compiling
        )
    )
    def test_moe(
        self,
        device: torch.device,
        dtype: torch.dtype,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size: int,
        is_glu: bool,
        kernel_backend: KernelBackend,
        is_compiling: bool,
    ) -> None:
        set_seed(_SEED)

        if num_experts_per_tok > num_experts:
            self.skipTest(
                f"skipping test since number of experts per token ({num_experts_per_tok}) is more than number of experts ({num_experts})"
            )

        with torch.device(device):
            moe = MoE_Cute(
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation_function=self.get_activation_function(is_glu=is_glu),
                is_glu=is_glu,
                add_bias=False,
                std=0.02,
            ).to(dtype=dtype)

        moe_custom = moe
        moe_torch = moe

        if is_compiling:
            moe_custom = torch.compile(moe_custom, fullgraph=True)

        x_torch = torch.randn(7, hidden_size, device=device, dtype=dtype, requires_grad=True)
        x_custom = x_torch.clone().detach().requires_grad_()

        y_torch = moe_torch(x_torch, kernel_backend=KernelBackend.torch)[0]
        y_custom = moe_custom(x_custom, kernel_backend=kernel_backend)[0]

        self.assert_equal_tensors(
            y_custom,
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
        weight_torch_grads = []
        for weight_name, weight in moe.named_parameters():
            if "gate" in weight_name:
                continue
            weight_torch_grads.append(weight.grad)

        moe.zero_grad()

        y_custom.sum().backward()
        weight_custom_grads = []
        for weight_name, weight in moe.named_parameters():
            if "gate" in weight_name:
                continue
            weight_custom_grads.append(weight.grad)

        self.assert_equal_tensors(
            x_custom.grad,
            x_torch.grad,
            False,
            atol_float32=6.5e-3,
            rtol_float32=0,
            atol_float16=4e-3,
            rtol_float16=0,
            atol_bfloat16=4e-2,
            rtol_bfloat16=0,
        )

        for weight_torch_grad, weight_custom_grad in zip(weight_torch_grads, weight_custom_grads):
            self.assert_equal_tensors(
                weight_custom_grad,
                weight_torch_grad,
                False,
                atol_float32=3e-2,
                rtol_float32=0,
                atol_float16=4e-3,
                rtol_float16=0,
                atol_bfloat16=4e-2,
                rtol_bfloat16=0,
            )
