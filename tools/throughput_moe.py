# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from functools import partial

import torch
import torch.nn.functional as F
from tabulate import tabulate

from cute_kernels import KernelBackend, MoE, device_synchronize


torch.backends.cuda.matmul.allow_tf32 = True

n = 100

with torch.device(torch.cuda.current_device()):
    moe = MoE(
        num_experts=128,
        num_experts_per_tok=8,
        hidden_size=1536,
        intermediate_size=512,
        activation_function=lambda x, y: x * F.silu(y),
        is_glu=True,
        add_bias=False,
        std=1,
    )

headers = [
    "dtype",
    "torch TFLOPs",
    "torch compile TFLOPs",
    "scattermoe TFLOPs",
    "scattermoe compile TFLOPs",
    "grouped gemm TFLOPs",
    "grouped gemm compile TFLOPs",
]
kernels = [
    partial(moe, kernel_backend=KernelBackend.torch),
    partial(torch.compile(moe), kernel_backend=KernelBackend.torch),
    partial(moe, kernel_backend=KernelBackend.triton),
    partial(torch.compile(moe), kernel_backend=KernelBackend.triton),
    partial(moe, kernel_backend=KernelBackend.cuda),
    partial(torch.compile(moe), kernel_backend=KernelBackend.cuda),
]

table = []

for dtype in [torch.bfloat16]:
    row = [str(dtype)]
    for kernel in kernels:
        x = torch.randn(4096, 4096, device=torch.cuda.current_device(), dtype=dtype)
        w = torch.randn(4096, 4096, device=torch.cuda.current_device(), dtype=dtype)

        for i in range(n):
            z = kernel(x, w, C=None, beta=0)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        s.record()
        for i in range(n):
            z = kernel(x, w, C=None, beta=0)
        e.record()

        device_synchronize()

        t = s.elapsed_time(e) / n / 1e3
        row.append(2 * x.size(0) * x.size(1) * w.size(0) / t / 1e12)

    table.append(row)


print(tabulate(table, headers=headers))
