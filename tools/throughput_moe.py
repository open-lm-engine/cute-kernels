# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from functools import partial

import torch
from tabulate import tabulate

from cute_kernels import KernelBackend, MoE, device_synchronize, swiglu_packed_cute


torch_profiler = torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=5, warmup=5, active=1, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("tmp"),
    record_shapes=True,
    profile_memory=True,
)


torch.backends.cuda.matmul.allow_tf32 = True

n = 100
dtype = torch.bfloat16

with torch.device(torch.cuda.current_device()):
    moe = MoE(
        num_experts=128,
        num_experts_per_tok=8,
        hidden_size=1536,
        intermediate_size=512,
        activation_function=swiglu_packed_cute,
        is_glu=True,
        add_bias=False,
        std=1,
    ).to(dtype)

headers = [
    "torch msec",
    "torch compile msec",
    "scattermoe msec",
    "scattermoe compile msec",
    "grouped gemm msec",
    "grouped gemm compile msec",
]
kernels = [
    partial(torch.compile(moe), kernel_backend=KernelBackend.torch),
    partial(torch.compile(moe), kernel_backend=KernelBackend.triton),
    partial(torch.compile(moe), kernel_backend=KernelBackend.cuda),
]

table = []

with torch.inference_mode():
    row = []
    for kernel in kernels:
        x = torch.randn(8, 4096, 1536, device=torch.cuda.current_device(), dtype=dtype)

        for i in range(n):
            z = kernel(x)
            torch_profiler.step()

#         s = torch.cuda.Event(enable_timing=True)
#         e = torch.cuda.Event(enable_timing=True)

#         s.record()
#         for i in range(n):
#             z = kernel(x)
#         e.record()

#         device_synchronize()

#         t = s.elapsed_time(e) / n / 1e3
#         row.append(t)

#     table.append(row)

# print(tabulate(table, headers=headers))
