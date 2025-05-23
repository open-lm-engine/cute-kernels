# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from functools import partial

import torch
from tabulate import tabulate

from cute_kernels import device_synchronize, fused_swiglu_cute, fused_swiglu_torch


torch._inductor.config.max_autotune_gemm_backends = "TRITON"
torch.backends.cuda.matmul.allow_tf32 = True

n = 100

headers = ["dtype", "torch TFLOPs", "torch compile TFLOPs", "triton memory efficient TFLOPs", "triton FLOPs"]
kernels = [
    fused_swiglu_torch,
    torch.compile(fused_swiglu_torch, mode="max-autotune"),
    partial(fused_swiglu_cute, memory_efficient=True),
    partial(fused_swiglu_cute, memory_efficient=False),
]

table = []

for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    row = [str(dtype)]
    for kernel in kernels:
        x = torch.randn(4096, 4096, device=torch.cuda.current_device(), dtype=dtype)
        Wu = torch.randn(4096, 4096, device=torch.cuda.current_device(), dtype=dtype)
        Wg = torch.randn(4096, 4096, device=torch.cuda.current_device(), dtype=dtype)
        Wd = torch.randn(4096, 4096, device=torch.cuda.current_device(), dtype=dtype)

        for i in range(n):
            z = kernel(x=x, gate_weight=Wg, up_weight=Wu, down_weight=Wd)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        s.record()
        for i in range(n):
            z = kernel(x=x, gate_weight=Wg, up_weight=Wu, down_weight=Wd)
        e.record()

        device_synchronize()

        t = s.elapsed_time(e) / n / 1e3
        flops = 2 * x.size(0) * x.size(1) * (Wu.size(0) + Wg.size(0))
        flops += 2 * x.size(0) * Wd.size(0) * Wd.size(1)
        row.append(flops / t / 1e12)

    table.append(row)


print(tabulate(table, headers=headers))
