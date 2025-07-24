# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from functools import partial

import torch
from tabulate import tabulate

from cute_kernels import KernelBackend, device_synchronize, fused_swiglu_cute


torch._inductor.config.max_autotune_gemm_backends = "TRITON"
torch.backends.cuda.matmul.allow_tf32 = True

n = 100

headers = ["dtype", "torch TFLOPs", "torch compile TFLOPs", "triton memory efficient TFLOPs", "triton FLOPs"]
kernels = [
    partial(
        fused_swiglu_cute, kernel_backend_forward=KernelBackend.torch, kernel_backend_backward=KernelBackend.torch
    ),
    torch.compile(fused_swiglu_cute, mode="max-autotune"),
    partial(fused_swiglu_cute, memory_efficient=True),
    partial(fused_swiglu_cute, memory_efficient=False),
]

table = []

T = 4096
H = 4096
I = 256
flops = 6 * T * H * I

_x = torch.randn(T, H, device=torch.cuda.current_device(), dtype=torch.float32)
_Wu = torch.randn(I, H, device=torch.cuda.current_device(), dtype=torch.float32)
_Wg = torch.randn(I, H, device=torch.cuda.current_device(), dtype=torch.float32)
_Wd = torch.randn(H, I, device=torch.cuda.current_device(), dtype=torch.float32)

with torch.no_grad():
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        x = _x.to(dtype)
        Wu = _Wu.to(dtype)
        Wg = _Wg.to(dtype)
        Wd = _Wd.to(dtype)

        row = [str(dtype)]
        for kernel in kernels:
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
            row.append(flops / t / 1e12)

        table.append(row)


print(tabulate(table, headers=headers))
