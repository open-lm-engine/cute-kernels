# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from functools import partial

import torch
from tabulate import tabulate

from cute_kernels import KernelBackend, continuous_count_cute, device_synchronize


torch._inductor.config.max_autotune_gemm_backends = "TRITON"
torch.backends.cuda.matmul.allow_tf32 = True

n = 100

headers = ["dtype", "torch BW", "CUDA BW"]
kernels = [
    partial(continuous_count_cute, kernel_backend=KernelBackend.torch),
    partial(continuous_count_cute, kernel_backend=KernelBackend.cuda),
]

table = []
E = 64
N = 64 * 4096

for dtype in [torch.int32, torch.long]:
    # output is uint32
    io = torch.tensor([], dtype=dtype).element_size() * N + 4 * E

    row = [str(dtype)]
    for kernel in kernels:
        x = torch.randint(0, E, (N,), device=torch.cuda.current_device(), dtype=dtype)

        for i in range(n):
            y = kernel(x, size=E)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        s.record()
        for i in range(n):
            y = kernel(x, size=E)
        e.record()

        device_synchronize()

        t = s.elapsed_time(e) / n / 1e3
        row.append(io / t / 1e12)

    table.append(row)


print(tabulate(table, headers=headers))
