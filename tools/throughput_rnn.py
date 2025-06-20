# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from functools import partial

import torch
from tabulate import tabulate

from cute_kernels import KernelBackend, device_synchronize, rnn_cute


n = 100

headers = ["dtype", "torch", "kernel"]
kernels = [partial(rnn_cute, kernel_backend=KernelBackend.torch), rnn_cute]

table = []

for dtype in [torch.float32]:
    row = [str(dtype)]
    for kernel in kernels:
        input = torch.randn(4, 128, 16, 16, device=torch.cuda.current_device(), dtype=dtype)
        weight = torch.randn(16, 16, 16, device=torch.cuda.current_device(), dtype=dtype)

        for i in range(n):
            z = kernel(input, weight)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        s.record()
        for i in range(n):
            z = kernel(input, weight)
        e.record()

        device_synchronize()

        row.append(s.elapsed_time(e) / n)
    table.append(row)


print(tabulate(table, headers=headers))
