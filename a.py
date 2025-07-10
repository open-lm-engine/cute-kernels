# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from cute_kernels import KernelBackend, bmm_cute


A = torch.empty(8, 1024, 768, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16).cuda()
B = torch.empty(8, 768, 1024, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16).cuda()
C = torch.empty(8, 1024, 1024, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16).cuda()

bmm_cute(A=A, B=B, C=C, kernel_backend=KernelBackend.cuda)
