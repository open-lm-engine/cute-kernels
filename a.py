# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from cute_kernels import KernelBackend, bmm_cute


A = torch.empty(8, 1024, 768, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16).cuda()
B = torch.empty(8, 768, 1024, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16).cuda()
C = torch.empty(8, 1024, 1024, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16).cuda()

bmm_cute(A=A, B=B, C=C, kernel_backend=KernelBackend.cuda)

# # Create and permute tensor A/B/C
# def create_and_permute_tensor(l, mode0, mode1, is_mode0_major, dtype):
#     # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
#     # else: (l, mode0, mode1) -> (mode0, mode1, l)
#     shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
#     permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)

#     return (
#         torch.empty(*shape, dtype=torch.int32)
#         .random_(-2, 2)
#         .to(dtype=dtype)
#         .permute(permute_order)
#         .cuda()
#     )

# L = 8
# M = 1024
# K = 768
# N = 1024

# a_major = "m"
# b_major = "n"

# a = create_and_permute_tensor(
#     L, M, K, a_major == "k", torch.bfloat16
# )
# b = create_and_permute_tensor(
#     L, N, K, b_major == "k", torch.bfloat16
# )

# print(a.size(), a.stride())
# print(b.size(), b.stride())

# A = A.permute(1, 2, 0)
# print(A.size(), A.stride())

# B = B.permute(2, 1, 0)
# print(B.size(), B.stride())
