# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...constants import LIBRARY_NAME
from ...kernel_backend import KernelBackend
from ...utils import cute_op
from .cuda_implementation import continuous_count_cuda


def _fake_bincount(x: torch.Tensor, minlength: int) -> torch.Tensor:
    return torch.empty(minlength, device=x.device, dtype=torch.int)


@cute_op(f"{LIBRARY_NAME}::bincount", mutates_args={}, fake_func=_fake_bincount)
def bincount(x: torch.Tensor, minlength: int) -> torch.Tensor:
    return x.bincount(minlength=minlength)


@torch.no_grad()
def continuous_count_cute(
    x: torch.Tensor, size: int, *, kernel_backend: KernelBackend = KernelBackend.cuda
) -> torch.Tensor:
    """counts the number of occurances of the values [0, 1, ..., `size`) in the input tensor (`size` is excluded).
        NOTE: the user is responsible for ensuring that the values lie in the valid range, any values outside this
        range are ignored and not counted.

    Args:
        x (torch.Tensor): input tensor
        size (int): values [0, 1, ..., `size`) are counted (`size` is excluded)
        kernel_backend (KernelBackend, optional): kernel backend to prioritize.
            Defaults to KernelBackend.cuda.

    Returns:
        torch.Tensor: output tensor
    """

    if size == 1:
        return torch.tensor([x.numel()], dtype=torch.uint32, device=x.device)

    assert x.dim() == 1, "x should be 1-dimensional"
    assert x.dtype in [torch.int32, torch.long]

    if kernel_backend == KernelBackend.torch:
        output = bincount(x=x, minlength=size).to(torch.uint32)
    elif kernel_backend == KernelBackend.cuda:
        output = torch.empty(size, dtype=torch.uint32, device=x.device)

        continuous_count_cuda(
            x=x,
            output=output,
            sorted_output=None,
            sorted_indices=None,
            C=size,
            THREAD_BLOCK_CLUSTER_SIZE=1,
            BLOCK_SIZE=1024,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output


@torch.no_grad()
def continuous_count_and_sort_cute(
    x: torch.Tensor, size: int, *, kernel_backend: KernelBackend = KernelBackend.cuda
) -> torch.Tensor:
    """counts the number of occurances of the values [0, 1, ..., `size`) in the input tensor (`size` is excluded).
        NOTE: the user is responsible for ensuring that the values lie in the valid range, any values outside this
        range are ignored and not counted.

    Args:
        x (torch.Tensor): input tensor
        size (int): values [0, 1, ..., `size`) are counted (`size` is excluded)
        kernel_backend (KernelBackend, optional): kernel backend to prioritize.
            Defaults to KernelBackend.cuda.

    Returns:
        torch.Tensor: output tensor
    """

    if size == 1:
        return torch.tensor([x.numel()], dtype=torch.uint32, device=x.device)

    assert x.dim() == 1, "x should be 1-dimensional"
    assert x.dtype in [torch.int32, torch.long]

    if kernel_backend == KernelBackend.torch:
        output = bincount(x=x, minlength=size).to(torch.uint32)
        sorted_output, sorted_indices = x.sort()
    elif kernel_backend == KernelBackend.cuda:
        output = torch.empty(size, dtype=torch.uint32, device=x.device)
        sorted_output = torch.empty_like(x)
        sorted_indices = torch.empty_like(x, dtype=torch.uint64)

        continuous_count_cuda(
            x=x,
            output=output,
            sorted_output=sorted_output,
            sorted_indices=sorted_indices,
            C=size,
            THREAD_BLOCK_CLUSTER_SIZE=1,
            BLOCK_SIZE=1024,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output, sorted_output, sorted_indices
