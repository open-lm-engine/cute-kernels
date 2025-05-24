# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...cutotune import CutoTuneConfig, CutoTuneParameter, cutotune
from ...kernel_backend import KernelBackend, is_cuda_kernel_backend_allowed, is_triton_kernel_backend_allowed
from ...math import ceil_divide
from ...utils import is_nvidia_gpu
from .cuda_implementation import add_scalar_cuda
from .torch_implementation import add_scalar_torch
from .triton_implementation import add_scalar_triton_kernel


@cutotune(
    configs=[
        CutoTuneConfig(
            {"kernel_backend": KernelBackend.cuda},
            condition=lambda **kwargs: is_cuda_kernel_backend_allowed(kwargs["kernel_backend"]) and is_nvidia_gpu(),
        ),
        CutoTuneConfig(
            {"kernel_backend": KernelBackend.triton},
            condition=lambda **kwargs: is_triton_kernel_backend_allowed(kwargs["kernel_backend"]),
        ),
    ],
    triggers={"x.dtype"},
)
def _forward(
    x: torch.Tensor, y: float, output: torch.Tensor, kernel_backend: KernelBackend | CutoTuneParameter
) -> None:
    if kernel_backend == KernelBackend.cuda:
        BLOCK_SIZE = 1024
        add_scalar_cuda(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE)
    elif kernel_backend == KernelBackend.triton:
        N = x.numel()
        BLOCK_SIZE = 4096
        NUM_WARPS = 32

        with torch.device(x.device):
            add_scalar_triton_kernel[ceil_divide(N, BLOCK_SIZE),](
                x_ptr=x, y=y, output_ptr=output, N=N, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS
            )
    else:
        raise ValueError("unexpected kernel_backend")


class _AddScalar_Cute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: float, kernel_backend: KernelBackend | CutoTuneParameter) -> torch.Tensor:
        output = torch.empty_like(x)
        _forward(x=x, y=y, output=output, kernel_backend=kernel_backend)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, None, None


def add_scalar_cute(
    x: torch.Tensor, y: int | float, *, kernel_backend: KernelBackend | CutoTuneParameter = CutoTuneParameter()
) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor): input tensor
        y (int | float): float value to add to `x`
        kernel_backend (KernelBackend | CutoTuneParameter, optional): kernel backend to prioritize.
            Defaults to CutoTuneParameter().

    Returns:
        torch.Tensor: output tensor
    """

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, (int, float)), "y needs to be a numeric type"
    assert isinstance(
        kernel_backend, (KernelBackend, CutoTuneParameter)
    ), f"unexpected kernel_backend ({kernel_backend})"

    if y == 0:
        return x

    return _AddScalar_Cute.apply(x, y, kernel_backend)
