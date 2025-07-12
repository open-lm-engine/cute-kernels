# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _add_scalar_cuda_kernel(gx: cute.Tensor, y: cute.Float32, gz: cute.Tensor, tv_layout: cute.Layout) -> None:
    THREAD_ID, _, _ = cute.arch.thread_idx()
    BLOCK_ID, _, _ = cute.arch.block_idx()

    block_x = gx[None, BLOCK_ID]
    block_z = gz[None, BLOCK_ID]

    thread_fragment_x = cute.composition(block_x, tv_layout)
    thread_fragment_z = cute.composition(block_z, tv_layout)

    thread_x = thread_fragment_x[THREAD_ID, None]
    thread_z = thread_fragment_z[THREAD_ID, None]

    thread_x = thread_x.load().to(cute.Float32)
    thread_x = thread_x + y
    thread_x = thread_x.to(gx.element_type)

    thread_z.store(thread_x)


@cute.jit
def _add_scalar_cuda_jit(mx: cute.Tensor, y: cute.Float32, mz: cute.Tensor) -> None:
    BLOCK_SIZE = 1024

    thread_layout = cute.make_layout((1, BLOCK_SIZE))
    value_layout = cute.make_layout((1, 128 // mx.element_type.width))

    tiler, tv_layout = cute.make_layout_tv(thread_layout, value_layout)

    gx = cute.zipped_divide(mx, tiler)
    gz = cute.zipped_divide(mz, tiler)

    _add_scalar_cuda_kernel(gx=gx, y=y, gz=gz, tv_layout=tv_layout).launch(
        grid=(cute.size(gx, mode=[1]), 1, 1), block=(BLOCK_SIZE, 1, 1)
    )


def add_scalar_cute_dsl(x: torch.Tensor, y: float, output: torch.Tensor) -> None:
    if x.dim() == 1:
        x = x.unsqueeze(0)
        output = output.unsqueeze(0)

    key = x.dtype

    x = from_dlpack(x, assumed_align=16)
    output = from_dlpack(output, assumed_align=16)

    kernel = add_scalar_cute_dsl._kernels.get(key, None)
    if kernel is None:
        kernel = cute.compile(_add_scalar_cuda_jit, x, y, output)
        add_scalar_cute_dsl._kernels[key] = kernel

    kernel(x, y, output)


add_scalar_cute_dsl._kernels = {}
