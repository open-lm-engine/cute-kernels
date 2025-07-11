# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import cutlass.cute as cute


@cute.kernel
def _add_scalar_cuda_kernel(gx: cute.Tensor, y: cute.Float32, gz: cute.Tensor, tv_layout: cute.Layout) -> None:
    THREAD_ID, _, _ = cute.arch.thread_idx()
    BLOCK_ID, _, _ = cute.arch.block_idx()

    block_x = gx[None, BLOCK_ID]
    block_z = gz[None, BLOCK_ID]

    thread_fragment_x = cute.composition(block_x, tv_layout)
    thread_fragment_z = cute.composition(block_z, tv_layout)

    thread_x = block_x.load()
    thread_y = cute.fragment_like(thread_x)
    thread_y.fill(y)

    thread_z = thread_x + thread_y


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
