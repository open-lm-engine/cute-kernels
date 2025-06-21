# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import time
from functools import partial

import torch

from cute_kernels import KernelBackend
from cute_kernels.modules.memory import EmbeddingBag


if __name__ == "__main__":
    # Correctness/speed check:
    B = 4096
    K = 1024**2
    bag_size = 32
    dim = 4096
    print(f"{B=} {K=} {bag_size=} {dim=}")
    torch.manual_seed(0)
    torch.set_default_device("cuda")
    # torch.set_default_dtype(torch.bfloat16)
    indices = torch.randint(0, K, [B, 10])

    model = EmbeddingBag(K, dim)

    # indices = torch.randint(0, K, [B, bag_size])
    per_sample_weights = torch.randn(indices.shape, requires_grad=True)
    gradient = torch.randn([B, dim])
    # Torch impl
    out_torch = model(indices, per_sample_weights, kernel_backend=KernelBackend.torch)
    out_torch.backward(gradient)
    wg = model.weight.grad
    wpsg = per_sample_weights.grad

    model.weight.grad = per_sample_weights.grad = None
    # xFormers impl
    out_xf = model(indices, per_sample_weights, kernel_backend=KernelBackend.triton)
    out_xf.backward(gradient)

    assert torch.allclose(out_torch, out_xf)
    assert torch.allclose(wg, model.weight.grad, atol=5e-2, rtol=1e-2)
    assert torch.allclose(wpsg, per_sample_weights.grad, atol=5e-2, rtol=1e-2)

    print("Correctness: PASS")

    for op, op_name in [
        (partial(model, kernel_backend=KernelBackend.torch), "F.embedding_bag"),
        (partial(model, kernel_backend=KernelBackend.triton), "xformers_embedding_bag"),
    ]:
        fn = lambda: op(indices, scores=per_sample_weights).backward(gradient)
        fn()
        # time
        REPEATS = 10
        torch.cuda.synchronize()
        begin = time.time()
        for _ in range(REPEATS):
            fn()
        torch.cuda.synchronize()
        dt = (time.time() - begin) / REPEATS
        print(f"{op_name}: {round(dt*1000, 2)}ms")
