// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#include <torch/extension.h>

void topk_softmax_forward_cuda(torch::Tensor &topk_weights,
                               torch::Tensor &topk_indices,
                               torch::Tensor &token_expert_indices,
                               torch::Tensor &gating_output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_softmax_forward_cuda", &topk_softmax_forward_cuda, "topk softmax (CUDA)");
}
