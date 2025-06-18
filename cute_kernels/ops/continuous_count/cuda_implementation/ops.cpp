// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#include <torch/extension.h>

void continuous_count_cuda(const torch::Tensor &x,
                           torch::Tensor &output,
                           std::optional<torch::Tensor> &_sorted_output,
                           std::optional<torch::Tensor> &_sorted_indices,
                           const uint &E,
                           const uint &BLOCK_SIZE);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("continuous_count_cuda", &continuous_count_cuda, "contiguous count (CUDA)");
}
