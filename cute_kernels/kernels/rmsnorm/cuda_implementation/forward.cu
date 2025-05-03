#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "include/cute_kernels.h"

namespace ck = cute_kernels;
namespace ck_mem = ck::memory;

using fp32 = ck::fp32;
using fp32_2 = ck::fp32_2;
using fp32_4 = ck::fp32_4;

using uint32 = ck::uint32;
using uint64 = ck::uint64;

template <typename scalar_t>
__global__ void _rmsnorm_forward_cuda_kernel(const scalar_t *x,
                                             const scalar_t *weight,
                                             scalar_t *output,
                                             const fp32 eps,
                                             scalar_t *rmsnorm_denominator,
                                             const uint32 B,
                                             const uint32 H) {
    constexpr int num_elements_per_thread = ck_mem::get_num_elements_for_vector_load_stores<scalar_t>();
    const uint32 H_vec = H / num_elements_per_thread;

    fp32 accumulator = 0;
    if (threadIdx.x < H_vec) {
        // compute RMSNorm's denominator locally for each thread
        const scalar_t *x_vec = ck_mem::load_128_bits<scalar_t>(x, thread_id);
        fp32 x_vec_fp32[num_elements_per_thread];

        for (uint32 i = 0; i < num_elements_per_thread; i++) {
            fp32 _x = ck::DType<scalar_t>::upcast(x_vec[i]);
            x_vec_fp32[i] = _x;
            accumulator += _x * _x;
        }
    }

    accumulator = 1 / accumulator;
}

void rmsnorm_forward_cuda(const torch::Tensor &x,
                          std::optional<torch::Tensor> &weight,
                          torch::Tensor &output,
                          const fp32 &eps,
                          std::optional<torch::Tensor> &rmsnorm_denominator,
                          const uint32 &BLOCK_SIZE_B,
                          const uint32 &BLOCK_SIZE_H) {
    CHECK_CUDA_TENSOR(x);
    CHECK_CUDA_TENSOR(output);

    if (weight.has_value()) {
        CHECK_CUDA_TENSOR(weight.value());
    }

    if (rmsnorm_denominator.has_value()) {
        CHECK_CUDA_TENSOR(rmsnorm_denominator);
    }

    CHECK_VALID_THREAD_BLOCK(BLOCK_SIZE_B * BLOCK_SIZE_H);

    uint32 B, H;
    std::tie(B, H) = x.sizes();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(x.scalar_type(), "rmsnorm_forward_cuda_kernel", ([&] {
                                       const uint32 num_elements_per_thread = 16 / sizeof(scalar_t);
                                       std::vector<ck::ChunkedArray<scalar_t>> x_chunks =
                                           ck::chunk_array<scalar_t>(x.data_ptr<scalar_t>(), total_elements);
                                       std::vector<ck::ChunkedArray<scalar_t>> output_chunks =
                                           ck::chunk_array<scalar_t>(output.data_ptr<scalar_t>(), total_elements);

                                       for (int i = 0; i < x_chunks.size(); i++) {
                                           ck::ChunkedArray<scalar_t> x_chunk = x_chunks[i];
                                           ck::ChunkedArray<scalar_t> output_chunk = output_chunks[i];

                                           const uint64 num_elements = x_chunk.num_elements;
                                           const uint32 NUM_BLOCKS =
                                               ck::ceil_divide<uint64>(num_elements, num_elements_per_block);

                                           _add_scalar_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                                               x_chunk.array, y, output_chunk.array, num_elements);
                                       }
                                   }));
}
