#include <cuda.h>

namespace cute_kernels::reductions {
    inline __device__ T warp_all_reduce(const T& x) {
        T y = 0;
        return y;
    }
}  // namespace cute_kernels::reductions
