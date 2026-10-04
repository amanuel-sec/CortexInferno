#include "layers/dense_layer.h"
#include "kernels/scalar_matmul_int8.h"

#if defined(CI_USE_NEON)
#include "kernels/neon_matmul_int8.h"
#endif
#if defined(CI_USE_DSP)
#include "kernels/dsp_matmul_int8.h"
#endif

namespace ci {

void DenseLayer::forward(const Tensor& input, Tensor& output,
                         ArenaAllocator& arena) {
    int M = 1; // Batch size
    int K = in_features_;
    int N = out_features_;

    // Allocate accumulator buffer
    auto* acc = arena.allocate_typed<int32_t>(M * N);

    // Dispatch to best available kernel
#if defined(CI_USE_NEON)
    kernels::neon_matmul_int8(input.data_as<int8_t>(), weights_, acc, M, N, K);
#elif defined(CI_USE_DSP)
    kernels::dsp_matmul_int8(input.data_as<int8_t>(), weights_, acc, M, N, K);
#else
    kernels::scalar_matmul_int8(input.data_as<int8_t>(), weights_, acc, M, N, K);
#endif

    // Requantize to INT8 output
    float eff_scale = (input.quant_params().scale * w_scale_) / out_scale_;
    kernels::requantize_matmul_int8(acc, output.data_as<int8_t>(),
                                    M, N, eff_scale, out_zp_, bias_);
}

} // namespace ci