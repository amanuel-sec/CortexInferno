#pragma once
#include <cstdint>

namespace ci { namespace kernels {

/// Scalar fallback INT8 matrix multiply.
/// A[M×K] × B_T[N×K] → C[M×N]  (B is pre-transposed)
/// Accumulates into INT32 to prevent overflow.
void scalar_matmul_int8(const int8_t* A, const int8_t* B_T,
                        int32_t* C, int M, int N, int K);

/// Dequantize INT32 accumulator → FLOAT32 output
void dequantize_matmul_result(const int32_t* acc, float* out,
                              int M, int N,
                              float scale_a, float scale_b,
                              int32_t zp_a, int32_t zp_b,
                              const float* bias, float out_scale, int32_t out_zp);

/// Requantize INT32 accumulator → INT8 output (fully quantized pipeline)
void requantize_matmul_int8(const int32_t* acc, int8_t* out,
                            int M, int N,
                            float effective_scale, int32_t out_zp,
                            const int32_t* bias);

}} // namespace ci::kernels