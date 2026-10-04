#pragma once
#include <cstdint>

namespace ci { namespace kernels {

/// ARM DSP-optimized INT8 matmul for Cortex-M4/M7.
/// Uses __SXTB16 and __SMLAD to process 2 multiply-accumulates
/// per instruction via dual 16-bit SIMD lanes.
void dsp_matmul_int8(const int8_t* A, const int8_t* B_T,
                     int32_t* C, int M, int N, int K);

}} // namespace ci::kernels