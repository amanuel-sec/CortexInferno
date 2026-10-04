#pragma once
#include <cstdint>

namespace ci { namespace kernels {

/// ARM NEON optimized INT8 matmul.
/// A[M×K] × B_T[N×K] → C[M×N]
/// Processes 16 elements per iteration using 128-bit NEON registers.
/// Requires K to be a multiple of 16 (pad if needed).
void neon_matmul_int8(const int8_t* A, const int8_t* B_T,
                      int32_t* C, int M, int N, int K);

}} // namespace ci::kernels