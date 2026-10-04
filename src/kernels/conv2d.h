#pragma once
#include <cstdint>

namespace ci { namespace kernels {

/// INT8 Conv2D: input[N,H,W,C_in] × weights[C_out,KH,KW,C_in] + bias → output[N,OH,OW,C_out]
/// Simplified: single batch (N=1), stride=1, no padding
void conv2d_int8(const int8_t* input, const int8_t* weights,
                 const int32_t* bias, int32_t* output,
                 int H, int W, int C_in,
                 int C_out, int KH, int KW,
                 int OH, int OW);

}} // namespace ci::kernels