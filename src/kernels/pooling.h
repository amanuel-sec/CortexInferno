#pragma once
#include <cstdint>

namespace ci { namespace kernels {

void maxpool2d_int8(const int8_t* input, int8_t* output,
                    int H, int W, int C, int KH, int KW,
                    int OH, int OW);

}} // namespace ci::kernels