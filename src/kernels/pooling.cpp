#include "kernels/pooling.h"
#include <algorithm>

namespace ci { namespace kernels {

void maxpool2d_int8(const int8_t* input, int8_t* output,
                    int H, int W, int C, int KH, int KW,
                    int OH, int OW) {
    for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
            for (int c = 0; c < C; ++c) {
                int8_t max_val = -128;
                for (int kh = 0; kh < KH; ++kh) {
                    for (int kw = 0; kw < KW; ++kw) {
                        int ih = oh * KH + kh;
                        int iw = ow * KW + kw;
                        if (ih < H && iw < W) {
                            max_val = std::max(max_val, input[(ih * W + iw) * C + c]);
                        }
                    }
                }
                output[(oh * OW + ow) * C + c] = max_val;
            }
        }
    }
}

}} // namespace ci::kernels