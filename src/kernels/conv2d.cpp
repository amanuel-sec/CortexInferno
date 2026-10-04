#include "kernels/conv2d.h"

namespace ci { namespace kernels {

void conv2d_int8(const int8_t* input, const int8_t* weights,
                 const int32_t* bias, int32_t* output,
                 int H, int W, int C_in,
                 int C_out, int KH, int KW,
                 int OH, int OW) {
    for (int co = 0; co < C_out; ++co) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                int32_t acc = bias ? bias[co] : 0;
                for (int kh = 0; kh < KH; ++kh) {
                    for (int kw = 0; kw < KW; ++kw) {
                        for (int ci = 0; ci < C_in; ++ci) {
                            int ih = oh + kh;
                            int iw = ow + kw;
                            int8_t in_val = input[(ih * W + iw) * C_in + ci];
                            int8_t w_val = weights[((co * KH + kh) * KW + kw) * C_in + ci];
                            acc += static_cast<int32_t>(in_val) * static_cast<int32_t>(w_val);
                        }
                    }
                }
                output[(oh * OW + ow) * C_out + co] = acc;
            }
        }
    }
}

}} // namespace ci::kernels