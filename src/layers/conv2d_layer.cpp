#include "layers/conv2d_layer.h"
#include "kernels/conv2d.h"
#include "kernels/quantize.h"
#include <cmath>

namespace ci {

void Conv2DLayer::forward(const Tensor& input, Tensor& output,
                          ArenaAllocator& arena) {
    auto s = input.shape();
    int H = s[1], W = s[2];
    int OH = H - kh_ + 1, OW = W - kw_ + 1;

    auto* acc = arena.allocate_typed<int32_t>(OH * OW * c_out_);

    kernels::conv2d_int8(input.data_as<int8_t>(), weights_, bias_, acc,
                         H, W, c_in_, c_out_, kh_, kw_, OH, OW);

    // Requantize
    float eff = (input.quant_params().scale * w_scale_) / out_scale_;
    for (int i = 0; i < OH * OW * c_out_; ++i) {
        float v = static_cast<float>(acc[i]) * eff + static_cast<float>(out_zp_);
        v = std::max(-128.0f, std::min(127.0f, std::round(v)));
        output.data_as<int8_t>()[i] = static_cast<int8_t>(v);
    }
}

} // namespace ci
