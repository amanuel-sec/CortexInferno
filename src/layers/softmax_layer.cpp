#include "layers/softmax_layer.h"
#include "kernels/activations.h"
#include "kernels/quantize.h"

namespace ci {

void SoftmaxLayer::forward(const Tensor& input, Tensor& output,
                           ArenaAllocator&) {
    size_t n = input.numel();
    // Dequantize to float first
    auto* fbuf = static_cast<float*>(output.data());
    kernels::dequantize_int8(input.data_as<int8_t>(), fbuf, n,
                             input.quant_params().scale,
                             input.quant_params().zero_point);
    // Softmax in-place
    kernels::softmax_float(fbuf, fbuf, n);
}

} // namespace ci