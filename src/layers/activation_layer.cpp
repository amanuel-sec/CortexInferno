#include "layers/activation_layer.h"
#include "kernels/activations.h"
#include <cstring>

namespace ci {

void ReLULayer::forward(const Tensor& input, Tensor& output,
                        ArenaAllocator&) {
    // In-place capable: if output.data == input.data, works fine
    if (output.data() != input.data())
        std::memcpy(output.data(), input.data(), input.numel());
    kernels::relu_int8(output.data_as<int8_t>(), output.numel(),
                       static_cast<int8_t>(output.quant_params().zero_point));
}

} // namespace ci
