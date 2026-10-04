#include "layers/pooling_layer.h"
#include "kernels/pooling.h"

namespace ci {

void MaxPool2DLayer::forward(const Tensor& input, Tensor& output,
                             ArenaAllocator&) {
    auto s = input.shape();
    int OH = s[1] / kh_, OW = s[2] / kw_;
    kernels::maxpool2d_int8(input.data_as<int8_t>(), output.data_as<int8_t>(),
                            s[1], s[2], s[3], kh_, kw_, OH, OW);
}

} // namespace ci