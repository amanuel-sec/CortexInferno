#include "layers/flatten_layer.h"

namespace ci {

void FlattenLayer::forward(const Tensor& input, Tensor& output,
                           ArenaAllocator&) {
    // Flatten is a no-op on data — just reshape
    output.set_data(const_cast<void*>(input.data()));
}

} // namespace ci
