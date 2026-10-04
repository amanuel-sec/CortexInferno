#pragma once
#include "core/tensor.h"
#include "core/allocator.h"
#include <cstdint>

namespace ci {

enum class LayerType : uint8_t {
    DENSE, CONV2D, MAXPOOL2D, RELU, SOFTMAX, FLATTEN
};

class Layer {
public:
    virtual ~Layer() = default;
    virtual void forward(const Tensor& input, Tensor& output,
                         ArenaAllocator& arena) = 0;
    virtual LayerType type() const = 0;
    virtual const char* name() const = 0;

    // For memory planning
    virtual size_t output_size_bytes(const TensorShape& input_shape) const = 0;
};

} // namespace ci