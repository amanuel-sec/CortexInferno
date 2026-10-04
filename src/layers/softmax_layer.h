#pragma once
#include "layers/layer.h"

namespace ci {

class SoftmaxLayer : public Layer {
public:
    void forward(const Tensor& input, Tensor& output,
                 ArenaAllocator&) override;
    LayerType type() const override { return LayerType::SOFTMAX; }
    const char* name() const override { return "Softmax"; }
    size_t output_size_bytes(const TensorShape& s) const override {
        return s.numel() * sizeof(float); // Softmax outputs float
    }
};

} // namespace ci