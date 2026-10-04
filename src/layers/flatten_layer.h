#pragma once
#include "layers/layer.h"

namespace ci {

class FlattenLayer : public Layer {
public:
    void forward(const Tensor& input, Tensor& output,
                 ArenaAllocator&) override;
    LayerType type() const override { return LayerType::FLATTEN; }
    const char* name() const override { return "Flatten"; }
    size_t output_size_bytes(const TensorShape& s) const override {
        return s.numel();
    }
};

} // namespace ci