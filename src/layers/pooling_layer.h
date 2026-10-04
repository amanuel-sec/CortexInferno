#pragma once
#include "layers/layer.h"

namespace ci {

class MaxPool2DLayer : public Layer {
public:
    MaxPool2DLayer(int kh, int kw) : kh_(kh), kw_(kw) {}
    void forward(const Tensor& input, Tensor& output,
                 ArenaAllocator& arena) override;
    LayerType type() const override { return LayerType::MAXPOOL2D; }
    const char* name() const override { return "MaxPool2D"; }
    size_t output_size_bytes(const TensorShape& s) const override {
        return static_cast<size_t>((s[1]/kh_) * (s[2]/kw_) * s[3]);
    }
private:
    int kh_, kw_;
};

} // namespace ci