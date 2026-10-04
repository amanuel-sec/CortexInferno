#pragma once
#include "layers/layer.h"

namespace ci {

class DenseLayer : public Layer {
public:
    DenseLayer(const int8_t* weights, const int32_t* bias,
               int in_features, int out_features,
               float w_scale, int32_t w_zp,
               float out_scale, int32_t out_zp)
        : weights_(weights), bias_(bias),
          in_features_(in_features), out_features_(out_features),
          w_scale_(w_scale), w_zp_(w_zp),
          out_scale_(out_scale), out_zp_(out_zp) {}

    void forward(const Tensor& input, Tensor& output,
                 ArenaAllocator& arena) override;
    LayerType type() const override { return LayerType::DENSE; }
    const char* name() const override { return "Dense"; }
    size_t output_size_bytes(const TensorShape&) const override {
        return static_cast<size_t>(out_features_);
    }

private:
    const int8_t* weights_;
    const int32_t* bias_;
    int in_features_, out_features_;
    float w_scale_;
    int32_t w_zp_;
    float out_scale_;
    int32_t out_zp_;
};

} // namespace ci