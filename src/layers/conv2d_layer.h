#pragma once
#include "layers/layer.h"

namespace ci {

class Conv2DLayer : public Layer {
public:
    Conv2DLayer(const int8_t* weights, const int32_t* bias,
                int kh, int kw, int c_in, int c_out,
                float w_scale, float out_scale, int32_t out_zp)
        : weights_(weights), bias_(bias),
          kh_(kh), kw_(kw), c_in_(c_in), c_out_(c_out),
          w_scale_(w_scale), out_scale_(out_scale), out_zp_(out_zp) {}

    void forward(const Tensor& input, Tensor& output,
                 ArenaAllocator& arena) override;
    LayerType type() const override { return LayerType::CONV2D; }
    const char* name() const override { return "Conv2D"; }
    size_t output_size_bytes(const TensorShape& s) const override {
        int oh = s[1] - kh_ + 1;
        int ow = s[2] - kw_ + 1;
        return static_cast<size_t>(oh * ow * c_out_);
    }

private:
    const int8_t* weights_;
    const int32_t* bias_;
    int kh_, kw_, c_in_, c_out_;
    float w_scale_, out_scale_;
    int32_t out_zp_;
};

} // namespace ci