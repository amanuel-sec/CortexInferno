#pragma once
#include <cstdint>

namespace mnist_model {

alignas(16) const int8_t dense1_weights[12544] = {0};
const int32_t dense1_bias[16] = {0};
constexpr float dense1_scale = 0.001539f;
constexpr int32_t dense1_zp = 127;

alignas(16) const int8_t dense2_weights[160] = {0};
const int32_t dense2_bias[10] = {0};
constexpr float dense2_scale = 0.002129f;
constexpr int32_t dense2_zp = 116;

struct ModelConfig {
    int num_layers = 2;
};
constexpr ModelConfig config;

} // namespace mnist_model
