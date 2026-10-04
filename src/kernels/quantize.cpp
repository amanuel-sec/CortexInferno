#include "kernels/quantize.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace ci { namespace kernels {

void compute_quant_params(const float* data, size_t n,
                          float& out_scale, int32_t& out_zp) {
    float min_val = *std::min_element(data, data + n);
    float max_val = *std::max_element(data, data + n);

    // Ensure zero is representable
    min_val = std::min(min_val, 0.0f);
    max_val = std::max(max_val, 0.0f);

    out_scale = (max_val - min_val) / 255.0f;
    if (out_scale == 0.0f) out_scale = 1.0f;
    out_zp = static_cast<int32_t>(std::round(-min_val / out_scale));
    out_zp = std::max(0, std::min(255, out_zp));
}

void quantize_int8(const float* input, int8_t* output, size_t n,
                   float scale, int32_t zero_point) {
    for (size_t i = 0; i < n; ++i) {
        float val = std::round(input[i] / scale) + static_cast<float>(zero_point);
        val = std::max(-128.0f, std::min(127.0f, val));
        output[i] = static_cast<int8_t>(val);
    }
}

void dequantize_int8(const int8_t* input, float* output, size_t n,
                     float scale, int32_t zero_point) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = (static_cast<float>(input[i]) - static_cast<float>(zero_point)) * scale;
    }
}

void quantize_int4(const float* input, uint8_t* output, size_t n,
                   float scale, int32_t zero_point) {
    for (size_t i = 0; i < n; ++i) {
        float val = std::round(input[i] / scale) + static_cast<float>(zero_point);
        int32_t q = static_cast<int32_t>(std::max(-8.0f, std::min(7.0f, val)));
        uint8_t nibble = static_cast<uint8_t>(q & 0x0F);

        if (i % 2 == 0) {
            output[i / 2] = nibble;        // Low nibble
        } else {
            output[i / 2] |= (nibble << 4); // High nibble
        }
    }
}

void dequantize_int4(const uint8_t* input, float* output, size_t n,
                     float scale, int32_t zero_point) {
    for (size_t i = 0; i < n; ++i) {
        uint8_t byte = input[i / 2];
        int32_t nibble;
        if (i % 2 == 0) {
            nibble = byte & 0x0F;
        } else {
            nibble = (byte >> 4) & 0x0F;
        }
        // Sign extend 4-bit to 32-bit
        if (nibble & 0x08) nibble |= 0xFFFFFFF0;

        output[i] = (static_cast<float>(nibble) - static_cast<float>(zero_point)) * scale;
    }
}

}} // namespace ci::kernels