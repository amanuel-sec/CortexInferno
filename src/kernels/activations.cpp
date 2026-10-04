#include "kernels/activations.h"
#include <cmath>
#include <algorithm>

namespace ci { namespace kernels {

void relu_int8(int8_t* data, size_t n, int8_t zero_point) {
    for (size_t i = 0; i < n; ++i) {
        if (data[i] < zero_point) data[i] = zero_point;
    }
}

void relu_float(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

void softmax_float(const float* input, float* output, size_t n) {
    float max_val = *std::max_element(input, input + n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for (size_t i = 0; i < n; ++i) {
        output[i] /= sum;
    }
}

}} // namespace ci::kernels