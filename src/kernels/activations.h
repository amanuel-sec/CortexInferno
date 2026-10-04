#pragma once
#include <cstdint>
#include <cstddef>

namespace ci { namespace kernels {

void relu_int8(int8_t* data, size_t n, int8_t zero_point);
void relu_float(float* data, size_t n);
void softmax_float(const float* input, float* output, size_t n);

}} // namespace ci::kernels