#pragma once
#include <cstdint>
#include <cstddef>

namespace ci { namespace kernels {

/// Quantize float32 → int8 using per-tensor affine quantization
void quantize_int8(const float* input, int8_t* output, size_t n,
                   float scale, int32_t zero_point);

/// Dequantize int8 → float32
void dequantize_int8(const int8_t* input, float* output, size_t n,
                     float scale, int32_t zero_point);

/// Quantize float32 → int4 (packed: 2 values per byte, low nibble first)
void quantize_int4(const float* input, uint8_t* output, size_t n,
                   float scale, int32_t zero_point);

/// Dequantize packed int4 → float32
void dequantize_int4(const uint8_t* input, float* output, size_t n,
                     float scale, int32_t zero_point);

/// Compute per-tensor quantization parameters from min/max
void compute_quant_params(const float* data, size_t n,
                          float& out_scale, int32_t& out_zp);

}} // namespace ci::kernels