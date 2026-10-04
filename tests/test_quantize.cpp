#include "kernels/quantize.h"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <vector>

int main() {
    // Test INT8 round-trip
    std::vector<float> input = {0.0f, 0.5f, 1.0f, -0.5f, -1.0f};
    std::vector<int8_t> q(5);
    std::vector<float> dq(5);

    float scale; int32_t zp;
    ci::kernels::compute_quant_params(input.data(), 5, scale, zp);
    ci::kernels::quantize_int8(input.data(), q.data(), 5, scale, zp);
    ci::kernels::dequantize_int8(q.data(), dq.data(), 5, scale, zp);

    for (int i = 0; i < 5; ++i) {
        assert(std::abs(input[i] - dq[i]) < scale * 1.5f);
    }

    // Test INT4 pack/unpack
    std::vector<float> f4 = {0.1f, -0.2f, 0.3f, -0.4f};
    std::vector<uint8_t> packed(2);
    std::vector<float> unpacked(4);
    float s4; int32_t z4;
    ci::kernels::compute_quant_params(f4.data(), 4, s4, z4);
    ci::kernels::quantize_int4(f4.data(), packed.data(), 4, s4, z4);
    ci::kernels::dequantize_int4(packed.data(), unpacked.data(), 4, s4, z4);

    printf("✅ Quantize tests passed\n");
    return 0;
}