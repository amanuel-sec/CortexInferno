#include "kernels/dsp_matmul_int8.h"

#if defined(CI_USE_DSP) && defined(__ARM_ARCH_7EM__)
#include <arm_acle.h> // ARM C Language Extensions for DSP intrinsics

namespace ci { namespace kernels {

void dsp_matmul_int8(const int8_t* A, const int8_t* B_T,
                     int32_t* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            const int8_t* a_ptr = &A[m * K];
            const int8_t* b_ptr = &B_T[n * K];
            int k = 0;

            // Process 4 elements at a time using DSP dual-MAC
            for (; k <= K - 4; k += 4) {
                // Pack 4 int8 into two 32-bit words (2 int16 each)
                int32_t a_word0 = __SXTB16(*(int32_t*)&a_ptr[k]);   // sign-extend bytes 0,2 to 16-bit
                int32_t a_word1 = __SXTB16(__ROR(*(int32_t*)&a_ptr[k], 8)); // bytes 1,3
                int32_t b_word0 = __SXTB16(*(int32_t*)&b_ptr[k]);
                int32_t b_word1 = __SXTB16(__ROR(*(int32_t*)&b_ptr[k], 8));

                // Dual signed multiply-accumulate
                acc = __SMLAD(a_word0, b_word0, acc);
                acc = __SMLAD(a_word1, b_word1, acc);
            }

            // Scalar tail
            for (; k < K; ++k) {
                acc += (int32_t)a_ptr[k] * (int32_t)b_ptr[k];
            }
            C[m * N + n] = acc;
        }
    }
}

}} // namespace ci::kernels

#else
namespace ci { namespace kernels {
void dsp_matmul_int8(const int8_t*, const int8_t*, int32_t*, int, int, int) {}
}} // namespace ci::kernels
#endif