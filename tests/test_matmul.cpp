#include "kernels/scalar_matmul_int8.h"
#include <cassert>
#include <cstdio>
#include <vector>

int main() {
    // 2×3 * 3×2 = 2×2
    // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
    // B_T = [[7,9,11],[8,10,12]]
    // C = [[58,64],[139,154]]
    int8_t A[] = {1,2,3, 4,5,6};
    int8_t BT[] = {7,9,11, 8,10,12};
    int32_t C[4] = {};

    ci::kernels::scalar_matmul_int8(A, BT, C, 2, 2, 3);

    assert(C[0] == 58);
    assert(C[1] == 64);
    assert(C[2] == 139);
    assert(C[3] == 154);

    printf("✅ Matmul tests passed\n");
    return 0;
}