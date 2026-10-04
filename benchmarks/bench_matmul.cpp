#include "kernels/scalar_matmul_int8.h"
#include <cstdio>
#include <chrono>
#include <vector>
#include <random>

int main() {
    const int M = 1, K = 784, N = 128;
    std::vector<int8_t> A(M*K), BT(N*K);
    std::vector<int32_t> C(M*N);

    std::mt19937 rng(42);
    for (auto& v : A) v = (int8_t)(rng() % 256 - 128);
    for (auto& v : BT) v = (int8_t)(rng() % 256 - 128);

    // Warmup
    ci::kernels::scalar_matmul_int8(A.data(), BT.data(), C.data(), M, N, K);

    // Benchmark
    const int iters = 1000;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        ci::kernels::scalar_matmul_int8(A.data(), BT.data(), C.data(), M, N, K);
    auto t1 = std::chrono::high_resolution_clock::now();

    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    printf("kernel,latency_us\n");
    printf("scalar_matmul_int8_1x784x128,%.1f\n", us);
    printf(" Benchmark: %.1f µs per inference (Dense 784→128)\n", us);
    return 0;
}