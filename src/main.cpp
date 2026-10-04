#include "cortex_inferno/cortex_inferno.h"
#include <cstdio>
#include <cstring>

int main() {
    printf("═══════════════════════════════════════════\n");
    printf(" CortexInferno v1.0 — TinyML Engine\n");
    printf("═══════════════════════════════════════════\n");

    // 1. Allocate arena (64KB for demo)
    static uint8_t arena_buf[64 * 1024] __attribute__((aligned(16)));
    ci::ArenaAllocator arena(arena_buf, sizeof(arena_buf));

    // 2. Build a simple 2-layer model: Dense(784→128) → ReLU → Dense(128→10) → Softmax
    // (Using dummy weights for demo — real weights come from model_converter.py)
    static int8_t w1[784 * 128] = {};
    static int32_t b1[128] = {};
    static int8_t w2[128 * 10] = {};
    static int32_t b2[10] = {};

    ci::GraphExecutor graph;

    ci::TensorShape s1_out; s1_out.ndim = 2; s1_out.dims = {1, 128};
    graph.add_layer({
        std::make_unique<ci::DenseLayer>(w1, b1, 784, 128, 0.01f, 0, 0.01f, 0),
        s1_out, ci::Dtype::INT8, {0.01f, 0}
    });

    ci::TensorShape s2_out = s1_out;
    graph.add_layer({
        std::make_unique<ci::ReLULayer>(),
        s2_out, ci::Dtype::INT8, {0.01f, 0}
    });

    ci::TensorShape s3_out; s3_out.ndim = 2; s3_out.dims = {1, 10};
    graph.add_layer({
        std::make_unique<ci::DenseLayer>(w2, b2, 128, 10, 0.01f, 0, 0.01f, 0),
        s3_out, ci::Dtype::INT8, {0.01f, 0}
    });

    ci::TensorShape s4_out; s4_out.ndim = 2; s4_out.dims = {1, 10};
    graph.add_layer({
        std::make_unique<ci::SoftmaxLayer>(),
        s4_out, ci::Dtype::FLOAT32, {}
    });

    graph.set_arena(&arena);

    // 3. Create dummy input
    static int8_t input_data[784] = {};
    ci::TensorShape in_shape; in_shape.ndim = 2; in_shape.dims = {1, 784};
    ci::Tensor input(input_data, in_shape, ci::Dtype::INT8, {0.01f, 0});

    // 4. Run inference
    arena.reset_hwm();
    ci::Tensor* result = graph.execute(input);

    printf("Inference complete.\n");
    printf("Arena peak usage: %zu / %zu bytes (%.1f%%)\n",
           arena.high_water_mark(), arena.capacity(),
           100.0f * arena.high_water_mark() / arena.capacity());

    if (result) {
        printf("Output classes: ");
        for (int i = 0; i < 10; ++i)
            printf("%.4f ", result->data_as<float>()[i]);
        printf("\n");
    }

    return 0;
}