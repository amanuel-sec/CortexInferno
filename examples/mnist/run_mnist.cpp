#include "cortex_inferno/cortex_inferno.h"
#include "mnist_weights.h"
#include <cstdio>

int main() {
    printf("🔥 CortexInferno MNIST Demo\n");

    static uint8_t arena_buf[32 * 1024];
    ci::ArenaAllocator arena(arena_buf, sizeof(arena_buf));

    ci::GraphExecutor graph;

    // Layer 1: Dense 784 -> 16
    ci::TensorShape s1; s1.ndim=2; s1.dims={1, 16};
    ci::LayerConfig cfg1;
    cfg1.layer = std::make_unique<ci::DenseLayer>(
        mnist_model::dense1_weights, mnist_model::dense1_bias,
        784, 16,
        mnist_model::dense1_scale, mnist_model::dense1_zp,
        0.01f, 0);
    cfg1.output_shape = s1;
    cfg1.output_dtype = ci::Dtype::INT8;
    cfg1.output_qp = {0.01f, 0};
    graph.add_layer(std::move(cfg1));

    // Layer 2: ReLU
    ci::LayerConfig cfg2;
    cfg2.layer = std::make_unique<ci::ReLULayer>();
    cfg2.output_shape = s1;
    cfg2.output_dtype = ci::Dtype::INT8;
    cfg2.output_qp = {0.01f, 0};
    graph.add_layer(std::move(cfg2));

    // Layer 3: Dense 16 -> 10
    ci::TensorShape s2; s2.ndim=2; s2.dims={1, 10};
    ci::LayerConfig cfg3;
    cfg3.layer = std::make_unique<ci::DenseLayer>(
        mnist_model::dense2_weights, mnist_model::dense2_bias,
        16, 10,
        mnist_model::dense2_scale, mnist_model::dense2_zp,
        0.01f, 0);
    cfg3.output_shape = s2;
    cfg3.output_dtype = ci::Dtype::INT8;
    cfg3.output_qp = {0.01f, 0};
    graph.add_layer(std::move(cfg3));

    // Layer 4: Softmax
    ci::LayerConfig cfg4;
    cfg4.layer = std::make_unique<ci::SoftmaxLayer>();
    cfg4.output_shape = s2;
    cfg4.output_dtype = ci::Dtype::FLOAT32;
    cfg4.output_qp = {};
    graph.add_layer(std::move(cfg4));

    graph.set_arena(&arena);

    static int8_t input_data[784] = {};
    ci::TensorShape in_s; in_s.ndim=2; in_s.dims={1, 784};
    ci::Tensor input(input_data, in_s, ci::Dtype::INT8, {0.01f, 0});

    ci::Tensor* result = graph.execute(input);
    if (result) {
        printf("Predictions: ");
        for (int i = 0; i < 10; ++i)
            printf("%.4f ", result->data_as<float>()[i]);
        printf("\n");
    }
    printf("Peak arena: %zu bytes\n", arena.high_water_mark());
    return 0;
}
