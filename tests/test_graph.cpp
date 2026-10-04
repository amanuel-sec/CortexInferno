#include "runtime/graph_executor.h"
#include "layers/dense_layer.h"
#include "layers/activation_layer.h"
#include <cstdio>
#include <cstring>
#include <cassert>

int main() {
    static uint8_t arena_buf[8192];
    ci::ArenaAllocator arena(arena_buf, sizeof(arena_buf));

    static int8_t w[4*3] = {1,0,0, 0,1,0, 0,0,1, 1,1,1};
    static int32_t b[3] = {0,0,0};

    ci::GraphExecutor graph;
    ci::TensorShape out_s; out_s.ndim=2; out_s.dims={1,3};
    
    ci::LayerConfig cfg;
    cfg.layer = std::make_unique<ci::DenseLayer>(w, b, 4, 3, 0.01f, 0, 0.01f, 0);
    cfg.output_shape = out_s;
    cfg.output_dtype = ci::Dtype::INT8;
    cfg.output_qp = {0.01f, 0};
    
    graph.add_layer(std::move(cfg));
    graph.set_arena(&arena);

    int8_t inp[4] = {10, 20, 30, 40};
    ci::TensorShape in_s; in_s.ndim=2; in_s.dims={1,4};
    ci::Tensor input(inp, in_s, ci::Dtype::INT8, {0.01f, 0});

    ci::Tensor* result = graph.execute(input);
    assert(result != nullptr);
    printf("✅ Graph executor tests passed\n");
    return 0;
}
