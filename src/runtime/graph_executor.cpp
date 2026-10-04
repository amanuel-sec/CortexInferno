#include "runtime/graph_executor.h"
#include <cstdio>

namespace ci {

void GraphExecutor::add_layer(LayerConfig config) {
    layers_.push_back(std::move(config));
}

Tensor* GraphExecutor::execute(const Tensor& input) {
    if (!arena_) return nullptr;

    ArenaScope scope(*arena_); // Auto-reset on return

    Tensor current = input;

    for (size_t i = 0; i < layers_.size(); ++i) {
        auto& lc = layers_[i];

        // Allocate output buffer from arena
        size_t out_bytes = lc.layer->output_size_bytes(current.shape());
        if (lc.output_dtype == Dtype::FLOAT32) out_bytes *= 4;
        void* out_data = arena_->allocate(out_bytes);
        if (!out_data) return nullptr; // OOM

        Tensor output(out_data, lc.output_shape, lc.output_dtype, lc.output_qp);

        lc.layer->forward(current, output, *arena_);
        current = output;
    }

    // Store final tensor (data pointer is still valid within ArenaScope)
    final_tensor_ = current;
    return &final_tensor_;
}

} // namespace ci