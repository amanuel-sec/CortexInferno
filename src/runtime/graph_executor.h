#pragma once
#include "layers/layer.h"
#include "core/allocator.h"
#include "core/tensor.h"
#include <vector>
#include <memory>

namespace ci {

struct LayerConfig {
    std::unique_ptr<Layer> layer;
    TensorShape output_shape;
    Dtype output_dtype;
    QuantParams output_qp;
};

class GraphExecutor {
public:
    void add_layer(LayerConfig config);

    /// Run full inference. Input tensor must have data set.
    /// Returns pointer to final output tensor.
    Tensor* execute(const Tensor& input);

    /// Get total arena memory required
    [[nodiscard]] size_t required_arena_size() const { return arena_size_; }

    /// Set the arena to use
    void set_arena(ArenaAllocator* arena) { arena_ = arena; }

    [[nodiscard]] size_t num_layers() const { return layers_.size(); }

private:
    std::vector<LayerConfig> layers_;
    ArenaAllocator* arena_ = nullptr;
    size_t arena_size_ = 0;
    std::vector<Tensor> intermediate_tensors_;

    Tensor final_tensor_;  // <-- ADD THIS LINE
};

} // namespace ci