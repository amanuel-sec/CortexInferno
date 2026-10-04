#pragma once
#include "runtime/graph_executor.h"

namespace ci {

struct InferenceConfig {
    uint8_t* arena_buffer;
    size_t arena_size;
    int task_priority;
    size_t stack_size;
};

/// Initialize the inference engine as an RTOS task
void inference_engine_init(GraphExecutor* executor, const InferenceConfig& cfg);

/// Trigger an inference (can be called from any RTOS task)
bool trigger_inference(const Tensor& input);

/// Get last inference result (blocking)
Tensor* get_inference_result();

} // namespace ci