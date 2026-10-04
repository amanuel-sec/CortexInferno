#include "runtime/rtos_interface.h"

#if defined(CI_USE_FREERTOS)
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"

namespace ci {

static GraphExecutor* g_executor = nullptr;
static ArenaAllocator* g_arena = nullptr;
static QueueHandle_t g_input_queue = nullptr;
static QueueHandle_t g_output_queue = nullptr;

static void inference_task(void* param) {
    (void)param;
    Tensor input;
    for (;;) {
        if (xQueueReceive(g_input_queue, &input, portMAX_DELAY) == pdTRUE) {
            Tensor* result = g_executor->execute(input);
            xQueueSend(g_output_queue, &result, portMAX_DELAY);
        }
    }
}

void inference_engine_init(GraphExecutor* executor, const InferenceConfig& cfg) {
    g_executor = executor;
    g_arena = new ArenaAllocator(cfg.arena_buffer, cfg.arena_size);
    executor->set_arena(g_arena);
    g_input_queue = xQueueCreate(1, sizeof(Tensor));
    g_output_queue = xQueueCreate(1, sizeof(Tensor*));
    xTaskCreate(inference_task, "Inference", cfg.stack_size,
                nullptr, cfg.task_priority, nullptr);
}

bool trigger_inference(const Tensor& input) {
    return xQueueSend(g_input_queue, &input, 0) == pdTRUE;
}

Tensor* get_inference_result() {
    Tensor* result = nullptr;
    xQueueReceive(g_output_queue, &result, portMAX_DELAY);
    return result;
}

} // namespace ci

#else
// Host/CI stub
namespace ci {
static GraphExecutor* g_exec = nullptr;
static ArenaAllocator* g_arena = nullptr;

void inference_engine_init(GraphExecutor* executor, const InferenceConfig& cfg) {
    g_exec = executor;
    g_arena = new ArenaAllocator(cfg.arena_buffer, cfg.arena_size);
    executor->set_arena(g_arena);
}
bool trigger_inference(const Tensor& input) {
    (void)input; return true;
}
Tensor* get_inference_result() { return nullptr; }
} // namespace ci
#endif