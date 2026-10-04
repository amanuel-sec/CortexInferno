#pragma once
// ─────────────────────────────────────────────────────────
// CortexInferno — Static Memory Planner
// ─────────────────────────────────────────────────────────
// Pre-computes arena offsets for all intermediate tensors
// so that inference runs with zero dynamic allocation.
// Uses a greedy linear-scan algorithm.
// ─────────────────────────────────────────────────────────

#include "core/tensor.h"
#include <vector>
#include <cstdint>

namespace ci {

struct TensorLifetime {
    int tensor_id;
    size_t size_bytes;
    size_t alignment;
    int birth_layer;  // First layer that writes this tensor
    int death_layer;  // Last layer that reads this tensor
};

struct AllocationPlan {
    int tensor_id;
    size_t offset;
    size_t size;
};

class MemoryPlanner {
public:
    /// Plan memory layout for a set of tensor lifetimes.
    /// Returns total arena size needed.
    size_t plan(const std::vector<TensorLifetime>& lifetimes,
                std::vector<AllocationPlan>& out_plan);

    [[nodiscard]] size_t peak_memory() const noexcept { return peak_; }

private:
    size_t peak_ = 0;
};

} // namespace ci