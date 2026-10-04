#include "core/memory_planner.h"
#include <algorithm>

namespace ci {

size_t MemoryPlanner::plan(const std::vector<TensorLifetime>& lifetimes,
                           std::vector<AllocationPlan>& out_plan) {
    out_plan.clear();
    out_plan.reserve(lifetimes.size());

    // Sort by birth_layer
    auto sorted = lifetimes;
    std::sort(sorted.begin(), sorted.end(),
              [](const TensorLifetime& a, const TensorLifetime& b) {
                  return a.birth_layer < b.birth_layer;
              });

    // Greedy first-fit allocation
    struct Block { size_t offset; size_t size; int death_layer; };
    std::vector<Block> active_blocks;
    size_t arena_end = 0;

    for (const auto& lt : sorted) {
        // Free expired blocks
        active_blocks.erase(
            std::remove_if(active_blocks.begin(), active_blocks.end(),
                [&](const Block& b) { return b.death_layer < lt.birth_layer; }),
            active_blocks.end());

        // First-fit search
        size_t aligned_offset = 0;
        bool found = false;

        // Sort active by offset for gap scanning
        std::sort(active_blocks.begin(), active_blocks.end(),
                  [](const Block& a, const Block& b) { return a.offset < b.offset; });

        size_t candidate = 0;
        for (const auto& blk : active_blocks) {
            size_t aligned = (candidate + lt.alignment - 1) & ~(lt.alignment - 1);
            // FIX 1: Changed lt.size to lt.size_bytes
            if (aligned + lt.size_bytes <= blk.offset) { 
                aligned_offset = aligned;
                found = true;
                break;
            }
            candidate = blk.offset + blk.size;
        }

        if (!found) {
            aligned_offset = (arena_end + lt.alignment - 1) & ~(lt.alignment - 1);
        }

        // FIX 2: Explicitly construct the structs for push_back
        out_plan.push_back(AllocationPlan{lt.tensor_id, aligned_offset, lt.size_bytes});
        active_blocks.push_back(Block{aligned_offset, lt.size_bytes, lt.death_layer});

        size_t end = aligned_offset + lt.size_bytes;
        if (end > arena_end) arena_end = end;
    }

    peak_ = arena_end;
    return arena_end;
}

} // namespace ci