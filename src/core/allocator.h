#pragma once
// ─────────────────────────────────────────────────────────
// CortexInferno — Fragmentation-Free Arena Allocator
// ─────────────────────────────────────────────────────────
// Designed for MCU environments where heap fragmentation
// is a fatal error. Uses a bump allocator with atomic
// high-water-mark tracking. Zero-cost reset between
// inference frames.
// ─────────────────────────────────────────────────────────

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <cstring>

namespace ci {

class ArenaAllocator {
public:
    /// Construct arena over a caller-provided buffer.
    /// @param buffer  Pointer to pre-allocated SRAM region
    /// @param size    Size in bytes
    explicit ArenaAllocator(uint8_t* buffer, size_t size) noexcept
        : buffer_(buffer), capacity_(size), offset_(0), hwm_(0) {}

    // Non-copyable, non-movable
    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    /// Allocate `size` bytes with given alignment.
    /// Returns nullptr on OOM. Thread-safe via atomic CAS.
    [[nodiscard]] void* allocate(size_t size, size_t alignment = 16) noexcept {
        if (size == 0) return nullptr;

        size_t current = offset_.load(std::memory_order_relaxed);
        size_t aligned, next;

        do {
            aligned = (current + alignment - 1) & ~(alignment - 1);
            next = aligned + size;
            if (next > capacity_) return nullptr;  // OOM
        } while (!offset_.compare_exchange_weak(
            current, next,
            std::memory_order_release,
            std::memory_order_relaxed));

        update_hwm(next);
        return buffer_ + aligned;
    }

    /// Typed allocation helper
    template <typename T>
    [[nodiscard]] T* allocate_typed(size_t count = 1) noexcept {
        return static_cast<T*>(allocate(count * sizeof(T), alignof(T)));
    }

    /// Reset arena for next inference frame. O(1).
    void reset() noexcept {
        offset_.store(0, std::memory_order_release);
    }

    /// Peak memory usage since last hwm reset
    [[nodiscard]] size_t high_water_mark() const noexcept {
        return hwm_.load(std::memory_order_relaxed);
    }

    /// Reset HWM counter (call at start of profiling run)
    void reset_hwm() noexcept {
        hwm_.store(0, std::memory_order_relaxed);
    }

    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] size_t used() const noexcept {
        return offset_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] size_t remaining() const noexcept {
        return capacity_ - used();
    }

private:
    void update_hwm(size_t val) noexcept {
        size_t cur = hwm_.load(std::memory_order_relaxed);
        while (val > cur &&
               !hwm_.compare_exchange_weak(cur, val,
                   std::memory_order_release,
                   std::memory_order_relaxed)) {}
    }

    uint8_t* const buffer_;
    const size_t capacity_;
    std::atomic<size_t> offset_;
    std::atomic<size_t> hwm_;
};

/// RAII guard that resets arena on scope exit
class ArenaScope {
public:
    explicit ArenaScope(ArenaAllocator& arena) noexcept : arena_(arena) {}
    ~ArenaScope() noexcept { arena_.reset(); }
    ArenaScope(const ArenaScope&) = delete;
    ArenaScope& operator=(const ArenaScope&) = delete;
private:
    ArenaAllocator& arena_;
};

} // namespace ci