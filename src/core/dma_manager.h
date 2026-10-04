#pragma once
// ─────────────────────────────────────────────────────────
// CortexInferno — DMA Double-Buffering State Machine
// ─────────────────────────────────────────────────────────
// Overlaps CPU compute (Layer N) with DMA load (Layer N+1)
// using ping-pong buffers. Interrupt-driven.
// ─────────────────────────────────────────────────────────

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <functional>

namespace ci {

class DMAManager {
public:
    enum class State : uint8_t {
        IDLE,
        LOADING,
        READY
    };

    using CompletionCallback = std::function<void()>;

    DMAManager() = default;

    /// Initialize with two pre-allocated ping-pong buffers
    void init(uint8_t* buf_a, uint8_t* buf_b, size_t buf_size) noexcept {
        buffers_[0] = buf_a;
        buffers_[1] = buf_b;
        buf_size_ = buf_size;
        active_idx_ = 0;
        state_.store(State::IDLE, std::memory_order_release);
    }

    /// Start async DMA load of `size` bytes from `src` into the
    /// INACTIVE buffer. Returns false if a transfer is already running.
    bool start_async_load(const uint8_t* src, size_t size) noexcept {
        State expected = State::IDLE;
        if (!state_.compare_exchange_strong(expected, State::LOADING,
                std::memory_order_acq_rel))
            return false; // Already loading

        load_src_ = src;
        load_size_ = size;
        uint8_t* dest = buffers_[1 - active_idx_];

        // ── Platform-specific DMA start ──
        // On STM32H7: HAL_DMA_Start_IT(&hdma, (uint32_t)src, (uint32_t)dest, size);
        // On host/CI:  fallback to memcpy
        platform_dma_start(src, dest, size);

        return true;
    }

    /// Call from DMA Transfer-Complete ISR
    void on_transfer_complete() noexcept {
        active_idx_ = 1 - active_idx_; // Swap
        state_.store(State::READY, std::memory_order_release);
        if (callback_) callback_();
    }

    /// Get pointer to the buffer ready for compute
    [[nodiscard]] uint8_t* get_compute_buffer() noexcept {
        return buffers_[active_idx_];
    }

    /// Mark compute done, transition to IDLE so next load can start
    void mark_compute_done() noexcept {
        state_.store(State::IDLE, std::memory_order_release);
    }

    [[nodiscard]] State state() const noexcept {
        return state_.load(std::memory_order_acquire);
    }

    void set_callback(CompletionCallback cb) { callback_ = cb; }

    /// Wait until current transfer completes (polling fallback)
    void wait_ready() noexcept {
        while (state_.load(std::memory_order_acquire) == State::LOADING) {
            // On real MCU: __WFI(); // Wait for interrupt
        }
    }

private:
    void platform_dma_start(const uint8_t* src, uint8_t* dest, size_t size) {
#if defined(CI_USE_SCALAR) || !defined(__arm__)
        // Host fallback: synchronous memcpy
        for (size_t i = 0; i < size; ++i) dest[i] = src[i];
        on_transfer_complete(); // Immediate completion
#else
        // Real STM32 HAL call would go here:
        // HAL_DMA_Start_IT(&hdma, (uint32_t)src, (uint32_t)dest, size);
        (void)src; (void)dest; (void)size;
#endif
    }

    uint8_t* buffers_[2] = {nullptr, nullptr};
    size_t buf_size_ = 0;
    int active_idx_ = 0;
    std::atomic<State> state_{State::IDLE};
    const uint8_t* load_src_ = nullptr;
    size_t load_size_ = 0;
    CompletionCallback callback_;
};

} // namespace ci