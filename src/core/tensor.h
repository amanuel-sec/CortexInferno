#pragma once
// ─────────────────────────────────────────────────────────
// CortexInferno — Lightweight Tensor Abstraction
// ─────────────────────────────────────────────────────────

#include <cstdint>
#include <cstddef>
#include <array>
#include <numeric>
#include <algorithm>

namespace ci {

enum class Dtype : uint8_t {
    FLOAT32 = 0,
    INT8    = 1,
    INT4    = 2,
    UINT8   = 3
};

inline size_t dtype_size(Dtype dt) noexcept {
    switch (dt) {
        case Dtype::FLOAT32: return 4;
        case Dtype::INT8:    return 1;
        case Dtype::INT4:    return 1; // 2 elements per byte
        case Dtype::UINT8:   return 1;
    }
    return 0;
}

static constexpr size_t MAX_DIMS = 4;

struct TensorShape {
    std::array<int, MAX_DIMS> dims{};
    int ndim = 0;

    [[nodiscard]] size_t numel() const noexcept {
        if (ndim == 0) return 0;
        size_t n = 1;
        for (int i = 0; i < ndim; ++i) n *= static_cast<size_t>(dims[i]);
        return n;
    }

    [[nodiscard]] int operator[](int i) const noexcept { return dims[i]; }
    [[nodiscard]] int& operator[](int i) noexcept { return dims[i]; }

    bool operator==(const TensorShape& o) const noexcept {
        if (ndim != o.ndim) return false;
        for (int i = 0; i < ndim; ++i)
            if (dims[i] != o.dims[i]) return false;
        return true;
    }
};

struct QuantParams {
    float scale = 1.0f;
    int32_t zero_point = 0;
};

class Tensor {
public:
    Tensor() = default;

    Tensor(void* data, TensorShape shape, Dtype dtype,
           QuantParams qp = {}) noexcept
        : data_(data), shape_(shape), dtype_(dtype), qp_(qp) {}

    [[nodiscard]] void* data() noexcept { return data_; }
    [[nodiscard]] const void* data() const noexcept { return data_; }

    template <typename T>
    [[nodiscard]] T* data_as() noexcept { return static_cast<T*>(data_); }

    template <typename T>
    [[nodiscard]] const T* data_as() const noexcept {
        return static_cast<const T*>(data_);
    }

    [[nodiscard]] TensorShape shape() const noexcept { return shape_; }
    [[nodiscard]] Dtype dtype() const noexcept { return dtype_; }
    [[nodiscard]] QuantParams quant_params() const noexcept { return qp_; }
    [[nodiscard]] size_t numel() const noexcept { return shape_.numel(); }

    [[nodiscard]] size_t byte_size() const noexcept {
        if (dtype_ == Dtype::INT4)
            return (numel() + 1) / 2;
        return numel() * dtype_size(dtype_);
    }

    void set_data(void* p) noexcept { data_ = p; }
    void set_quant_params(QuantParams qp) noexcept { qp_ = qp; }

private:
    void* data_ = nullptr;
    TensorShape shape_{};
    Dtype dtype_ = Dtype::FLOAT32;
    QuantParams qp_{};
};

} // namespace ci