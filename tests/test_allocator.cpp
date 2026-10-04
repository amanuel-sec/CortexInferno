#include "core/allocator.h"
#include <cassert>
#include <cstdio>

int main() {
    uint8_t buf[1024];
    ci::ArenaAllocator arena(buf, sizeof(buf));

    // Basic allocation
    void* p1 = arena.allocate(64, 16);
    assert(p1 != nullptr);
    assert(reinterpret_cast<uintptr_t>(p1) % 16 == 0);

    void* p2 = arena.allocate(128, 16);
    assert(p2 != nullptr);
    assert(p2 != p1);

    // HWM tracking
    assert(arena.high_water_mark() >= 192);

    // Reset
    arena.reset();
    assert(arena.used() == 0);
    assert(arena.high_water_mark() >= 192); // HWM persists

    // OOM
    arena.reset();
    void* big = arena.allocate(2048);
    assert(big == nullptr);

    printf("✅ Allocator tests passed\n");
    return 0;
}