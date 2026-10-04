cmake_minimum_required(VERSION 3.20)
project(CortexInferno VERSION 1.0.0 LANGUAGES C CXX ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# ── Options ──────────────────────────────────────────────
option(CI_BUILD          "Build for CI (host x86 with scalar fallback)" OFF)
option(ENABLE_NEON       "Enable ARM NEON SIMD kernels"                OFF)
option(ENABLE_DSP        "Enable ARM DSP intrinsics (Cortex-M4/M7)"    OFF)
option(ENABLE_FREERTOS   "Build with FreeRTOS integration"             OFF)
option(BUILD_TESTS       "Build unit tests"                            ON)
option(BUILD_BENCHMARKS  "Build micro-benchmarks"                      ON)
option(BUILD_EXAMPLES    "Build example applications"                  ON)

# ── Detect target architecture ──────────────────────────
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|armv8")
        set(ENABLE_NEON ON CACHE BOOL "" FORCE)
        message(STATUS "🔥 Detected AArch64 — enabling NEON")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|cortex-m")
        set(ENABLE_DSP ON CACHE BOOL "" FORCE)
        message(STATUS "🔥 Detected ARMv7-M — enabling DSP intrinsics")
    endif()
endif()

if(CI_BUILD)
    message(STATUS "🏗️  CI Build — scalar kernels only, tests enabled")
    set(BUILD_TESTS ON)
    set(BUILD_BENCHMARKS ON)
endif()

# ── Compiler flags ───────────────────────────────────────
add_compile_options(-Wall -Wextra -Wpedantic  -fno-exceptions -fno-rtti)

if(ENABLE_NEON)
    add_compile_options(-mfpu=neon-vfpv4 -mfloat-abi=hard)
    add_compile_definitions(CI_USE_NEON=1)
endif()

if(ENABLE_DSP)
    add_compile_definitions(CI_USE_DSP=1)
endif()

if(NOT ENABLE_NEON AND NOT ENABLE_DSP)
    add_compile_definitions(CI_USE_SCALAR=1)
endif()

# ── Core library ─────────────────────────────────────────
set(CORE_SOURCES
    src/core/allocator.cpp
    src/core/tensor.cpp
    src/core/dma_manager.cpp
    src/core/memory_planner.cpp
)

set(KERNEL_SOURCES
    src/kernels/scalar_matmul_int8.cpp
    src/kernels/quantize.cpp
    src/kernels/activations.cpp
    src/kernels/conv2d.cpp
    src/kernels/pooling.cpp
)

if(ENABLE_NEON)
    list(APPEND KERNEL_SOURCES src/kernels/neon_matmul_int8.cpp)
endif()
if(ENABLE_DSP)
    list(APPEND KERNEL_SOURCES src/kernels/dsp_matmul_int8.cpp)
endif()

set(LAYER_SOURCES
    src/layers/dense_layer.cpp
    src/layers/conv2d_layer.cpp
    src/layers/pooling_layer.cpp
    src/layers/activation_layer.cpp
    src/layers/flatten_layer.cpp
    src/layers/softmax_layer.cpp
)

set(RUNTIME_SOURCES
    src/runtime/graph_executor.cpp
    src/runtime/rtos_interface.cpp
)

add_library(cortex_inferno STATIC
    ${CORE_SOURCES}
    ${KERNEL_SOURCES}
    ${LAYER_SOURCES}
    ${RUNTIME_SOURCES}
)

target_include_directories(cortex_inferno PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ── Main executable ──────────────────────────────────────
add_executable(cortex_inferno_demo src/main.cpp)
target_link_libraries(cortex_inferno_demo PRIVATE cortex_inferno)

# ── Tests ────────────────────────────────────────────────
if(BUILD_TESTS)
    enable_testing()
    # Minimal test framework (no external deps for embedded)
    add_executable(test_allocator tests/test_allocator.cpp)
    target_link_libraries(test_allocator PRIVATE cortex_inferno)
    add_test(NAME AllocatorTest COMMAND test_allocator)

    add_executable(test_quantize tests/test_quantize.cpp)
    target_link_libraries(test_quantize PRIVATE cortex_inferno)
    add_test(NAME QuantizeTest COMMAND test_quantize)

    add_executable(test_matmul tests/test_matmul.cpp)
    target_link_libraries(test_matmul PRIVATE cortex_inferno)
    add_test(NAME MatmulTest COMMAND test_matmul)

    add_executable(test_graph tests/test_graph.cpp)
    target_link_libraries(test_graph PRIVATE cortex_inferno)
    add_test(NAME GraphTest COMMAND test_graph)
endif()

# ── Benchmarks ───────────────────────────────────────────
if(BUILD_BENCHMARKS)
    add_executable(bench_matmul benchmarks/bench_matmul.cpp)
    target_link_libraries(bench_matmul PRIVATE cortex_inferno)
endif()

# ── Examples ─────────────────────────────────────────────
if(BUILD_EXAMPLES)
    add_executable(mnist_demo examples/mnist/run_mnist.cpp)
    target_link_libraries(mnist_demo PRIVATE cortex_inferno)
endif()

message(STATUS "═══════════════════════════════════════")
message(STATUS " CortexInferno Configuration")
message(STATUS " NEON:    ${ENABLE_NEON}")
message(STATUS " DSP:     ${ENABLE_DSP}")
message(STATUS " FreeRTOS:${ENABLE_FREERTOS}")
message(STATUS " Tests:   ${BUILD_TESTS}")
message(STATUS "═══════════════════════════════════════")