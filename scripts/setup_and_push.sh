#!/usr/bin/env bash
set -euo pipefail

echo "🔥 CortexInferno — Setup & Push to GitHub"
echo "═══════════════════════════════════════════"

# 1. Generate demo weights
echo "📦 Generating demo model weights..."
python3 tools/model_converter.py --demo --output examples/mnist/mnist_weights.h

# 2. Build & test (host scalar)
echo "🏗️  Building (CI mode)..."
cmake -B build -DCI_BUILD=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

echo "🧪 Running tests..."
cd build && ctest --output-on-failure && cd ..

echo "📊 Running benchmarks..."
./build/bench_matmul > benchmarks/results.csv
cat benchmarks/results.csv

# 3. Git init & push
echo ""
read -p "📌 Enter your GitHub repo URL (e.g., git@github.com:user/CortexInferno.git): " REPO_URL

git init
git add -A
git commit -m "🔥 Initial commit: CortexInferno v1.0

Production-grade TinyML inference engine from scratch.

Features:
- Fragmentation-free arena allocator with atomic HWM tracking
- ARM NEON SIMD INT8 matrix multiplication (4×4 tiled)
- ARM DSP dual-MAC optimized kernels (Cortex-M4/M7)
- DMA double-buffering state machine for compute/memory overlap
- INT8 and INT4 quantization with per-tensor calibration
- Complete layer library: Dense, Conv2D, MaxPool, ReLU, Softmax, Flatten
- Graph executor with static memory planning
- FreeRTOS task integration
- Python model converter (ONNX → C-header)
- CI/CD with cross-compilation and static analysis

Target: STM32H7 (Cortex-M7), ESP32-S3, AArch64 Edge"

git branch -M main
git remote add origin "$REPO_URL"
git push -u origin main

echo ""
echo "✅ CortexInferno pushed to $REPO_URL"
echo "🚀 You are now in the top 0.01%."