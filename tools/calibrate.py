#!/usr/bin/env python3
"""
CortexInferno Quantization Calibrator
─────────────────────────────────────
Runs sample data through an ONNX model to collect per-tensor
activation ranges for post-training quantization.
"""

import argparse
import numpy as np

def calibrate(model_path: str, num_samples: int = 100):
    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ pip install onnxruntime")
        return

    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape

    # Replace dynamic dims
    input_shape = [1 if (d is None or d == 'batch') else d for d in input_shape]

    print(f"📊 Calibrating with {num_samples} random samples...")
    print(f"   Input: {input_name} shape={input_shape}")

    all_outputs = {}
    for i in range(num_samples):
        dummy = np.random.randn(*input_shape).astype(np.float32)
        outputs = sess.run(None, {input_name: dummy})
        for j, out in enumerate(outputs):
            if j not in all_outputs:
                all_outputs[j] = {'min': float('inf'), 'max': float('-inf')}
            all_outputs[j]['min'] = min(all_outputs[j]['min'], float(out.min()))
            all_outputs[j]['max'] = max(all_outputs[j]['max'], float(out.max()))

    print("\n📋 Per-tensor activation ranges:")
    for j, stats in all_outputs.items():
        scale = (stats['max'] - stats['min']) / 255.0
        zp = int(round(-stats['min'] / scale)) if scale > 0 else 0
        print(f"   Output {j}: min={stats['min']:.4f} max={stats['max']:.4f} "
              f"scale={scale:.6f} zp={zp}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--samples', type=int, default=100)
    args = parser.parse_args()
    calibrate(args.model, args.samples)