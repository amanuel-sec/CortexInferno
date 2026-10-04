#!/usr/bin/env python3
"""
CortexInferno Benchmark Visualizer
───────────────────────────────────
Parses benchmark CSV output and generates comparison charts.
"""

import argparse
import csv
import sys

def visualize(csv_path: str, output_png: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("❌ pip install matplotlib")
        return

    kernels = []
    latencies = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernels.append(row['kernel'])
            latencies.append(float(row['latency_us']))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(kernels, latencies, color=['#2196F3', '#FF9800', '#4CAF50', '#F44336'])
    ax.set_xlabel('Latency (µs)')
    ax.set_title('CortexInferno Kernel Benchmarks')

    for bar, val in zip(bars, latencies):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}µs', va='center')

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    print(f"✅ Saved chart to {output_png}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--output', default='benchmark.png')
    args = parser.parse_args()
    visualize(args.csv, args.output)