#!/usr/bin/env python3
"""分析 benchmark 结果并生成简历 bullet。"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as cfg  # noqa: E402


def load_result(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Analyze Qwen2-0.5B vLLM benchmark results.")
    parser.add_argument("--input", default=os.path.join(cfg.RESULTS_DIR, "offline_throughput.json"))
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(PROJECT_ROOT, input_path)
    if not os.path.exists(input_path):
        print(f"result not found: {input_path}")
        print("run first: python scripts/run_offline_benchmark.py")
        return 2

    result = load_result(input_path)
    records = result.get("records", [])
    if not records:
        print("empty records")
        return 2

    print("metadata:")
    for key, value in result.get("metadata", {}).items():
        print(f"  {key}: {value}")

    print("\nrecords:")
    print(f"{'batch':>6} {'elapsed_s':>10} {'tok/s':>12} {'req/s':>10}")
    for record in records:
        print(f"{record['batch_size']:>6} {record['elapsed_s']:>10.3f} {record['tok_per_s']:>12.2f} {record['req_per_s']:>10.2f}")

    best = max(records, key=lambda r: r["tok_per_s"])
    base = min(records, key=lambda r: r["batch_size"])
    speedup = best["tok_per_s"] / base["tok_per_s"] if base["tok_per_s"] > 0 else 0.0

    print("\nresume bullet template:")
    print(
        f"- 基于 vLLM 部署 Qwen2-0.5B-Instruct 推理服务，离线 batch 压测显示：batch={base['batch_size']} 时 "
        f"{base['tok_per_s']:.1f} tok/s，batch={best['batch_size']} 时 {best['tok_per_s']:.1f} tok/s"
        f"（{speedup:.2f}x）；max_model_len={result['metadata']['max_model_len']}，"
        f"prefix_caching={result['metadata']['enable_prefix_caching']}，enforce_eager={result['metadata']['enforce_eager']}。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
