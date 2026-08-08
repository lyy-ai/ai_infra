#!/usr/bin/env python3
"""运行 Qwen2-0.5B vLLM 离线吞吐 benchmark。"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as cfg  # noqa: E402
from src.offline_benchmark import run_offline_benchmark  # noqa: E402
from src.utils import save_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run Qwen2-0.5B vLLM offline throughput benchmark.")
    parser.add_argument("--output", default=os.path.join(cfg.RESULTS_DIR, "offline_throughput.json"))
    args = parser.parse_args()

    result = run_offline_benchmark()
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(PROJECT_ROOT, output_path)
    save_json(result, output_path)

    print("saved:", output_path)
    for record in result["records"]:
        print(
            f"batch={record['batch_size']:>3} "
            f"elapsed={record['elapsed_s']:.3f}s "
            f"tok/s={record['tok_per_s']:.2f} "
            f"req/s={record['req_per_s']:.2f}"
        )


if __name__ == "__main__":
    main()
