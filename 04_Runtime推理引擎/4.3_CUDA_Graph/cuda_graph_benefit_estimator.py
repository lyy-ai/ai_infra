#!/usr/bin/env python3
"""
CUDA Graph 收益估算：Kernel Launch Overhead、静态图捕获与重放。

运行方式：
    python 4.3_CUDA_Graph/cuda_graph_benefit_estimator.py
"""


OPS = [
    ("qkv_proj", 18.0),
    ("attention", 42.0),
    ("out_proj", 12.0),
    ("layernorm", 4.0),
    ("mlp_gate_up", 20.0),
    ("mlp_down", 16.0),
    ("residual_add", 3.0),
]

LAUNCH_OVERHEAD_US = 5.0
SYNC_OVERHEAD_US = 3.0
GRAPH_CAPTURE_US = 120.0
GRAPH_REPLAY_LAUNCH_US = 8.0


def no_graph_time_us() -> float:
    """无 CUDA Graph：每个 kernel 都有 launch 开销。"""
    kernel = sum(t for _, t in OPS)
    launch = len(OPS) * LAUNCH_OVERHEAD_US
    return kernel + launch + SYNC_OVERHEAD_US


def graph_time_us(replays: int) -> float:
    """CUDA Graph：capture 一次，replay N 次。"""
    kernel = sum(t for _, t in OPS) * replays
    replay = replays * (GRAPH_REPLAY_LAUNCH_US + SYNC_OVERHEAD_US)
    return GRAPH_CAPTURE_US + kernel + replay


def main():
    print("=" * 78)
    print("CUDA Graph Benefit Estimator")
    print("=" * 78)
    print(f"ops: {len(OPS)}, launch overhead: {LAUNCH_OVERHEAD_US} us/kernel")
    print(f"capture cost: {GRAPH_CAPTURE_US} us, replay launch: {GRAPH_REPLAY_LAUNCH_US} us")

    print("\nOps:")
    for name, time_us in OPS:
        print(f"  {name:<14} {time_us:>6.1f} us")

    baseline = no_graph_time_us()
    print("\n" + "=" * 78)
    print("No graph per iteration:")
    print(f"  {baseline:.1f} us")
    print("\nGraph total time by replays:")
    print(f"{'replays':>8} {'graph_total_us':>16} {'graph_per_iter':>16} {'speedup':>10}")
    print("-" * 78)

    breakeven = None
    for replays in [1, 2, 4, 8, 16, 32, 64, 128]:
        total = graph_time_us(replays)
        per_iter = total / replays
        speedup = baseline / per_iter
        if breakeven is None and speedup >= 1.0:
            breakeven = replays
        print(f"{replays:>8} {total:>16.1f} {per_iter:>16.2f} {speedup:>10.2f}")

    print("-" * 78)
    print(f"break-even replays: {breakeven}")
    print("\nTakeaway:")
    print("- CUDA Graph helps most when there are many small kernels and frequent launches.")
    print("- Static shape is required: batch size, seq length bucket, memory addresses must be fixed per captured graph.")
    print("- Amortize capture cost over many replays; one-off dynamic shapes should not capture.")


if __name__ == "__main__":
    main()
