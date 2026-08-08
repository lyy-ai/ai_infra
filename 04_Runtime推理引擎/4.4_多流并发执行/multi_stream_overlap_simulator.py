#!/usr/bin/env python3
"""
多流并发执行模拟器：CUDA Stream、异步执行、计算/通信重叠、事件同步。

运行方式：
    python 4.4_多流并发执行/multi_stream_overlap_simulator.py
"""


STAGES = [
    ("h2d_copy", 6.0),
    ("compute", 24.0),
    ("d2h_copy", 4.0),
]
NUM_FRAMES = 6
FRAME_INTERVAL_MS = 24.0


def simulate_single_stream():
    """单 stream：copy/compute/copy 串行。"""
    engine_free = 0.0
    rows = []
    for frame in range(NUM_FRAMES):
        arrival = frame * FRAME_INTERVAL_MS
        start = max(arrival, engine_free)
        end = start + sum(cost for _, cost in STAGES)
        rows.append((frame, start, end))
        engine_free = end
    return rows


def simulate_two_streams():
    """多 stream：H2D、compute、D2H 分开发射，相邻帧 overlap。"""
    h2d_free = 0.0
    compute_free = 0.0
    d2h_free = 0.0
    rows = []
    for frame in range(NUM_FRAMES):
        arrival = frame * FRAME_INTERVAL_MS
        h2d_start = max(arrival, h2d_free)
        h2d_end = h2d_start + STAGES[0][1]
        h2d_free = h2d_end

        compute_start = max(h2d_end, compute_free)
        compute_end = compute_start + STAGES[1][1]
        compute_free = compute_end

        d2h_start = max(compute_end, d2h_free)
        d2h_end = d2h_start + STAGES[2][1]
        d2h_free = d2h_end
        rows.append((frame, h2d_start, d2h_end))
    return rows


def print_rows(title, rows):
    print(f"\n[{title}]")
    print("-" * 72)
    print(f"{'frame':>6} {'start':>10} {'end':>10} {'latency':>10}")
    print("-" * 72)
    for frame, start, end in rows:
        print(f"{frame:>6} {start:>10.1f} {end:>10.1f} {end-start:>10.1f}")
    total = max(end for _, _, end in rows)
    fps = len(rows) / (total / 1000.0) if total > 0 else 0.0
    print("-" * 72)
    print(f"total span: {total:.1f} ms, approx throughput: {fps:.2f} FPS")


def main():
    print("=" * 72)
    print("Multi-Stream Overlap Simulator")
    print("=" * 72)
    for name, cost in STAGES:
        print(f"stage {name:<10} {cost:>5.1f} ms")

    single = simulate_single_stream()
    dual = simulate_two_streams()
    print_rows("Single Stream", single)
    print_rows("Two Streams", dual)

    print("\nTakeaway:")
    print("- Multi-stream helps when H2D/D2H copies can overlap with compute of other frames.")
    print("- Events are needed to express dependencies: compute waits H2D, D2H waits compute.")
    print("- If compute dominates, overlap mainly improves throughput, not single-frame latency.")


if __name__ == "__main__":
    main()
