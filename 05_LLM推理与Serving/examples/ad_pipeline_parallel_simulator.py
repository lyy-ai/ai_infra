#!/usr/bin/env python3
"""
自动驾驶推理流水线并行优化模拟器。

本脚本对比两种执行方式：
1. Sequential：一帧的 preprocess -> BEV perception -> prediction -> PlanNN -> control 全部完成后，才处理下一帧。
2. Pipelined：不同阶段使用不同执行单元，帧与帧之间流水重叠。

无需 GPU，可直接运行。

运行方式：
    python examples/ad_pipeline_parallel_simulator.py
"""


STAGES = [
    ("preprocess_cpu", 8.0),
    ("bev_perception_gpu", 28.0),
    ("prediction", 10.0),
    ("planning_plannn", 6.0),
    ("control", 2.0),
]

FRAME_INTERVAL_MS = 33.0
NUM_FRAMES = 10


def simulate_sequential():
    """顺序执行：总时延是 stage 之和，吞吐受 total latency 限制。"""
    engine_available = 0.0
    rows = []
    for frame_id in range(NUM_FRAMES):
        arrival = frame_id * FRAME_INTERVAL_MS
        start = max(arrival, engine_available)
        end = start + sum(stage_time for _, stage_time in STAGES)
        engine_available = end
        rows.append((frame_id, arrival, start, end))
    return rows


def simulate_pipelined():
    """流水执行：每个 stage 有自己的可用时间，吞吐受最慢 stage 限制。"""
    stage_available = {name: 0.0 for name, _ in STAGES}
    rows = []
    for frame_id in range(NUM_FRAMES):
        arrival = frame_id * FRAME_INTERVAL_MS
        cursor = arrival
        stage_starts = []
        for name, stage_time in STAGES:
            start = max(cursor, stage_available[name])
            end = start + stage_time
            stage_available[name] = end
            stage_starts.append((name, start, end))
            cursor = end
        rows.append((frame_id, arrival, stage_starts[0][1], stage_starts[-1][2]))
    return rows


def print_rows(title, rows):
    """打印帧调度结果。"""
    print(f"\n[{title}]")
    print("-" * 84)
    print(f"{'frame':>6} {'arrival':>10} {'start':>10} {'end':>10} {'latency':>10}")
    print("-" * 84)
    for frame_id, arrival, start, end in rows:
        print(f"{frame_id:>6} {arrival:>10.1f} {start:>10.1f} {end:>10.1f} {end - arrival:>10.1f}")
    total_time = max(end for _, _, _, end in rows)
    completed_span = total_time - rows[0][1]
    fps = len(rows) / (completed_span / 1000.0) if completed_span > 0 else 0.0
    avg_latency = sum(end - arrival for _, arrival, _, end in rows) / len(rows)
    print("-" * 84)
    print(f"avg latency: {avg_latency:.1f} ms")
    print(f"approx throughput: {fps:.2f} FPS")


def main():
    print("=" * 84)
    print("Autonomous Driving Inference Pipeline Simulator")
    print("=" * 84)
    print("stages:")
    for name, stage_time in STAGES:
        print(f"  {name:<22} {stage_time:>5.1f} ms")
    print(f"frame interval: {FRAME_INTERVAL_MS:.1f} ms, frames: {NUM_FRAMES}")

    sequential = simulate_sequential()
    pipelined = simulate_pipelined()
    print_rows("Sequential", sequential)
    print_rows("Pipelined", pipelined)

    print("\n" + "=" * 84)
    print("Takeaway")
    print("=" * 84)
    print("- Sequential latency is the sum of all stages; throughput is capped by total latency.")
    print("- Pipelining keeps latency similar for one frame, but throughput is capped by the slowest stage.")
    print("- To improve AD latency further, optimize the slowest stage, overlap CPU/GPU work, or reduce frame rate adaptively.")


if __name__ == "__main__":
    main()
