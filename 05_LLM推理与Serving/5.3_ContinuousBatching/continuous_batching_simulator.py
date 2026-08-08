#!/usr/bin/env python3
"""
Continuous Batching 模拟器。

本脚本用纯 Python 模拟 Static Batching 与 Continuous Batching 的调度过程，
直观对比两种批处理策略在延迟、吞吐、GPU 利用率上的差异。

运行方式：
    python examples/continuous_batching_simulator.py

无需 GPU 或 vLLM。
"""
import random
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Request:
    """模拟请求。"""
    id: int
    arrival_time: int          # 到达时间（iteration 为单位）
    num_tokens: int            # 需要生成的 token 数
    generated_tokens: int = 0
    completed: bool = False
    completion_time: int = 0


def generate_requests(n: int, seed: int = 42) -> List[Request]:
    """生成一组随机请求。"""
    random.seed(seed)
    requests = []
    current_time = 0
    for i in range(n):
        # 到达间隔随机 0-3 个 iteration
        current_time += random.randint(0, 3)
        # 生成长度随机 5-50 个 token
        num_tokens = random.randint(5, 50)
        requests.append(Request(id=i, arrival_time=current_time, num_tokens=num_tokens))
    return requests


def static_batching(requests: List[Request], max_batch_size: int = 4) -> Tuple[List[Tuple[int, int, int]], int]:
    """
    静态批处理模拟。

    逻辑：
    1. 等待 batch 满（max_batch_size）或没有新请求。
    2. 执行 batch 直到最长请求完成。
    3. 处理下一个 batch。
    """
    time = 0
    total = len(requests)
    completed = 0
    queue = sorted(requests, key=lambda r: r.arrival_time)
    results = []

    while completed < total:
        # 收集一个 batch
        batch = []
        while len(batch) < max_batch_size and queue:
            if queue[0].arrival_time <= time:
                batch.append(queue.pop(0))
            else:
                break

        if not batch:
            # 没有请求可处理，跳到下一个请求到达时间
            if queue:
                time = queue[0].arrival_time
            else:
                break
            continue

        # 等待 batch 中最长请求完成
        max_tokens = max(r.num_tokens for r in batch)
        time += max_tokens

        for r in batch:
            r.generated_tokens = r.num_tokens
            r.completed = True
            r.completion_time = time
            results.append((r.id, time, r.num_tokens))
            completed += 1

    return results, time


def continuous_batching(requests: List[Request], max_batch_size: int = 4) -> Tuple[List[Tuple[int, int, int]], int]:
    """
    持续批处理模拟（FCFS 策略）。

    逻辑：
    1. 每个 iteration 开始前，将已到达的请求加入 running batch。
    2. 每个 iteration 中，所有 running 请求生成 1 个 token。
    3. 完成的请求立即离开，释放位置给新请求。
    """
    time = 0
    total = len(requests)
    completed = 0
    queue = sorted(requests, key=lambda r: r.arrival_time)
    running: List[Request] = []
    results = []

    while completed < total:
        # 将新到达请求加入 running batch
        while len(running) < max_batch_size and queue:
            if queue[0].arrival_time <= time:
                running.append(queue.pop(0))
            else:
                break

        if running:
            # 所有 running 请求前进一个 token
            time += 1
            for r in running[:]:
                r.generated_tokens += 1
                if r.generated_tokens >= r.num_tokens:
                    r.completed = True
                    r.completion_time = time
                    results.append((r.id, time, r.num_tokens))
                    running.remove(r)
                    completed += 1
        else:
            # 没有请求在运行，跳到下一个请求到达时间
            time = queue[0].arrival_time

    return results, time


def continuous_batching_with_preemption(
    requests: List[Request],
    max_batch_size: int = 4,
    preempt_penalty: int = 2,
) -> Tuple[List[Tuple[int, int, int]], int]:
    """
    持续批处理 + 简单抢占策略模拟。

    抢占策略：
    - 如果 running batch 已满，且仍有等待请求，
      抢占当前运行中生成 token 最多的请求（相当于“最老”的请求）。
    - 被抢占的请求回到等待队列，需要付出 preempt_penalty 的迭代代价
      （模拟 KV Cache swap 或 recompute 开销）。
    - 原始 arrival_time 保持不变，用于计算真实延迟。

    注意：这是简化模拟，真实系统的抢占需要 swap KV Cache。
    """
    time = 0
    total = len(requests)
    completed = 0

    # 等待队列：每个请求记录 (ready_time, request)
    # ready_time 表示该请求最早可以重新被调度的时间
    queue = [(r.arrival_time, r) for r in requests]
    queue.sort(key=lambda x: x[0])

    running: List[Request] = []
    results = []

    while completed < total:
        # 收集当前可运行的等待请求
        waiting = [r for ready_time, r in queue if ready_time <= time]
        queue = [(ready_time, r) for ready_time, r in queue if ready_time > time]

        # 尝试加入新请求到 running batch
        while len(running) < max_batch_size and waiting:
            running.append(waiting.pop(0))

        # 如果 running 已满，但仍有等待请求，执行抢占
        while len(running) >= max_batch_size and waiting:
            # 抢占运行最久的请求（生成的 token 最多）
            longest = max(running, key=lambda r: r.generated_tokens)
            running.remove(longest)
            # 被抢占的请求需要付出额外代价，ready_time = 当前时间 + 惩罚
            queue.append((time + preempt_penalty, longest))
            queue.sort(key=lambda x: x[0])
            # 加入新请求
            running.append(waiting.pop(0))

        if running:
            time += 1
            for r in running[:]:
                r.generated_tokens += 1
                if r.generated_tokens >= r.num_tokens:
                    r.completed = True
                    r.completion_time = time
                    results.append((r.id, time, r.num_tokens))
                    running.remove(r)
                    completed += 1
        else:
            # 没有请求在运行，跳到下一个请求到达时间
            if queue:
                time = queue[0][0]
            else:
                break

    return results, time


def compute_metrics(requests: List[Request], total_time: int) -> dict:
    """计算平均延迟、吞吐等指标。"""
    total_tokens = sum(r.num_tokens for r in requests)
    latencies = [r.completion_time - r.arrival_time for r in requests]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    throughput = total_tokens / total_time if total_time > 0 else 0

    return {
        "total_time": total_time,
        "total_tokens": total_tokens,
        "avg_latency": avg_latency,
        "max_latency": max_latency,
        "throughput": throughput,
    }


def print_request_schedule(requests: List[Request], results: List[Tuple[int, int, int]], title: str):
    """打印每个请求的完成情况。"""
    print(f"\n[{title}] Request Completion Details")
    print("-" * 60)
    print(f"{'ID':>4} {'Arrival':>8} {'Tokens':>8} {'Completion':>12} {'Latency':>10}")
    # 按完成顺序
    for req_id, completion_time, num_tokens in results:
        req = next(r for r in requests if r.id == req_id)
        latency = completion_time - req.arrival_time
        print(f"{req_id:>4} {req.arrival_time:>8} {num_tokens:>8} {completion_time:>12} {latency:>10}")


def run_comparison(num_requests: int = 20, max_batch_size: int = 4):
    """对比三种调度策略。"""
    print("=" * 70)
    print(f"Continuous Batching Simulator")
    print(f"Num requests: {num_requests}, Max batch size: {max_batch_size}")
    print("=" * 70)

    requests = generate_requests(num_requests)
    print("\nGenerated requests:")
    print(f"{'ID':>4} {'Arrival':>8} {'Num Tokens':>12}")
    for r in requests:
        print(f"{r.id:>4} {r.arrival_time:>8} {r.num_tokens:>12}")

    # 1. Static Batching
    requests_static = [Request(r.id, r.arrival_time, r.num_tokens) for r in requests]
    results_static, total_time_static = static_batching(requests_static, max_batch_size)
    metrics_static = compute_metrics(requests_static, total_time_static)
    print_request_schedule(requests_static, results_static, "Static Batching")

    # 2. Continuous Batching (FCFS)
    requests_continuous = [Request(r.id, r.arrival_time, r.num_tokens) for r in requests]
    results_continuous, total_time_continuous = continuous_batching(requests_continuous, max_batch_size)
    metrics_continuous = compute_metrics(requests_continuous, total_time_continuous)
    print_request_schedule(requests_continuous, results_continuous, "Continuous Batching (FCFS)")

    # 3. Continuous Batching + Preemption
    requests_preempt = [Request(r.id, r.arrival_time, r.num_tokens) for r in requests]
    results_preempt, total_time_preempt = continuous_batching_with_preemption(requests_preempt, max_batch_size)
    metrics_preempt = compute_metrics(requests_preempt, total_time_preempt)
    print_request_schedule(requests_preempt, results_preempt, "Continuous Batching + Preemption")

    # 汇总对比
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Metric':>30} {'Static':>12} {'Continuous':>12} {'+Preemption':>12}")
    print("-" * 70)
    print(f"{'Total time (iterations)':>30} {metrics_static['total_time']:>12} {metrics_continuous['total_time']:>12} {metrics_preempt['total_time']:>12}")
    print(f"{'Avg latency (iterations)':>30} {metrics_static['avg_latency']:>12.2f} {metrics_continuous['avg_latency']:>12.2f} {metrics_preempt['avg_latency']:>12.2f}")
    print(f"{'Max latency (iterations)':>30} {metrics_static['max_latency']:>12} {metrics_continuous['max_latency']:>12} {metrics_preempt['max_latency']:>12}")
    print(f"{'Throughput (tokens/iter)':>30} {metrics_static['throughput']:>12.2f} {metrics_continuous['throughput']:>12.2f} {metrics_preempt['throughput']:>12.2f}")

    if metrics_continuous['total_time'] > 0:
        speedup_vs_static = metrics_static['total_time'] / metrics_continuous['total_time']
        print(f"\nContinuous Batching speedup vs Static: {speedup_vs_static:.2f}x")

    print("=" * 70)


if __name__ == "__main__":
    run_comparison(num_requests=20, max_batch_size=4)
