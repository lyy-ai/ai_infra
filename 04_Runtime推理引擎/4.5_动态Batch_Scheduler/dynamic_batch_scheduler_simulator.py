#!/usr/bin/env python3
"""
动态 Batch Scheduler 模拟器：动态批处理策略、请求队列管理、负载均衡与资源利用率。

运行方式：
    python 4.5_动态Batch_Scheduler/dynamic_batch_scheduler_simulator.py
"""
import random
from dataclasses import dataclass
from typing import List


MAX_BATCH = 8
MAX_TOKENS_PER_BATCH = 4096
BASE_BATCH_OVERHEAD = 8.0
TOKEN_COMPUTE_US = 0.02


@dataclass
class Request:
    id: int
    arrival: int
    tokens: int
    priority: int = 1
    start: int = 0
    finish: int = 0


def generate_requests(n=24, seed=3) -> List[Request]:
    random.seed(seed)
    requests = []
    t = 0
    for i in range(n):
        t += random.randint(0, 3)
        tokens = random.choice([128, 256, 512, 1024, 2048])
        priority = 0 if random.random() < 0.2 else 1
        requests.append(Request(i, t, tokens, priority))
    return requests


def batch_compute_ms(batch: List[Request]) -> float:
    """批处理耗时：batch 越大吞吐越高，但单 batch 耗时并非线性增长。"""
    total_tokens = sum(req.tokens for req in batch)
    return BASE_BATCH_OVERHEAD + total_tokens * TOKEN_COMPUTE_US / 1000.0


def can_add(batch: List[Request], req: Request) -> bool:
    return len(batch) < MAX_BATCH and sum(r.tokens for r in batch) + req.tokens <= MAX_TOKENS_PER_BATCH


def simulate(requests: List[Request], policy: str):
    time = 0
    queue = sorted(requests, key=lambda r: (r.arrival, r.id))
    done = []
    batches = 0
    busy_time = 0.0

    while len(done) < len(requests):
        arrived = [r for r in queue if r.arrival <= time and r not in done]
        if not arrived:
            future = [r.arrival for r in queue if r not in done]
            time = min(future) if future else time + 1
            continue

        if policy == "fcfs":
            ordered = sorted(arrived, key=lambda r: (r.arrival, r.id))
        elif policy == "sjf":
            ordered = sorted(arrived, key=lambda r: (r.tokens, r.arrival))
        else:
            ordered = sorted(arrived, key=lambda r: (r.priority, r.arrival))

        batch = []
        for req in ordered:
            if can_add(batch, req):
                batch.append(req)

        if not batch:
            time += 1
            continue

        cost = batch_compute_ms(batch)
        start = time
        finish = time + cost
        batches += 1
        busy_time += cost
        for req in batch:
            req.start = start
            req.finish = finish
            done.append(req)
        time = finish

    waits = [r.start - r.arrival for r in requests]
    e2e = [r.finish - r.arrival for r in requests]
    total_tokens = sum(r.tokens for r in requests)
    total_time = max(r.finish for r in requests)
    return {
        "total_time": total_time,
        "avg_wait": sum(waits) / len(waits),
        "avg_e2e": sum(e2e) / len(e2e),
        "throughput": total_tokens / total_time if total_time else 0,
        "batches": batches,
        "utilization": busy_time / total_time if total_time else 0,
    }


def print_result(name, result):
    print(f"\n[{name}]")
    for key in ["total_time", "avg_wait", "avg_e2e", "throughput", "batches", "utilization"]:
        print(f"  {key:>12}: {result[key]:.2f}")


def main():
    requests = generate_requests()
    print("=" * 78)
    print("Dynamic Batch Scheduler Simulator")
    print("=" * 78)
    print(f"requests={len(requests)}, max_batch={MAX_BATCH}, max_tokens_per_batch={MAX_TOKENS_PER_BATCH}")
    print("request tokens:")
    for req in requests:
        print(f"  id={req.id:>2} arrival={req.arrival:>3} tokens={req.tokens:>4} priority={req.priority}")

    for policy in ["fcfs", "sjf", "priority"]:
        print_result(policy, simulate(requests, policy))

    print("\nTakeaway:")
    print("- Dynamic batching improves throughput by sharing fixed batch overhead across requests.")
    print("- SJF/priority can reduce average wait but may starve long requests without aging.")
    print("- Token budget is as important as request count: many long requests can fill one batch quickly.")


if __name__ == "__main__":
    main()
