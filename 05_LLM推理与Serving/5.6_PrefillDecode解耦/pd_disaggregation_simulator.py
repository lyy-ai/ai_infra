#!/usr/bin/env python3
"""
Prefill/Decode 解耦（PD Disaggregation）调度模拟器。

本脚本用纯 Python 对比两种推理服务方式：
1. Coupled：Prefill 与 Decode 共享同一个执行资源，长 Prefill 会阻塞 Decode。
2. Disaggregated：Prefill 与 Decode 在不同 worker 上并行执行，KV 通过传输交给 Decode。

无需 GPU 或 vLLM，可直接运行。

运行方式：
    python examples/pd_disaggregation_simulator.py
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


PREFILL_CHUNK_TOKENS = 256
KV_TRANSFER_TOKENS_PER_ITER = 1024


@dataclass
class Request:
    """模拟请求，时间单位统一为 iteration。"""

    id: int
    arrival_time: int
    prompt_tokens: int
    decode_tokens: int
    prefilled_tokens: int = 0
    transfer_remaining: int = 0
    decode_generated: int = 0
    prefill_done_time: Optional[int] = None
    first_token_time: Optional[int] = None
    completion_time: Optional[int] = None
    decode_token_times: List[int] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return self.completion_time is not None


def clone_requests(requests: List[Request]) -> List[Request]:
    """复制请求，避免多个实验共享状态。"""
    return [
        Request(
            id=r.id,
            arrival_time=r.arrival_time,
            prompt_tokens=r.prompt_tokens,
            decode_tokens=r.decode_tokens,
        )
        for r in requests
    ]


def generate_workload() -> List[Request]:
    """构造一组长短不一的请求，突出长 prefill 对 decode 的影响。"""
    return [
        Request(id=0, arrival_time=0, prompt_tokens=1024, decode_tokens=64),
        Request(id=1, arrival_time=1, prompt_tokens=256, decode_tokens=32),
        Request(id=2, arrival_time=2, prompt_tokens=2048, decode_tokens=48),
        Request(id=3, arrival_time=10, prompt_tokens=512, decode_tokens=64),
        Request(id=4, arrival_time=12, prompt_tokens=768, decode_tokens=32),
        Request(id=5, arrival_time=20, prompt_tokens=128, decode_tokens=16),
    ]


def simulate_coupled(requests: List[Request]) -> Tuple[List[Request], int, int]:
    """
    Coupled 模拟：每个 iteration 只能做一件事——要么 prefill 一个 chunk，要么 decode 一轮。

    当 running 中已有 decode 请求时，如果新请求进入 prefill，decode 会被暂停，
    这会造成 TPOT 抖动和 TTFT 排队。
    """
    time = 0
    next_idx = 0
    waiting_prefill: List[Request] = []
    running_decode: List[Request] = []
    decode_stall_iterations = 0

    requests = sorted(requests, key=lambda r: (r.arrival_time, r.id))

    while not all(r.completed for r in requests):
        while next_idx < len(requests) and requests[next_idx].arrival_time <= time:
            waiting_prefill.append(requests[next_idx])
            next_idx += 1

        if waiting_prefill:
            req = waiting_prefill[0]
            req.prefilled_tokens += PREFILL_CHUNK_TOKENS
            if running_decode:
                decode_stall_iterations += 1
            if req.prefilled_tokens >= req.prompt_tokens:
                req.prefill_done_time = time
                waiting_prefill.pop(0)
                running_decode.append(req)
        elif running_decode:
            for req in list(running_decode):
                req.decode_generated += 1
                req.decode_token_times.append(time)
                if req.first_token_time is None:
                    req.first_token_time = time
                if req.decode_generated >= req.decode_tokens:
                    req.completion_time = time
                    running_decode.remove(req)
        else:
            if next_idx < len(requests):
                time = requests[next_idx].arrival_time
            else:
                break

        time += 1

    return requests, decode_stall_iterations, time


def simulate_disaggregated(requests: List[Request]) -> Tuple[List[Request], int, int]:
    """
    Disaggregated 模拟：Prefill worker、KV transfer、Decode worker 在同一个时间轴上并行推进。

    每个 iteration：
    - transfer 先推进已完成的 KV 传输；
    - prefill worker 处理一个 chunk；
    - decode worker 对 ready 的请求批量 decode 一个 token。
    """
    time = 0
    next_idx = 0
    prefill_queue: List[Request] = []
    transferring: List[Request] = []
    running_decode: List[Request] = []

    requests = sorted(requests, key=lambda r: (r.arrival_time, r.id))

    while not all(r.completed for r in requests):
        while next_idx < len(requests) and requests[next_idx].arrival_time <= time:
            prefill_queue.append(requests[next_idx])
            next_idx += 1

        for req in list(transferring):
            req.transfer_remaining -= 1
            if req.transfer_remaining <= 0:
                transferring.remove(req)
                running_decode.append(req)

        if prefill_queue:
            req = prefill_queue[0]
            req.prefilled_tokens += PREFILL_CHUNK_TOKENS
            if req.prefilled_tokens >= req.prompt_tokens:
                req.prefill_done_time = time
                req.transfer_remaining = math.ceil(req.prompt_tokens / KV_TRANSFER_TOKENS_PER_ITER)
                prefill_queue.pop(0)
                transferring.append(req)

        for req in list(running_decode):
            req.decode_generated += 1
            req.decode_token_times.append(time)
            if req.first_token_time is None:
                req.first_token_time = time
            if req.decode_generated >= req.decode_tokens:
                req.completion_time = time
                running_decode.remove(req)

        if not prefill_queue and not transferring and not running_decode and next_idx < len(requests):
            time = requests[next_idx].arrival_time
        else:
            time += 1

    return requests, 0, time


def percentile(values: List[float], pct: float) -> float:
    """简单 percentile 实现，避免引入额外依赖。"""
    if not values:
        return 0.0
    values = sorted(values)
    idx = max(0, min(len(values) - 1, math.ceil(pct / 100.0 * len(values)) - 1))
    return values[idx]


def compute_metrics(requests: List[Request], total_time: int, decode_stall_iterations: int) -> dict:
    """计算 TTFT、TPOT、吞吐与 decode stall 指标。"""
    ttfts = []
    end_to_end = []
    decode_gaps = []
    total_decode_tokens = 0

    for req in requests:
        total_decode_tokens += req.decode_tokens
        if req.first_token_time is not None:
            ttfts.append(req.first_token_time - req.arrival_time)
        if req.completion_time is not None:
            end_to_end.append(req.completion_time - req.arrival_time)
        times = req.decode_token_times
        decode_gaps.extend([b - a for a, b in zip(times, times[1:])])

    return {
        "total_time": total_time,
        "avg_ttft": sum(ttfts) / len(ttfts) if ttfts else 0.0,
        "p95_ttft": percentile(ttfts, 95),
        "avg_e2e": sum(end_to_end) / len(end_to_end) if end_to_end else 0.0,
        "avg_tpot_gap": sum(decode_gaps) / len(decode_gaps) if decode_gaps else 0.0,
        "max_tpot_gap": max(decode_gaps) if decode_gaps else 0,
        "decode_stall_iterations": decode_stall_iterations,
        "throughput": total_decode_tokens / total_time if total_time > 0 else 0.0,
    }


def print_workload(requests: List[Request]):
    """打印工作负载。"""
    print("\nWorkload")
    print("-" * 72)
    print(f"{'id':>4} {'arrival':>8} {'prompt':>8} {'decode':>8}")
    for req in requests:
        print(f"{req.id:>4} {req.arrival_time:>8} {req.prompt_tokens:>8} {req.decode_tokens:>8}")


def print_per_request(coupled: List[Request], disaggregated: List[Request]):
    """打印每个请求在两种模式下的 TTFT / completion。"""
    print("\nPer-request Results")
    print("-" * 72)
    print(f"{'id':>4} {'coupled_ttft':>13} {'pd_ttft':>10} {'coupled_e2e':>12} {'pd_e2e':>10}")
    disagg_by_id = {r.id: r for r in disaggregated}
    for req in coupled:
        other = disagg_by_id[req.id]
        c_ttft = req.first_token_time - req.arrival_time
        d_ttft = other.first_token_time - other.arrival_time
        c_e2e = req.completion_time - req.arrival_time
        d_e2e = other.completion_time - other.arrival_time
        print(f"{req.id:>4} {c_ttft:>13} {d_ttft:>10} {c_e2e:>12} {d_e2e:>10}")


def print_summary(coupled_metrics: dict, disagg_metrics: dict):
    """打印汇总对比。"""
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"{'metric':>28} {'coupled':>14} {'disaggregated':>14}")
    print("-" * 72)
    print(f"{'total time (iterations)':>28} {coupled_metrics['total_time']:>14} {disagg_metrics['total_time']:>14}")
    print(f"{'avg TTFT (iterations)':>28} {coupled_metrics['avg_ttft']:>14.2f} {disagg_metrics['avg_ttft']:>14.2f}")
    print(f"{'p95 TTFT (iterations)':>28} {coupled_metrics['p95_ttft']:>14.2f} {disagg_metrics['p95_ttft']:>14.2f}")
    print(f"{'avg E2E latency':>28} {coupled_metrics['avg_e2e']:>14.2f} {disagg_metrics['avg_e2e']:>14.2f}")
    print(f"{'avg TPOT gap':>28} {coupled_metrics['avg_tpot_gap']:>14.2f} {disagg_metrics['avg_tpot_gap']:>14.2f}")
    print(f"{'max TPOT gap':>28} {coupled_metrics['max_tpot_gap']:>14} {disagg_metrics['max_tpot_gap']:>14}")
    print(f"{'decode stall iterations':>28} {coupled_metrics['decode_stall_iterations']:>14} {disagg_metrics['decode_stall_iterations']:>14}")
    print(f"{'decode throughput (tok/iter)':>28} {coupled_metrics['throughput']:>14.2f} {disagg_metrics['throughput']:>14.2f}")
    print("-" * 72)
    print("Note: time is measured in simulator iterations; lower TTFT/TPOT gap is better.")


def run_comparison():
    """运行 coupled vs disaggregated 对比实验。"""
    print("=" * 72)
    print("Prefill/Decode Disaggregation Simulator")
    print(f"prefill chunk: {PREFILL_CHUNK_TOKENS} tokens/iteration")
    print(f"KV transfer speed: {KV_TRANSFER_TOKENS_PER_ITER} tokens/iteration")
    print("=" * 72)

    workload = generate_workload()
    print_workload(workload)

    coupled, coupled_stalls, coupled_time = simulate_coupled(clone_requests(workload))
    disaggregated, _, disagg_time = simulate_disaggregated(clone_requests(workload))

    coupled_metrics = compute_metrics(coupled, coupled_time, coupled_stalls)
    disagg_metrics = compute_metrics(disaggregated, disagg_time, 0)

    print_per_request(coupled, disaggregated)
    print_summary(coupled_metrics, disagg_metrics)


if __name__ == "__main__":
    run_comparison()
