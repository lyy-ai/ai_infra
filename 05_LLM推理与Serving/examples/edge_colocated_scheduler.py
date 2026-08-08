#!/usr/bin/env python3
"""
端侧共线多模态模型部署调度示例。

本脚本模拟多个模型共用同一台端侧加速器：
- safety_monitor：高优先级、常驻、必须准时
- asr / tts：中优先级语音链路
- planner_llm：规划模型，pinned，不能被卸载
- vlm：大体量多模态模型，内存不足时可能被拒绝或延后

无需 GPU，可直接运行。

运行方式：
    python examples/edge_colocated_scheduler.py
"""
from dataclasses import dataclass
from typing import Dict, List, Optional


MEMORY_CAPACITY_GB = 8.0


@dataclass
class Job:
    name: str
    release: int
    exec_ms: int
    deadline: int
    priority: int
    mem_gb: float
    pinned: bool = False
    start: Optional[int] = None
    end: Optional[int] = None
    rejected: bool = False


def build_jobs() -> List[Job]:
    """构造 1 秒内的多模态任务流。"""
    jobs = []
    for t in range(0, 1000, 20):
        jobs.append(Job("safety", t, 3, t + 10, 0, 0.5, pinned=True))
    for t in range(0, 1000, 100):
        jobs.append(Job("asr", t, 12, t + 50, 2, 1.5))
    for t in [150, 450, 750]:
        jobs.append(Job("tts", t, 30, t + 120, 2, 1.0))
    for t in [200, 700]:
        jobs.append(Job("planner", t, 80, t + 300, 1, 4.0, pinned=True))
    jobs.append(Job("vlm", 300, 120, 300 + 500, 3, 6.0))
    return sorted(jobs, key=lambda j: (j.release, j.priority))


def schedule(jobs: List[Job], memory_limit: bool) -> Dict[str, float]:
    """简化调度：单加速器，高优先级 ready job 先跑；memory_limit 模式做内存准入。"""
    time = 0
    loaded: Dict[str, float] = {}
    running: Optional[Job] = None
    safety_misses = 0

    jobs = [Job(**vars(job)) for job in jobs]
    pending = jobs[:]

    while pending or running:
        ready = [job for job in pending if job.release <= time]
        if running is None and ready:
            ready.sort(key=lambda job: (job.priority, job.release))
            job = ready[0]
            if memory_limit:
                needed = job.mem_gb
                used = sum(loaded.values())
                if job.name not in loaded and used + needed > MEMORY_CAPACITY_GB:
                    candidates = sorted(
                        [(name, mem) for name, mem in loaded.items() if name != job.name and name != "safety" and name != "planner"],
                        key=lambda item: item[1],
                    )
                    for name, mem in candidates:
                        if used + needed <= MEMORY_CAPACITY_GB:
                            break
                        used -= mem
                        del loaded[name]
                if job.name not in loaded and sum(loaded.values()) + needed > MEMORY_CAPACITY_GB:
                    job.rejected = True
                    pending.remove(job)
                    time += 1
                    continue
            running = job
            running.start = time
            loaded.setdefault(job.name, job.mem_gb)
            pending.remove(job)

        if running is not None:
            time += running.exec_ms
            running.end = time
            if running.name == "safety" and running.end > running.deadline:
                safety_misses += 1
            running = None
        else:
            next_release = min(job.release for job in pending) if pending else time + 1
            time = max(time + 1, next_release)

    completed = [job for job in jobs if job.end is not None]
    rejected = [job for job in jobs if job.rejected]
    waits = [job.start - job.release for job in completed]
    unique_mem: Dict[str, float] = {}
    for job in jobs:
        unique_mem[job.name] = max(unique_mem.get(job.name, 0.0), job.mem_gb)
    max_mem = MEMORY_CAPACITY_GB if memory_limit else sum(unique_mem.values())

    return {
        "completed": len(completed),
        "rejected": len(rejected),
        "rejected_names": [job.name for job in rejected],
        "safety_misses": safety_misses,
        "avg_wait": sum(waits) / len(waits) if waits else 0.0,
        "max_mem_gb": max_mem,
    }


def main():
    print("=" * 84)
    print("Edge Co-located Multimodal Scheduler")
    print("=" * 84)
    print(f"memory capacity: {MEMORY_CAPACITY_GB:.1f} GiB")
    print("models: safety(0.5,pinned), planner(4.0,pinned), asr(1.5), tts(1.0), vlm(6.0)")

    no_limit = schedule(build_jobs(), memory_limit=False)
    limited = schedule(build_jobs(), memory_limit=True)

    print("\n" + "=" * 84)
    print("Results")
    print("=" * 84)
    print(f"{'metric':>24} {'no_limit':>14} {'memory_limit':>14}")
    print("-" * 84)
    print(f"{'completed jobs':>24} {no_limit['completed']:>14} {limited['completed']:>14}")
    print(f"{'rejected jobs':>24} {no_limit['rejected']:>14} {limited['rejected']:>14}")
    print(f"{'safety misses':>24} {no_limit['safety_misses']:>14} {limited['safety_misses']:>14}")
    print(f"{'avg wait (ms)':>24} {no_limit['avg_wait']:>14.2f} {limited['avg_wait']:>14.2f}")
    print(f"{'max mem (GiB)':>24} {no_limit['max_mem_gb']:>14.2f} {limited['max_mem_gb']:>14.2f}")
    print(f"rejected in memory_limit: {limited['rejected_names']}")

    print("\nTakeaway:")
    print("- On edge devices, admission control is part of the model: not every model should be resident.")
    print("- Safety-critical tasks must be pinned and protected from large VLM/LLM jobs.")
    print("- Rejecting or deferring a heavy VLM job can be better than missing a safety deadline.")


if __name__ == "__main__":
    main()
