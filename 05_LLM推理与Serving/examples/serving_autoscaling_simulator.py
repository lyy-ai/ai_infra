#!/usr/bin/env python3
"""
负载均衡与弹性扩缩容模拟器。

本脚本模拟一个简化的推理服务集群：
1. 请求按时间到达，每个请求需要若干 service_time。
2. Router 使用 round-robin 或 least-queue 策略分发请求。
3. Autoscaler 根据平均队列深度弹性增加/减少实例，实例启动需要 warmup 时间。

无需 GPU，可直接运行。

运行方式：
    python examples/serving_autoscaling_simulator.py
"""
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional


SERVICE_MIN_ITERS = 5
SERVICE_MAX_ITERS = 12
STARTUP_ITERS = 20
SCALE_COOLDOWN = 10
SCALE_OUT_QUEUE = 4
SCALE_IN_QUEUE = 1
MIN_INSTANCES = 1
MAX_INSTANCES = 6


@dataclass
class Request:
    id: int
    arrival_time: int
    service_time: int
    start_time: Optional[int] = None
    completion_time: Optional[int] = None


@dataclass
class Instance:
    id: int
    startup_remaining: int = 0
    busy_until: int = 0
    queue: Deque[Request] = None
    completed: int = 0

    def __post_init__(self):
        if self.queue is None:
            self.queue = deque()

    @property
    def ready(self) -> bool:
        return self.startup_remaining == 0

    def load(self, time: int) -> int:
        """粗略负载：排队数 + 是否忙碌。"""
        return len(self.queue) + (1 if self.busy_until > time else 0)


def generate_requests() -> List[Request]:
    """生成低负载 + 突发流量 + 回落的请求曲线。"""
    random.seed(7)
    requests = []
    req_id = 0
    for time in range(0, 180):
        if 60 <= time < 120:
            arrivals = 1 if time % 1 == 0 else 0
        elif 30 <= time < 60:
            arrivals = 1 if time % 2 == 0 else 0
        else:
            arrivals = 1 if time % 4 == 0 else 0
        for _ in range(arrivals):
            requests.append(Request(req_id, time, random.randint(SERVICE_MIN_ITERS, SERVICE_MAX_ITERS)))
            req_id += 1
    return requests


def choose_instance(instances: List[Instance], policy: str, time: int, rr_state: dict) -> Optional[Instance]:
    """根据策略选择一个 ready 实例。"""
    ready = [inst for inst in instances if inst.ready]
    if not ready:
        return None
    if policy == "least_queue":
        return min(ready, key=lambda inst: inst.load(time))
    idx = rr_state.get("idx", 0) % len(ready)
    rr_state["idx"] = idx + 1
    return ready[idx]


def maybe_autoscale(instances: List[Instance], time: int, enabled: bool, events: List[str], next_id: dict, cooldown: dict):
    """按平均队列深度扩缩容。"""
    if not enabled or time < cooldown.get("next", 0):
        return
    ready = [inst for inst in instances if inst.ready]
    avg_queue = (sum(len(inst.queue) for inst in ready) / len(ready)) if ready else float("inf")

    if avg_queue > SCALE_OUT_QUEUE and len(instances) < MAX_INSTANCES:
        inst = Instance(next_id["id"], startup_remaining=STARTUP_ITERS)
        next_id["id"] += 1
        instances.append(inst)
        cooldown["next"] = time + SCALE_COOLDOWN
        events.append(f"t={time}: scale-out -> instance {inst.id} (startup {STARTUP_ITERS} iters)")
    elif avg_queue < SCALE_IN_QUEUE and len(instances) > MIN_INSTANCES:
        idle = [inst for inst in instances if inst.ready and not inst.queue and inst.busy_until <= time]
        if idle:
            victim = idle[0]
            instances.remove(victim)
            cooldown["next"] = time + SCALE_COOLDOWN
            events.append(f"t={time}: scale-in -> remove instance {victim.id}")


def simulate(policy: str, autoscaling: bool, initial_instances: int = 1) -> dict:
    """运行一次模拟。"""
    requests = generate_requests()
    instances = [Instance(i) for i in range(initial_instances)]
    next_id = {"id": initial_instances}
    rr_state = {"idx": 0}
    cooldown = {"next": 0}
    events: List[str] = []
    time = 0
    idx = 0
    max_queue = 0

    while idx < len(requests) or any(inst.queue or inst.busy_until > time for inst in instances):
        for inst in instances:
            if inst.startup_remaining > 0:
                inst.startup_remaining -= 1

        while idx < len(requests) and requests[idx].arrival_time <= time:
            inst = choose_instance(instances, policy, time, rr_state)
            if inst is None:
                break
            inst.queue.append(requests[idx])
            idx += 1

        for inst in instances:
            if not inst.ready:
                continue
            if inst.busy_until <= time and inst.queue:
                req = inst.queue.popleft()
                req.start_time = time
                inst.busy_until = time + req.service_time
            if inst.busy_until == time:
                inst.completed += 1

        for req in requests:
            if req.start_time is not None and req.completion_time is None and req.start_time + req.service_time <= time:
                req.completion_time = req.start_time + req.service_time

        maybe_autoscale(instances, time, autoscaling, events, next_id, cooldown)
        max_queue = max(max_queue, sum(len(inst.queue) for inst in instances))
        time += 1

    completed = [req for req in requests if req.completion_time is not None]
    waits = [req.start_time - req.arrival_time for req in completed]
    waits_sorted = sorted(waits)
    p95 = waits_sorted[max(0, int(0.95 * len(waits_sorted)) - 1)] if waits_sorted else 0
    return {
        "policy": policy,
        "autoscaling": autoscaling,
        "total_time": time,
        "requests": len(requests),
        "completed": len(completed),
        "avg_wait": sum(waits) / len(waits) if waits else 0,
        "p95_wait": p95,
        "max_queue": max_queue,
        "final_instances": len(instances),
        "events": events,
    }


def print_result(result: dict):
    """打印单次实验结果。"""
    print("\n" + "=" * 78)
    print(f"policy={result['policy']}, autoscaling={result['autoscaling']}")
    print("=" * 78)
    print(f"requests: {result['completed']}/{result['requests']}")
    print(f"total time: {result['total_time']} iterations")
    print(f"avg wait: {result['avg_wait']:.2f} iterations")
    print(f"p95 wait: {result['p95_wait']} iterations")
    print(f"max queue: {result['max_queue']}")
    print(f"final instances: {result['final_instances']}")
    for event in result["events"]:
        print(f"  {event}")


def main():
    print("=" * 78)
    print("Load Balancing and Autoscaling Simulator")
    print("=" * 78)
    print(f"burst traffic: t=60..119, startup={STARTUP_ITERS} iters, max_instances={MAX_INSTANCES}")

    fixed = simulate("round_robin", autoscaling=False, initial_instances=2)
    elastic = simulate("least_queue", autoscaling=True, initial_instances=1)
    print_result(fixed)
    print_result(elastic)

    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    print("- Fixed capacity is simple but suffers during bursts.")
    print("- Least-queue routing sends new requests to the least loaded ready instance.")
    print("- Autoscaling absorbs bursts but pays startup warmup time; cooldown avoids thrashing.")


if __name__ == "__main__":
    main()
