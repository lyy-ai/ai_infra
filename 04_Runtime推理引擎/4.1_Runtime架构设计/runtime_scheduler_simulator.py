#!/usr/bin/env python3
"""
Runtime 架构设计模拟器：Session、Graph Executor、Tensor 生命周期与调度瓶颈。

运行方式：
    python 4.1_Runtime架构设计/runtime_scheduler_simulator.py
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Node:
    name: str
    op_type: str
    compute_us: float
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    memory_mb: float = 0.0


def demo_graph() -> List[Node]:
    """一个小型推理图：输入 -> conv/attn -> mlp -> 输出。"""
    return [
        Node("input", "io", 0.0, outputs=["x"], memory_mb=64),
        Node("qkv", "matmul", 40.0, inputs=["x"], outputs=["q", "k", "v"], memory_mb=192),
        Node("attn", "attention", 80.0, inputs=["q", "k", "v"], outputs=["attn_out"], memory_mb=128),
        Node("proj", "matmul", 30.0, inputs=["attn_out"], outputs=["proj_out"], memory_mb=64),
        Node("mlp", "mlp", 60.0, inputs=["proj_out"], outputs=["y"], memory_mb=128),
        Node("output", "io", 0.0, inputs=["y"], outputs=[], memory_mb=0),
    ]


def tensor_lifetimes(nodes: List[Node]) -> Dict[str, Tuple[int, int]]:
    """计算每个 tensor 的 first_write 与 last_read（按节点执行序号）。"""
    first_write: Dict[str, int] = {}
    last_read: Dict[str, int] = {}
    for idx, node in enumerate(nodes):
        for tensor in node.outputs:
            first_write.setdefault(tensor, idx)
        for tensor in node.inputs:
            last_read[tensor] = idx
    return {tensor: (first_write[tensor], last_read.get(tensor, first_write[tensor])) for tensor in first_write}


def peak_memory(nodes: List[Node], reuse: bool) -> float:
    """估算峰值显存；reuse=True 表示 tensor 生命周期结束后可复用。"""
    lifetimes = tensor_lifetimes(nodes)
    tensor_size = {}
    for node in nodes:
        if node.outputs:
            share = node.memory_mb / len(node.outputs) if node.memory_mb else 0.0
            for tensor in node.outputs:
                tensor_size[tensor] = share

    if not reuse:
        return sum(tensor_size.values())

    peak = 0.0
    for idx, node in enumerate(nodes):
        alive = 0.0
        for tensor, (start, end) in lifetimes.items():
            if start <= idx <= end:
                alive += tensor_size.get(tensor, 0.0)
        peak = max(peak, alive)
    return peak


def critical_path(nodes: List[Node]) -> float:
    """按依赖关系计算关键路径长度。"""
    node_map = {node.name: node for node in nodes}
    tensor_producer = {}
    for node in nodes:
        for tensor in node.outputs:
            tensor_producer[tensor] = node.name

    memo: Dict[str, float] = {}

    def dp(name: str) -> float:
        if name in memo:
            return memo[name]
        node = node_map[name]
        dep_cost = 0.0
        for tensor in node.inputs:
            producer = tensor_producer.get(tensor)
            if producer:
                dep_cost = max(dep_cost, dp(producer))
        memo[name] = dep_cost + node.compute_us
        return memo[name]

    return max(dp(node.name) for node in nodes)


def simulate_sessions(base_peak_mb: float, weight_mb: float, sessions: List[int]):
    """模拟多 session 下的显存占用。"""
    print("\nSession scaling")
    print("-" * 72)
    print(f"{'sessions':>10} {'weights_mb':>12} {'activation_mb':>14} {'total_mb':>12}")
    print("-" * 72)
    for n in sessions:
        total = weight_mb + n * base_peak_mb
        print(f"{n:>10} {weight_mb:>12.1f} {n * base_peak_mb:>14.1f} {total:>12.1f}")


def main():
    nodes = demo_graph()
    total_compute = sum(node.compute_us for node in nodes)
    cp = critical_path(nodes)
    no_reuse = peak_memory(nodes, reuse=False)
    with_reuse = peak_memory(nodes, reuse=True)

    print("=" * 72)
    print("Runtime Architecture Simulator")
    print("=" * 72)
    print("Graph nodes:")
    for node in nodes:
        print(f"  {node.name:<8} {node.op_type:<10} compute={node.compute_us:>5.1f}us mem={node.memory_mb:>5.1f}MB")

    print("\nTensor lifetimes:")
    for tensor, (start, end) in tensor_lifetimes(nodes).items():
        print(f"  {tensor:<10} write@node {start}, last_read@node {end}")

    print("\n" + "=" * 72)
    print("Analysis")
    print("=" * 72)
    print(f"total compute: {total_compute:.1f} us")
    print(f"critical path: {cp:.1f} us")
    print(f"peak activation memory (no reuse): {no_reuse:.1f} MB")
    print(f"peak activation memory (with reuse): {with_reuse:.1f} MB")
    print(f"memory saved by lifetime reuse: {(1 - with_reuse / no_reuse) * 100:.1f}%")

    simulate_sessions(base_peak_mb=with_reuse, weight_mb=1024.0, sessions=[1, 2, 4, 8])

    print("\nTakeaway:")
    print("- Session shares model weights, but each session owns activations/KV-like state.")
    print("- Graph executor bottleneck is critical path, not just total compute.")
    print("- Tensor lifetime analysis enables memory reuse and lower peak memory.")


if __name__ == "__main__":
    main()
