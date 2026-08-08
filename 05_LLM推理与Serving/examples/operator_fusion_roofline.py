#!/usr/bin/env python3
"""
算子融合与 Roofline 估算示例。

本脚本用简化 roofline 模型估算：
1. 未融合算子：每个 kernel 都有启动开销，且中间结果要读写显存。
2. 融合算子：减少 kernel launch 次数与中间显存读写。

无需 GPU，可直接运行。

运行方式：
    python examples/operator_fusion_roofline.py
"""


MEMORY_BW_GBPS = 1000.0
KERNEL_LAUNCH_US = 5.0


OPS = [
    # name, compute_us, memory_mb
    ("image_backbone_conv", 120.0, 256.0),
    ("permute_reshape", 8.0, 192.0),
    ("bev_pooling", 45.0, 384.0),
    ("deformable_attn", 90.0, 320.0),
    ("layernorm", 6.0, 96.0),
    ("planning_mlp", 35.0, 128.0),
    ("nms_postprocess", 18.0, 64.0),
]

FUSION_GROUPS = [
    ["image_backbone_conv"],
    ["permute_reshape", "bev_pooling"],
    ["deformable_attn", "layernorm"],
    ["planning_mlp", "nms_postprocess"],
]


def memory_time_us(memory_mb: float) -> float:
    """按显存带宽估算访存时间。"""
    return memory_mb * 1000.0 / MEMORY_BW_GBPS


def unfused_time_us(ops) -> float:
    """未融合总时间：每个 op 取 compute/memory 较大者，再加 launch 开销。"""
    total = 0.0
    for _, compute_us, memory_mb in ops:
        total += max(compute_us, memory_time_us(memory_mb)) + KERNEL_LAUNCH_US
    return total


def fused_time_us(ops, groups) -> float:
    """融合总时间：同组算子共享一次 launch，并减少中间显存读写。"""
    op_map = {name: (compute_us, memory_mb) for name, compute_us, memory_mb in ops}
    total = 0.0
    for group in groups:
        compute = sum(op_map[name][0] for name in group) * 0.95
        memory = sum(op_map[name][1] for name in group)
        if len(group) > 1:
            memory *= 0.65
        total += max(compute, memory_time_us(memory)) + KERNEL_LAUNCH_US
    return total


def main():
    print("=" * 84)
    print("Operator Fusion Roofline Estimator")
    print("=" * 84)
    print(f"memory bandwidth: {MEMORY_BW_GBPS:.0f} GB/s, kernel launch: {KERNEL_LAUNCH_US:.1f} us")

    print("\nPer-op estimate (unfused)")
    print("-" * 84)
    print(f"{'op':>22} {'compute_us':>12} {'memory_us':>12} {'bound':>10}")
    print("-" * 84)
    for name, compute_us, memory_mb in OPS:
        mem_us = memory_time_us(memory_mb)
        bound = "memory" if mem_us > compute_us else "compute"
        print(f"{name:>22} {compute_us:>12.1f} {mem_us:>12.1f} {bound:>10}")

    unfused = unfused_time_us(OPS)
    fused = fused_time_us(OPS, FUSION_GROUPS)

    print("\n" + "=" * 84)
    print("Summary")
    print("=" * 84)
    print(f"unfused total: {unfused:.1f} us")
    print(f"fused total:   {fused:.1f} us")
    print(f"speedup:       {unfused / fused:.2f}x")
    print("\nTakeaway:")
    print("- Memory-bound ops such as permute/reshape/bev_pooling are prime fusion targets.")
    print("- Fusion helps by removing launches and intermediate tensor traffic, not by changing math.")
    print("- Custom kernels must be verified for numeric parity, determinism, and worst-case latency.")


if __name__ == "__main__":
    main()
