#!/usr/bin/env python3
"""
BEV/PlanNN 显存与算力估算示例。

本脚本用于教学估算自动驾驶 BEV 感知 + PlanNN 规划模型在不同结构配置下的：
1. BEV feature 显存占用
2. dense attention vs deformable attention 的近似 FLOPs
3. 减小 BEV 分辨率、通道数、历史帧、量化位宽后的收益

无需 GPU，可直接运行。

运行方式：
    python examples/bev_memory_compute_estimator.py
"""
from dataclasses import dataclass, replace


@dataclass
class BEVConfig:
    name: str
    bev_h: int
    bev_w: int
    channels: int
    history_frames: int
    dtype_bytes: int
    deformable_points: int = 8
    planning_input_dim: int = 512
    planning_hidden_dim: int = 1024
    planning_layers: int = 4


BASELINE = BEVConfig(
    name="baseline_fp16",
    bev_h=200,
    bev_w=200,
    channels=256,
    history_frames=4,
    dtype_bytes=2,
)

OPTIMIZED = BEVConfig(
    name="optimized_int8",
    bev_h=128,
    bev_w=128,
    channels=128,
    history_frames=2,
    dtype_bytes=1,
)


def bev_feature_mib(cfg: BEVConfig) -> float:
    """BEV feature 显存，单位 MiB。"""
    num_bytes = cfg.bev_h * cfg.bev_w * cfg.channels * cfg.history_frames * cfg.dtype_bytes
    return num_bytes / (1024 ** 2)


def dense_attention_gflops(cfg: BEVConfig) -> float:
    """BEV 自注意力近似 FLOPs：4*N*C^2 + 2*N^2*C。"""
    n = cfg.bev_h * cfg.bev_w
    c = cfg.channels
    flops = 4 * n * c * c + 2 * n * n * c
    return flops * cfg.history_frames / 1e9


def deformable_attention_gflops(cfg: BEVConfig) -> float:
    """Deformable attention 近似 FLOPs：4*N*C^2 + 2*N*K*C。"""
    n = cfg.bev_h * cfg.bev_w
    c = cfg.channels
    k = cfg.deformable_points * cfg.history_frames
    flops = 4 * n * c * c + 2 * n * k * c
    return flops / 1e9


def planning_gflops(cfg: BEVConfig) -> float:
    """PlanNN MLP/规划头近似 FLOPs。"""
    flops = 2 * cfg.planning_layers * cfg.planning_input_dim * cfg.planning_hidden_dim
    flops += 2 * cfg.planning_hidden_dim * cfg.planning_hidden_dim * cfg.planning_layers
    return flops / 1e9


def print_config(cfg: BEVConfig):
    """打印单个配置估算结果。"""
    dense = dense_attention_gflops(cfg)
    deform = deformable_attention_gflops(cfg)
    plan = planning_gflops(cfg)
    print(f"\n[{cfg.name}]")
    print(f"  BEV: {cfg.bev_h}x{cfg.bev_w}, channels={cfg.channels}, history={cfg.history_frames}, dtype={cfg.dtype_bytes}B")
    print(f"  BEV feature memory: {bev_feature_mib(cfg):.2f} MiB")
    print(f"  dense attention: {dense:.2f} GFLOPs")
    print(f"  deformable attention: {deform:.2f} GFLOPs")
    print(f"  PlanNN head: {plan:.2f} GFLOPs")
    print(f"  total(deformable): {deform + plan:.2f} GFLOPs")
    return deform + plan


def main():
    print("=" * 78)
    print("BEV/PlanNN Memory and Compute Estimator")
    print("=" * 78)
    baseline_total = print_config(BASELINE)
    optimized_total = print_config(OPTIMIZED)

    print("\n" + "=" * 78)
    print("Comparison")
    print("=" * 78)
    mem_ratio = bev_feature_mib(OPTIMIZED) / bev_feature_mib(BASELINE)
    compute_ratio = optimized_total / baseline_total
    print(f"BEV feature memory ratio: {mem_ratio:.3f}x")
    print(f"deformable+planning compute ratio: {compute_ratio:.3f}x")
    print("\nTakeaway:")
    print("- BEV resolution and channel count dominate both memory and attention compute.")
    print("- Deformable/sparse attention avoids the N^2 term of dense BEV self-attention.")
    print("- INT8/FP8 reduce memory traffic, but accuracy must be validated with calibration/QAT.")


if __name__ == "__main__":
    main()
