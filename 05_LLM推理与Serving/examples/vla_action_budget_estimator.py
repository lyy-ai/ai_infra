#!/usr/bin/env python3
"""
VLA（Vision-Language-Action）动作 token 预算估算示例。

本脚本估算机器人/自动驾驶动作模型中：
1. 每个动作 chunk 需要多少动作 token
2. 在给定控制频率下，推理引擎需要多快的 decode token/s
3. 缩短 action horizon、减少 action dim、降低每维 token 数对延迟预算的影响

无需 GPU，可直接运行。

运行方式：
    python examples/vla_action_budget_estimator.py
"""
from dataclasses import dataclass, replace


@dataclass
class VLAConfig:
    name: str
    action_dim: int
    bins_per_dim: int
    tokens_per_dim: int
    horizon_steps: int
    control_hz: float
    prompt_image_tokens: int


def action_tokens(cfg: VLAConfig) -> int:
    """一个 action chunk 的动作 token 数。"""
    return cfg.action_dim * cfg.tokens_per_dim * cfg.horizon_steps


def chunk_budget_ms(cfg: VLAConfig) -> float:
    """一个 chunk 覆盖的物理时间，即生成下一个 chunk 的硬预算。"""
    return cfg.horizon_steps / cfg.control_hz * 1000.0


def required_decode_tps(cfg: VLAConfig) -> float:
    """为在 budget 内生成动作 token 所需的 decode token/s。"""
    return action_tokens(cfg) / (chunk_budget_ms(cfg) / 1000.0)


def print_config(cfg: VLAConfig):
    """打印单个 VLA 配置估算。"""
    act = action_tokens(cfg)
    budget = chunk_budget_ms(cfg)
    tps = required_decode_tps(cfg)
    total_seq = cfg.prompt_image_tokens + act
    print(f"\n[{cfg.name}]")
    print(f"  action_dim={cfg.action_dim}, bins={cfg.bins_per_dim}, tokens/dim={cfg.tokens_per_dim}")
    print(f"  horizon={cfg.horizon_steps} steps @ {cfg.control_hz:.1f} Hz -> budget {budget:.1f} ms")
    print(f"  prompt/image tokens: {cfg.prompt_image_tokens}")
    print(f"  action tokens per chunk: {act}")
    print(f"  total tokens per chunk: {total_seq}")
    print(f"  required decode speed for action tokens: {tps:.2f} tok/s")
    return tps


def main():
    print("=" * 84)
    print("VLA Action Token Budget Estimator")
    print("=" * 84)

    configs = [
        VLAConfig("vla_7d_h50", action_dim=7, bins_per_dim=256, tokens_per_dim=1, horizon_steps=50, control_hz=50, prompt_image_tokens=512),
        VLAConfig("vla_7d_h10", action_dim=7, bins_per_dim=256, tokens_per_dim=1, horizon_steps=10, control_hz=50, prompt_image_tokens=512),
        VLAConfig("vla_14d_h50", action_dim=14, bins_per_dim=256, tokens_per_dim=1, horizon_steps=50, control_hz=50, prompt_image_tokens=512),
        VLAConfig("vla_14d_h10_2tok", action_dim=14, bins_per_dim=256, tokens_per_dim=2, horizon_steps=10, control_hz=50, prompt_image_tokens=512),
    ]

    for cfg in configs:
        print_config(cfg)

    print("\n" + "=" * 84)
    print("Takeaway")
    print("=" * 84)
    print("- Long action horizons amortize prefill but increase the time before the next replanning update.")
    print("- More action dimensions or more tokens per dimension directly increase decode work per chunk.")
    print("- On edge VLA, common levers are shorter chunks, coarser action bins, action caching, and smaller image/prompt tokens.")


if __name__ == "__main__":
    main()
