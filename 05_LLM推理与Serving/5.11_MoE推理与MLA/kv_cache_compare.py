#!/usr/bin/env python3
"""
MHA / GQA / MQA / MLA 的 KV Cache 显存定量对比。

以 DeepSeek-V3 规模参数为例：
- 61 层，hidden 7168，128 个注意力头（d_head = 56）
- MLA：每层每 token 存 d_c=512 维 latent + 64 维 rope 分量，共 576 维
- BF16，每元素 2 字节

输出：
1. 每 token 每层 / 每 token 全模型 的 KV Cache 字节数对比表。
2. 4k / 32k / 128k / 1M 上下文长度下的单请求 KV Cache 占用表。
3. "一张 80GB 卡在给定 KV 预算下能容纳多少条 128k 上下文并发"的估算。

只依赖 stdlib + numpy，可直接运行。

运行方式：
    python kv_cache_compare.py
"""
import numpy as np


# ---------------- DeepSeek-V3 规模配置 ----------------
N_LAYERS = 61
HIDDEN = 7168
N_HEADS = 128               # MHA 注意力头数
D_HEAD = HIDDEN // N_HEADS  # 56
GQA_KV_HEADS = 8
MQA_KV_HEADS = 1
MLA_LATENT = 512            # d_c
MLA_ROPE = 64               # rope 分量
DTYPE_BYTES = 2             # BF16

CONTEXT_LENGTHS = [4 * 1024, 32 * 1024, 128 * 1024, 1024 * 1024]


def kv_bytes_per_token_mha(n_kv_heads: int) -> int:
    """MHA/GQA/MQA：每 token 全模型 KV 字节数 = 2(K,V) × 层数 × kv头数 × d_head × 2B。"""
    return 2 * N_LAYERS * n_kv_heads * D_HEAD * DTYPE_BYTES


def kv_bytes_per_token_mla() -> int:
    """MLA：每 token 全模型 = 层数 × (latent + rope) × 2B，与 head 数无关。"""
    return N_LAYERS * (MLA_LATENT + MLA_ROPE) * DTYPE_BYTES


STRUCTURES = {
    "MHA-128": kv_bytes_per_token_mha(N_HEADS),
    "GQA-8":   kv_bytes_per_token_mha(GQA_KV_HEADS),
    "MQA-1":   kv_bytes_per_token_mha(MQA_KV_HEADS),
    "MLA-576": kv_bytes_per_token_mla(),
}


def fmt_gib(n_bytes: float) -> str:
    """格式化为 GiB / MiB / KiB。"""
    gib = n_bytes / (1024 ** 3)
    if gib >= 1.0:
        return f"{gib:10.2f} GiB"
    mib = n_bytes / (1024 ** 2)
    if mib >= 1.0:
        return f"{mib:10.2f} MiB"
    return f"{n_bytes / 1024:10.2f} KiB"


def print_per_token_table():
    """打印每 token KV Cache 对比表。"""
    print("=" * 72)
    print("每 token KV Cache 字节数对比（61 层，BF16）")
    print("=" * 72)
    print(f"{'结构':>10} {'公式':>30} {'每层每token':>14} {'全模型每token':>16} {'相对MHA':>10}")
    print("-" * 72)
    formulas = {
        "MHA-128": "2×128×56×2B ×61L",
        "GQA-8": "2×8×56×2B ×61L",
        "MQA-1": "2×1×56×2B ×61L",
        "MLA-576": "(512+64)×2B ×61L",
    }
    base = STRUCTURES["MHA-128"]
    for name, total in STRUCTURES.items():
        per_layer = total // N_LAYERS
        print(f"{name:>10} {formulas[name]:>30} {per_layer:>12d} B {total:>14d} B "
              f"{total / base:>10.4f}")
    print("-" * 72)
    print("Note: MLA 的 576 维与 head 数无关；MQA 最小但信息容量受限，")
    print("      MLA 用 512 维 latent 在'保质量'与'省显存'之间取得平衡。")


def print_context_table():
    """打印不同上下文长度下单请求 KV Cache 占用。"""
    print("\n" + "=" * 72)
    print("单请求 KV Cache 占用随上下文长度的变化")
    print("=" * 72)
    header = f"{'结构':>10}" + "".join(f"{('ctx=' + str(c // 1024) + 'k'):>16}" for c in CONTEXT_LENGTHS)
    print(header)
    print("-" * 72)
    for name, per_token in STRUCTURES.items():
        row = f"{name:>10}"
        for ctx in CONTEXT_LENGTHS:
            row += f"{fmt_gib(per_token * ctx):>16}"
        print(row)
    print("-" * 72)
    print("Note: KV Cache 随上下文长度线性增长；MHA 在 128k 时已超 200 GiB，")
    print("      单卡 HBM 完全放不下，必须靠 GQA/MLA 压缩或显存分层（见 5.12 节）。")


def print_capacity_estimate():
    """估算 80GB 卡在给定 KV 预算下的 128k 并发容量。"""
    hbm_total = 80 * (1024 ** 3)
    print("\n" + "=" * 72)
    print("一张 80GB 卡能容纳多少条 128k 上下文并发？")
    print("=" * 72)
    print(f"{'KV预算(GiB)':>12}" + "".join(f"{name:>14}" for name in STRUCTURES))
    print("-" * 72)
    ctx = 128 * 1024
    for budget_gib in [20, 40, 60]:
        budget = budget_gib * (1024 ** 3)
        if budget > hbm_total:
            continue
        row = f"{budget_gib:>12d}"
        for per_token in STRUCTURES.values():
            n_req = budget / (per_token * ctx)
            row += f"{n_req:>14.1f}"
        print(row)
    print("-" * 72)
    print("Note: 实际部署中 80GB 还要扣掉权重与激活，KV 预算通常只有几十 GiB；")
    print("      且本表假设该卡独占这些 KV（DP attention 下每卡需复制完整 KV，")
    print("      TP 下 MHA/GQA 可按卡数切分而 MLA 无法按 head 切分，只能复制）。")
    mla_per_req = STRUCTURES["MLA-576"] * ctx / (1024 ** 3)
    print(f"\n结论: 128k 单请求 MLA KV 约 {mla_per_req:.1f} GiB，")
    print(f"      60 GiB KV 预算下 MLA 可并发约 {60 / mla_per_req:.0f} 条，")
    print(f"      而 MHA 连 1 条都放不下（需要约 "
          f"{STRUCTURES['MHA-128'] * ctx / (1024 ** 3):.0f} GiB）。")


def run_analysis():
    """运行完整 KV Cache 对比分析。"""
    print("KV Cache 结构对比分析（DeepSeek-V3 规模：61 层 / hidden 7168 / 128 头）")
    print(f"d_head = {D_HEAD}，MLA latent = {MLA_LATENT} + rope {MLA_ROPE} = "
          f"{MLA_LATENT + MLA_ROPE} 维，BF16\n")
    print_per_token_table()
    print_context_table()
    print_capacity_estimate()


if __name__ == "__main__":
    run_analysis()
