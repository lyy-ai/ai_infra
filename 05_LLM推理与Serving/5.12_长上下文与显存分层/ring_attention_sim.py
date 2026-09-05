#!/usr/bin/env python3
"""
Ring Attention / Context Parallelism 时间线模拟器。

模拟 P 张卡做 ring attention 的执行时间线：
- 序列维均分到 P 卡，每卡持有 block = seq_len / P 的 Q、K、V；
- 共 P 步：每步本地 Q block 对当前持有的 K/V block 做 partial attention
  （online softmax 累积），同时把 K/V block 发给环上下一卡；
- 通信与计算重叠：每步耗时 = max(计算时间, 通信时间)；
- 因果注意力平均计算量减半（zigzag 切分后各卡负载均衡，教学上取 0.5 系数）。

输出：
1. 单步计算/通信时间的 roofline 判断（通信能否被计算掩盖）；
2. 随 P 与 seq_len 变化的可扩展性表格（总耗时、并行效率、每卡 KV 占用）。

只依赖 stdlib，可直接运行。

运行方式：
    python ring_attention_sim.py
"""
import math


# ---------------- 模型与硬件配置 ----------------
HIDDEN = 7168           # hidden size（DeepSeek-V3 规模）
N_LAYERS = 61           # 层数
N_HEADS = 128           # Q head 数
N_KV_HEADS = 8          # GQA-8 的 KV head 数
D_HEAD = HIDDEN // N_HEADS  # 56
DTYPE_BYTES = 2

GPU_FLOPS = 6e14        # 有效 BF16 算力（约 600 TFLOPS，教学估算值）
NVLINK_BW = 450e9       # 机内 NVLink 有效带宽（GB/s 量级）
RDMA_BW = 50e9          # 跨节点 RDMA 单卡有效带宽（GB/s 量级）
INTRA_NODE_GPUS = 8     # 机内 GPU 数，超过则 ring 需跨节点

CAUSAL_FACTOR = 0.5     # 因果掩码 + zigzag 均衡后的平均计算系数


def kv_block_bytes(block_tokens: int) -> int:
    """单层一个 K/V block 的字节数。"""
    return 2 * block_tokens * N_KV_HEADS * D_HEAD * DTYPE_BYTES


def ring_step_times(block_tokens: int, p: int) -> tuple:
    """
    返回单层单步 (计算时间, 通信时间)（秒）。

    计算：partial attention flops ≈ CAUSAL_FACTOR × 4 × b^2 × hidden
         （QK^T 与 PV 各 2*b*b*d_head per head，共 4*b*b*hidden）
    通信：发送一个单层 K/V block = 2 × b × n_kv_heads × d_head × 2B
    """
    b = block_tokens
    flops = CAUSAL_FACTOR * 4.0 * b * b * HIDDEN
    bw = NVLINK_BW if p <= INTRA_NODE_GPUS else RDMA_BW
    return flops / GPU_FLOPS, kv_block_bytes(b) / bw


def simulate(seq_len: int, p: int) -> dict:
    """模拟全模型一层 ring attention × N_LAYERS 的总耗时与并行效率。"""
    b = seq_len // p
    t_comp, t_comm = ring_step_times(b, p)
    step = max(t_comp, t_comm)              # 重叠执行
    total = p * step * N_LAYERS
    ideal = p * t_comp * N_LAYERS           # 通信完全被掩盖的理想值
    return {
        "block": b,
        "t_comp": t_comp,
        "t_comm": t_comm,
        "hidden": t_comm <= t_comp,
        "total": total,
        "efficiency": ideal / total if total > 0 else 0.0,
        "kv_per_gpu_gib": kv_block_bytes(b) * N_LAYERS / (1024 ** 3),
    }


def print_roofline_detail():
    """打印单步计算 vs 通信的 roofline 细节。"""
    print("=" * 88)
    print("单步 roofline：通信能否被计算掩盖？（单层视角，全模型共 61 层）")
    print(f"(算力 {GPU_FLOPS / 1e12:.0f} TFLOPS | NVLink {NVLINK_BW / 1e9:.0f} GB/s | "
          f"RDMA {RDMA_BW / 1e9:.0f} GB/s | GQA-8)")
    print("=" * 88)
    print(f"{'block tokens':>13} {'互联':>8} {'单步计算(us)':>13} "
          f"{'单步通信(us)':>13} {'通信/计算':>10} {'是否掩盖':>8}")
    print("-" * 88)
    for b, p in [(1024, 8), (4096, 8), (2048, 32), (8192, 32), (16384, 64), (65536, 64)]:
        t_comp, t_comm = ring_step_times(b, p)
        link = "NVLink" if p <= INTRA_NODE_GPUS else "RDMA"
        ratio = t_comm / t_comp
        print(f"{b:>13} {link:>8} {t_comp * 1e6:>13.1f} {t_comm * 1e6:>13.1f} "
              f"{ratio:>10.2f} {'是' if ratio <= 1 else '否':>8}")
    print("-" * 88)
    be_nv = GPU_FLOPS * 2 * N_KV_HEADS * D_HEAD * DTYPE_BYTES / (CAUSAL_FACTOR * 4 * HIDDEN * NVLINK_BW)
    be_rdma = GPU_FLOPS * 2 * N_KV_HEADS * D_HEAD * DTYPE_BYTES / (CAUSAL_FACTOR * 4 * HIDDEN * RDMA_BW)
    print(f"Note: 通信掩盖的 break-even block ≈ NVLink {be_nv:.0f} tokens / "
          f"RDMA {be_rdma:.0f} tokens；")
    print("      block 越大计算/通信比越高，跨节点 RDMA 带宽低一个量级，")
    print("      需要 9 倍大的 block 才能掩盖——长序列 + 大 P 时更容易扩展。")


def print_scalability_table():
    """打印随 P 与 seq_len 变化的可扩展性表格。"""
    seq_lens = [16 * 1024, 32 * 1024, 128 * 1024, 512 * 1024, 1024 * 1024, 4 * 1024 * 1024]
    ps = [8, 16, 32, 64]
    print("\n" + "=" * 88)
    print("可扩展性：全模型总耗时(秒) 与并行效率，随 P 与 seq_len 变化")
    print("=" * 88)
    print(f"{'seq_len':>10}" + "".join(f"{('P=' + str(p)):>18}" for p in ps))
    print("-" * 88)
    for L in seq_lens:
        row = f"{L // 1024:>8d}k "
        for p in ps:
            r = simulate(L, p)
            mark = "" if r["hidden"] else "*"   # * 表示通信未被完全掩盖
            row += f"{r['total']:>8.2f}s/{r['efficiency'] * 100:>5.0f}%{mark:>2}"
        print(row)
    print("-" * 88)
    print("格式: 总耗时/并行效率，* 表示通信暴露（comm > compute，效率受损）。")

    print("\n每卡 KV 占用（GiB，全模型 61 层）与单卡基线对比")
    print("-" * 88)
    print(f"{'seq_len':>10} {'单卡':>10}" + "".join(f"{('P=' + str(p)):>12}" for p in ps))
    for L in seq_lens:
        single = kv_block_bytes(L) * N_LAYERS / (1024 ** 3)
        row = f"{L // 1024:>8d}k {single:>10.1f}"
        for p in ps:
            row += f"{simulate(L, p)['kv_per_gpu_gib']:>12.2f}"
        print(row)
    print("-" * 88)
    print("Note: ring attention 把每卡 KV 降为 1/P，用通信换显存；")
    print("      单卡放不下的百万级上下文（GQA-8 下 1M 约 107 GiB）可摊到多卡。")


def print_timeline_example():
    """打印一个 P=4 的简化时间线，帮助理解 ring 的执行过程。"""
    print("\nP=4 ring attention 时间线示例（每卡 block = L/4）")
    print("-" * 88)
    print("  step | GPU0 持有KV | GPU1 持有KV | GPU2 持有KV | GPU3 持有KV | 动作")
    print("  -----+------------+------------+------------+------------+------------------")
    for s in range(4):
        holders = [(i - s) % 4 for i in range(4)]
        print(f"   {s}   |     B{holders[0]}      |     B{holders[1]}      |"
              f"     B{holders[2]}      |     B{holders[3]}      | 本地 partial attn + 传 KV 给下一卡")
    print("  -----+------------+------------+------------+------------+------------------")
    print("  4 步后：每张卡的 Q block 都见过全部 4 个 KV block，经 online softmax 归并得到正确输出。")


def run_simulation():
    """运行完整 ring attention 分析。"""
    print("Ring Attention / Context Parallelism 模拟器")
    print(f"模型: hidden={HIDDEN}  heads={N_HEADS}  kv_heads={N_KV_HEADS} (GQA-8)  BF16")
    print(f"假设: 通信与计算完全重叠（每步 = max(comp, comm)），zigzag 均衡后因果系数 = {CAUSAL_FACTOR}\n")
    print_timeline_example()
    print()
    print_roofline_detail()
    print_scalability_table()


if __name__ == "__main__":
    run_simulation()
