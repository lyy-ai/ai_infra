# DiT 并行加速比模拟（17.3 节配套）
# 运行: python parallel_dit_sim.py
# 只依赖标准库
#
# 模拟 CFG 并行与 USP 序列并行在不同卡数下的端到端加速比：
# - CFG 并行：两次前向分两组卡，通信近零，理想 2x
# - USP：序列切 P 份，计算降为 1/P，但引入 ring/all-to-all 通信
#   通信是否暴露取决于 块计算时间 vs 块传输时间

SEQ_LEN = 100_000        # 视频 DiT 序列长度
HIDDEN = 3072
N_LAYERS = 40
STEPS = 30
GPU_TFLOPS = 300.0       # 有效算力（约 30% MFU 的 H100 级）
NVLINK_GBS = 450.0       # 有效互联带宽（约半双工折算）


def compute_time_per_step(seq):
    n, d = seq, HIDDEN
    flops = (4 * n * d * d + 2 * n * n * d) * N_LAYERS
    return flops / (GPU_TFLOPS * 1e12)


def main():
    t_step_single = compute_time_per_step(SEQ_LEN)
    print("=" * 76)
    print(f"视频 DiT 并行模拟（序列 {SEQ_LEN:,}，{STEPS} 步）")
    print(f"单卡单步耗时 ≈ {t_step_single*1000:.0f} ms，全步数 ≈ {t_step_single*STEPS:.1f} s")
    print("=" * 76)

    print("\n[1] CFG 并行（2 组卡，有条件/无条件各跑一组）")
    t_cfg = t_step_single * STEPS  # 两次前向并行 -> 等效单前向时间
    t_serial = t_step_single * 2 * STEPS
    print(f"  串行两次前向: {t_serial:.1f}s  ->  CFG 并行: {t_cfg:.1f}s  (加速 {t_serial/t_cfg:.1f}x，通信≈0)")

    print("\n[2] USP 序列并行（序列切 P 份，ring 传 KV，含通信开销）")
    print(f"{'P(卡数)':>8}{'每卡序列':>10}{'计算时间':>10}{'通信时间':>10}{'总耗时':>10}{'加速比':>8}{'效率':>8}")
    print("-" * 76)
    for p in (1, 2, 4, 8, 16):
        seq_per = SEQ_LEN // p
        # 每卡计算：线性层随 1/P 下降；attention 是本地 Q 块 × 全量 KV（ring 轮换），总量不变
        linear_flops = 4 * seq_per * HIDDEN * HIDDEN * N_LAYERS
        attn_flops = 2 * seq_per * SEQ_LEN * HIDDEN * N_LAYERS
        t_comp = (linear_flops + attn_flops) / (GPU_TFLOPS * 1e12) * STEPS
        # 通信：每层 ring 传 K/V ≈ 2 × seq_per × hidden × 2B × 2(K,V)
        comm_bytes = 2 * seq_per * HIDDEN * 2 * 2 * N_LAYERS
        t_comm = comm_bytes / (NVLINK_GBS * 1e9) * STEPS
        # 通信与计算部分重叠，按 50% 暴露
        t_total = t_comp + t_comm * 0.5
        print(f"{p:>8}{seq_per:>10,}{t_comp:>9.1f}s{t_comm*0.5:>9.1f}s{t_total:>9.1f}s"
              f"{t_step_single*STEPS/t_total:>7.1f}x{t_step_single*STEPS/t_total/p:>7.0%}")

    print("-" * 76)
    print("\n[3] 组合：CFG 2 组 × USP 8 卡 = 16 卡")
    seq_per = SEQ_LEN // 8
    linear_flops = 4 * seq_per * HIDDEN * HIDDEN * N_LAYERS
    attn_flops = 2 * seq_per * SEQ_LEN * HIDDEN * N_LAYERS
    t_comp = (linear_flops + attn_flops) / (GPU_TFLOPS * 1e12) * STEPS
    comm_bytes = 2 * seq_per * HIDDEN * 2 * 2 * N_LAYERS
    t_total = t_comp + (comm_bytes / (NVLINK_GBS * 1e9) * STEPS) * 0.5
    print(f"  端到端 ≈ {t_total:.1f}s，相对单卡串行 {t_serial:.1f}s 加速 {t_serial/t_total:.1f}x")

    print("\n读法：")
    print("1. CFG 并行是白送的 2x（零通信），只要卡够就先开")
    print("2. 本例 DiT 计算/通信比高，USP 接近线性加速；序列更短或带宽更低时效率会下降")
    print("3. DiT 无因果掩码，序列维负载天然均衡，不需要 LLM 的 zigzag 切分")
    print("4. 组合并行是生产形态：CFG 分组 × 组内 USP")


if __name__ == "__main__":
    main()
