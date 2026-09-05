# 6.10 弹性训练与容错：Checkpoint 成本与 goodput 模型
#
# 运行：
#   cd /data/liyangyang/ai_infra/06_分布式训练
#   /data/qwen35_env/bin/python 6.10_弹性训练与容错/checkpoint_cost_model.py
#
# 功能：
#   给定集群规模、单卡年故障率、模型+优化器状态大小、存储带宽、checkpoint 间隔，
#   计算集群期望故障间隔（MTBF）、单次故障平均损失进度、
#   同步 vs 异步 checkpoint 的 goodput，并扫描不同间隔找最优。
import numpy as np

# ---------------- 集群与模型参数 ----------------
N_GPUS = 10_000              # 集群卡数
ANNUAL_FAIL_RATE = 0.20      # 等效年化中断率（每张卡）：
                             # 单 GPU 年化故障率仅约 3%，但中断来自 GPU/NVLink/网卡/
                             # 存储/软件多层，等效年化中断率可到约 20%
                             # （参考公开数据：Llama-3 16k 卡 54 天数百次中断，约每天 8 次）
STATE_SIZE_TB = 10.0         # 模型+优化器状态总量（671B 模型约 10 TB 量级）
STORAGE_BW_TB_S = 0.5        # 聚合存储带宽（TB/s）
ASYNC_SNAPSHOT_S = 5.0       # 异步 checkpoint 的阻塞时间（内存快照，秒级）
RESTART_MIN = 10.0           # 故障后重启+加载 checkpoint 的耗时（分钟）

HOURS_PER_YEAR = 8760.0


def cluster_mtbf_hours(n_gpus, annual_fail_rate):
    """集群期望故障间隔：MTBF = 8760 / (N x 单卡年化故障率)"""
    failures_per_hour = n_gpus * annual_fail_rate / HOURS_PER_YEAR
    return 1.0 / failures_per_hour


def goodput(interval_h, ckpt_overhead_h, mtbf_h, restart_h):
    """
    goodput 模型（教学简化）：
      每个 checkpoint 间隔 I 内：
        - 发生故障的期望次数 I/M
        - 每次故障平均损失进度 I/2（上次 ckpt 之后已训练的时间）+ 重启时间 R
        - checkpoint 本身阻塞 C
      goodput = 有效训练时间 / 墙钟时间 = (I - I/M*(I/2 + R)) / (I + C)
    """
    lost_per_interval = (interval_h / mtbf_h) * (interval_h / 2.0 + restart_h)
    useful = interval_h - lost_per_interval
    wall = interval_h + ckpt_overhead_h
    return useful / wall


def main():
    mtbf_h = cluster_mtbf_hours(N_GPUS, ANNUAL_FAIL_RATE)
    restart_h = RESTART_MIN / 60.0
    sync_overhead_h = (STATE_SIZE_TB / STORAGE_BW_TB_S) / 3600.0   # 落盘阻塞，小时
    async_overhead_h = ASYNC_SNAPSHOT_S / 3600.0                   # 内存快照阻塞，小时

    print("=" * 78)
    print("Part 1: 集群故障基本面")
    print("=" * 78)
    print(f"  集群规模            : {N_GPUS:,} 卡")
    print(f"  等效年化中断率/卡   : {ANNUAL_FAIL_RATE:.0%}（含 GPU/网络/存储/软件全故障层）")
    print(f"  集群期望故障间隔    : {mtbf_h:.2f} 小时（约每天 {24.0/mtbf_h:.1f} 次中断）")
    print(f"  模型+优化器状态     : {STATE_SIZE_TB:.1f} TB")
    print(f"  聚合存储带宽        : {STORAGE_BW_TB_S:.2f} TB/s")
    print(f"  同步 ckpt 阻塞      : {sync_overhead_h*3600:.1f} 秒/次")
    print(f"  异步 ckpt 阻塞      : {ASYNC_SNAPSHOT_S:.1f} 秒/次（仅内存快照）")
    print(f"  故障重启+加载       : {RESTART_MIN:.0f} 分钟/次")
    print()

    intervals_h = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    print("=" * 78)
    print("Part 2: 扫描 checkpoint 间隔，对比同步 vs 异步的 goodput")
    print("=" * 78)
    header = (f"{'间隔':>7} | {'期望损失进度/次':>14} | {'同步 goodput':>12} | {'异步 goodput':>12}")
    print(header)
    print("-" * len(header))

    best_sync, best_async = None, None
    for iv in intervals_h:
        expected_lost_min = iv / 2.0 * 60.0  # 单次故障平均损失的进度（分钟）
        gp_sync = goodput(iv, sync_overhead_h, mtbf_h, restart_h)
        gp_async = goodput(iv, async_overhead_h, mtbf_h, restart_h)
        label = f"{iv*60:.0f}min" if iv < 1 else f"{iv:.0f}h"
        print(f"{label:>7} | {expected_lost_min:>12.1f}min | {gp_sync:>11.1%} | {gp_async:>11.1%}")
        if best_sync is None or gp_sync > best_sync[1]:
            best_sync = (iv, gp_sync)
        if best_async is None or gp_async > best_async[1]:
            best_async = (iv, gp_async)

    print()
    print(f"  同步方案最优: 间隔 {best_sync[0]*60:.0f} 分钟, goodput {best_sync[1]:.1%}")
    print(f"  异步方案最优: 间隔 {best_async[0]*60:.0f} 分钟, goodput {best_async[1]:.1%}")
    print()
    print("解读:")
    print("  1) 间隔越小，单次故障损失越小，但 checkpoint 开销越大 -> 存在最优间隔；")
    print("  2) 异步 checkpoint 把阻塞从数十秒压到秒级后，最优间隔变密、goodput 更高；")
    print("  3) 万卡集群 MTBF 只有数小时，间隔大到 4-8h 时期望故障次数超过 1，")
    print("     goodput 被故障损失主导、急剧恶化。")
    print()

    # 规模敏感性
    print("=" * 78)
    print("Part 3: 集群规模敏感性（固定异步方案，间隔 15 分钟）")
    print("=" * 78)
    header = f"{'卡数':>8} | {'MTBF':>10} | {'每天中断':>9} | {'异步 goodput':>12}"
    print(header)
    print("-" * len(header))
    for n in [1_000, 4_000, 10_000, 32_000]:
        m = cluster_mtbf_hours(n, ANNUAL_FAIL_RATE)
        gp = goodput(0.25, async_overhead_h, m, restart_h)
        print(f"{n:>8,} | {m:>8.2f} h | {24.0/m:>8.1f} 次 | {gp:>11.1%}")
    print()
    print("解读: 规模越大 MTBF 越短，同样的容错策略 goodput 越低；")
    print("      超大规模必须进一步压缩重启时间（in-memory 副本、热备机）才能守住 goodput。")


if __name__ == "__main__":
    main()
