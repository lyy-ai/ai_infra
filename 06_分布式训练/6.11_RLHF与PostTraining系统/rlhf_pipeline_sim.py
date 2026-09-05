# 6.11 RLHF 与 Post-Training 系统：一轮迭代的耗时流水线模拟
#
# 运行：
#   cd /data/liyangyang/ai_infra/06_分布式训练
#   /data/qwen35_env/bin/python 6.11_RLHF与PostTraining系统/rlhf_pipeline_sim.py
#
# 功能：
#   模拟一次 RLHF 迭代的各阶段耗时（rollout 长尾 / reward / 训练 step / 权重同步），
#   对比三档优化：
#     A. 无优化（静态批处理 + 过 CPU 的朴素权重同步）
#     B. 开启动态批处理（持续批处理消除长尾空等）
#     C. 再叠加权重同步分桶流水化（gather 与广播重叠、GPU-GPU 直传）
import numpy as np

# ---------------- 模拟参数 ----------------
N_PROMPTS = 512            # 一轮的 prompt 数
GROUP_SIZE = 16            # GRPO 每 prompt 采样的 response 数
N_SEQ = N_PROMPTS * GROUP_SIZE  # 总生成序列数 = 8192

# response 长度分布：lognormal，均值约 1000 token，长尾可到数千
LEN_MU, LEN_SIGMA = np.log(768), 0.7
LEN_MAX = 8192

TOKENS_PER_SEC_PER_SEQ = 15.0   # 单序列解码速度（tok/s，大批量大模型下的近似）
DYN_BATCH_UTIL = 0.85           # 持续批处理下引擎的槽位利用率

REWARD_SEC = 5.0           # reward 打分（RM 前向，批量）
ADV_SEC = 1.0              # 组内优势计算（轻量）
TRAIN_SEC = 15.0           # actor 训练 step（FSDP/Megatron）
SYNC_NAIVE_SEC = 120.0     # 朴素权重同步：串行 gather + 过 CPU
SYNC_PIPELINED_SEC = 15.0  # 分桶流水 + GPU-GPU 直传


def sample_lengths(rng):
    lens = rng.lognormal(LEN_MU, LEN_SIGMA, size=N_SEQ)
    return np.clip(lens, 16, LEN_MAX).astype(int)


def rollout_static(lens):
    """静态批处理：整批一起开始，等最长序列完成（makespan 由长尾决定）"""
    return lens.max() / TOKENS_PER_SEC_PER_SEQ


def rollout_dynamic(lens):
    """持续批处理：完成一条立刻补一条，总耗时由 token 总量和利用率决定"""
    total_tokens = lens.sum()
    capacity = N_SEQ * TOKENS_PER_SEC_PER_SEQ * DYN_BATCH_UTIL
    return total_tokens / capacity


def run_config(name, rollout_fn, sync_sec, lens):
    t_rollout = rollout_fn(lens)
    stages = {
        "rollout": t_rollout,
        "reward": REWARD_SEC,
        "advantage": ADV_SEC,
        "train": TRAIN_SEC,
        "weight_sync": sync_sec,
    }
    total = sum(stages.values())
    # GPU 有效工作时间：理想 rollout（无长尾空等）+ reward + train
    busy = rollout_dynamic(lens) + REWARD_SEC + ADV_SEC + TRAIN_SEC
    util = busy / total
    return name, stages, total, util


def main():
    rng = np.random.default_rng(42)
    lens = sample_lengths(rng)
    print("=" * 80)
    print("RLHF 一轮迭代耗时模拟")
    print(f"  prompt 数={N_PROMPTS}, GRPO group size={GROUP_SIZE}, 总序列数={N_SEQ}")
    print(f"  response 长度: 均值={lens.mean():.0f}, p50={np.percentile(lens,50):.0f}, "
          f"p99={np.percentile(lens,99):.0f}, max={lens.max()} tokens")
    ideal = lens.sum() / (N_SEQ * TOKENS_PER_SEC_PER_SEQ)
    print(f"  理想 rollout 耗时（无长尾）= {ideal:.1f}s, "
          f"静态批处理 makespan = {rollout_static(lens):.1f}s "
          f"(长尾放大 {rollout_static(lens)/ideal:.1f}x)")
    print("=" * 80)
    print()

    configs = [
        ("A 无优化(静态批处理+朴素同步)", rollout_static, SYNC_NAIVE_SEC),
        ("B +动态批处理", rollout_dynamic, SYNC_NAIVE_SEC),
        ("C +权重同步流水化", rollout_dynamic, SYNC_PIPELINED_SEC),
    ]

    results = []
    stage_names = ["rollout", "reward", "advantage", "train", "weight_sync"]
    header = f"{'配置':<26} | " + " | ".join(f"{s:>11}" for s in stage_names) + f" | {'总耗时':>8} | {'rollout占比':>10} | {'GPU利用率':>9}"
    print(header)
    print("-" * len(header))
    for name, rfn, sync in configs:
        name, stages, total, util = run_config(name, rfn, sync, lens)
        results.append((name, total))
        row = f"{name:<26} | " + " | ".join(f"{stages[s]:>9.1f}s" for s in stage_names)
        row += f" | {total:>6.1f}s | {stages['rollout']/total:>9.0%} | {util:>8.0%}"
        print(row)

    print()
    speedup_b = results[0][1] / results[1][1]
    speedup_c = results[0][1] / results[2][1]
    print(f"加速比: B 相对 A = {speedup_b:.2f}x, C 相对 A = {speedup_c:.2f}x")
    print()
    print("解读:")
    print("  1) 无优化时 rollout 因长尾空等占到大头，叠加朴素权重同步后利用率极低；")
    print("  2) 动态批处理把 makespan 从'最长序列'降到'总 token 量'级别，是 rollout 侧第一优化；")
    print("     此时权重同步成为新瓶颈（短板效应）；")
    print("  3) 权重同步分桶流水化把百 GB 级同步从数十秒压到十秒级；")
    print("  4) 全部优化后 rollout 重新占 50-80% —— 这就是'推理优化能力决定 RLHF 吞吐'的含义。")


if __name__ == "__main__":
    main()
