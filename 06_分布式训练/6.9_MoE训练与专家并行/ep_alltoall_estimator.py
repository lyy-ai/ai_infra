# 6.9 MoE 训练与专家并行：all-to-all 通信估算与负载不均模拟
#
# 运行：
#   cd /data/liyangyang/ai_infra/06_分布式训练
#   /data/qwen35_env/bin/python 6.9_MoE训练与专家并行/ep_alltoall_estimator.py
#
# 功能：
#   1) 估算 EP=8/64/256 下单层 MoE 的 all-to-all 通信量与耗时，
#      区分节点内 NVLink 与跨节点 IB，与一层计算时间对比给出通信占比。
#   2) 模拟随机均匀路由下专家负载不均（max/mean）随 token 总数的变化。
import numpy as np

# ---------------- 模型与硬件参数（DeepSeek-V3 量级） ----------------
HIDDEN = 7168            # hidden size
TOPK = 8                 # top-k 路由
N_EXPERTS = 256          # 路由专家数
BYTES = 2                # BF16
N_LAYERS_MOE = 61        # 用于把激活参数摊到单层（近似）

ACTIVATED_PARAMS = 37e9  # 每 token 激活参数
GPU_TFLOPS = 989.0       # H100 BF16 稠密峰值
MFU = 0.35               # 假设训练 MFU

BW_NVLINK_GBS = 900.0    # 节点内 NVLink 有效带宽（每 GPU）
BW_IB_GBS = 50.0         # 跨节点 IB 有效带宽（每 GPU，400Gbps 网卡折算）

TOKENS_PER_GPU = 8192    # 每卡每 micro-batch 的 token 数


def flops_per_token_per_layer():
    """单层 MoE 每 token 的计算量（FLOPs），由激活参数摊到层数近似"""
    return 2.0 * ACTIVATED_PARAMS / N_LAYERS_MOE


def comm_bytes_per_gpu(tokens_per_gpu):
    """单层 dispatch+combine 两次 all-to-all 的单卡通信字节数（近似）"""
    return 2 * tokens_per_gpu * TOPK * HIDDEN * BYTES


def estimate_ep(ep_size, tokens_per_gpu):
    """返回 (带宽 GB/s, 通信耗时 ms, 计算耗时 ms, 通信占比)"""
    if ep_size <= 8:
        bw = BW_NVLINK_GBS
        domain = "节点内 NVLink"
    else:
        bw = BW_IB_GBS
        domain = "跨节点 IB"
    comm_bytes = comm_bytes_per_gpu(tokens_per_gpu)
    t_comm = comm_bytes / (bw * 1e9) * 1e3  # ms
    eff_flops = GPU_TFLOPS * MFU * 1e12
    t_comp = tokens_per_gpu * flops_per_token_per_layer() / eff_flops * 1e3  # ms
    return domain, comm_bytes, t_comm, t_comp, t_comm / t_comp


def part1_alltoall_table():
    print("=" * 78)
    print("Part 1: 不同 EP size 下单层 MoE 的 all-to-all 通信估算")
    print(f"  参数: 每卡 token 数={TOKENS_PER_GPU}, top-k={TOPK}, hidden={HIDDEN}, BF16")
    print(f"  通信量近似: 2 x tokens x top-k x hidden x bytes（dispatch + combine）")
    print("=" * 78)
    header = f"{'EP size':>8} | {'通信域':<14} | {'通信量/卡':>10} | {'带宽':>9} | {'通信耗时':>9} | {'计算耗时':>9} | {'通信占比':>8}"
    print(header)
    print("-" * len(header))
    for ep in [8, 64, 256]:
        domain, comm_bytes, t_comm, t_comp, ratio = estimate_ep(ep, TOKENS_PER_GPU)
        print(f"{ep:>8} | {domain:<14} | {comm_bytes/1e9:>8.2f} GB | {('900' if ep<=8 else '50')+' GB/s':>9} | "
              f"{t_comm:>7.1f} ms | {t_comp:>7.1f} ms | {ratio:>7.0%}")
    print()
    print("解读: EP<=8 落在 NVLink 域内，通信只占个位数百分比；")
    print("      EP 跨节点后同样字节数走 IB，通信反超计算，必须靠重叠（DualPipe/DeepEP）掩盖。")
    print()


def part2_load_imbalance():
    print("=" * 78)
    print("Part 2: 随机均匀路由下的专家负载不均（max/mean）")
    print(f"  专家数={N_EXPERTS}, top-k={TOPK}, 每个 token 独立均匀随机选 {TOPK} 个专家")
    print("=" * 78)
    rng = np.random.default_rng(42)
    header = f"{'token 总数':>12} | {'平均每专家负载':>14} | {'最热专家负载':>12} | {'max/mean':>9}"
    print(header)
    print("-" * len(header))
    for total_tokens in [1_000, 10_000, 100_000, 1_000_000]:
        # 每个 token 产生 topk 次独立选择，等价于 total_tokens*topk 次均匀抽样
        draws = rng.integers(0, N_EXPERTS, size=total_tokens * TOPK)
        loads = np.bincount(draws, minlength=N_EXPERTS)
        mean = loads.mean()
        print(f"{total_tokens:>12,} | {mean:>14.1f} | {loads.max():>12,} | {loads.max()/mean:>9.3f}")
    print()
    print("解读: token 越多 max/mean 越接近 1（大数定律），大 batch 对 MoE 更友好；")
    print("      小 batch 下最热专家负载可达均值 1.5 倍以上，通信和计算都由它决定（木桶效应）。")
    print("      真实路由远非均匀（马太效应），负载不均会比随机情形严重得多，")
    print("      这正是 auxiliary loss / aux-loss-free bias 存在的理由。")
    print()


def main():
    part1_alltoall_table()
    part2_load_imbalance()


if __name__ == "__main__":
    main()
