# 2.6 集合通信基础：Ring/Tree AllReduce 通信量与耗时
#
# 运行：
#   /data/qwen35_env/bin/python 1.10_集合通信基础/collective_comm_calc.py

import math


def ring_allreduce(size_gb, n_ranks):
    steps = 2 * (n_ranks - 1)
    per_rank_gb = 2 * (n_ranks - 1) / n_ranks * size_gb
    return steps, per_rank_gb


def tree_allreduce(size_gb, n_ranks):
    steps = 2 * math.ceil(math.log2(n_ranks))
    per_rank_gb = 2 * size_gb
    return steps, per_rank_gb


def time_ms(data_gb, bw_gbps):
    return data_gb / bw_gbps * 1000


def main():
    S, N = 1.0, 8
    print(f"梯度 {S}GB, {N} ranks\n")
    for name, fn in [("Ring", ring_allreduce), ("Tree", tree_allreduce)]:
        steps, per = fn(S, N)
        print(f"{name}: steps={steps}, 每rank通信量={per:.2f}GB")
    print()
    for bw_name, bw in [("NVLink 600GB/s", 600), ("IB HDR 25GB/s", 25)]:
        _, per = ring_allreduce(S, N)
        print(f"{bw_name:<16}: 同步耗时 ≈ {time_ms(per, bw):.1f} ms")
    print("\n结论：Ring 通信量与 N 无关；NVLink 比 IB 快一个数量级 → TP 节点内、DP/PP 跨节点")


if __name__ == "__main__":
    main()
