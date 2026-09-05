#!/usr/bin/env python3
"""
MoE 专家并行（EP）负载分布与 all-to-all 通信量模拟器。

模拟 N 个 token 经 top-k 路由到 E 个专家、专家分布在 P 张卡上的场景：
1. 随机路由（router logits 无干预）vs bias 均衡路由（aux-loss-free 风格动态偏置）。
2. 打印专家负载直方统计：max/mean 比、空闲专家数、各卡负载。
3. 估算 dispatch + combine 两次 all-to-all 通信量随 EP size 的变化。

只依赖 stdlib + numpy，可直接运行。

运行方式：
    python moe_ep_simulator.py
"""
import numpy as np


# ---------------- 模型/工作负载配置（以 DeepSeek-V3 为参照） ----------------
HIDDEN = 7168            # hidden size
DTYPE_BYTES = 2          # BF16
E = 256                  # 路由专家数
TOP_K = 8                # 每 token 路由专家数
N_TOKENS = 4096          # 一个 decode step 的 token 数（大 batch decode）
BIAS_ROUNDS = 20         # bias 动态调整轮数（模拟多个 step 的在线调整）
BIAS_LR = 0.02           # bias 更新步长（相对单位）
SEED = 42


def route_tokens(logits: np.ndarray, top_k: int) -> np.ndarray:
    """对 [N, E] 的 router logits 取 top-k，返回 [N, k] 专家编号。"""
    return np.argpartition(logits, -top_k, axis=1)[:, -top_k:]


def expert_loads(routing: np.ndarray, e: int) -> np.ndarray:
    """统计每个专家收到的 token 数。"""
    return np.bincount(routing.reshape(-1), minlength=e).astype(np.float64)


def balance_bias(logits: np.ndarray, top_k: int, rounds: int, lr: float) -> np.ndarray:
    """
    aux-loss-free 风格 bias 均衡：多轮观察负载，给冷门专家加 bias、热门专家减 bias。

    每轮用固定 logits + 当前 bias 路由，统计负载偏差，然后按偏差更新 bias。
    真实系统中 bias 在推理过程中随每个 step 的统计在线更新，此处用多轮迭代模拟。
    """
    bias = np.zeros(logits.shape[1], dtype=np.float64)
    e = logits.shape[1]
    for _ in range(rounds):
        routing = route_tokens(logits + bias, top_k)
        loads = expert_loads(routing, e)
        mean = loads.mean()
        # 负载高于均值的专家 bias 下调，低于均值的上调
        bias += lr * (mean - loads) / max(mean, 1.0)
    return bias


def gpu_loads(loads: np.ndarray, p: int) -> np.ndarray:
    """专家均分到 p 张卡（连续切块），返回每张卡的总 token 负载。"""
    per_gpu = np.array_split(loads, p)
    return np.array([g.sum() for g in per_gpu])


def print_load_stats(name: str, loads: np.ndarray, p: int):
    """打印专家级与卡级负载统计。"""
    mean = loads.mean()
    idle = int((loads == 0).sum())
    g = gpu_loads(loads, p)
    g_mean = g.mean()
    print(f"\n[{name}]  EP size = {p}")
    print("-" * 72)
    print(f"  专家级: mean={mean:8.1f}  max={loads.max():8.0f}  "
          f"max/mean={loads.max() / max(mean, 1e-9):6.2f}  空闲专家数={idle:4d} / {len(loads)}")
    print(f"  分位数: p50={np.percentile(loads, 50):6.0f}  p90={np.percentile(loads, 90):6.0f}  "
          f"p99={np.percentile(loads, 99):6.0f}")
    print(f"  卡级  : mean={g_mean:8.1f}  max={g.max():8.0f}  "
          f"max/mean={g.max() / max(g_mean, 1e-9):6.2f}  "
          f"(all-to-all 同步点由最忙的卡决定，卡级 max/mean 直接反映 GPU 利用率损失)")


def alltoall_volume_gb(n_tokens: int, top_k: int, hidden: int, p: int) -> float:
    """
    估算一次 MoE 层 dispatch + combine 的 all-to-all 通信量（GB）。

    单方向：每个 token 的 top_k 份 hidden state 中，约 (P-1)/P 发往远端。
    dispatch 与 combine 各一次，共乘 2。
    """
    single_dir = n_tokens * top_k * hidden * DTYPE_BYTES * (p - 1) / p
    return 2 * single_dir / (1024 ** 3)


def print_comm_table(n_tokens: int, top_k: int, hidden: int):
    """打印 all-to-all 通信量随 EP size 的变化。"""
    nvlink_gbs = 450.0   # 机内 NVLink 有效带宽量级（GB/s，教学估算值）
    rdma_gbs = 50.0      # 跨节点 RDMA 单卡有效带宽量级（GB/s，教学估算值）
    print("\n" + "=" * 72)
    print("All-to-all 通信量随 EP size 的变化（单次 MoE 层，dispatch + combine）")
    print("=" * 72)
    print(f"{'EP size':>8} {'本地命中比例':>12} {'通信量(GB)':>12} "
          f"{'NVLink 时间(us)':>15} {'RDMA 时间(us)':>14}")
    print("-" * 72)
    for p in [1, 4, 8, 16, 32, 64, 128, 256]:
        vol = alltoall_volume_gb(n_tokens, top_k, hidden, p)
        local_ratio = 1.0 / p
        t_nv = vol / nvlink_gbs * 1e6 if p > 1 else 0.0
        t_rdma = vol / rdma_gbs * 1e6 if p > 8 else 0.0
        rdma_str = f"{t_rdma:14.1f}" if p > 8 else f"{'(机内)':>14}"
        print(f"{p:>8} {local_ratio:>12.3f} {vol:>12.3f} {t_nv:>15.1f} {rdma_str}")
    print("-" * 72)
    print("Note: EP 越大，本地命中比例越低，通信量趋近全部 token 都跨卡；")
    print("      EP 超过机内 8 卡后需跨节点 RDMA，带宽骤降一个量级，")
    print("      这就是 DeepEP 需要 low-latency 模式（通信与计算 SM 分离重叠）的原因。")


def print_histogram(loads: np.ndarray, bins: int = 10):
    """打印专家负载的文本直方图。"""
    hist, edges = np.histogram(loads, bins=bins)
    print("\n专家负载直方图（均衡路由，每个 # 约代表 "
          f"{max(1, int(hist.max() / 50))} 个专家）")
    print("-" * 72)
    for i in range(bins):
        bar = "#" * int(50 * hist[i] / max(hist.max(), 1))
        print(f"  [{edges[i]:7.1f}, {edges[i + 1]:7.1f}) {hist[i]:4d} {bar}")


def run_simulation():
    """运行随机路由 vs bias 均衡路由的对比实验。"""
    rng = np.random.default_rng(SEED)

    print("=" * 72)
    print("MoE 专家并行（EP）负载分布模拟器")
    print(f"配置: tokens={N_TOKENS}  experts={E}  top_k={TOP_K}  hidden={HIDDEN}")
    print(f"总路由次数 = {N_TOKENS * TOP_K}，平均每专家 {N_TOKENS * TOP_K / E:.1f} 个 token")
    print("=" * 72)

    # router logits：真实 router 存在偏好，给 logits 加固定偏置模拟热门/冷门专家
    popularity = rng.normal(0.0, 1.0, size=E)
    logits = rng.normal(0.0, 1.0, size=(N_TOKENS, E)) + popularity

    # 1) 随机路由（无干预）
    routing_rand = route_tokens(logits, TOP_K)
    loads_rand = expert_loads(routing_rand, E)

    # 2) bias 均衡路由
    bias = balance_bias(logits, TOP_K, BIAS_ROUNDS, BIAS_LR)
    routing_bal = route_tokens(logits + bias, TOP_K)
    loads_bal = expert_loads(routing_bal, E)

    ep = 32
    print_load_stats("随机路由（无干预）", loads_rand, ep)
    print_load_stats("bias 均衡路由", loads_bal, ep)
    print_histogram(loads_bal)

    # 小 batch 对照：Decode batch 变小时负载不均急剧恶化
    print("\n" + "=" * 72)
    print("Decode batch 大小对负载不均的影响（随机路由，EP=32）")
    print("=" * 72)
    print(f"{'tokens':>8} {'平均/专家':>10} {'max/mean':>10} {'空闲专家':>10} {'卡级 max/mean':>14}")
    print("-" * 72)
    for n in [256, 512, 1024, 4096, 16384]:
        sub_logits = rng.normal(0.0, 1.0, size=(n, E)) + popularity
        sub_loads = expert_loads(route_tokens(sub_logits, TOP_K), E)
        g = gpu_loads(sub_loads, ep)
        print(f"{n:>8} {n * TOP_K / E:>10.1f} "
              f"{sub_loads.max() / max(sub_loads.mean(), 1e-9):>10.2f} "
              f"{int((sub_loads == 0).sum()):>10d} "
              f"{g.max() / max(g.mean(), 1e-9):>14.2f}")
    print("-" * 72)
    print("Note: batch 越小，每专家平均 token 越少，相对波动越大，空闲专家越多；")
    print("      这就是 MoE Decode 需要大 batch + 大 EP 的根本原因。")

    print_comm_table(N_TOKENS, TOP_K, HIDDEN)


if __name__ == "__main__":
    run_simulation()
