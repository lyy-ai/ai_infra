#!/usr/bin/env python3
"""
KV Cache 三级存储（HBM / CPU DDR / NVMe SSD）offload 模拟器。

模拟场景：
- 多租户长上下文服务：若干"系统前缀组"（共享 prefix）+ 每请求独有后缀，
  KV 以 block（16 token）为单位管理（与 PagedAttention 对齐）。
- 三级存储：HBM（有限）、DDR（有限）、SSD（近似无限），层级间 LRU 降级/提升。
- 每次访问一个 KV block：
    * 命中 HBM：只付 HBM 读取时间（极快）；
    * 命中 DDR：付 PCIe 传输时间并提升到 HBM；
    * 命中 SSD：付 SSD 读取时间并提升到 HBM；
    * 未命中  ：重新 prefill 计算该 block（最贵），写入 HBM。
- 统计各级命中率，以及 prefill/预取延迟对 TTFT 的影响。

只依赖 stdlib，可直接运行。

运行方式：
    python kv_offload_model.py
"""
import math
import random
from collections import OrderedDict


# ---------------- 模型与硬件配置（以 DeepSeek-V3 + MLA 为参照） ----------------
KV_BYTES_PER_TOKEN = 70272      # MLA：61 层 × 576 维 × 2B
BLOCK_TOKENS = 16               # KV block 大小（token）
BLOCK_BYTES = KV_BYTES_PER_TOKEN * BLOCK_TOKENS  # 约 1.07 MiB

HBM_BW = 3e12                   # HBM 约 3 TB/s
PCIE_BW = 64e9                  # PCIe 约 64 GB/s（双向量级）
SSD_BW = 7e9                    # NVMe SSD 约 7 GB/s

# 重新 prefill 一个 block 的计算时间（教学估算：大模型单请求 prefill 约 20k tok/s）
PREFILL_TOKENS_PER_SEC = 20000.0
PREFILL_BLOCK_SEC = BLOCK_TOKENS / PREFILL_TOKENS_PER_SEC  # 约 0.8 ms

# 存储容量（block 数），SSD 近似无限
# DDR 故意设为较小值（约 8.8 GiB），使 HBM 不足时前缀 block 会被挤到 SSD，展示三级行为
DDR_CAPACITY_BLOCKS = 8 * 1024
SSD_CAPACITY_BLOCKS = 10 ** 9

# 工作负载
N_REQUESTS = 300
N_PREFIX_GROUPS = 12            # 共享系统前缀的组数（模拟多租户共用 system prompt / RAG 文档）
PREFIX_LEN_TOKENS = 32 * 1024   # 每组共享前缀长度
UNIQ_MIN, UNIQ_MAX = 1024, 16 * 1024  # 每请求独有部分长度范围
SEED = 7


class TieredKVStore:
    """三级 KV 存储：每层一个 LRU（OrderedDict），访问未命中高层级时向下查找并提升。"""

    def __init__(self, hbm_blocks: int, ddr_blocks: int):
        self.hbm_cap = hbm_blocks
        self.ddr_cap = ddr_blocks
        self.hbm: OrderedDict = OrderedDict()
        self.ddr: OrderedDict = OrderedDict()
        self.ssd: OrderedDict = OrderedDict()
        self.hits = {"hbm": 0, "ddr": 0, "ssd": 0, "miss": 0}

    def _insert_hbm(self, key):
        self.hbm[key] = True
        if len(self.hbm) > self.hbm_cap:
            evicted, _ = self.hbm.popitem(last=False)  # LRU 逐出，降级到 DDR
            self._insert_ddr(evicted)

    def _insert_ddr(self, key):
        self.ddr[key] = True
        if len(self.ddr) > self.ddr_cap:
            evicted, _ = self.ddr.popitem(last=False)  # 继续降级到 SSD
            self.ssd[evicted] = True
            if len(self.ssd) > SSD_CAPACITY_BLOCKS:
                self.ssd.popitem(last=False)

    def access(self, key) -> float:
        """访问一个 KV block，返回该次访问的耗时（秒）。"""
        if key in self.hbm:
            self.hbm.move_to_end(key)
            self.hits["hbm"] += 1
            return BLOCK_BYTES / HBM_BW
        if key in self.ddr:
            del self.ddr[key]
            self._insert_hbm(key)
            self.hits["ddr"] += 1
            return BLOCK_BYTES / PCIE_BW          # 经 PCIe 取回
        if key in self.ssd:
            del self.ssd[key]
            self._insert_hbm(key)
            self.hits["ssd"] += 1
            return BLOCK_BYTES / SSD_BW           # 经 SSD 读回
        self.hits["miss"] += 1
        self._insert_hbm(key)
        return PREFILL_BLOCK_SEC                  # 未命中：重新计算 prefill


def generate_workload() -> list:
    """生成请求流：每条请求 = 某个共享前缀组 + 独有后缀，返回 block key 序列列表。"""
    rng = random.Random(SEED)
    requests = []
    for rid in range(N_REQUESTS):
        group = rng.randrange(N_PREFIX_GROUPS)
        uniq_tokens = rng.randint(UNIQ_MIN, UNIQ_MAX)
        keys = []
        n_prefix_blocks = PREFIX_LEN_TOKENS // BLOCK_TOKENS
        for b in range(n_prefix_blocks):
            keys.append(("prefix", group, b))      # 共享前缀 block：跨请求可复用
        n_uniq_blocks = math.ceil(uniq_tokens / BLOCK_TOKENS)
        for b in range(n_uniq_blocks):
            keys.append(("uniq", rid, b))          # 独有 block：不可复用
        requests.append(keys)
    return requests


def simulate(hbm_capacity_gib: float) -> dict:
    """用给定 HBM 容量跑一遍工作负载，返回命中率与 TTFT 统计。"""
    hbm_blocks = int(hbm_capacity_gib * (1024 ** 3) / BLOCK_BYTES)
    store = TieredKVStore(hbm_blocks, DDR_CAPACITY_BLOCKS)
    requests = generate_workload()

    ttfts = []
    for keys in requests:
        ttft = sum(store.access(k) for k in keys)
        ttfts.append(ttft)

    total = sum(store.hits.values())
    ttfts_sorted = sorted(ttfts)
    p95 = ttfts_sorted[int(0.95 * len(ttfts_sorted)) - 1]
    return {
        "hbm_gib": hbm_capacity_gib,
        "hit_rates": {k: v / total for k, v in store.hits.items()},
        "avg_ttft": sum(ttfts) / len(ttfts),
        "p95_ttft": p95,
        "total": total,
    }


def print_result(res: dict):
    """打印单个实验的结果。"""
    h = res["hit_rates"]
    print(f"{res['hbm_gib']:>12.0f} "
          f"{h['hbm'] * 100:>11.1f}% "
          f"{h['ddr'] * 100:>11.1f}% "
          f"{h['ssd'] * 100:>11.1f}% "
          f"{h['miss'] * 100:>11.1f}% "
          f"{res['avg_ttft'] * 1000:>13.1f} "
          f"{res['p95_ttft'] * 1000:>13.1f}")


def run_simulation():
    """运行不同 HBM 预算下的对比实验。"""
    print("=" * 88)
    print("KV Cache 三级存储 offload 模拟器（MLA：每 token KV "
          f"{KV_BYTES_PER_TOKEN} B，block = {BLOCK_TOKENS} token ≈ "
          f"{BLOCK_BYTES / 1024:.0f} KiB）")
    print(f"工作负载: {N_REQUESTS} 请求 × ({N_PREFIX_GROUPS} 组共享 32k 前缀 + 独有后缀)")
    print(f"访问耗时量级: HBM {BLOCK_BYTES / HBM_BW * 1e6:.1f} us/block | "
          f"DDR(PCIe) {BLOCK_BYTES / PCIE_BW * 1e6:.1f} us/block | "
          f"SSD {BLOCK_BYTES / SSD_BW * 1e6:.1f} us/block | "
          f"重算 prefill {PREFILL_BLOCK_SEC * 1e3:.2f} ms/block")
    print("=" * 88)

    print(f"\n{'HBM(GiB)':>12} {'HBM命中率':>12} {'DDR命中率':>12} "
          f"{'SSD命中率':>12} {'未命中重算':>12} {'avg TTFT(ms)':>14} {'p95 TTFT(ms)':>14}")
    print("-" * 88)
    for hbm_gib in [1, 4, 16, 64, 256]:
        print_result(simulate(float(hbm_gib)))
    print("-" * 88)
    print("Note: 共享前缀在 HBM 放得下时命中率极高、TTFT 几乎只剩独有部分的计算；")
    print("      HBM 不足时前缀 block 被逐出到 DDR/SSD，命中低层级仍远好于重新 prefill；")
    print("      SSD 命中比 DDR 慢约一个数量级，这就是'让绝大多数访问命中高层级'的原因。")

    # 单请求视角：一条 128k 请求在不同命中层级下的 TTFT 构成
    print("\n单条 128k 请求的 TTFT 构成（8192 个 block 全部来自同一层级时的极限情况）")
    print("-" * 88)
    n_blocks = 128 * 1024 // BLOCK_TOKENS
    rows = [
        ("全部命中 HBM", n_blocks * BLOCK_BYTES / HBM_BW),
        ("全部命中 DDR", n_blocks * BLOCK_BYTES / PCIE_BW),
        ("全部命中 SSD", n_blocks * BLOCK_BYTES / SSD_BW),
        ("全部重算 prefill", n_blocks * PREFILL_BLOCK_SEC),
    ]
    for name, sec in rows:
        print(f"  {name:<14}: {sec:>10.2f} s")
    print("-" * 88)
    print("Note: 真实系统中 DDR/SSD 命中可与 prefill 计算流水线重叠，")
    print("      表中为串行极限值，用于体现层级间带宽差对 TTFT 的量级影响。")


if __name__ == "__main__":
    run_simulation()
