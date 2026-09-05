# mig_partition_planner.py
#
# A100 80GB MIG 装箱规划教学模拟器：
#   给定一组推理任务（显存 / 算力需求档位），分别用三种方案做承载规划：
#     A. 整卡直通   ：一个任务独占一张卡
#     B. 时间片共享 ：按显存装箱，多任务共用一张卡（计算时分复用，有争抢）
#     C. MIG        ：按硬件 profile（1g.10gb/2g.20gb/3g.40gb/4g.40gb/7g.80gb）
#                     装箱，硬件级隔离，但受 7-slice 几何约束
#   对比：所需卡数、承载任务数、SLA 风险任务数、显存利用率。
#
# 运行：
#   cd /data/liyangyang/ai_infra/09_企业级工程体系/9.6_GPU集群调度与平台
#   /data/liyangyang/qwen35_env/bin/python mig_partition_planner.py

from dataclasses import dataclass

GPU_MEM = 80.0       # A100 80GB
GPU_SLICES = 7       # MIG 共 7 个 slice

# MIG profile：名字 -> (slice 数, 显存 GB, 算力占比)
PROFILES = {
    "1g.10gb": (1, 10, 1 / 7),
    "2g.20gb": (2, 20, 2 / 7),
    "3g.40gb": (3, 40, 3 / 7),
    "4g.40gb": (4, 40, 4 / 7),
    "7g.80gb": (7, 80, 7 / 7),
}


@dataclass
class Task:
    name: str
    mem: float        # 显存需求 GB
    compute: float    # 算力需求（占整卡比例，0~1）
    tier: str


def gen_workload():
    """典型推理平台任务画像：大量小模型 + 若干中模型 + 少量大模型。"""
    tasks = []
    for i in range(20):
        tasks.append(Task(f"small-{i:02d}", mem=6, compute=0.12, tier="小模型(≤10GB)"))
    for i in range(12):
        tasks.append(Task(f"medium-{i:02d}", mem=16, compute=0.25, tier="中模型(≤20GB)"))
    for i in range(8):
        tasks.append(Task(f"large-{i:02d}", mem=30, compute=0.40, tier="大模型(≤40GB)"))
    for i in range(3):
        tasks.append(Task(f"xl-{i:02d}", mem=70, compute=0.90, tier="超大模型(>40GB)"))
    return tasks


# ---------------------------------------------------------------------------
# 方案 A：整卡直通
# ---------------------------------------------------------------------------

def plan_passthrough(tasks):
    """一个任务一张卡。隔离性最好，浪费最大。"""
    gpus = [[t] for t in tasks]     # 每任务一卡
    return gpus, {"sla_risk": 0, "note": "一任务一卡，无争抢；卡数 = 任务数"}


# ---------------------------------------------------------------------------
# 方案 B：时间片共享（按显存装箱，计算争抢产生 SLA 风险）
# ---------------------------------------------------------------------------

def plan_timeslice(tasks):
    """first-fit decreasing 按显存装箱（sum(mem) <= 80GB）。

    SLA 风险模型：同一卡上所有任务算力需求之和 > 1.0 时，时分复用导致
    每个任务拿不到承诺算力，该卡上全部任务标记为 SLA 风险。
    """
    gpus = []                        # 元素：[task, ...]
    for t in sorted(tasks, key=lambda x: -x.mem):
        placed = False
        for g in gpus:
            if sum(x.mem for x in g) + t.mem <= GPU_MEM:
                g.append(t)
                placed = True
                break
        if not placed:
            gpus.append([t])
    sla_risk = sum(len(g) for g in gpus if sum(x.compute for x in g) > 1.0)
    n_over = sum(1 for g in gpus if sum(x.compute for x in g) > 1.0)
    return gpus, {"sla_risk": sla_risk,
                  "note": f"{n_over} 张卡算力超卖（sum(compute)>1），其上任务有争抢风险"}


# ---------------------------------------------------------------------------
# 方案 C：MIG（profile 装箱，硬件隔离）
# ---------------------------------------------------------------------------

def pick_profile(task):
    """选满足显存与算力需求的最小 profile。"""
    for name, (slices, mem, comp) in PROFILES.items():
        if mem >= task.mem and comp >= task.compute:
            return name, slices
    return None, None


def plan_mig(tasks):
    """MIG 装箱：每任务映射到最小可用 profile，再按 slice 装箱（容量 7）。

    注：真实 MIG 还有 placement 约束（如两个 3g 不能随意拼接），此处用
    slice 容量做近似，教学上足够说明"粒度固定带来的卡内碎片"。
    """
    instances = []                   # (profile_name, slices, task)
    for t in tasks:
        name, slices = pick_profile(t)
        if name is None:
            raise ValueError(f"{t.name} 超出单卡能力")
        instances.append((name, slices, t))
    # first-fit decreasing 按 slice 数装箱
    gpus = []                        # 元素：[(name, slices, task), ...]
    for inst in sorted(instances, key=lambda x: -x[1]):
        placed = False
        for g in gpus:
            if sum(x[1] for x in g) + inst[1] <= GPU_SLICES:
                g.append(inst)
                placed = True
                break
        if not placed:
            gpus.append([inst])
    # MIG 硬件隔离：无 SLA 争抢风险；统计卡内浪费的 slice
    wasted = sum(GPU_SLICES - sum(x[1] for x in g) for g in gpus)
    return gpus, {"sla_risk": 0,
                  "note": f"硬件隔离无争抢；几何约束共浪费 {wasted} 个 slice（卡内碎片）"}


def mem_util(gpus, per_gpu_mem_used):
    used = sum(per_gpu_mem_used(g) for g in gpus)
    return used / (len(gpus) * GPU_MEM)


def main():
    tasks = gen_workload()
    print("=" * 78)
    print("A100 80GB 推理任务承载规划：整卡直通 vs 时间片共享 vs MIG")
    print("=" * 78)
    tiers = {}
    for t in tasks:
        tiers.setdefault(t.tier, 0)
        tiers[t.tier] += 1
    print("任务画像：" + "，".join(f"{k} x{v}" for k, v in tiers.items()))
    print(f"共 {len(tasks)} 个任务\n")

    plans = [
        ("A. 整卡直通", *plan_passthrough(tasks),
         lambda g: sum(t.mem for t in g)),
        ("B. 时间片共享", *plan_timeslice(tasks),
         lambda g: sum(t.mem for t in g)),
        ("C. MIG 切分", *plan_mig(tasks),
         lambda g: sum(PROFILES[x[0]][1] for x in g)),
    ]

    print(f"  {'方案':<16}{'所需卡数':>10}{'承载任务数':>12}{'SLA风险任务':>12}"
          f"{'显存利用率':>12}   说明")
    print("  " + "-" * 74)
    for name, gpus, meta, mem_fn in plans:
        carried = sum(len(g) for g in gpus)
        util = mem_util(gpus, mem_fn)
        print(f"  {name:<16}{len(gpus):>10}{carried:>12}{meta['sla_risk']:>12}"
              f"{util:>12.1%}   {meta['note']}")

    # 展示 MIG 前几张卡的装箱细节
    _, mig_gpus, _, _ = plans[2]
    print()
    print("MIG 装箱明细（前 6 张卡）：")
    for i, g in enumerate(mig_gpus[:6]):
        desc = " + ".join(x[0] for x in g)
        used = sum(x[1] for x in g)
        print(f"  GPU-{i}: {desc:<40} 用 {used}/7 slice")

    total_compute = sum(t.compute for t in tasks)
    print()
    print("=" * 78)
    print("结论")
    print("=" * 78)
    print(f"1. 整卡直通最安全也最浪费：43 个任务要 43 张卡，显存利用率仅 22%。")
    print(f"2. 时间片共享'10 张卡装下一切'是假象：任务总算力需求约 {total_compute:.1f} 卡，")
    print(f"   塞进 10 张卡必然超卖（sum(compute)>1），高峰期延迟抖动直接打到 SLA。")
    print(f"   时间片只隔离显存，不隔离算力——这是它便宜的根本原因。")
    print(f"3. MIG 的 13 张卡才是诚实容量：按硬件 profile 分配算力（1 slice ≈ 1/7 卡），")
    print(f"   不超卖、故障隔离，SLA 可承诺；代价是粒度固定——7-slice 几何约束")
    print(f"   会产生卡内碎片（本例浪费 2 个 slice），且超大模型仍要独占整卡（7g）。")
    print(f"4. 生产实践：在线推理多租户用 MIG 拿隔离性，开发/离线任务用时间片拿密度，")
    print(f"   关键大模型（XL 档）整卡直通——三种方案通常同时存在于一个集群。")


if __name__ == "__main__":
    main()
