# gpu_scheduling_sim.py
#
# GPU 集群调度教学模拟器：
#   在一个 64 卡集群（8 台 x 8 卡 NVLink 域）上，模拟一批不同卡数需求
#   （1/2/4/8 卡）、不同优先级（在线推理=高优 / 训练=低优）任务的到达与调度，
#   对比两种策略：
#     A. 朴素 FIFO        ：严格按到达顺序、允许跨机拼凑、队头阻塞
#     B. 拓扑感知 gang     ：任务必须整机内放置（gang + NVLink 域亲和）、
#                           best-fit bin-packing、允许回填（backfill）、
#                           高优先级优先
#   指标：平均/P95 等待时间、高优任务等待、集群碎片率（有卡但凑不齐
#   8 卡域的机器时间占比）、总体利用率。
#
# 运行：
#   cd /data/liyangyang/ai_infra/09_企业级工程体系/9.6_GPU集群调度与平台
#   /data/liyangyang/qwen35_env/bin/python gpu_scheduling_sim.py

import numpy as np
from dataclasses import dataclass, field

N_NODES = 8
GPUS_PER_NODE = 8
TOTAL_GPUS = N_NODES * GPUS_PER_NODE


@dataclass
class Job:
    jid: int
    arrival: float
    req: int          # 需求卡数：1/2/4/8
    duration: float
    prio: int         # 1=高优（在线推理），0=低优（训练）
    kind: str
    start: float = -1.0
    end: float = -1.0


def gen_workload(n_jobs=160, seed=7, horizon=100.0):
    """生成任务流：8 卡整机训练为主负载，1/2 卡推理任务穿插其中。"""
    rng = np.random.default_rng(seed)
    jobs = []
    t = 0.0
    for i in range(n_jobs):
        t += rng.exponential(horizon / n_jobs)
        r = rng.random()
        if r < 0.35:
            req, kind, prio = 1, "推理", 1
            dur = rng.exponential(3.0) + 0.5
        elif r < 0.55:
            req, kind, prio = 2, "推理", 1
            dur = rng.exponential(4.0) + 0.5
        elif r < 0.75:
            req, kind, prio = 4, "训练", 0
            dur = rng.exponential(8.0) + 1.0
        else:
            req, kind, prio = 8, "训练", 0
            dur = rng.exponential(12.0) + 2.0
        jobs.append(Job(i, t, req, dur, prio, kind))
    return jobs


class Cluster:
    """节点级资源视图：nodes[i] = 第 i 台机器的空闲卡数。"""

    def __init__(self):
        self.nodes = [GPUS_PER_NODE] * N_NODES
        # 时间加权统计
        self._last_t = 0.0
        self._frag_time = 0.0      # 处于"有卡但凑不齐 8 卡域"状态的机时
        self._busy_gpu_time = 0.0  # 忙碌卡时

    def _accumulate(self, t):
        dt = t - self._last_t
        if dt > 0:
            partial = sum(1 for f in self.nodes if 0 < f < GPUS_PER_NODE)
            self._frag_time += partial * dt
            self._busy_gpu_time += sum(GPUS_PER_NODE - f for f in self.nodes) * dt
            self._last_t = t

    def frag_ratio(self):
        total = N_NODES * self._last_t
        return self._frag_time / total if total > 0 else 0.0

    def utilization(self):
        total = TOTAL_GPUS * self._last_t
        return self._busy_gpu_time / total if total > 0 else 0.0


def simulate(jobs, policy):
    """离散事件模拟：policy ∈ {"fifo", "topo"}，返回任务列表与集群统计。"""
    cluster = Cluster()
    queue, running = [], []
    pending = sorted(jobs, key=lambda j: j.arrival)

    def alloc_scatter(req):
        alloc = {}
        left = req
        for i in range(N_NODES):
            take = min(cluster.nodes[i], left)
            if take:
                cluster.nodes[i] -= take
                alloc[i] = take
                left -= take
            if left == 0:
                break
        return alloc

    def alloc_topo(req):
        best, best_after = -1, GPUS_PER_NODE + 1
        for i in range(N_NODES):
            if cluster.nodes[i] >= req and cluster.nodes[i] - req < best_after:
                best, best_after = i, cluster.nodes[i] - req
        if best < 0:
            return None
        cluster.nodes[best] -= req
        return {best: req}

    while pending or queue or running:
        next_arrival = pending[0].arrival if pending else float("inf")
        next_finish = min((r[0] for r in running), default=float("inf"))
        t = min(next_arrival, next_finish)
        cluster._accumulate(t)

        # 完成事件：还卡
        still = []
        for end, job, alloc in running:
            if end <= t + 1e-9:
                for node, cnt in alloc.items():
                    cluster.nodes[node] += cnt
            else:
                still.append((end, job, alloc))
        running = still

        # 到达事件
        while pending and pending[0].arrival <= t + 1e-9:
            queue.append(pending.pop(0))

        # 调度
        if policy == "fifo":
            # 队头阻塞：头任务放不下则全员等待（无 backfill）
            while queue and sum(cluster.nodes) >= queue[0].req:
                job = queue.pop(0)
                alloc = alloc_scatter(job.req)
                job.start = t
                job.end = t + job.duration
                running.append((job.end, job, alloc))
        else:
            # 拓扑感知 gang + backfill：按（优先级, 到达）扫描，能整机放下就调度
            queue.sort(key=lambda j: (-j.prio, j.arrival))
            rest = []
            for job in queue:
                alloc = alloc_topo(job.req)
                if alloc is not None:
                    job.start = t
                    job.end = t + job.duration
                    running.append((job.end, job, alloc))
                else:
                    rest.append(job)   # 留在队列，继续扫描后面的任务（回填）
            queue = rest

    return jobs, cluster


def report(name, jobs, cluster):
    waits = np.array([j.start - j.arrival for j in jobs])
    hi = waits[[j.prio == 1 for j in jobs]]
    lo = waits[[j.prio == 0 for j in jobs]]
    print(f"  {name:<30}{waits.mean():>10.2f}{np.percentile(waits, 95):>10.2f}"
          f"{hi.mean():>10.2f}{lo.mean():>10.2f}{cluster.frag_ratio():>10.1%}"
          f"{cluster.utilization():>10.1%}")


def main():
    jobs_template = gen_workload()
    demand = sum(j.req * j.duration for j in jobs_template)
    horizon = max(j.arrival for j in jobs_template)
    print("=" * 78)
    print("GPU 集群调度模拟：朴素 FIFO vs 拓扑感知 bin-packing + gang scheduling")
    print("=" * 78)
    print(f"集群规模：{N_NODES} 台 x {GPUS_PER_NODE} 卡 = {TOTAL_GPUS} 卡（整机 = 一个 NVLink 域）")
    print(f"任务流：{len(jobs_template)} 个任务，到达窗口 {horizon:.0f}s")
    print(f"卡数需求分布：" + ", ".join(
        f"{r}卡x{sum(1 for j in jobs_template if j.req == r)}" for r in (1, 2, 4, 8)))
    print(f"供给负载（需求卡时 / 窗口容量）：{demand / (TOTAL_GPUS * horizon):.1%}")
    print()
    print(f"  {'调度策略':<30}{'平均等待s':>10}{'P95等待s':>10}{'高优等待s':>10}"
          f"{'低优等待s':>10}{'碎片率':>10}{'利用率':>10}")
    print("  " + "-" * 74)

    import copy
    for policy, name in [("fifo", "A. 朴素 FIFO（跨机拼凑+队头阻塞）"),
                         ("topo", "B. 拓扑感知 gang + bin-packing + 回填")]:
        js = copy.deepcopy(jobs_template)
        js, cluster = simulate(js, policy)
        report(name, js, cluster)

    print()
    print("=" * 78)
    print("结论")
    print("=" * 78)
    print("1. 朴素 FIFO 的队头阻塞：一个 8 卡任务凑不齐卡时，后面所有任务陪等，")
    print("   平均等待与 P95 等待被显著拉高。")
    print("2. FIFO 跨机拼凑看似不浪费卡，但 8 卡任务被拆成 4+4 跨机后 NVLink")
    print("   域内带宽优势全失（训练吞吐腰斩）——这笔损失不计入上表，是隐性成本。")
    print("3. 拓扑感知策略把任务限制在整机内（gang），用 best-fit bin-packing 把")
    print("   小任务集中塞到少数机器、把完整机器留给 8 卡任务，碎片率显著下降；")
    print("   backfill 让小任务绕过暂时放不下的大任务填缝，高优任务等待大幅降低。")
    print("4. 真实系统（Volcano/Kueue + 拓扑感知插件）正是这套思路的工程化。")


if __name__ == "__main__":
    main()
