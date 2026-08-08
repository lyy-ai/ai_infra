#!/usr/bin/env python3
"""生成 Runtime 推理引擎专题演示图。运行：python tools/generate_runtime_diagrams.py"""
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
else:
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BG = "#0b1220"
PANEL = "#111c33"
TEXT = "#e5eefc"
MUTED = "#9fb3d1"
CYAN = "#22d3ee"
GREEN = "#34d399"
ORANGE = "#fb923c"
PURPLE = "#a78bfa"
RED = "#f87171"
YELLOW = "#facc15"
BLUE = "#60a5fa"


def new_fig(title, subtitle=""):
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.6, 8.35, title, color=TEXT, fontsize=26, fontweight="bold", va="top")
    if subtitle:
        ax.text(0.62, 7.82, subtitle, color=MUTED, fontsize=12.5, va="top")
    return fig, ax


def save(fig, rel_path):
    path = os.path.join(ROOT, rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("saved", rel_path)


def box(ax, x, y, w, h, title, body="", fc=PANEL, ec=CYAN, title_color=TEXT, body_color=MUTED, lw=1.8):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                           linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", color=title_color, fontsize=15, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center", color=body_color, fontsize=10.5)
    return patch


def arrow(ax, x1, y1, x2, y2, color=CYAN, lw=2.2, style="-|>", ms=18, alpha=1.0):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                          linewidth=lw, color=color, alpha=alpha)
    ax.add_patch(arr)
    return arr


def label(ax, x, y, text, color=MUTED, size=11, ha="center", va="center", weight="normal"):
    ax.text(x, y, text, color=color, fontsize=size, ha=ha, va=va, fontweight=weight)


def cover():
    fig, ax = new_fig("Runtime 推理引擎", "从架构设计到性能优化：Session / Graph / Memory / CUDA Graph / Stream / Scheduler / Stability")
    items = [
        ("5.1", "Runtime 架构", CYAN), ("5.2", "内存池", GREEN), ("5.3", "CUDA Graph", ORANGE),
        ("5.4", "多流并发", PURPLE), ("5.5", "动态 Batch", YELLOW), ("5.6", "Relax/稳定性", RED),
    ]
    x = 1.0
    for num, name, color in items:
        box(ax, x, 4.4, 2.25, 1.35, num, name, fc="#0f1b31", ec=color)
        x += 2.45
    arrow(ax, 2.2, 3.7, 13.8, 3.7, color=MUTED, lw=1.4, style="-", ms=1, alpha=0.6)
    label(ax, 8, 3.35, "契约 → 资源 → 性能 → 调度 → 多租户 → 稳定性", color=TEXT, size=15, weight="bold")
    save(fig, "images/runtime_cover.png")


def d_5_1_arch():
    fig, ax = new_fig("5.1 Runtime 架构设计", "共享权重、隔离状态、复用内存、按依赖调度")
    box(ax, 0.8, 5.6, 3.0, 1.35, "API / Serving", "请求、鉴权、路由", ec=BLUE)
    box(ax, 4.4, 5.6, 3.0, 1.35, "Scheduler", "动态 batch / QoS / aging", ec=YELLOW)
    box(ax, 8.0, 5.6, 3.0, 1.35, "Session Manager", "Create / Bind / Cancel / Release", ec=CYAN)
    box(ax, 11.6, 5.6, 3.6, 1.35, "Graph Executor", "ready queue / launch / completion", ec=ORANGE)
    box(ax, 3.0, 2.7, 3.2, 1.35, "Tensor Manager", "first_write / last_read / reuse", ec=GREEN)
    box(ax, 7.0, 2.7, 3.2, 1.35, "Memory Pools", "weight / KV / activation / workspace", ec=GREEN)
    box(ax, 11.0, 2.7, 3.2, 1.35, "Device Backend", "CUDA kernel / stream / event / graph", ec=PURPLE)
    arrow(ax, 3.8, 6.25, 4.4, 6.25)
    arrow(ax, 7.4, 6.25, 8.0, 6.25)
    arrow(ax, 11.0, 6.25, 11.6, 6.25)
    arrow(ax, 9.5, 5.6, 4.6, 4.05, color=GREEN)
    arrow(ax, 12.8, 5.6, 8.6, 4.05, color=GREEN)
    arrow(ax, 13.0, 5.6, 12.6, 4.05, color=PURPLE)
    label(ax, 8, 1.55, "关键路径 ≠ 总计算量；先看依赖与生命周期，再谈 kernel 优化", color=TEXT, size=14, weight="bold")
    save(fig, "4.1_Runtime架构设计/images/5_1_runtime_architecture.png")


def d_5_1_lifecycle():
    fig, ax = new_fig("5.1 Tensor 生命周期", "first_write → last_read：过了 last_read 才能复用")
    tensors = [("x", 0, 1, CYAN), ("q/k/v", 1, 2, GREEN), ("attn_out", 2, 3, ORANGE), ("proj_out", 3, 4, PURPLE), ("y", 4, 5, YELLOW)]
    for i, (name, start, end, color) in enumerate(tensors):
        y = 6.6 - i * 0.85
        ax.add_patch(Rectangle((start + 2.0, y), end - start + 0.85, 0.42, facecolor=color, alpha=0.85, edgecolor="none"))
        label(ax, 1.2, y + 0.21, name, color=TEXT, size=11, ha="right")
        label(ax, start + 2.0, y + 0.58, f"write@{start}", color=MUTED, size=8.5)
        label(ax, end + 2.85, y + 0.58, f"last_read@{end}", color=MUTED, size=8.5)
    for n in range(6):
        label(ax, n + 2.42, 2.0, f"node{n}", color=MUTED, size=9)
    ax.add_patch(Rectangle((2.0, 1.15), 4.85, 0.08, facecolor=RED, alpha=0.9))
    label(ax, 8.6, 1.24, "peak = 任一时刻 alive tensor 的最大总和；no_reuse 576MB → reuse 320MB（-44.4%）", color=TEXT, size=12.5, ha="left")
    save(fig, "4.1_Runtime架构设计/images/5_1_tensor_lifecycle.png")


def d_5_2_pools():
    fig, ax = new_fig("5.2 内存池设计", "热路径不分配：分池 + size class + 按 scope Reset")
    pools = [
        ("Weight Pool", "进程级\n加载后不释放", BLUE, 1.0),
        ("KV Pool", "请求/前缀级\nblock 分配归还", PURPLE, 4.1),
        ("Activation Pool", "graph/step 级\n生命周期复用", GREEN, 7.2),
        ("Workspace Pool", "node/step 级\nArena Reset", ORANGE, 10.3),
        ("IO / Zero-Copy", "小控制信息\n大块仍走 pinned async", CYAN, 13.4),
    ]
    for name, body, color, x in pools:
        box(ax, x, 4.6, 2.6, 1.6, name, body, fc="#0f1b31", ec=color)
    label(ax, 8, 3.55, "分配请求必须带：bytes + alignment + lifetime + pool + owner + group", color=TEXT, size=14, weight="bold")
    box(ax, 3.0, 1.35, 4.4, 1.35, "热路径", "pool lookup / Reset\n无 cudaMalloc / 无同步", ec=GREEN)
    box(ax, 8.6, 1.35, 4.4, 1.35, "冷路径", "预分配 / defrag / 扩容\nOOM 降级与审计", ec=ORANGE)
    save(fig, "4.2_内存池设计/images/5_2_memory_pools.png")


def d_5_2_fragmentation():
    fig, ax = new_fig("5.2 显存碎片", "总空闲够 ≠ 大块能分出来")
    # left: contiguous fragmented
    label(ax, 3.6, 6.8, "连续 first-fit：外部碎片 46.2%，512MB 分配失败", color=RED, size=13, weight="bold")
    x = 0.8
    segs = [(128, GREEN), (64, BG), (256, GREEN), (128, BG), (128, GREEN), (256, BG)]
    for size, color in segs:
        w = size / 1024 * 5.6
        ax.add_patch(Rectangle((x, 4.7), w, 0.85, facecolor=color, edgecolor=MUTED, linewidth=0.8))
        if color == BG:
            label(ax, x + w / 2, 5.12, "hole", color=MUTED, size=8)
        x += w
    box(ax, 5.2, 3.5, 1.7, 0.75, "需要 512MB", "失败", fc="#2a1620", ec=RED)
    # right: size class
    label(ax, 11.7, 6.8, "size-class pool：同类复用，cudaMalloc 5 次", color=GREEN, size=13, weight="bold")
    classes = [("64", 5), ("128", 4), ("256", 3), ("512", 1)]
    y = 5.7
    for cls, n in classes:
        label(ax, 8.2, y, cls + "B", color=TEXT, size=10, ha="right")
        for i in range(n):
            ax.add_patch(Rectangle((8.5 + i * 0.55, y - 0.18), 0.45, 0.36, facecolor=GREEN, edgecolor="none", alpha=0.85))
        y -= 0.65
    label(ax, 8, 1.6, "监控：total_free / largest_free_block / fragmentation / pool_hit_rate / failed_alloc", color=TEXT, size=12.5)
    save(fig, "4.2_内存池设计/images/5_2_fragmentation.png")


def d_5_3_flow():
    fig, ax = new_fig("5.3 CUDA Graph", "把 N 次 kernel launch 变成 1 次 graph launch；前提是 shape/地址固定")
    label(ax, 3.6, 6.9, "Eager：每个 op 都 launch", color=ORANGE, size=13, weight="bold")
    for i in range(5):
        box(ax, 0.9 + i * 1.15, 5.4, 0.95, 0.75, f"op{i}", "", ec=ORANGE)
        if i:
            arrow(ax, 0.9 + (i - 1) * 1.15 + 0.95, 5.78, 0.9 + i * 1.15, 5.78, color=ORANGE, lw=1.4)
    label(ax, 3.6, 4.75, "CPU launch × N，p99 抖动", color=MUTED, size=11)
    label(ax, 11.8, 6.9, "Capture once → Replay N times", color=GREEN, size=13, weight="bold")
    box(ax, 8.6, 5.25, 2.0, 1.0, "capture", "记录图 + 依赖", ec=CYAN)
    arrow(ax, 10.6, 5.75, 11.3, 5.75, color=GREEN)
    box(ax, 11.3, 5.25, 2.4, 1.0, "graphExec", "shape bucket + IO binding", ec=GREEN)
    arrow(ax, 13.7, 5.75, 14.4, 5.75, color=GREEN)
    box(ax, 14.4, 5.25, 1.0, 1.0, "replay", "1 launch", ec=GREEN)
    box(ax, 3.0, 2.0, 4.6, 1.4, "接入前提", "IO info 固定\nshape bucket 固定\n内存地址固定", ec=CYAN)
    box(ax, 8.6, 2.0, 4.6, 1.4, "必须保留", "eager fallback\nfallback_rate 监控", ec=RED)
    save(fig, "4.3_CUDA_Graph/images/5_3_cuda_graph_flow.png")


def d_5_3_speedup():
    replays = [1, 2, 4, 8, 16, 32, 64, 128]
    speedup = [0.62, 0.82, 0.98, 1.09, 1.15, 1.18, 1.20, 1.21]
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    ax = fig.add_axes([0.09, 0.14, 0.86, 0.72], facecolor=BG)
    ax.plot(replays, speedup, marker="o", color=CYAN, linewidth=3, markersize=8, label="graph speedup")
    ax.axhline(1.0, color=RED, linestyle="--", linewidth=2, label="break-even")
    ax.axvline(8, color=YELLOW, linestyle=":", linewidth=2)
    ax.text(8.4, 0.7, "break-even ≈ 8 replays", color=YELLOW, fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_xticks(replays)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(0.5, 1.3)
    ax.set_xlabel("replay 次数", color=TEXT, fontsize=13)
    ax.set_ylabel("speedup vs eager", color=TEXT, fontsize=13)
    ax.set_title("CUDA Graph 收益摊销：capture 成本必须靠 replay 打平", color=TEXT, fontsize=22, fontweight="bold", pad=20)
    ax.grid(color="#233552", alpha=0.7)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color("#33415c")
    ax.legend(facecolor=PANEL, edgecolor="#33415c", labelcolor=TEXT)
    save(fig, "4.3_CUDA_Graph/images/5_3_graph_speedup.png")


def d_5_4_events():
    fig, ax = new_fig("5.4 Stream 与 Event", "同 stream 靠顺序，跨 stream 靠 event")
    label(ax, 2.0, 7.0, "copy_stream", color=CYAN, size=13, ha="left", weight="bold")
    label(ax, 2.0, 4.4, "compute_stream", color=GREEN, size=13, ha="left", weight="bold")
    box(ax, 2.2, 6.55, 2.3, 0.85, "H2D", "MemcpyAsync", ec=CYAN)
    box(ax, 8.0, 6.55, 2.3, 0.85, "D2H", "MemcpyAsync", ec=CYAN)
    box(ax, 5.0, 3.95, 3.0, 0.9, "kernel", "compute", ec=GREEN)
    box(ax, 4.75, 5.55, 0.55, 0.45, "e1", "", fc=YELLOW, ec=YELLOW)
    box(ax, 8.35, 5.0, 0.55, 0.45, "e2", "", fc=YELLOW, ec=YELLOW)
    arrow(ax, 4.5, 6.98, 4.85, 5.75, color=YELLOW)
    arrow(ax, 5.3, 5.55, 5.3, 4.85, color=YELLOW)
    arrow(ax, 8.0, 4.4, 8.55, 5.0, color=YELLOW)
    arrow(ax, 8.9, 5.0, 8.9, 6.55, color=YELLOW)
    label(ax, 5.9, 5.25, "Record(e_h2d)", color=MUTED, size=9)
    label(ax, 9.7, 4.65, "Record(e_compute)", color=MUTED, size=9)
    label(ax, 8, 2.4, "错误：只 record 不 wait；用 cudaDeviceSynchronize 代替依赖图；buffer 未等完成就复用", color=RED, size=12.5)
    save(fig, "4.4_多流并发执行/images/5_4_stream_events.png")


def d_5_4_overlap():
    fig, ax = new_fig("5.4 多流 Overlap", "单帧 latency 不变，吞吐/FPS 提升")
    label(ax, 2.0, 7.0, "single: span 204ms ≈ 29.4 FPS", color=ORANGE, size=12.5, ha="left", weight="bold")
    for i in range(6):
        x = 1.0 + i * 2.35
        box(ax, x, 6.1, 0.55, 0.55, "H", "", ec=CYAN, lw=1.2)
        box(ax, x + 0.55, 6.1, 1.3, 0.55, "C", "", ec=GREEN, lw=1.2)
        box(ax, x + 1.85, 6.1, 0.5, 0.55, "D", "", ec=PURPLE, lw=1.2)
    label(ax, 2.0, 4.2, "two streams: span 154ms ≈ 39.0 FPS", color=GREEN, size=12.5, ha="left", weight="bold")
    for i in range(6):
        x = 1.0 + i * 1.68
        box(ax, x, 3.2, 0.55, 0.55, "H", "", ec=CYAN, lw=1.2)
        box(ax, x + 0.55, 3.2, 1.3, 0.55, "C", "", ec=GREEN, lw=1.2)
        box(ax, x + 1.85, 3.2, 0.5, 0.55, "D", "", ec=PURPLE, lw=1.2)
    label(ax, 8, 1.5, "结论：overlap 主要提升 throughput；latency 是否下降看最慢 stage 与排队", color=TEXT, size=13)
    save(fig, "4.4_多流并发执行/images/5_4_overlap_timeline.png")


def d_5_5_queues():
    fig, ax = new_fig("5.5 动态 Batch Scheduler", "按成本组 batch，按 QoS 保公平，按资源防 OOM")
    box(ax, 0.9, 5.7, 3.0, 1.35, "waiting queue", "arrival / cost estimate", ec=BLUE)
    box(ax, 4.6, 5.7, 3.4, 1.35, "scheduler", "token budget + KV blocks\nmax_num_seqs + aging", ec=YELLOW)
    box(ax, 8.7, 5.7, 3.0, 1.35, "running batch", "prefill + decode", ec=GREEN)
    box(ax, 12.4, 5.7, 2.7, 1.35, "done / emit", "token / metrics", ec=PURPLE)
    box(ax, 4.6, 3.3, 3.4, 1.1, "blocked / preempted", "swap / recompute / defer", ec=RED)
    arrow(ax, 3.9, 6.38, 4.6, 6.38)
    arrow(ax, 8.0, 6.38, 8.7, 6.38)
    arrow(ax, 11.7, 6.38, 12.4, 6.38)
    arrow(ax, 6.3, 5.7, 6.3, 4.4, color=RED)
    arrow(ax, 8.0, 3.85, 8.7, 5.95, color=RED)
    label(ax, 8, 1.8, "入 batch 条件：len<max_num_seqs 且 tokens≤budget 且 kv_blocks≥need 且 workspace≥need 且 QoS 允许", color=TEXT, size=12.5)
    save(fig, "4.5_动态Batch_Scheduler/images/5_5_scheduler_queues.png")


def d_5_5_tradeoff():
    import numpy as np
    x = np.linspace(1, 64, 200)
    throughput = 1 - np.exp(-x / 12)
    latency = 0.25 + (x / 64) ** 2 * 2.2
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    ax = fig.add_axes([0.09, 0.15, 0.84, 0.68], facecolor=BG)
    ax.plot(x, throughput, color=GREEN, lw=3, label="throughput（边际递减）")
    ax.plot(x, latency, color=ORANGE, lw=3, label="batch latency / TPOT 风险")
    ax.axvspan(8, 24, color=CYAN, alpha=0.15)
    ax.text(16, 1.35, "sweet spot\n按 token budget/KV 调整", color=CYAN, ha="center", fontsize=13)
    ax.set_xlabel("effective batch / token budget", color=TEXT, fontsize=13)
    ax.set_title("Dynamic Batch Tradeoff：吞吐收益 vs 延迟/显存风险", color=TEXT, fontsize=22, fontweight="bold", pad=20)
    ax.grid(color="#233552", alpha=0.7)
    ax.tick_params(colors=MUTED)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#33415c")
    ax.legend(facecolor=PANEL, edgecolor="#33415c", labelcolor=TEXT, loc="lower right")
    save(fig, "4.5_动态Batch_Scheduler/images/5_5_batch_tradeoff.png")


def d_5_6_order():
    fig, ax = new_fig("5.6 Runtime 特性开发顺序", "先契约正确，再资源可控，再性能优化，再多租户，最后稳定性兜底")
    steps = [("1 IO info", CYAN), ("2 memory pool", GREEN), ("3 cudaGraph", ORANGE), ("4 dynamic batch", YELLOW), ("5 groupContext", PURPLE), ("6 stability", RED)]
    x = 0.8
    for name, color in steps:
        box(ax, x, 4.7, 2.25, 1.25, name, "", fc="#0f1b31", ec=color)
        if x > 1:
            arrow(ax, x - 0.25, 5.32, x, 5.32, color=MUTED, lw=1.8)
        x += 2.55
    notes = [
        "shape/dtype/layout", "固定地址/分池", "capture+replay+fallback", "token budget/QoS", "quota/隔离/回收", "watchdog/cancel/soak",
    ]
    x = 1.9
    for note, (_, color) in zip(notes, steps):
        label(ax, x, 3.95, note, color=color, size=10.5)
        x += 2.55
    box(ax, 3.0, 1.6, 10.0, 1.2, "评审模板", "每个 feature 都必须写清 goal / interface / risks / tests，否则不进开发", ec=BLUE)
    save(fig, "4.6_Relay_Relax_Runtime特性开发/images/5_6_feature_order.png")


def d_5_6_group():
    fig, ax = new_fig("5.6 groupContext", "多业务共线：quota、隔离、优先级、回收审计")
    groups = [("Group A 在线主链路", "高优先级\nguaranteed quota", GREEN, 1.0), ("Group B 内部工具", "中优先级\nlimited quota", YELLOW, 5.6), ("Group C 离线/实验", "低优先级\n可被抢占/隔离", RED, 10.2)]
    for name, body, color, x in groups:
        box(ax, x, 5.2, 4.0, 1.6, name, body, fc="#0f1b31", ec=color)
    box(ax, 5.2, 2.8, 5.6, 1.2, "Runtime Resource Manager", "memory quota + max_num_seqs + token budget + scheduler priority", ec=CYAN)
    for x in [3.0, 7.6, 12.2]:
        arrow(ax, x, 5.2, 8.0, 4.0, color=MUTED, lw=1.6)
    label(ax, 8, 1.65, "删除 group 前必须审计：session/kernel/event/graph exec/KV blocks/显存是否全部回收", color=TEXT, size=12.5)
    save(fig, "4.6_Relay_Relax_Runtime特性开发/images/5_6_group_context.png")


def d_resume_stack():
    fig, ax = new_fig("简历项目：自研推理 Runtime", "用最小可编译证据支撑执行层闭环")
    layers = [("IO contract", CYAN), ("Memory pool", GREEN), ("Graph executor", ORANGE), ("Stream/CUDA Graph", PURPLE), ("Dynamic batch", YELLOW), ("Stability", RED)]
    y = 6.7
    for name, color in layers:
        box(ax, 1.0, y, 4.0, 0.75, name, "", fc="#0f1b31", ec=color)
        y -= 0.9
    evidence = [("C++ demo", "ArenaAllocator / SizeClassPool"), ("Python simulators", "scheduler / pool / graph / stream / batch"), ("Metrics", "p99 / fragmentation / fallback rate / throughput")]
    y = 6.5
    for name, body in evidence:
        box(ax, 6.2, y, 8.6, 0.95, name, body, ec=BLUE)
        y -= 1.35
    label(ax, 8, 1.1, "简历 bullet = 分层架构 + 指标口径 + 可复现脚本 + 正确性保障（IO info/fallback/回收审计）", color=TEXT, size=13)
    save(fig, "简历项目/images/resume_runtime_stack.png")


def main():
    cover()
    d_5_1_arch(); d_5_1_lifecycle()
    d_5_2_pools(); d_5_2_fragmentation()
    d_5_3_flow(); d_5_3_speedup()
    d_5_4_events(); d_5_4_overlap()
    d_5_5_queues(); d_5_5_tradeoff()
    d_5_6_order(); d_5_6_group()
    d_resume_stack()


if __name__ == "__main__":
    main()
