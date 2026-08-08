#!/usr/bin/env python3
"""生成分布式训练专题演示图。运行：
python tools/generate_distributed_training_diagrams.py
"""
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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
    fig, ax = new_fig("分布式训练", "从并行策略到框架实战：DP / ZeRO / TP / PP / 3D / 通信优化")
    items = [
        ("7.1", "总论", CYAN), ("7.2", "数据并行", GREEN), ("7.3", "ZeRO", ORANGE),
        ("7.4", "TP/SP", PURPLE), ("7.5", "PP", YELLOW), ("7.6", "3D 并行", RED),
        ("7.7", "框架实战", BLUE), ("7.8", "通信优化", CYAN),
    ]
    x = 0.8
    for num, name, color in items:
        box(ax, x, 4.4, 1.75, 1.25, num, name, fc="#0f1b31", ec=color)
        x += 1.95
    arrow(ax, 1.7, 3.75, 15.3, 3.75, color=MUTED, lw=1.4, style="-", ms=1, alpha=0.6)
    label(ax, 8, 3.35, "显存分析 → 并行策略 → 框架配置 → 通信优化", color=TEXT, size=15, weight="bold")
    save(fig, "images/distributed_training_cover.png")


def d_7_1():
    fig, ax = new_fig("7.1 分布式训练总论", "显存分析 + 并行策略全景")
    box(ax, 1.0, 5.6, 3.0, 1.35, "Params", "4P / 2P", ec=CYAN)
    box(ax, 4.6, 5.6, 3.0, 1.35, "Gradients", "4P", ec=GREEN)
    box(ax, 8.2, 5.6, 3.0, 1.35, "Optimizer States", "8P (Adam)", ec=ORANGE)
    box(ax, 11.8, 5.6, 3.0, 1.35, "Activations", "与长度/层数相关", ec=PURPLE)
    box(ax, 3.0, 3.0, 3.5, 1.25, "数据并行", "切分数据", ec=BLUE)
    box(ax, 6.8, 3.0, 3.5, 1.25, "模型并行", "TP / PP", ec=YELLOW)
    box(ax, 10.6, 3.0, 3.5, 1.25, "混合并行", "3D", ec=RED)
    label(ax, 8, 1.6, "训练显存 ≈ 16P + Activations；并行策略取决于单卡能否放下模型", color=TEXT, size=13, weight="bold")
    save(fig, "6.1_分布式训练总论/images/7_1_distributed_overview.png")


def d_7_2():
    fig, ax = new_fig("7.2 数据并行", "DP / DDP / FSDP")
    box(ax, 0.8, 5.6, 3.0, 1.35, "DP", "单进程多卡\n低效", ec=RED)
    box(ax, 4.4, 5.6, 3.0, 1.35, "DDP", "多进程 + AllReduce\n标准", ec=GREEN)
    box(ax, 8.0, 5.6, 3.0, 1.35, "FSDP", "参数/梯度/状态分片\n省显存", ec=CYAN)
    box(ax, 11.6, 5.6, 3.0, 1.35, "ZeRO-3", "DeepSpeed 版", ec=ORANGE)
    arrow(ax, 3.8, 6.25, 4.4, 6.25, color=GREEN, lw=1.5, alpha=0.6)
    arrow(ax, 7.4, 6.25, 8.0, 6.25, color=CYAN, lw=1.5, alpha=0.6)
    arrow(ax, 11.0, 6.25, 11.6, 6.25, color=ORANGE, lw=1.5, alpha=0.6)
    box(ax, 3.0, 3.0, 4.5, 1.25, "DDP 核心", "多进程 + bucket + overlap", ec=GREEN)
    box(ax, 8.5, 3.0, 4.5, 1.25, "FSDP 核心", "all-gather + reduce-scatter", ec=CYAN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=GREEN)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=CYAN)
    label(ax, 8, 1.6, "能放下模型用 DDP；放不下用 FSDP/ZeRO-3", color=TEXT, size=13, weight="bold")
    save(fig, "6.2_数据并行/images/7_2_data_parallelism.png")


def d_7_3():
    fig, ax = new_fig("7.3 ZeRO 系列", "用通信换显存")
    stages = [
        ("DDP", "16P", RED), ("ZeRO-1", "8P", ORANGE),
        ("ZeRO-2", "4P", YELLOW), ("ZeRO-3", "16P/N", GREEN),
        ("Offload", "CPU/NVMe", CYAN),
    ]
    x = 0.8
    for name, body, color in stages:
        box(ax, x, 5.6, 2.7, 1.35, name, body, fc="#0f1b31", ec=color)
        x += 2.95
    box(ax, 3.0, 3.0, 4.5, 1.25, "切分内容", "OS / +G / +P", ec=ORANGE)
    box(ax, 8.5, 3.0, 4.5, 1.25, "代价", "更多 AllGather\nReduceScatter", ec=RED)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=ORANGE)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=RED)
    label(ax, 8, 1.6, "ZeRO 核心：消除数据并行中的模型状态冗余", color=TEXT, size=13, weight="bold")
    save(fig, "6.3_ZeRO系列/images/7_3_zero.png")


def d_7_4():
    fig, ax = new_fig("7.4 张量并行与序列并行", "TP / SP / GQA")
    box(ax, 0.8, 5.6, 3.5, 1.35, "Column Parallel", "W 按列切分\nAllGather 输出", ec=CYAN)
    box(ax, 4.8, 5.6, 3.5, 1.35, "Row Parallel", "W 按行切分\nAllReduce 输出", ec=GREEN)
    box(ax, 8.8, 5.6, 3.0, 1.35, "TP", "节点内 NVLink\n高带宽", ec=ORANGE)
    box(ax, 12.3, 5.6, 3.0, 1.35, "SP", "沿序列维度切分\n激活显存", ec=PURPLE)
    box(ax, 3.0, 3.0, 4.5, 1.25, "限制", "TP 不能跨机\nNVLink vs IB", ec=RED)
    box(ax, 8.5, 3.0, 4.5, 1.25, "GQA/MQA", "K/V 共享\n需要广播", ec=YELLOW)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=RED)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=YELLOW)
    label(ax, 8, 1.6, "TP 切分单层参数，SP 切分序列激活，两者配合降低显存", color=TEXT, size=13, weight="bold")
    save(fig, "6.4_张量并行与序列并行/images/7_4_tensor_sequence_parallel.png")


def d_7_5():
    fig, ax = new_fig("7.5 流水线并行", "GPipe / 1F1B / 气泡")
    box(ax, 0.8, 5.6, 3.0, 1.35, "GPipe", "fill-flush\n气泡大", ec=ORANGE)
    box(ax, 4.4, 5.6, 3.0, 1.35, "1F1B", "交错 F/B\n气泡小", ec=GREEN)
    box(ax, 8.0, 5.6, 3.0, 1.35, "Interleaved", "多层交错\n进一步降气泡", ec=CYAN)
    box(ax, 11.6, 5.6, 3.0, 1.35, "Bubble Rate", "(P-1)/M", ec=RED)
    box(ax, 3.0, 3.0, 4.5, 1.25, "权衡", "气泡 vs 激活显存", ec=ORANGE)
    box(ax, 8.5, 3.0, 4.5, 1.25, "优化", "增 micro-batch\nactivation ckpt", ec=GREEN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=ORANGE)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=GREEN)
    label(ax, 8, 1.6, "PP 按层切分，micro-batch 越多气泡越小但激活越多", color=TEXT, size=13, weight="bold")
    save(fig, "6.5_流水线并行/images/7_5_pipeline_parallel.png")


def d_7_6():
    fig, ax = new_fig("7.6 3D 并行与混合训练", "TP × PP × DP")
    box(ax, 0.8, 5.6, 3.0, 1.35, "TP=8", "节点内", ec=CYAN)
    box(ax, 4.4, 5.6, 3.0, 1.35, "PP=4", "跨节点", ec=GREEN)
    box(ax, 8.0, 5.6, 3.0, 1.35, "DP=2", "剩余卡", ec=ORANGE)
    box(ax, 11.6, 5.6, 3.0, 1.35, "64 GPUs", "8×4×2", ec=PURPLE)
    box(ax, 3.0, 3.0, 4.5, 1.25, "混合精度", "FP16/BF16/FP8\nLoss Scaling", ec=BLUE)
    box(ax, 8.5, 3.0, 4.5, 1.25, "显存优化", "Grad Accum\nActivation Checkpoint", ec=YELLOW)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=BLUE)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=YELLOW)
    label(ax, 8, 1.6, "3D 并行 = 显存、通信、计算的平衡艺术", color=TEXT, size=13, weight="bold")
    save(fig, "6.6_3D并行与混合训练策略/images/7_6_3d_parallel.png")


def d_7_7():
    fig, ax = new_fig("7.7 训练框架实战", "Megatron / DeepSpeed / FSDP")
    box(ax, 0.8, 5.6, 3.5, 1.35, "Megatron-LM", "TP + PP\nNVIDIA", ec=CYAN)
    box(ax, 4.8, 5.6, 3.0, 1.35, "DeepSpeed", "ZeRO 系列\nOffload", ec=GREEN)
    box(ax, 8.3, 5.6, 3.0, 1.35, "PyTorch FSDP", "原生 ZeRO-3\n易集成", ec=ORANGE)
    box(ax, 11.8, 5.6, 3.0, 1.35, "组合", "Megatron+DS\nMegatron+FSDP", ec=PURPLE)
    box(ax, 3.0, 3.0, 4.5, 1.25, "选型依据", "模型规模 / 集群 / 生态", ec=BLUE)
    box(ax, 8.5, 3.0, 4.5, 1.25, "配置重点", "stage / wrap / bucket", ec=GREEN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=BLUE)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=GREEN)
    label(ax, 8, 1.6, "框架只是工具，理解并行策略才能用好框架", color=TEXT, size=13, weight="bold")
    save(fig, "6.7_训练框架实战/images/7_7_training_frameworks.png")


def d_7_8():
    fig, ax = new_fig("7.8 多卡训练通信优化", "NCCL / Ring / Overlap")
    box(ax, 0.8, 5.6, 2.8, 1.35, "NCCL", "集合通信库\nbackend", ec=BLUE)
    box(ax, 4.1, 5.6, 2.8, 1.35, "Ring AllReduce", "带宽最优\n2B 通信量", ec=CYAN)
    box(ax, 7.4, 5.6, 2.8, 1.35, "Bucket", "梯度分桶\n提前 AllReduce", ec=GREEN)
    box(ax, 10.7, 5.6, 2.8, 1.35, "Overlap", "通信与计算\n并行", ec=ORANGE)
    box(ax, 14.0, 5.6, 1.6, 1.35, "压缩", "Top-k\n量化", ec=RED)
    box(ax, 3.0, 3.0, 4.5, 1.25, "瓶颈", "跨机 IB\n慢速网络", ec=RED)
    box(ax, 8.5, 3.0, 4.5, 1.25, "优化", "NVLink / 多流\n梯度压缩", ec=GREEN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=RED)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=GREEN)
    label(ax, 8, 1.6, "通信优化：先隐藏，再压缩，最后才考虑降带宽", color=TEXT, size=13, weight="bold")
    save(fig, "6.8_多卡训练通信优化/images/7_8_communication_opt.png")


def d_resume_stack():
    fig, ax = new_fig("简历项目：分布式训练", "并行策略 + 集群性能证据")
    layers = [("并行策略设计", CYAN), ("框架配置", GREEN), ("通信优化", ORANGE), ("性能指标", PURPLE)]
    y = 6.7
    for name, color in layers:
        box(ax, 1.0, y, 4.0, 0.75, name, "", fc="#0f1b31", ec=color)
        y -= 0.9
    evidence = [
        ("DDP/ZeRO demos", "显存估算 / 加速比"),
        ("3D topo design", "TP/PP/DP 拓扑"),
        ("Nsight/NCCL", "通信 timeline / overlap"),
    ]
    y = 6.5
    for name, body in evidence:
        box(ax, 6.2, y, 8.6, 0.95, name, body, ec=BLUE)
        y -= 1.35
    label(ax, 8, 1.1, "简历 bullet = 并行策略 + 集群规模 + baseline + 吞吐指标 + 加速比", color=TEXT, size=13)
    save(fig, "简历项目/images/distributed_training_project_stack.png")


def main():
    cover()
    d_7_1()
    d_7_2()
    d_7_3()
    d_7_4()
    d_7_5()
    d_7_6()
    d_7_7()
    d_7_8()
    d_resume_stack()


if __name__ == "__main__":
    main()
