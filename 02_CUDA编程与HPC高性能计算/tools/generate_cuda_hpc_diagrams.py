#!/usr/bin/env python3
"""生成 CUDA 与 HPC 高性能计算专题演示图。运行：
python tools/generate_cuda_hpc_diagrams.py
"""
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
    fig, ax = new_fig("CUDA 编程与 HPC 高性能计算",
                      "从编程模型到性能优化：CUDA 基础 / 优化 / Reduce / GEMM / Attention / CPU 并行 / 工具链")
    items = [
        ("3.1", "CUDA 体系", CYAN), ("3.2", "性能优化", GREEN), ("3.3", "Reduce", ORANGE),
        ("3.4", "GEMM", PURPLE), ("3.5", "Attention", YELLOW), ("3.6", "CPU 并行", RED),
        ("3.7", "HPC 实战", BLUE), ("3.8", "性能分析", CYAN),
    ]
    x = 0.8
    for num, name, color in items:
        box(ax, x, 4.4, 1.75, 1.25, num, name, fc="#0f1b31", ec=color)
        x += 1.95
    arrow(ax, 1.7, 3.75, 15.3, 3.75, color=MUTED, lw=1.4, style="-", ms=1, alpha=0.6)
    label(ax, 8, 3.35, "GPU 编程模型 → 经典算子 → CPU 并行 → 业务实战 → 性能分析", color=TEXT, size=15, weight="bold")
    save(fig, "images/cuda_hpc_cover.png")


def d_3_1():
    fig, ax = new_fig("3.1 CUDA 编程体系", "Grid / Block / Thread / 内存层级 / Warp")
    box(ax, 1.0, 5.8, 2.8, 1.35, "Grid", "所有线程", ec=BLUE)
    box(ax, 4.6, 5.8, 2.8, 1.35, "Block", "共享内存 + 同步", ec=CYAN)
    box(ax, 8.2, 5.8, 2.8, 1.35, "Warp", "32 线程", ec=ORANGE)
    box(ax, 11.8, 5.8, 2.8, 1.35, "Thread", "执行单元", ec=GREEN)
    arrow(ax, 3.8, 6.48, 4.6, 6.48, color=CYAN, lw=1.4, alpha=0.6)
    arrow(ax, 7.4, 6.48, 8.2, 6.48, color=ORANGE, lw=1.4, alpha=0.6)
    arrow(ax, 11.0, 6.48, 11.8, 6.48, color=GREEN, lw=1.4, alpha=0.6)

    mems = [
        ("Global", "全局内存\n容量大/高延迟", BLUE, 1.0),
        ("Constant", "常量内存\n只读/广播", CYAN, 3.8),
        ("Shared", "共享内存\nBlock 内/低延迟", ORANGE, 6.6),
        ("Register", "寄存器\n线程私有/最快", GREEN, 9.4),
    ]
    for name, body, color, x in mems:
        box(ax, x, 3.2, 2.6, 1.35, name, body, fc="#0f1b31", ec=color)
    label(ax, 8, 1.7, "关键：合并访问、避免 Bank Conflict、合理 Occupancy、分支发散", color=TEXT, size=13, weight="bold")
    save(fig, "2.1_CUDA编程体系/images/3_1_cuda_programming_model.png")


def d_3_2():
    fig, ax = new_fig("3.2 CUDA 性能优化基础", "内存 / Shuffle / Occupancy / 同步")
    opts = [
        ("Coalesced", "合并访问", CYAN), ("Shared Tiling", "共享内存预取", GREEN),
        ("Vectorized", "float4 加载", ORANGE), ("Warp Shuffle", "Warp 内通信", PURPLE),
        ("Occupancy", "资源占用率", YELLOW), ("Atomic", "原子操作最小化", RED),
    ]
    x = 0.8
    for name, body, color in opts:
        box(ax, x, 5.6, 2.3, 1.35, name, body, fc="#0f1b31", ec=color)
        x += 2.55
    box(ax, 3.0, 3.0, 4.5, 1.25, "Memory-bound", "优化带宽利用率\n减少数据搬运", ec=CYAN)
    box(ax, 8.5, 3.0, 4.5, 1.25, "Compute-bound", "压榨 Tensor Core\n增加 FMA 密度", ec=ORANGE)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=CYAN)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=ORANGE)
    label(ax, 8, 1.6, "先用 Roofline 判断瓶颈，再针对性优化", color=TEXT, size=13, weight="bold")
    save(fig, "2.2_CUDA性能优化基础/images/3_2_cuda_performance_opt.png")


def d_3_3():
    fig, ax = new_fig("3.3 Reduce 算子实现", "atomic -> shared -> shuffle -> 多级")
    stages = [
        ("Atomic", "全局原子加", RED), ("Shared Tree", "共享内存树形归约", ORANGE),
        ("Warp Shuffle", "Warp 内寄存器交换", GREEN), ("Multi-level", "Block + Grid 两级", CYAN),
    ]
    x = 1.0
    for name, body, color in stages:
        box(ax, x, 5.6, 3.4, 1.35, name, body, fc="#0f1b31", ec=color)
        x += 3.6
    arrow(ax, 4.4, 6.25, 4.6, 6.25, color=CYAN, lw=1.5, alpha=0.6)
    arrow(ax, 8.0, 6.25, 8.2, 6.25, color=CYAN, lw=1.5, alpha=0.6)
    arrow(ax, 11.6, 6.25, 11.8, 6.25, color=CYAN, lw=1.5, alpha=0.6)
    box(ax, 3.0, 3.0, 4.5, 1.25, "瓶颈", "全局内存带宽\n原子操作序列化", ec=RED)
    box(ax, 8.5, 3.0, 4.5, 1.25, "目标", "把通信限制在片上\n减少 HBM 读写", ec=GREEN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=RED)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=GREEN)
    label(ax, 8, 1.6, "Reduce 核心：先在局部归约，最后少量全局合并", color=TEXT, size=13, weight="bold")
    save(fig, "2.3_经典算子实现-Reduce/images/3_3_reduce.png")


def d_3_4():
    fig, ax = new_fig("3.4 GEMM 算子实现", "Tiling -> Shared Memory -> Tensor Core")
    box(ax, 0.8, 5.6, 3.0, 1.35, "Naive", "每个线程读 A 行 + B 列\n无复用", ec=RED)
    box(ax, 4.4, 5.6, 3.0, 1.35, "Tiling", "C tile = A tile * B tile\n复用共享内存", ec=ORANGE)
    box(ax, 8.0, 5.6, 3.0, 1.35, "Shared Mem", "A_tile/B_tile -> SMEM\n线程协作加载", ec=GREEN)
    box(ax, 11.6, 5.6, 3.0, 1.35, "Tensor Core", "WMMA / m16n8k16\n硬件矩阵乘累加", ec=CYAN)
    arrow(ax, 3.8, 6.25, 4.4, 6.25, color=CYAN, lw=1.5, alpha=0.6)
    arrow(ax, 7.4, 6.25, 8.0, 6.25, color=CYAN, lw=1.5, alpha=0.6)
    arrow(ax, 11.0, 6.25, 11.6, 6.25, color=CYAN, lw=1.5, alpha=0.6)
    box(ax, 3.0, 3.0, 4.5, 1.25, "计算强度", "I = 2MNK / (MK+KN+MN)", ec=BLUE)
    box(ax, 8.5, 3.0, 4.5, 1.25, "目标", "把数据复用提升到片上\n隐藏 HBM 延迟", ec=GREEN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=BLUE)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=GREEN)
    label(ax, 8, 1.6, "手写 GEMM 通常只有 cuBLAS 30%~70%，但理解优化过程很关键", color=TEXT, size=13, weight="bold")
    save(fig, "2.4_经典算子实现-GEMM/images/3_4_gemm.png")


def d_3_5():
    fig, ax = new_fig("3.5 Attention 算子", "FlashAttention / Decoding / Paged / FlashInfer")
    items = [
        ("FlashAttn V1", "Tiling + Online Softmax", CYAN),
        ("FlashAttn V2", "Warp 划分优化", GREEN),
        ("FlashAttn V3", "Hopper FP8/TMA", ORANGE),
        ("Flash-Decoding", "长序列 decode", PURPLE),
        ("PagedAttention", "KV Cache 分页", YELLOW),
        ("FlashInfer", "可组合引擎", RED),
    ]
    x = 0.8
    for name, body, color in items:
        box(ax, x, 5.6, 2.3, 1.35, name, body, fc="#0f1b31", ec=color)
        x += 2.55
    box(ax, 3.0, 3.0, 4.5, 1.25, "核心思想", "避免显式 N x N\n把计算留在 SRAM", ec=CYAN)
    box(ax, 8.5, 3.0, 4.5, 1.25, "工程收益", "减少 HBM 访问\n支持更长序列", ec=GREEN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=CYAN)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=GREEN)
    label(ax, 8, 1.6, "FlashAttention 不减少 FLOPs，而是减少 HBM 访问", color=TEXT, size=13, weight="bold")
    save(fig, "2.5_Attention算子/images/3_5_attention.png")


def d_3_6():
    fig, ax = new_fig("3.6 多线程并行优化", "OpenMP / NUMA / SIMD")
    box(ax, 1.0, 5.6, 3.5, 1.35, "OpenMP", "parallel for / reduction\nschedule / affinity", ec=BLUE)
    box(ax, 5.8, 5.6, 3.5, 1.35, "NUMA", "本地内存 + 绑核\n避免跨 socket", ec=CYAN)
    box(ax, 10.6, 5.6, 3.5, 1.35, "SIMD", "SSE/AVX/AVX-512\n向量化计算", ec=GREEN)
    arrow(ax, 4.5, 6.25, 5.8, 6.25, color=CYAN, lw=1.5, alpha=0.6)
    arrow(ax, 9.3, 6.25, 10.6, 6.25, color=CYAN, lw=1.5, alpha=0.6)
    box(ax, 3.0, 3.0, 4.5, 1.25, "适用场景", "预处理/后处理\n不适合 GPU 的算子", ec=ORANGE)
    box(ax, 8.5, 3.0, 4.5, 1.25, "关键指标", "扩展性 / 远端内存\n向量化比率", ec=GREEN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=ORANGE)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=GREEN)
    label(ax, 8, 1.6, "CPU 优化与 GPU 优化互补，不是替代", color=TEXT, size=13, weight="bold")
    save(fig, "2.6_多线程并行优化/images/3_6_cpu_parallel.png")


def d_3_7():
    fig, ax = new_fig("3.7 HPC 算子开发实战", "分块 / 向量化 / 流水线 / 多平台")
    box(ax, 0.8, 5.6, 3.0, 1.35, "CUDA A100", "Tensor Core\nWMMA/async", ec=CYAN)
    box(ax, 4.4, 5.6, 3.0, 1.35, "Orin/P1X", "低功耗/INT8\n算子融合", ec=GREEN)
    box(ax, 8.0, 5.6, 3.0, 1.35, "Ascend/PPU", "自定义 DSL\nLayout 适配", ec=ORANGE)
    box(ax, 11.6, 5.6, 3.0, 1.35, "业务算子", "MMBEV/Adam\n训练+推理", ec=PURPLE)
    box(ax, 3.0, 3.0, 4.5, 1.25, "核心技术", "Tiling / Vectorization\nPipelining / Cache", ec=BLUE)
    box(ax, 8.5, 3.0, 4.5, 1.25, "迁移关键", "数值对齐 / Layout\n计算单元差异", ec=RED)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=BLUE)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=RED)
    label(ax, 8, 1.6, "HPC 算子开发 = 理解硬件 + 算法等价 + 性能验证", color=TEXT, size=13, weight="bold")
    save(fig, "2.7_HPC算子开发实战/images/3_7_hpc_practice.png")


def d_3_8():
    fig, ax = new_fig("3.8 性能分析工具链", "Nsight / PyTorch Profiler / Roofline")
    box(ax, 0.8, 5.6, 3.2, 1.35, "Nsight Systems", "系统级 Timeline\nCPU-GPU 交互", ec=BLUE)
    box(ax, 4.4, 5.6, 3.2, 1.35, "Nsight Compute", "Kernel 级下钻\nSOL 面板", ec=CYAN)
    box(ax, 8.0, 5.6, 3.2, 1.35, "PyTorch Profiler", "Python 层热点\n算子耗时", ec=GREEN)
    box(ax, 11.6, 5.6, 3.2, 1.35, "Roofline", "计算强度判断\nbound 类型", ec=ORANGE)
    box(ax, 3.0, 3.0, 4.5, 1.25, "先用 Systems 定位慢 kernel", ec=BLUE)
    box(ax, 8.5, 3.0, 4.5, 1.25, "再用 Compute 下钻优化", ec=CYAN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=BLUE)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=CYAN)
    label(ax, 8, 1.6, "无工具不调优：指标驱动，而不是感觉驱动", color=TEXT, size=13, weight="bold")
    save(fig, "2.8_性能分析工具链/images/3_8_profiler_toolchain.png")


def d_resume_stack():
    fig, ax = new_fig("简历项目：CUDA 与 HPC", "用硬件特性 + 性能证据支撑项目叙述")
    layers = [("CUDA 编程模型", CYAN), ("Reduce / GEMM / Attention", GREEN), ("CPU 并行优化", ORANGE), ("性能分析工具链", PURPLE)]
    y = 6.7
    for name, color in layers:
        box(ax, 1.0, y, 4.0, 0.75, name, "", fc="#0f1b31", ec=color)
        y -= 0.9
    evidence = [
        ("CUDA demos", "threading / reduce / sgemm"),
        ("Nsight metrics", "Memory SOL / Compute SOL / Roofline"),
        ("Resume bullets", "shape / precision / hardware / baseline / diff"),
    ]
    y = 6.5
    for name, body in evidence:
        box(ax, 6.2, y, 8.6, 0.95, name, body, ec=BLUE)
        y -= 1.35
    label(ax, 8, 1.1, "简历 bullet = 硬件特性 + 优化策略 + 指标 + 数值对齐", color=TEXT, size=13)
    save(fig, "简历项目/images/cuda_hpc_project_stack.png")


def main():
    cover()
    d_3_1()
    d_3_2()
    d_3_3()
    d_3_4()
    d_3_5()
    d_3_6()
    d_3_7()
    d_3_8()
    d_resume_stack()


if __name__ == "__main__":
    main()
