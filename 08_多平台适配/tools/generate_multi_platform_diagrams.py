#!/usr/bin/env python3
"""生成多平台适配专题演示图。运行：
python tools/generate_multi_platform_diagrams.py
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
    fig, ax = new_fig("多平台适配", "NVIDIA GPU / Jetson Orin / Ascend NPU / 自研芯片 / Benchmark")
    items = [
        ("9.1", "NVIDIA GPU", CYAN), ("9.2", "Jetson Orin", GREEN),
        ("9.3", "Ascend NPU", ORANGE), ("9.4", "自研芯片", PURPLE),
        ("9.5", "Benchmark", RED),
    ]
    x = 1.2
    for num, name, color in items:
        box(ax, x, 4.4, 2.6, 1.25, num, name, fc="#0f1b31", ec=color)
        x += 3.0
    arrow(ax, 2.0, 3.75, 15.0, 3.75, color=MUTED, lw=1.4, style="-", ms=1, alpha=0.6)
    label(ax, 8, 3.35, "平台特性 → 部署策略 → 算子适配 → 一致性验证 → Benchmark", color=TEXT, size=15, weight="bold")
    save(fig, "images/multi_platform_cover.png")


def d_9_1():
    fig, ax = new_fig("9.1 NVIDIA GPU 部署", "TensorRT + A100/H100 + 多卡")
    box(ax, 0.8, 5.6, 2.8, 1.35, "A100", "Ampere\n80GB HBM2e", ec=CYAN)
    box(ax, 4.0, 5.6, 2.8, 1.35, "H100", "Hopper\nFP8 / NVLink", ec=GREEN)
    box(ax, 7.2, 5.6, 2.8, 1.35, "TensorRT", "层融合 / 精度\nkernel 调优", ec=ORANGE)
    box(ax, 10.4, 5.6, 2.8, 1.35, "TRT-LLM", "PagedAttention\nInflight Batch", ec=PURPLE)
    box(ax, 13.6, 5.6, 1.8, 1.35, "TP/PP", "多卡部署", ec=YELLOW)
    box(ax, 3.0, 3.0, 4.5, 1.25, "优化维度", "精度 / batch / KV Cache\n多卡并行", ec=CYAN)
    box(ax, 8.5, 3.0, 4.5, 1.25, "关键指标", "吞吐 / 延迟 / 显存", ec=GREEN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=CYAN)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=GREEN)
    label(ax, 8, 1.6, "NVIDIA 生态成熟：TensorRT / TRT-LLM 是性能关键", color=TEXT, size=13, weight="bold")
    save(fig, "8.1_NVIDIA_GPU部署/images/9_1_nvidia_gpu.png")


def d_9_2():
    fig, ax = new_fig("9.2 边缘部署 - Jetson Orin", "量化 / 流式 / 功耗")
    box(ax, 0.8, 5.6, 2.8, 1.35, "Orin Nano", "7 - 15W\n统一内存", ec=CYAN)
    box(ax, 4.0, 5.6, 2.8, 1.35, "Orin NX", "10 - 25W\n中等算力", ec=GREEN)
    box(ax, 7.2, 5.6, 2.8, 1.35, "Orin AGX", "15 - 60W\n最高算力", ec=ORANGE)
    box(ax, 10.4, 5.6, 2.8, 1.35, "INT4 / AWQ", "显存压缩", ec=PURPLE)
    box(ax, 13.6, 5.6, 1.8, 1.35, "Stream", "低 TBT", ec=YELLOW)
    box(ax, 3.0, 3.0, 4.5, 1.25, "权衡", "精度 vs 延迟 vs 功耗", ec=RED)
    box(ax, 8.5, 3.0, 4.5, 1.25, "指标", "TTFT / TBT / 功耗", ec=GREEN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=RED)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=GREEN)
    label(ax, 8, 1.6, "边缘部署核心：在满足功耗墙的前提下尽量降低 TBT", color=TEXT, size=13, weight="bold")
    save(fig, "8.2_边缘部署_Jetson_Orin/images/9_2_jetson_orin.png")


def d_9_3():
    fig, ax = new_fig("9.3 Ascend NPU 适配", "CANN / ATC / 算子迁移")
    box(ax, 0.8, 5.6, 2.8, 1.35, "CANN", "达芬奇架构\n软件栈", ec=ORANGE)
    box(ax, 4.0, 5.6, 2.8, 1.35, "ATC", "ONNX → .om\n离线模型", ec=CYAN)
    box(ax, 7.2, 5.6, 2.8, 1.35, "AscendCL", "推理运行时\n类 CUDA", ec=GREEN)
    box(ax, 10.4, 5.6, 2.8, 1.35, "TBE", "自定义算子\nDSL", ec=PURPLE)
    box(ax, 13.6, 5.6, 1.8, 1.35, "msprof", "性能分析", ec=YELLOW)
    box(ax, 3.0, 3.0, 4.5, 1.25, "迁移要点", "数据排布 / 动态 shape\n精度差异", ec=RED)
    box(ax, 8.5, 3.0, 4.5, 1.25, "调优方向", "算子融合 / 精度\n内存排布", ec=GREEN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=RED)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=GREEN)
    label(ax, 8, 1.6, "Ascend 迁移关键：CANN 版本一致 + 算子支持清单", color=TEXT, size=13, weight="bold")
    save(fig, "8.3_Ascend_NPU适配/images/9_3_ascend_npu.png")


def d_9_4():
    fig, ax = new_fig("9.4 自研芯片算子适配", "接口对齐 / 一致性验证")
    box(ax, 0.8, 5.6, 2.8, 1.35, "CPU Reference", "黄金标准", ec=CYAN)
    box(ax, 4.0, 5.6, 2.8, 1.35, "Kernel", "自研芯片\n微码/DMA", ec=GREEN)
    box(ax, 7.2, 5.6, 2.8, 1.35, "Interface", "对齐 PyTorch\nONNX", ec=ORANGE)
    box(ax, 10.4, 5.6, 2.8, 1.35, "Consistency", "多平台误差\n对比", ec=PURPLE)
    box(ax, 13.6, 5.6, 1.8, 1.35, "Profiling", "性能定位", ec=YELLOW)
    box(ax, 3.0, 3.0, 4.5, 1.25, "接口字段", "shape / dtype\nstream / attr", ec=CYAN)
    box(ax, 8.5, 3.0, 4.5, 1.25, "误差指标", "abs / relative\ncosine", ec=GREEN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=CYAN)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=GREEN)
    label(ax, 8, 1.6, "自研芯片：先正确、再对齐、最后优化性能", color=TEXT, size=13, weight="bold")
    save(fig, "8.4_自研芯片算子适配/images/9_4_custom_chip.png")


def d_9_5():
    fig, ax = new_fig("9.5 Benchmark 方法论", "统一标准 / 自动报告")
    box(ax, 0.8, 5.6, 2.8, 1.35, "统一配置", "模型/精度/输入", ec=CYAN)
    box(ax, 4.0, 5.6, 2.8, 1.35, "Warmup", "预热 5-10 轮", ec=GREEN)
    box(ax, 7.2, 5.6, 2.8, 1.35, "多轮采样", "去抖动", ec=ORANGE)
    box(ax, 10.4, 5.6, 2.8, 1.35, "指标", "延迟/吞吐/功耗", ec=PURPLE)
    box(ax, 13.6, 5.6, 1.8, 1.35, "报告", "MD/CSV/图表", ec=YELLOW)
    box(ax, 3.0, 3.0, 4.5, 1.25, "陷阱", "动态 shape / 未 warmup\n忽略功耗", ec=RED)
    box(ax, 8.5, 3.0, 4.5, 1.25, "价值", "选型依据 / 优化追踪", ec=GREEN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=RED)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=GREEN)
    label(ax, 8, 1.6, "Benchmark 不是跑分，而是为决策提供可信数据", color=TEXT, size=13, weight="bold")
    save(fig, "8.5_Benchmark方法论/images/9_5_benchmark.png")


def d_resume_stack():
    fig, ax = new_fig("简历项目：多平台适配", "端侧部署 + Benchmark + 算子一致性")
    items = [
        ("Jetson 7B 部署", "INT4 / 流式 / 25W", CYAN),
        ("多平台 Benchmark", "A100/Orin/Ascend", GREEN),
        ("自研芯片算子", "PPU/P1X 一致性", ORANGE),
    ]
    y = 6.5
    for title, body, color in items:
        box(ax, 1.0, y, 4.2, 1.0, title, body, fc="#0f1b31", ec=color)
        y -= 1.35
    evidence = [
        ("量化收益", "14GB → 4GB"),
        ("效率提升", "测试效率 +40%"),
        ("误差收敛", "1e-2 → 1e-4"),
    ]
    x = 6.2
    for title, body in evidence:
        box(ax, x, 6.5, 2.5, 3.5, title, body, ec=BLUE)
        x += 2.85
    label(ax, 8, 1.1, "简历 bullet = 平台 + 技术方案 + 量化指标 + 业务价值", color=TEXT, size=13)
    save(fig, "简历项目/images/multi_platform_project_stack.png")


def main():
    cover()
    d_9_1()
    d_9_2()
    d_9_3()
    d_9_4()
    d_9_5()
    d_resume_stack()


if __name__ == "__main__":
    main()
