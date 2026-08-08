#!/usr/bin/env python3
"""生成综合项目实战演示图。运行：
python tools/generate_comprehensive_project_diagrams.py
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


def box(ax, x, y, w, h, title, body="", fc=PANEL, ec=CYAN, lw=1.8):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                           linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", color=TEXT, fontsize=15, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center", color=MUTED, fontsize=10.5)
    return patch


def arrow(ax, x1, y1, x2, y2, color=CYAN, lw=2.2, style="-|>", ms=18, alpha=1.0):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                          linewidth=lw, color=color, alpha=alpha)
    ax.add_patch(arr)
    return arr


def label(ax, x, y, text, color=MUTED, size=11, ha="center", va="center", weight="normal"):
    ax.text(x, y, text, color=color, fontsize=size, ha=ha, va=va, fontweight=weight)


def cover():
    fig, ax = new_fig("综合项目：MMBEV 端到端多平台部署优化",
                      "业务模型分析 → HPC算子 → 编译优化 → Runtime → 量化 → 推理引擎 → 验证")
    stages = [
        ("模型分析", CYAN), ("HPC算子", GREEN), ("编译优化", ORANGE),
        ("Runtime", PURPLE), ("INT8量化", YELLOW), ("推理引擎", RED), ("多平台验证", BLUE),
    ]
    x = 0.6
    for name, color in stages:
        box(ax, x, 4.4, 2.05, 1.25, name, "", fc="#0f1b31", ec=color)
        if x < 13:
            arrow(ax, x + 2.05, 5.0, x + 2.2, 5.0, color=MUTED, lw=1.5, alpha=0.6)
        x += 2.2
    label(ax, 8, 3.5, "Orin 100ms → 30ms | 吞吐 ×3 | 显存 -30% | INT8 损失 <1% | 利用率 85%",
          color=TEXT, size=15, weight="bold")
    save(fig, "images/comprehensive_project_cover.png")


def d_pipeline():
    fig, ax = new_fig("端到端延迟演进（Orin）", "各优化阶段贡献")
    stages = [("原始", 100, RED), ("HPC算子", 60, GREEN), ("编译", 52, ORANGE),
              ("Runtime", 42, PURPLE), ("INT8", 32, YELLOW), ("引擎", 30, CYAN)]
    x = 0.8
    for name, lat, color in stages:
        h = lat / 100.0 * 4.0
        box(ax, x, 1.8, 2.1, h, f"{lat}ms", name, fc="#0f1b31", ec=color)
        if x < 12:
            arrow(ax, x + 2.1, 6.6, x + 2.3, 6.6, color=MUTED, lw=1.5, alpha=0.6)
        x += 2.4
    label(ax, 8, 1.2, "目标 < 50ms ✓ | 100ms → 30ms | 吞吐提升 3 倍", color=TEXT, size=13, weight="bold")
    save(fig, "10.1_综合项目_MMBEV端到端多平台部署优化/images/mmbev_latency_evolution.png")


def d_modules():
    fig, ax = new_fig("五大优化模块", "算子 / 编译 / Runtime / 量化 / 引擎")
    rows = [
        ("HPC算子", "BEV Pooling warp reduce\nLN+SiLU 融合\n跨平台接口对齐", "+65%", GREEN),
        ("编译优化", "Laser 一键编译\n图优化 -23% ops\nplugin 兼容", "CI 2天→2h", ORANGE),
        ("Runtime", "内存池\nCUDA Graph\n异步流水线", "-40% 延迟", PURPLE),
        ("INT8量化", "分层保护\nper-channel\nNDS 损失 0.4%", "×2 速度", YELLOW),
        ("推理引擎", "6/8 路并行\n软硬协同\n三平台验证", "85% 利用率", CYAN),
    ]
    y = 6.3
    for title, body, result, color in rows:
        box(ax, 0.9, y, 8.6, 1.0, title, body, fc="#0f1b31", ec=color)
        box(ax, 10.0, y, 4.8, 1.0, result, "", ec=color)
        y -= 1.15
    label(ax, 8, 0.9, "五个模块环环相扣，全部落到业务指标", color=TEXT, size=13, weight="bold")
    save(fig, "10.1_综合项目_MMBEV端到端多平台部署优化/images/mmbev_modules.png")


def d_validation():
    fig, ax = new_fig("多平台验证矩阵", "功能 / 精度 / 性能 / 一致性 / 稳定性")
    plats = [("Orin", "30ms", "30 FPS", "85%", GREEN),
             ("A100", "10ms", "100 FPS", "85%", CYAN),
             ("Ascend", "12ms", "83 FPS", "82%", ORANGE)]
    x = 1.0
    for name, lat, fps, util, color in plats:
        box(ax, x, 5.2, 4.2, 1.6, name, f"{lat} | {fps}\nutil {util}", fc="#0f1b31", ec=color)
        x += 4.8
    box(ax, 2.0, 2.6, 5.5, 1.25, "精度验证", "NDS 损失 0.4% < 1%\nTensor Diff < 1e-2", ec=GREEN)
    box(ax, 8.5, 2.6, 5.5, 1.25, "稳定性验证", "7×24h 通过\n量产验证 ✓", ec=BLUE)
    label(ax, 8, 1.6, "三平台全部达标，部署自动驾驶业务", color=TEXT, size=13, weight="bold")
    save(fig, "10.1_综合项目_MMBEV端到端多平台部署优化/images/mmbev_validation.png")


def d_resume_stack():
    fig, ax = new_fig("简历项目：MMBEV 端到端部署优化", "一个项目串起全部知识点")
    items = [("算子", "+65%"), ("延迟", "100→30ms"), ("显存", "-30%"),
             ("量化", "×2 加速"), ("利用率", "85%"), ("量产", "通过")]
    x = 0.9
    for title, body in items:
        box(ax, x, 5.6, 2.3, 1.4, title, body, fc="#0f1b31", ec=GREEN)
        x += 2.5
    label(ax, 8, 4.3, "简历 bullet = 业务背景 + 五模块行动 + 量化成果 + 量产验证",
          color=TEXT, size=14, weight="bold")
    label(ax, 8, 2.9, "STAR: 100ms 无法满足实时性 → 负责端到端优化 → 算子/编译/Runtime/量化/引擎 → 30ms + 3×吞吐 + 量产",
          color=MUTED, size=11.5)
    save(fig, "简历项目/images/comprehensive_project_stack.png")


def main():
    cover()
    d_pipeline()
    d_modules()
    d_validation()
    d_resume_stack()


if __name__ == "__main__":
    main()
