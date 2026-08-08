#!/usr/bin/env python3
"""生成企业级工程体系专题演示图。运行：
python tools/generate_enterprise_engineering_diagrams.py
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
    fig, ax = new_fig("企业级工程体系", "从工具到平台：CI/CD / 性能回归 / Profiling / 质量保障 / 监控告警")
    items = [
        ("10.1", "CI/CD", CYAN), ("10.2", "性能回归", GREEN),
        ("10.3", "Profiling", ORANGE), ("10.4", "质量保障", PURPLE),
        ("10.5", "监控告警", RED),
    ]
    x = 1.2
    for num, name, color in items:
        box(ax, x, 4.4, 2.6, 1.25, num, name, fc="#0f1b31", ec=color)
        x += 3.0
    arrow(ax, 2.0, 3.75, 15.0, 3.75, color=MUTED, lw=1.4, style="-", ms=1, alpha=0.6)
    label(ax, 8, 3.35, "代码 → 编译 → 测试 → 回归 → 上线 → 监控", color=TEXT, size=15, weight="bold")
    save(fig, "images/enterprise_engineering_cover.png")


def d_10_1():
    fig, ax = new_fig("10.1 CI/CD 与自动化", "流水线 / 测试 / 版本管理")
    stages = [
        ("Lint", CYAN), ("Build", GREEN), ("Accuracy", ORANGE),
        ("Benchmark", PURPLE), ("Release", YELLOW),
    ]
    x = 0.8
    for name, color in stages:
        box(ax, x, 5.6, 2.6, 1.25, name, "", ec=color)
        if x < 11:
            arrow(ax, x + 2.6, 6.2, x + 3.0, 6.2, color=MUTED, lw=1.5, alpha=0.6)
        x += 3.0
    box(ax, 3.0, 3.0, 4.5, 1.25, "版本管理", "代码 / 模型 / 产物", ec=CYAN)
    box(ax, 8.5, 3.0, 4.5, 1.25, "关键产物", "engine + config\n+ checksum", ec=GREEN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=CYAN)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=GREEN)
    label(ax, 8, 1.6, "AI Infra CI/CD 要保证功能、精度、性能三重正确", color=TEXT, size=13, weight="bold")
    save(fig, "9.1_CI_CD与自动化/images/10_1_cicd.png")


def d_10_2():
    fig, ax = new_fig("10.2 性能回归平台", "Benchmark / 门禁 / 定位")
    box(ax, 0.8, 5.6, 2.8, 1.35, "Trigger", "git / cron\nmanual", ec=CYAN)
    box(ax, 4.0, 5.6, 2.8, 1.35, "Benchmark", "多平台\n多指标", ec=GREEN)
    box(ax, 7.2, 5.6, 2.8, 1.35, "Compare", "vs baseline\n阈值判定", ec=ORANGE)
    box(ax, 10.4, 5.6, 2.8, 1.35, "Block / Pass", "门禁", ec=RED)
    box(ax, 13.6, 5.6, 1.8, 1.35, "Bisect", "定位", ec=PURPLE)
    box(ax, 3.0, 3.0, 4.5, 1.25, "阈值策略", "吞吐/延迟\n>5% block", ec=RED)
    box(ax, 8.5, 3.0, 4.5, 1.25, "定位工具", "git bisect\nNsight", ec=PURPLE)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=RED)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=PURPLE)
    label(ax, 8, 1.6, "性能回归 = 自动检测 + 及时 block + 快速定位", color=TEXT, size=13, weight="bold")
    save(fig, "9.2_性能回归平台/images/10_2_regression.png")


def d_10_3():
    fig, ax = new_fig("10.3 Profiling 平台", "统一 schema / Nsight / 可视化")
    box(ax, 0.8, 5.6, 2.8, 1.35, "Nsight Systems", "timeline", ec=CYAN)
    box(ax, 4.0, 5.6, 2.8, 1.35, "Nsight Compute", "kernel", ec=GREEN)
    box(ax, 7.2, 5.6, 2.8, 1.35, "PyTorch Prof", "Python栈", ec=ORANGE)
    box(ax, 10.4, 5.6, 2.8, 1.35, "msprof", "Ascend", ec=PURPLE)
    box(ax, 13.6, 5.6, 1.8, 1.35, "自研", "custom", ec=YELLOW)
    box(ax, 3.0, 3.0, 4.5, 1.25, "统一 Schema", "event / duration\nstream / commit", ec=CYAN)
    box(ax, 8.5, 3.0, 4.5, 1.25, "可视化", "timeline / flamegraph\n历史趋势", ec=GREEN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=CYAN)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=GREEN)
    label(ax, 8, 1.6, "Profiling 平台把多工具数据变成统一语言", color=TEXT, size=13, weight="bold")
    save(fig, "9.3_Profiling平台/images/10_3_profiling.png")


def d_10_4():
    fig, ax = new_fig("10.4 质量保障系统", "日志 / Tensor Diff / 一致性")
    box(ax, 0.8, 5.6, 2.8, 1.35, "Logs", "结构化\n可检索", ec=CYAN)
    box(ax, 4.0, 5.6, 2.8, 1.35, "Tensor Diff", "精度对比\n逐层热力图", ec=GREEN)
    box(ax, 7.2, 5.6, 2.8, 1.35, "Cross-Platform", "多平台\n一致性", ec=ORANGE)
    box(ax, 10.4, 5.6, 2.8, 1.35, "Threshold", "FP16/INT8\n不同标准", ec=PURPLE)
    box(ax, 13.6, 5.6, 1.8, 1.35, "CI Gate", "自动 block", ec=RED)
    box(ax, 3.0, 3.0, 4.5, 1.25, "日志价值", "定位 + 审计", ec=CYAN)
    box(ax, 8.5, 3.0, 4.5, 1.25, "Diff 价值", "量化/编译\n精度不丢", ec=GREEN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=CYAN)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=GREEN)
    label(ax, 8, 1.6, "质量保障 = 可观测 + 可对比 + 可拦截", color=TEXT, size=13, weight="bold")
    save(fig, "9.4_质量保障系统/images/10_4_quality.png")


def d_10_5():
    fig, ax = new_fig("10.5 监控与告警", "GPU / SLO / 异常检测")
    box(ax, 0.8, 5.6, 2.8, 1.35, "GPU Metrics", "util / mem\ntemp / power", ec=CYAN)
    box(ax, 4.0, 5.6, 2.8, 1.35, "SLO", "P99 / TTFT\nerror rate", ec=GREEN)
    box(ax, 7.2, 5.6, 2.8, 1.35, "异常检测", "阈值 / 基线\n变化率", ec=ORANGE)
    box(ax, 10.4, 5.6, 2.8, 1.35, "Alert", "分级收敛\n带 context", ec=RED)
    box(ax, 13.6, 5.6, 1.8, 1.35, "Dashboard", "Grafana", ec=PURPLE)
    box(ax, 3.0, 3.0, 4.5, 1.25, "目标", "提前发现\n不是事后救火", ec=CYAN)
    box(ax, 8.5, 3.0, 4.5, 1.25, "驱动优化", "数据反馈\n容量规划", ec=GREEN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=CYAN)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=GREEN)
    label(ax, 8, 1.6, "监控告警要 actionable，避免告警疲劳", color=TEXT, size=13, weight="bold")
    save(fig, "9.5_监控与告警/images/10_5_monitoring.png")


def d_resume_stack():
    fig, ax = new_fig("简历项目：企业级工程体系", "平台化能力 + 量化收益")
    items = [
        ("CI/CD 平台", "编译失败率↓", CYAN),
        ("性能回归平台", "定位时间↓", GREEN),
        ("质量监控系统", "跨平台误差↓", ORANGE),
    ]
    y = 6.5
    for title, body, color in items:
        box(ax, 1.0, y, 4.2, 1.0, title, body, fc="#0f1b31", ec=color)
        y -= 1.35
    evidence = [
        ("失败率", "12% → 2%"),
        ("定位", "1天 → 2h"),
        ("误差", "1e-2 → 1e-4"),
    ]
    x = 6.2
    for title, body in evidence:
        box(ax, x, 6.5, 2.5, 3.5, title, body, ec=BLUE)
        x += 2.85
    label(ax, 8, 1.1, "简历 bullet = 平台 + 流程 + 指标 + 业务价值", color=TEXT, size=13)
    save(fig, "简历项目/images/enterprise_engineering_project_stack.png")


def main():
    cover()
    d_10_1()
    d_10_2()
    d_10_3()
    d_10_4()
    d_10_5()
    d_resume_stack()


if __name__ == "__main__":
    main()
