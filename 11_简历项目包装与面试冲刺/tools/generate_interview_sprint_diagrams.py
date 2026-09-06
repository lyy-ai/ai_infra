#!/usr/bin/env python3
"""生成面试冲刺专题演示图。运行：
python tools/generate_interview_sprint_diagrams.py
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
    fig, ax = new_fig("简历项目包装与面试冲刺", "简历写作 → 高频题拆解 → 模拟面试 → 复盘迭代")
    items = [("简历写作", "STAR + 量化", CYAN), ("高频题", "六大模块 19 题", GREEN),
             ("模拟面试", "四岗位专项", ORANGE), ("系统设计", "结构化答题", PURPLE),
             ("复盘", "错题本迭代", RED)]
    x = 0.9
    for name, body, color in items:
        box(ax, x, 4.4, 2.75, 1.35, name, body, fc="#0f1b31", ec=color)
        x += 3.0
    label(ax, 8, 3.5, "技术 → 表达 → 演练 → offer", color=TEXT, size=15, weight="bold")
    save(fig, "images/interview_sprint_cover.png")


def d_1():
    fig, ax = new_fig("1. 简历写作指导", "STAR 法则 + 四条黄金法则")
    star = [("S", "背景\n业务+约束", CYAN), ("T", "任务\n你负责什么", GREEN),
            ("A", "行动\n技术栈关键词", ORANGE), ("R", "成果\n全部量化", RED)]
    x = 1.0
    for s, body, color in star:
        box(ax, x, 5.4, 3.2, 1.4, s, body, fc="#0f1b31", ec=color)
        if x < 11:
            arrow(ax, x + 3.2, 6.1, x + 3.5, 6.1, color=MUTED, lw=1.5, alpha=0.6)
        x += 3.6
    rules = ["量化成果", "突出技术栈", "业务价值", "分层描述"]
    x = 1.0
    for r in rules:
        box(ax, x, 3.0, 3.2, 1.0, r, "", ec=BLUE)
        x += 3.6
    label(ax, 8, 1.8, "问题→行动→结果；数字真实可复现；按 JD 定制", color=TEXT, size=13, weight="bold")
    save(fig, "11.1_简历写作指导/images/1_resume_writing.png")


def d_2():
    fig, ax = new_fig("2. 高频面试题拆解", "六大模块答题框架")
    mods = [("算子", "GEMM / FA / Bank", CYAN), ("编译器", "Pass / TVM / 动态shape", GREEN),
            ("Runtime", "内存池 / Graph / 调度", ORANGE), ("量化", "PTQ / SQ / 精度", PURPLE),
            ("分布式", "ZeRO / 3D / 显存", YELLOW), ("LLM推理", "PA / CB / KV / SD", RED)]
    x, y = 0.9, 5.6
    for name, body, color in mods:
        box(ax, x, y, 4.6, 1.25, name, body, fc="#0f1b31", ec=color)
        y -= 1.5
        if y < 1.4 and x < 6:
            x, y = 8.5, 5.6
    label(ax, 8, 0.8, "一句话框架 → 分层展开 → 准备追问 → 联系自己项目", color=TEXT, size=13, weight="bold")
    save(fig, "11.2_高频面试题拆解/images/2_interview_questions.png")


def d_3():
    fig, ax = new_fig("3. 模拟面试", "四岗位专项 + 系统设计")
    roles = [("算子岗", "深挖+手写kernel", CYAN), ("编译器岗", "深挖+架构问答", GREEN),
             ("推理框架岗", "深挖+vLLM场景", ORANGE), ("训练岗", "深挖+拓扑计算", PURPLE)]
    x = 0.9
    for name, body, color in roles:
        box(ax, x, 5.4, 3.5, 1.35, name, body, fc="#0f1b31", ec=color)
        x += 3.8
    steps = ["澄清需求", "高层架构", "模块深入", "权衡扩展", "总结"]
    x = 0.9
    for i, s in enumerate(steps):
        box(ax, x, 2.9, 2.75, 1.0, f"{i+1}", s, ec=BLUE)
        x += 3.0
    label(ax, 8, 1.9, "系统设计 = 澄清 → 架构 → 深入 → 权衡；每次面试后复盘迭代", color=TEXT, size=13, weight="bold")
    save(fig, "11.3_模拟面试/images/3_mock_interview.png")


def d_resume_stack():
    fig, ax = new_fig("简历自查清单", "投递前最后一遍检查")
    checks = [
        ("每条 bullet 有数字", GREEN), ("JD 关键词重合 >70%", CYAN),
        ("业务价值说清楚", ORANGE), ("背景→职责→成果", PURPLE),
        ("数字能现场复算", YELLOW), ("3 个深挖故事备好", RED),
    ]
    y = 6.3
    for i, (text, color) in enumerate(checks):
        box(ax, 1.5, y, 12.5, 0.85, f"✓ {text}", "", fc="#0f1b31", ec=color)
        y -= 1.0
    label(ax, 8, 0.7, "面试能力 = 知识 × 表达 × 演练次数", color=TEXT, size=13, weight="bold")
    save(fig, "简历项目/images/interview_checklist.png")


def main():
    cover()
    d_1()
    d_2()
    d_3()
    d_resume_stack()


if __name__ == "__main__":
    main()
