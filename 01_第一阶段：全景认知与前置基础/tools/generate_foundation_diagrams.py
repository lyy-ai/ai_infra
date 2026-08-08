#!/usr/bin/env python3
"""生成第一阶段全景认知与前置基础演示图。运行：
python tools/generate_foundation_diagrams.py
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
    fig, ax = new_fig("第一阶段：全景认知与前置基础", "全景认知（1.1-1.4）+ 前置基础（2.1-2.7）")
    c1 = [("1.1", "AI Infra", CYAN), ("1.2", "大厂组织", GREEN), ("1.3", "岗位路线", ORANGE), ("1.4", "学习路线", PURPLE)]
    x = 1.2
    for num, name, color in c1:
        box(ax, x, 5.5, 3.0, 1.2, num, name, fc="#0f1b31", ec=color)
        x += 3.5
    c2 = [("2.1", "编程", CYAN), ("2.2", "数学", GREEN), ("2.3", "Transformer", ORANGE),
          ("2.4", "PyTorch", PURPLE), ("2.5", "GPU", YELLOW), ("2.6", "集合通信", RED), ("2.7", "工具链", BLUE)]
    x = 0.7
    for num, name, color in c2:
        box(ax, x, 3.2, 2.05, 1.1, num, name, fc="#0f1b31", ec=color)
        x += 2.2
    label(ax, 8, 2.0, "核心思维：任何技术都是在 计算/通信/显存 三角上取舍", color=TEXT, size=15, weight="bold")
    save(fig, "images/foundation_cover.png")


def d_1_1():
    fig, ax = new_fig("1.1 什么是 AI Infra", "五大方向 + 不可能三角")
    dirs = [("训练", "ZeRO/3D并行", CYAN), ("推理", "TRT/vLLM", GREEN), ("编译", "TVM/MLIR", ORANGE),
            ("Runtime", "内存池/Graph", PURPLE), ("芯片适配", "CANN/算子库", YELLOW)]
    x = 0.8
    for name, body, color in dirs:
        box(ax, x, 5.5, 2.85, 1.3, name, body, fc="#0f1b31", ec=color)
        x += 3.05
    box(ax, 5.5, 3.4, 5.0, 1.1, "计算", "FLOPS 上限", ec=CYAN)
    box(ax, 2.2, 1.4, 5.0, 1.1, "通信", "带宽效率", ec=GREEN)
    box(ax, 8.8, 1.4, 5.0, 1.1, "显存", "容量+带宽", ec=ORANGE)
    arrow(ax, 6.0, 3.4, 5.0, 2.5, color=MUTED, lw=1.5, alpha=0.6)
    arrow(ax, 10.0, 3.4, 11.0, 2.5, color=MUTED, lw=1.5, alpha=0.6)
    label(ax, 8, 0.8, "每个优化 = 三角上的取舍（ZeRO 用通信换显存）", color=TEXT, size=13, weight="bold")
    save(fig, "1.1_什么是AI_Infra/images/1_1_what_is_ai_infra.png")


def d_1_2():
    fig, ax = new_fig("1.2 大厂 AI Infra 组织架构", "业务驱动组织形态")
    orgs = [("NVIDIA", "定义标准", CYAN), ("字节AML", "规模迭代", GREEN), ("阿里PAI", "平台产品", ORANGE),
            ("百度", "框架+芯片", PURPLE), ("自动驾驶", "实时量产", YELLOW)]
    y = 6.2
    for name, feat, color in orgs:
        box(ax, 1.5, y, 4.5, 0.9, name, "", fc="#0f1b31", ec=color)
        box(ax, 6.5, y, 8.0, 0.9, feat, "", ec=color)
        y -= 1.1
    label(ax, 8, 0.6, "选团队 = 选技术栈：想去哪类公司，重点准备对应方向", color=TEXT, size=13, weight="bold")
    save(fig, "1.2_大厂AI_Infra组织架构/images/1_2_org_landscape.png")


def d_1_3():
    fig, ax = new_fig("1.3 岗位路线图", "初级 → 中级 → 高级")
    levels = [("初级 20-35k", "部署/TensorRT/CUDA\n会用+基础扎实", CYAN),
              ("中级 35-60k", "Infra/编译器/Serving\n独立负责+原理深度", GREEN),
              ("高级 60-100k", "架构师/平台负责人\n架构+业务转化", ORANGE)]
    x = 0.9
    for name, body, color in levels:
        box(ax, x, 4.4, 4.5, 1.6, name, body, fc="#0f1b31", ec=color)
        if x < 10:
            arrow(ax, x + 4.5, 5.2, x + 4.9, 5.2, color=MUTED, lw=1.5, alpha=0.6)
        x += 4.9
    label(ax, 8, 3.4, "薪资 ∝ 可量化的影响力：延迟降多少、成本省多少", color=TEXT, size=14, weight="bold")
    save(fig, "1.3_AI_Infra岗位路线图/images/1_3_career_roadmap.png")


def d_1_4():
    fig, ax = new_fig("1.4 学习路线总览", "前置基础 → 核心技术 → 项目实战 → 面试")
    stages = [("前置基础", CYAN), ("CUDA/算子", GREEN), ("分布式训练", ORANGE),
              ("推理部署", PURPLE), ("项目实战", YELLOW), ("面试冲刺", RED)]
    x = 0.7
    for name, color in stages:
        box(ax, x, 5.2, 2.35, 1.2, name, "", fc="#0f1b31", ec=color)
        if x < 13:
            arrow(ax, x + 2.35, 5.8, x + 2.55, 5.8, color=MUTED, lw=1.5, alpha=0.6)
        x += 2.55
    label(ax, 8, 3.8, "学习总纲：每学一个技术，填一行取舍表", color=TEXT, size=14, weight="bold")
    label(ax, 8, 2.9, "ZeRO：牺牲通信 → 换取显存 | INT8：牺牲精度 → 换取速度+显存", color=MUTED, size=12)
    save(fig, "1.4_学习路线总览/images/1_4_learning_path.png")


def d_2_1():
    fig, ax = new_fig("2.1 编程语言基础", "Python × C++ 双语世界")
    box(ax, 1.0, 5.4, 6.3, 1.5, "Python 进阶", "装饰器 / 生成器\n多进程(GIL)", ec=CYAN)
    box(ax, 8.6, 5.4, 6.3, 1.5, "C++ 核心", "指针 / 内存管理\n编译链接 / 模板", ec=GREEN)
    box(ax, 1.0, 3.0, 6.3, 1.3, "pybind11", "Python ↔ C++ 桥\n自定义算子之路", ec=ORANGE)
    box(ax, 8.6, 3.0, 6.3, 1.3, "Linux", "命令行 / Shell\n进程 / 环境配置", ec=PURPLE)
    label(ax, 8, 1.9, "框架胶水层用 Python，性能关键路径用 C++", color=TEXT, size=13, weight="bold")
    save(fig, "1.5_编程语言基础/images/2_1_languages.png")


def d_2_2():
    fig, ax = new_fig("2.2 数学基础", "工程所需的最低必要数学")
    items = [("线性代数", "GEMM=2MNK\n分块=tile/TP根基", CYAN),
             ("概率统计", "softmax减max\n交叉熵融合", GREEN),
             ("微积分", "链式法则\n=反向传播=autograd", ORANGE)]
    x = 0.9
    for name, body, color in items:
        box(ax, x, 4.6, 4.5, 1.7, name, body, fc="#0f1b31", ec=color)
        x += 5.0
    label(ax, 8, 3.6, "不推导论文，但要心算维度/显存/FLOPs", color=TEXT, size=13, weight="bold")
    save(fig, "1.6_数学基础/images/2_2_math.png")


def d_2_3():
    fig, ax = new_fig("2.3 Transformer 架构详解", "O(N²) 与 KV Cache 决定优化方向")
    comps = [("Attention", "O(N²)\nFlashAttn存在原因", CYAN), ("FFN", "参数2/3\n量化/TP主战场", GREEN),
             ("RoPE", "旋转编码\n长序列需插值", ORANGE), ("Pre-Norm", "残差+LN\n融合kernel", PURPLE)]
    x = 0.8
    for name, body, color in comps:
        box(ax, x, 5.4, 3.55, 1.5, name, body, fc="#0f1b31", ec=color)
        x += 3.8
    variants = [("MHA", "1×"), ("MQA", "1/h"), ("GQA", "g/h"), ("MLA", "低秩")]
    x = 1.5
    for name, body in variants:
        box(ax, x, 2.9, 3.0, 1.0, name, f"KV Cache {body}", ec=YELLOW)
        x += 3.3
    label(ax, 8, 1.9, "演进主线 = 压缩 KV Cache；decode 是 memory-bound", color=TEXT, size=13, weight="bold")
    save(fig, "1.7_Transformer架构详解/images/2_3_transformer.png")


def d_2_4():
    fig, ax = new_fig("2.4 PyTorch 框架", "Tensor/autograd + Module + 调试")
    items = [("Tensor", "dtype/device\nview vs copy", CYAN), ("autograd", "保存激活\n=训练显存来源", GREEN),
             ("Module", "parameters\nstate_dict", ORANGE), ("调试", "memory_summary\nprofiler", PURPLE)]
    x = 0.8
    for name, body, color in items:
        box(ax, x, 4.8, 3.55, 1.5, name, body, fc="#0f1b31", ec=color)
        x += 3.8
    label(ax, 8, 3.7, "profiler 找 top 算子 = 一切性能优化的第一步", color=TEXT, size=13, weight="bold")
    save(fig, "1.8_PyTorch框架/images/2_4_pytorch.png")


def d_2_5():
    fig, ax = new_fig("2.5 GPU 硬件概论", "存储层次 + Memory Wall")
    mems = [("寄存器", "~1 cycle", CYAN), ("Shared/L1", "~30 cyc", GREEN),
            ("L2", "~200 cyc", ORANGE), ("HBM", "~400+ cyc", PURPLE), ("DDR", "微秒级", RED)]
    x = 0.8
    for name, body, color in mems:
        box(ax, x, 5.4, 2.85, 1.3, name, body, fc="#0f1b31", ec=color)
        x += 3.05
    box(ax, 2.0, 2.6, 5.5, 1.3, "Memory Wall", "算术强度 < 拐点\n→ memory-bound", ec=YELLOW)
    box(ax, 8.5, 2.6, 5.5, 1.3, "互联拓扑", "NVLink 600GB/s\nIB 25GB/s → 差10x", ec=BLUE)
    label(ax, 8, 1.5, "优化第一性原理：数据尽量留在离计算近的地方", color=TEXT, size=13, weight="bold")
    save(fig, "1.9_GPU硬件概论/images/2_5_gpu_hardware.png")


def d_2_6():
    fig, ax = new_fig("2.6 集合通信基础", "原语 + 算法 + NCCL")
    prims = [("AllReduce", "DDP梯度", CYAN), ("AllGather", "FSDP参数", GREEN), ("ReduceScatter", "ZeRO分片", ORANGE)]
    x = 1.2
    for name, body, color in prims:
        box(ax, x, 5.4, 4.2, 1.3, name, body, fc="#0f1b31", ec=color)
        x += 4.6
    box(ax, 2.0, 2.7, 5.5, 1.3, "Ring", "≈2S 与N无关\n带宽最优", ec=YELLOW)
    box(ax, 8.5, 2.7, 5.5, 1.3, "Tree", "log N 步\n延迟最优", ec=PURPLE)
    label(ax, 8, 1.6, "AllReduce = ReduceScatter + AllGather", color=TEXT, size=13, weight="bold")
    save(fig, "1.10_集合通信基础/images/2_6_collective_comm.png")


def d_2_7():
    fig, ax = new_fig("2.7 工程工具链", "调试四件套 + Git + 芯片")
    tools = [("gdb", "崩溃", CYAN), ("perf", "CPU热点", GREEN), ("nsys", "timeline", ORANGE), ("ncu", "kernel细节", PURPLE)]
    x = 1.2
    for name, body, color in tools:
        box(ax, x, 5.4, 3.2, 1.3, name, body, fc="#0f1b31", ec=color)
        x += 3.6
    chips = [("A100", CYAN), ("Orin", GREEN), ("Ascend", ORANGE), ("P1X/PPU", PURPLE)]
    x = 1.2
    for name, color in chips:
        box(ax, x, 2.9, 3.2, 1.0, name, "", ec=color)
        x += 3.6
    label(ax, 8, 1.9, "调试决策树：nsys全局 → ncu kernel → perf CPU → gdb 崩溃", color=TEXT, size=13, weight="bold")
    save(fig, "1.11_工程工具链/images/2_7_toolchain.png")


def d_resume():
    fig, ax = new_fig("前置基础自查清单", "必会心算 + 概念一句话")
    calcs = [("训练显存", "≈16P"), ("KV/token", "2L·kvh·d·2B"), ("Ring通信", "≈2S"),
             ("decode下界", "权重/带宽"), ("roofline", "算力/带宽")]
    x = 0.9
    for name, body in calcs:
        box(ax, x, 5.4, 2.85, 1.3, name, body, fc="#0f1b31", ec=GREEN)
        x += 3.05
    concepts = ["不可能三角", "FlashAttention", "GQA", "Memory Wall", "NVLink vs IB"]
    x = 0.9
    for name in concepts:
        box(ax, x, 3.1, 2.85, 1.0, f"✓ {name}", "", ec=CYAN)
        x += 3.05
    label(ax, 8, 2.1, "基础融入项目 bullet 的技术深度，而不是单独罗列", color=TEXT, size=13, weight="bold")
    save(fig, "简历项目/images/foundation_checklist.png")


def main():
    cover()
    d_1_1(); d_1_2(); d_1_3(); d_1_4()
    d_2_1(); d_2_2(); d_2_3(); d_2_4(); d_2_5(); d_2_6(); d_2_7()
    d_resume()


if __name__ == "__main__":
    main()
