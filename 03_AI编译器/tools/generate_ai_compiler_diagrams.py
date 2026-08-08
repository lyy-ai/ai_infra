#!/usr/bin/env python3
"""生成 AI 编译器专题演示图。运行：python tools/generate_ai_compiler_diagrams.py"""
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
    fig, ax = new_fig("AI 编译器", "从编译原理到工程实践：前端 → 中端 Pass → 后端 Codegen → 多平台部署 → 编译耗时优化")
    items = [
        ("4.1", "AI 编译器基础", CYAN), ("4.2", "TVM 架构", GREEN), ("4.3", "编译前端", ORANGE),
        ("4.4", "中端 Pass", PURPLE), ("4.5", "编译后端", YELLOW), ("4.6", "ONNX→TRT→TVM", RED),
        ("4.7", "Laser 框架", BLUE), ("4.8", "Relay→Relax", CYAN), ("4.9", "编译耗时优化", GREEN),
    ]
    x = 0.8
    for num, name, color in items:
        box(ax, x, 4.4, 1.55, 1.25, num, name, fc="#0f1b31", ec=color)
        x += 1.75
    arrow(ax, 1.6, 3.75, 14.4, 3.75, color=MUTED, lw=1.4, style="-", ms=1, alpha=0.6)
    label(ax, 8, 3.35, "Pipeline: 前端解析 → 中端图优化 → 后端代码生成 → 产物打包 → 性能/耗时优化", color=TEXT, size=15, weight="bold")
    save(fig, "images/ai_compiler_cover.png")


def d_4_1_pipeline():
    fig, ax = new_fig("4.1 AI 编译器 Pipeline", "前端 → 中端 → 后端，量化/HPC/多平台协同")
    box(ax, 0.8, 5.6, 4.2, 1.35, "FrontEnd", "ONNX / TorchScript / 动态 shape / 自定义算子", ec=BLUE)
    box(ax, 5.8, 5.6, 4.2, 1.35, "MiddleEnd", "DCE / FoldConstant / FuseOps / Layout", ec=YELLOW)
    box(ax, 10.8, 5.6, 4.2, 1.35, "BackEnd", "CUDA / CUTLASS / TensorRT / NPU / CPU", ec=GREEN)
    arrow(ax, 5.0, 6.25, 5.8, 6.25)
    arrow(ax, 10.0, 6.25, 10.8, 6.25)
    box(ax, 3.0, 3.0, 3.2, 1.25, "量化", "Q/DQ → 融合 INT8 kernel", ec=CYAN)
    box(ax, 6.4, 3.0, 3.2, 1.25, "HPC", "CUTLASS / cuBLAS / oneDNN", ec=ORANGE)
    box(ax, 9.8, 3.0, 3.2, 1.25, "多平台", "GPU / NPU / CPU 同出", ec=PURPLE)
    arrow(ax, 2.5, 5.6, 4.0, 4.25, color=CYAN)
    arrow(ax, 7.2, 5.6, 8.0, 4.25, color=ORANGE)
    arrow(ax, 12.5, 5.6, 11.0, 4.25, color=PURPLE)
    label(ax, 8, 1.8, "核心：统一 IR + 可复用 Pass，把模型自动转换为高性能、可移植硬件代码", color=TEXT, size=14, weight="bold")
    save(fig, "3.1_AI编译器基础/images/4_1_ai_compiler_pipeline.png")


def d_4_2_tvm():
    fig, ax = new_fig("4.2 TVM 架构", "Relay 静态图 / Relax 动态图 / TIR 底层 IR / Runtime")
    layers = [
        ("FrontEnd", "ONNX / PyTorch / TensorFlow", BLUE, 6.8),
        ("Relay / Relax", "高层 IR：静态图 / 动态 shape", CYAN, 5.2),
        ("TIR + Schedule", "底层 IR：循环 / Buffer / 线程", GREEN, 3.6),
        ("Runtime / RPC", "Module / PackedFunc / DeviceAPI", ORANGE, 2.0),
    ]
    for name, body, color, y in layers:
        box(ax, 3.0, y, 10.0, 1.15, name, body, fc="#0f1b31", ec=color)
    for y in [6.8, 5.2, 3.6]:
        arrow(ax, 8.0, y, 8.0, y - 0.55, color=CYAN, lw=1.8)
    label(ax, 8, 0.8, "Relax 用 call_tir 显式连接 TIR，Relay 是隐式 lower", color=TEXT, size=13, weight="bold")
    save(fig, "3.2_TVM架构拆解/images/4_2_tvm_architecture.png")


def d_4_3_frontend():
    fig, ax = new_fig("4.3 编译前端", "格式解析 + 算子映射 + 动态 shape + 自定义算子")
    box(ax, 0.8, 5.6, 3.0, 1.35, "ONNX / TorchScript", "模型文件输入", ec=BLUE)
    box(ax, 4.2, 5.6, 3.0, 1.35, "Parser", "图结构 / 节点 / 边 / 权重", ec=CYAN)
    box(ax, 7.6, 5.6, 3.0, 1.35, "Op Mapping", "算子名 → TVM 算子", ec=YELLOW)
    box(ax, 11.0, 5.6, 3.8, 1.35, "IRModule", "Relay/Relax + 动态 shape", ec=GREEN)
    arrow(ax, 3.8, 6.25, 4.2, 6.25)
    arrow(ax, 7.2, 6.25, 7.6, 6.25)
    arrow(ax, 10.6, 6.25, 11.0, 6.25)
    box(ax, 3.0, 3.0, 4.0, 1.25, "动态 Shape", "batch / seq_len 符号化", ec=CYAN)
    box(ax, 8.0, 3.0, 4.0, 1.25, "自定义算子", "convert_map / compute / schedule", ec=ORANGE)
    arrow(ax, 5.5, 5.6, 5.5, 4.25, color=CYAN)
    arrow(ax, 10.0, 5.6, 10.0, 4.25, color=ORANGE)
    label(ax, 8, 1.6, "前端失败常见原因：算子未映射、动态 shape 未声明、自定义算子未注册", color=TEXT, size=13, weight="bold")
    save(fig, "3.3_编译前端/images/4_3_frontend_pipeline.png")


def d_4_4_pass():
    fig, ax = new_fig("4.4 编译中端 - 图优化 Pass", "DCE / 常量折叠 / 融合 / 布局 / 内存规划")
    passes = [
        ("DCE", "死代码消除", CYAN), ("FoldConst", "常量折叠", GREEN),
        ("FuseOps", "算子融合", ORANGE), ("Layout", "布局变换", PURPLE),
        ("Memory", "内存规划", YELLOW), ("Legalize", "后端合法化", RED),
    ]
    x = 0.8
    for name, body, color in passes:
        box(ax, x, 5.6, 2.3, 1.35, name, body, fc="#0f1b31", ec=color)
        x += 2.55
    arrow(ax, 3.0, 6.25, 3.4, 6.25, color=CYAN, lw=1.4, alpha=0.6)
    arrow(ax, 5.5, 6.25, 5.9, 6.25, color=CYAN, lw=1.4, alpha=0.6)
    arrow(ax, 8.05, 6.25, 8.45, 6.25, color=CYAN, lw=1.4, alpha=0.6)
    arrow(ax, 10.6, 6.25, 11.0, 6.25, color=CYAN, lw=1.4, alpha=0.6)
    arrow(ax, 13.15, 6.25, 13.55, 6.25, color=CYAN, lw=1.4, alpha=0.6)
    box(ax, 3.0, 3.0, 4.5, 1.25, "冗余算子", "Cast / Reshape / Tuple / Identity / Transpose", ec=RED)
    box(ax, 8.5, 3.0, 4.0, 1.25, "Pass 顺序", "先 DCE → 再 Fuse → 再 Layout → 再 Legalize", ec=BLUE)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=RED)
    arrow(ax, 11.0, 5.6, 11.0, 4.25, color=BLUE)
    label(ax, 8, 1.6, "MoE 等特殊架构：需定制路由优化、分支合并、动态维度推导 Pass", color=TEXT, size=13, weight="bold")
    save(fig, "3.4_编译中端-图优化Pass/images/4_4_middle_end_pass.png")


def d_4_5_backend():
    fig, ax = new_fig("4.5 编译后端", "CUDA / CUTLASS / TensorRT / NPU / CPU")
    backends = [
        ("CUDA Codegen", "通用可定制", CYAN), ("CUTLASS", "高性能 GEMM", GREEN),
        ("TensorRT", "工业部署", ORANGE), ("NPU SDK", "DSA 指令", PURPLE),
        ("CPU LLVM", "x86/ARM SIMD", YELLOW),
    ]
    x = 0.8
    for name, body, color in backends:
        box(ax, x, 5.6, 2.9, 1.35, name, body, fc="#0f1b31", ec=color)
        x += 3.1
    box(ax, 3.0, 3.0, 4.5, 1.25, "TIR Lowering", "高层 IR → 循环/Buffer/线程", ec=BLUE)
    box(ax, 8.5, 3.0, 4.5, 1.25, "Codegen + Pack", "CUDA C++ / nvcc / .so", ec=GREEN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=BLUE)
    arrow(ax, 11.0, 5.6, 11.0, 4.25, color=GREEN)
    label(ax, 8, 1.6, "选择后端 = 权衡性能/可控性/动态 shape/可调试性", color=TEXT, size=13, weight="bold")
    save(fig, "3.5_编译后端/images/4_5_backend_pipeline.png")


def d_4_6_onnx_trt_tvm():
    fig, ax = new_fig("4.6 ONNX → TVM → TensorRT/CUTLASS", "端到端编译 + 动态 batch")
    box(ax, 0.8, 5.6, 2.5, 1.35, "ONNX", "onnx-simplifier", ec=BLUE)
    box(ax, 3.7, 5.6, 2.5, 1.35, "TVM FE", "Relay/Relax IR", ec=CYAN)
    box(ax, 6.6, 5.6, 2.5, 1.35, "MiddleEnd", "DCE/Fuse/Layout", ec=YELLOW)
    box(ax, 9.5, 5.6, 2.5, 1.35, "Partition", "TRT / CUTLASS", ec=ORANGE)
    box(ax, 12.4, 5.6, 2.6, 1.35, "Build", ".so / .trt / .aom", ec=GREEN)
    arrow(ax, 3.3, 6.25, 3.7, 6.25)
    arrow(ax, 6.2, 6.25, 6.6, 6.25)
    arrow(ax, 9.1, 6.25, 9.5, 6.25)
    arrow(ax, 12.0, 6.25, 12.4, 6.25)
    box(ax, 3.0, 3.0, 4.5, 1.25, "Relax 动态", "原生符号 shape", ec=CYAN)
    box(ax, 8.5, 3.0, 4.5, 1.25, "Relay Bucket", "预编译多个尺寸", ec=GREEN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=CYAN)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=GREEN)
    label(ax, 8, 1.6, "TensorRT 需 explicit batch + profile；Relax 原生动态更适合 LLM", color=TEXT, size=13, weight="bold")
    save(fig, "3.6_ONNX到TensorRT到TVM编译流程/images/4_6_onnx_tensorrt_tvm_pipeline.png")


def d_4_7_laser():
    fig, ax = new_fig("4.7 Laser 编译框架", "多平台同出 + HPC Plugin + CI/CD")
    stages = [
        ("Loader", "ONNX/PT", BLUE), ("IR Builder", "Relay/Relax", CYAN),
        ("Pass", "DCE/Fuse/Layout", YELLOW), ("Backend", "GPU/NPU/CPU", ORANGE),
        ("Codegen", "CUDA/TRT/NPU", GREEN), ("Pack", ".so/.aom", PURPLE),
        ("CI/CD", "自动触发", RED),
    ]
    x = 0.8
    for name, body, color in stages:
        box(ax, x, 5.6, 2.0, 1.35, name, body, fc="#0f1b31", ec=color)
        x += 2.15
    arrow(ax, 2.8, 6.25, 3.0, 6.25, color=CYAN, lw=1.4, alpha=0.6)
    arrow(ax, 5.0, 6.25, 5.15, 6.25, color=CYAN, lw=1.4, alpha=0.6)
    arrow(ax, 7.15, 6.25, 7.3, 6.25, color=CYAN, lw=1.4, alpha=0.6)
    arrow(ax, 9.3, 6.25, 9.45, 6.25, color=CYAN, lw=1.4, alpha=0.6)
    arrow(ax, 11.45, 6.25, 11.6, 6.25, color=CYAN, lw=1.4, alpha=0.6)
    arrow(ax, 13.6, 6.25, 13.75, 6.25, color=CYAN, lw=1.4, alpha=0.6)
    box(ax, 3.0, 3.0, 4.5, 1.25, "HPC Plugin", "CUTLASS/cuBLAS/oneDNN", ec=ORANGE)
    box(ax, 8.5, 3.0, 4.5, 1.25, "多平台同出", "同 IR 多后端产物", ec=GREEN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=ORANGE)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=GREEN)
    label(ax, 8, 1.6, "核心价值：一份模型配置，GPU/NPU/CPU 产物并行生成", color=TEXT, size=13, weight="bold")
    save(fig, "3.7_Laser编译框架与自动化编译/images/4_7_laser_pipeline.png")


def d_4_8_migrate():
    fig, ax = new_fig("4.8 Relay → Relax Pass 迁移", "IR 抽象差异 + BlockBuilder + call_tir")
    box(ax, 0.8, 5.6, 3.5, 1.35, "Relay", "静态 / 函数式 / ExprMutator", ec=BLUE)
    box(ax, 6.2, 5.6, 3.5, 1.35, "Relax", "动态 / 绑定式 / BlockBuilder", ec=CYAN)
    box(ax, 11.6, 5.6, 3.4, 1.35, "allspark", "Legalize / call_tir", ec=ORANGE)
    arrow(ax, 4.3, 6.25, 6.2, 6.25, color=CYAN, lw=2.5)
    arrow(ax, 9.7, 6.25, 11.6, 6.25, color=ORANGE, lw=2.5)
    changes = [
        ("relay.Function", "relax.Function + BlockBuilder", CYAN),
        ("relay.Call", "relax.call_tir / call_pure_packed", GREEN),
        ("checked_type", "struct_info", ORANGE),
        ("ExprMutator", "BlockBuilder + pattern rewrite", PURPLE),
    ]
    y = 3.5
    for old, new, color in changes:
        box(ax, 1.0, y, 6.2, 0.65, old, "", ec=color, title_color=MUTED, body_color=MUTED, lw=1.2)
        box(ax, 8.0, y, 6.8, 0.65, new, "", ec=color, title_color=MUTED, body_color=MUTED, lw=1.2)
        arrow(ax, 7.2, y + 0.32, 8.0, y + 0.32, color=color, lw=1.5)
        y -= 0.85
    label(ax, 8, 0.9, "迁移不是 API 改名，而是 IR 语义和 Pass 写法的重新适配", color=TEXT, size=13, weight="bold")
    save(fig, "3.8_Relay到Relax_Pass迁移实战/images/4_8_relay_relax_migrate.png")


def d_4_9_compile_time():
    fig, ax = new_fig("4.9 编译耗时分析", "优化等级 / 分支爆炸 / 寄存器压力 / 并发资源")
    bottlenecks = [
        ("-O2 → -O0", "中间产物编译等级", CYAN),
        ("21 层 where", "tir.Simplify 路径爆炸", ORANGE),
        ("main 常量密集", "寄存器压力 / IR 体积", RED),
        ("16 核并发", "CPU/内存资源抢占", PURPLE),
    ]
    x = 1.0
    for name, body, color in bottlenecks:
        box(ax, x, 5.6, 3.2, 1.35, name, body, fc="#0f1b31", ec=color)
        x += 3.5
    box(ax, 3.0, 3.0, 4.5, 1.25, "日志分析", "阶段拆分 / time -v / perf", ec=BLUE)
    box(ax, 8.5, 3.0, 4.5, 1.25, "根因定位", "单任务 vs 多任务资源", ec=GREEN)
    arrow(ax, 5.25, 5.6, 5.25, 4.25, color=BLUE)
    arrow(ax, 10.75, 5.6, 10.75, 4.25, color=GREEN)
    label(ax, 8, 1.6, "先拆分阶段、再定位根因，最后针对性优化，不要盲目加机器", color=TEXT, size=13, weight="bold")
    save(fig, "3.9_编译耗时优化方案/images/4_9_compile_time_analysis.png")


def d_resume_stack():
    fig, ax = new_fig("简历项目：AI 编译器", "用最小工程证据支撑编译全链路闭环")
    layers = [("前端解析", CYAN), ("中端 Pass", GREEN), ("后端 Codegen", ORANGE), ("多平台部署", PURPLE), ("编译耗时优化", RED)]
    y = 6.7
    for name, color in layers:
        box(ax, 1.0, y, 4.0, 0.75, name, "", fc="#0f1b31", ec=color)
        y -= 0.9
    evidence = [
        ("Python 脚本", "ONNX 解析 / Pass 改写 / 后端编译"),
        ("TVM demo", "Relay/Relax/TIR 对比 / call_tir"),
        ("Metrics", "kernel 数量 / 延迟 / 编译时间 / 数值误差"),
    ]
    y = 6.5
    for name, body in evidence:
        box(ax, 6.2, y, 8.6, 0.95, name, body, ec=BLUE)
        y -= 1.35
    label(ax, 8, 1.1, "简历 bullet = 分层链路 + 指标口径 + 可复现脚本 + 数值正确性保障", color=TEXT, size=13)
    save(fig, "简历项目/images/ai_compiler_project_stack.png")


def main():
    cover()
    d_4_1_pipeline()
    d_4_2_tvm()
    d_4_3_frontend()
    d_4_4_pass()
    d_4_5_backend()
    d_4_6_onnx_trt_tvm()
    d_4_7_laser()
    d_4_8_migrate()
    d_4_9_compile_time()
    d_resume_stack()


if __name__ == "__main__":
    main()
