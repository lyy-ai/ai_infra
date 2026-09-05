# 2.9 Triton 编程实战：Triton vs CUDA vs cuBLAS 选型决策估算器
#
# 运行：
#   cd /data/ai_infra/02_CUDA编程与HPC高性能计算
#   /data/qwen35_env/bin/python 2.9_Triton编程实战/triton_vs_cuda_decision.py
#
# 说明：本脚本是一个"决策树式"教学估算器，输入算子特征（是否需要融合、
# shape 是否多变、性能要求档位、是否依赖新硬件特性），输出建议的技术路线
# 及理由。规则来自 2.9 节讲义的工程实践总结，仅供学习参考。
import unicodedata

CHOICES = ("Triton", "手写 CUDA / CUTLASS", "直接调 cuBLAS/cuDNN")


def pad(s, width):
    """按显示宽度对齐（中文算 2 列），保证表格整齐。"""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return s + " " * max(0, width - w)


def decide(op):
    """决策树：按优先级逐级判断，返回 (建议, [理由列表])。"""
    reasons = []

    # 第 0 级：标准 GEMM / 卷积，有成熟闭源库
    if op["is_standard_gemm"]:
        reasons.append("标准 GEMM 是 cuBLAS 的主场，数十年调优 + 按 shape 自动选核")
        reasons.append("Triton/手写只在 shape 极特殊或需要融合前后操作时才有意义")
        return CHOICES[2], reasons

    # 第 1 级：依赖 Hopper/Blackwell 新硬件特性
    if op["needs_new_hw_feature"]:
        reasons.append("依赖 TMA / Thread Block Cluster / warp specialization 等新特性")
        reasons.append("Triton 对硬件新特性的支持滞后于 CUDA C++，编译器抽象成为束缚")
        if op["perf_tier"] == "极致":
            reasons.append("性能档位为'极致'，最后一英里的寄存器/warp 级控制只有手写能做")
        return CHOICES[1], reasons

    # 第 2 级：极致性能 + compute-bound 大算子
    if op["perf_tier"] == "极致" and not op["is_fused"]:
        reasons.append("非融合的核心大算子 + 极致性能要求，5% 差距在数据中心规模下不可接受")
        reasons.append("建议 CUTLASS 模板起步，而非完全从零手写")
        return CHOICES[1], reasons

    # 第 3 级：Triton 的甜区
    if op["is_fused"]:
        reasons.append("融合算子多为 memory-bound，一次读写即可打满带宽，Triton 与手写几乎无差距")
    else:
        reasons.append("非极致性能档位的算子，Triton 通常可达手写的 80-95%")
    if op["shape_varies"]:
        reasons.append("shape 多变（如 LLM 解码），@triton.autotune 按 shape 缓存配置，省去手工调参")
    if op["needs_portability"]:
        reasons.append("需要跨 NVIDIA/AMD/国产芯片部署，Triton 一份源码多后端编译")
    reasons.append("Python 生态与框架同语言，开发/调试/迭代成本最低")
    return CHOICES[0], reasons


CASES = [
    {
        "name": "RMSNorm + residual 融合（vLLM 推理）",
        "is_standard_gemm": False, "is_fused": True, "shape_varies": True,
        "perf_tier": "正常", "needs_new_hw_feature": False, "needs_portability": False,
    },
    {
        "name": "fused MoE（路由 + 专家 GEMM 融合）",
        "is_standard_gemm": False, "is_fused": True, "shape_varies": True,
        "perf_tier": "高", "needs_new_hw_feature": False, "needs_portability": False,
    },
    {
        "name": "训练主干 BF16 GEMM（4096 方阵）",
        "is_standard_gemm": True, "is_fused": False, "shape_varies": False,
        "perf_tier": "极致", "needs_new_hw_feature": False, "needs_portability": False,
    },
    {
        "name": "FlashAttention-3 级极致 attention（H100）",
        "is_standard_gemm": False, "is_fused": True, "shape_varies": True,
        "perf_tier": "极致", "needs_new_hw_feature": True, "needs_portability": False,
    },
    {
        "name": "新 attention 变体的研究原型",
        "is_standard_gemm": False, "is_fused": True, "shape_varies": True,
        "perf_tier": "正常", "needs_new_hw_feature": False, "needs_portability": True,
    },
]


def print_case(idx, op):
    choice, reasons = decide(op)
    feat = (f"融合={'是' if op['is_fused'] else '否'}, "
            f"shape多变={'是' if op['shape_varies'] else '否'}, "
            f"性能档位={op['perf_tier']}, "
            f"新硬件特性={'是' if op['needs_new_hw_feature'] else '否'}")
    print(f"  案例 {idx}: {op['name']}")
    print(f"    特征: {feat}")
    print(f"    建议: {choice}")
    for r in reasons:
        print(f"      - {r}")
    print()


def print_summary():
    rows = []
    for op in CASES:
        choice, _ = decide(op)
        rows.append((op["name"], choice))
    print("=" * 74)
    print("决策汇总表")
    print("=" * 74)
    print("  " + pad("算子案例", 40) + " | " + pad("建议路线", 22))
    print("  " + "-" * 66)
    for name, choice in rows:
        print("  " + pad(name, 40) + " | " + pad(choice, 22))
    print()
    print("口诀：能调库就不写；没有库先 Triton；profile 证明差距大且占比高，")
    print("      或必须吃 TMA/warp specialization 等新硬件红利，再上 CUTLASS/手写。")


def main():
    print("=" * 74)
    print("Triton vs CUDA vs cuBLAS 选型决策树（教学估算器）")
    print("=" * 74)
    print()
    for i, op in enumerate(CASES, 1):
        print_case(i, op)
    print_summary()


if __name__ == "__main__":
    main()
