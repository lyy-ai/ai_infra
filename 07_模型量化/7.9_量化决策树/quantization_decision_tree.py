# quantization_decision_tree.py


def recommend_quantization(
    hardware,
    bottleneck,
    accuracy_sensitivity,
    tuning_cost,
    context_length=None,
):
    """
    根据硬件、瓶颈、精度敏感度、调优成本，推荐量化方案。
    """
    recommendation = []
    reasons = []

    # 1. 根据硬件过滤
    if hardware == "NVIDIA GPU":
        recommendation.append("W4A16 / W8A16 / INT8")
    elif hardware == "国产 GPU/NPU":
        recommendation.append("INT8 (厂商适配版)")
    elif hardware == "手机/边缘":
        recommendation.append("INT4/INT8 + MLC-LLM / llama.cpp")
    elif hardware == "CPU":
        recommendation.append("INT8 / ONNX Runtime / llama.cpp")
    else:
        return "未知硬件", ["请选择 NVIDIA GPU / 国产 GPU/NPU / 手机/边缘 / CPU"]

    # 2. 根据瓶颈补充
    if bottleneck == "权重显存":
        recommendation.append("AWQ / GPTQ / W4A16")
        reasons.append("权重量化能显著降低模型加载显存")
    elif bottleneck == "KV Cache 显存":
        if context_length and context_length >= 32000:
            recommendation.append("KIVI 2-bit KV Cache")
            reasons.append("长上下文下 KV Cache 是主要瓶颈，2-bit 压缩比最高")
        else:
            recommendation.append("KV Cache INT8")
            reasons.append("INT8 KV Cache 可减半显存，精度损失小")
    elif bottleneck == "推理延迟":
        recommendation.append("INT8 Tensor Core Kernel")
        reasons.append("INT8 Tensor Core 提供低延迟矩阵乘")
    elif bottleneck == "吞吐":
        recommendation.append("INT8 + PagedAttention + Continuous Batching")
        reasons.append("低比特 + 高效调度提升并发")
    elif bottleneck == "综合":
        recommendation.append("W4A16 + KIVI 2-bit / KV INT8 组合")
        reasons.append("同时压缩权重和 KV Cache")

    # 3. 根据精度要求调整
    if accuracy_sensitivity == "极高":
        recommendation = ["FP16 / BF16", "如必须量化，使用 QAT INT8"]
        reasons.append("精度优先，不建议使用低比特 PTQ")
    elif accuracy_sensitivity == "高":
        recommendation.append("优先 AWQ / QAT W8A16")
        reasons.append("AWQ 保护重要权重，QAT 进一步恢复精度")
    elif accuracy_sensitivity == "中":
        recommendation.append("GPTQ / W4A16 / INT8")
        reasons.append("在压缩和精度间取得平衡")
    elif accuracy_sensitivity == "低":
        recommendation.append("可尝试 W4A8 或更激进 KV Cache 量化")
        reasons.append("允许更大精度损失以换取极限压缩")

    # 4. 根据调优成本调整
    if tuning_cost == "零" and "QAT" in str(recommendation):
        recommendation = [r for r in recommendation if "QAT" not in r]
        reasons.append("移除 QAT，改用 PTQ 方案以降低实施成本")
    elif tuning_cost == "可训练":
        recommendation.append("QAT 可作为精度兜底方案")
        reasons.append("有训练资源时，QAT 通常能获得最佳精度")

    return recommendation, reasons


def interactive_decision_tree():
    """交互式决策树"""
    print("=== 量化方案选型决策树 ===\n")

    print("可选硬件：NVIDIA GPU / 国产 GPU/NPU / 手机/边缘 / CPU")
    hardware = input("请输入硬件平台：").strip()

    print("\n可选瓶颈：权重显存 / KV Cache 显存 / 推理延迟 / 吞吐 / 综合")
    bottleneck = input("请输入主要瓶颈：").strip()

    print("\n可选精度敏感度：极高 / 高 / 中 / 低")
    accuracy = input("请输入精度敏感度：").strip()

    print("\n可选调优成本：零 / 少量 / 可训练")
    cost = input("请输入可接受的调优成本：").strip()

    context_len = None
    if bottleneck in ["KV Cache 显存", "综合"]:
        context_len = input("请输入典型上下文长度（可选，回车跳过）：").strip()
        context_len = int(context_len) if context_len else None

    rec, reasons = recommend_quantization(hardware, bottleneck, accuracy, cost, context_len)

    print("\n=== 推荐方案 ===")
    for r in rec:
        print(f"  - {r}")
    print("\n=== 推荐理由 ===")
    for r in reasons:
        print(f"  - {r}")


def demo_cases():
    """预设几个典型场景的推荐"""
    cases = [
        ("NVIDIA GPU", "权重显存", "高", "零"),
        ("NVIDIA GPU", "KV Cache 显存", "高", "零", 100000),
        ("手机/边缘", "综合", "中", "零"),
        ("NVIDIA GPU", "吞吐", "中", "少量"),
    ]
    print("=== 典型场景推荐 ===\n")
    for case in cases:
        if len(case) == 5:
            hardware, bottleneck, accuracy, cost, ctx = case
            rec, reasons = recommend_quantization(hardware, bottleneck, accuracy, cost, ctx)
        else:
            hardware, bottleneck, accuracy, cost = case
            rec, reasons = recommend_quantization(hardware, bottleneck, accuracy, cost)
        print(f"场景：{hardware} | {bottleneck} | 精度{accuracy} | 成本{cost}")
        print(f"推荐：{', '.join(rec)}")
        print(f"理由：{reasons[0]}")
        print()


if __name__ == "__main__":
    demo_cases()
    print("\n")
    # 取消下面这行注释可以运行交互式版本
    # interactive_decision_tree()
