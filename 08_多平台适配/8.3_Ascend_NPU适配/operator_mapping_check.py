# 9.3 Ascend NPU 适配：算子映射检查
#
# 运行：
#   cd /data/liyangyang/ai_infra/08_多平台适配
#   /data/liyangyang/qwen35_env/bin/python 8.3_Ascend_NPU适配/operator_mapping_check.py


def check_operators(onnx_ops, cann_support):
    print(f"{'ONNX Op':>20} | {'CANN Support':>14} | {'Risk':>10}")
    print("-" * 50)
    risks = []
    for op in onnx_ops:
        supported = op in cann_support
        risk = "low" if supported else "high"
        if not supported:
            risks.append(op)
        print(f"{op:>20} | {'yes' if supported else 'no':>14} | {risk:>10}")
    print()
    if risks:
        print(f"High-risk ops ({len(risks)}): {', '.join(risks)}")
    else:
        print("All ops are supported by CANN.")
    return len(risks) == 0


def main():
    onnx_ops = ["Conv", "MatMul", "LayerNormalization", "Softmax", "GELU", "RoPE", "FlashAttention"]
    cann_support = {"Conv", "MatMul", "LayerNormalization", "Softmax", "GELU"}
    check_operators(onnx_ops, cann_support)


if __name__ == "__main__":
    main()
