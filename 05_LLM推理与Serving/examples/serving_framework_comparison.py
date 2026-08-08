#!/usr/bin/env python3
"""
主流 LLM 推理框架对比与选型示例。

本脚本不启动任何服务，只用内置的定性评分表打印：
1. vLLM / TGI / TensorRT-LLM / SGLang / LMDeploy / DeepSpeed-MII 的能力矩阵
2. 不同业务场景下的推荐排序

运行方式：
    python examples/serving_framework_comparison.py

注意：评分是教学用定性参考，真实选型请结合最新版本文档与压测结果。
"""


FRAMEWORKS = {
    "vLLM": {
        "ease": 5,
        "openai_api": 5,
        "throughput": 5,
        "prefix_cache": 5,
        "structured_output": 4,
        "quantization": 4,
        "multimodal": 4,
        "pd_disagg": 4,
        "nvidia_perf": 4,
        "hf_integration": 4,
        "community": 5,
        "note": "Python 友好、OpenAI 兼容、PagedAttention/APC 生态成熟",
    },
    "TGI": {
        "ease": 4,
        "openai_api": 4,
        "throughput": 4,
        "prefix_cache": 4,
        "structured_output": 4,
        "quantization": 4,
        "multimodal": 3,
        "pd_disagg": 3,
        "nvidia_perf": 3,
        "hf_integration": 5,
        "community": 4,
        "note": "Hugging Face 生态集成好，适合 transformers 模型快速服务化",
    },
    "TensorRT-LLM": {
        "ease": 2,
        "openai_api": 3,
        "throughput": 5,
        "prefix_cache": 4,
        "structured_output": 3,
        "quantization": 5,
        "multimodal": 4,
        "pd_disagg": 4,
        "nvidia_perf": 5,
        "hf_integration": 3,
        "community": 4,
        "note": "NVIDIA GPU 极致性能，但构建/调优成本更高",
    },
    "SGLang": {
        "ease": 4,
        "openai_api": 4,
        "throughput": 5,
        "prefix_cache": 5,
        "structured_output": 5,
        "quantization": 4,
        "multimodal": 4,
        "pd_disagg": 4,
        "nvidia_perf": 4,
        "hf_integration": 4,
        "community": 4,
        "note": "RadixAttention 与结构化输出强，适合 Agent/复杂 prompt 程序",
    },
    "LMDeploy": {
        "ease": 4,
        "openai_api": 4,
        "throughput": 4,
        "prefix_cache": 4,
        "structured_output": 3,
        "quantization": 4,
        "multimodal": 3,
        "pd_disagg": 3,
        "nvidia_perf": 4,
        "hf_integration": 4,
        "community": 3,
        "note": "InternLM/国产模型支持友好，部署链路相对完整",
    },
    "DeepSpeed-MII": {
        "ease": 3,
        "openai_api": 3,
        "throughput": 4,
        "prefix_cache": 3,
        "structured_output": 3,
        "quantization": 4,
        "multimodal": 3,
        "pd_disagg": 3,
        "nvidia_perf": 3,
        "hf_integration": 4,
        "community": 3,
        "note": "DeepSpeed 生态相关，适合已有 DeepSpeed 技术栈的团队",
    },
}

SCENARIOS = {
    "openai_compatible_fast": {
        "desc": "想要最快暴露 OpenAI 兼容 API",
        "weights": {"ease": 3, "openai_api": 3, "throughput": 2, "community": 2},
    },
    "max_nvidia_perf": {
        "desc": "NVIDIA GPU 上追求极限性能与量化",
        "weights": {"nvidia_perf": 4, "throughput": 3, "quantization": 3, "pd_disagg": 2},
    },
    "hf_ecosystem": {
        "desc": "深度使用 Hugging Face/transformers 生态",
        "weights": {"hf_integration": 4, "ease": 2, "openai_api": 2, "community": 2},
    },
    "structured_output_agent": {
        "desc": "Agent/工具调用，强依赖 JSON/grammar 结构化输出",
        "weights": {"structured_output": 4, "openai_api": 2, "throughput": 2, "prefix_cache": 1},
    },
    "long_context_high_concurrency": {
        "desc": "长上下文 + 高并发，关注 KV/Prefix/PD 解耦",
        "weights": {"throughput": 3, "prefix_cache": 3, "pd_disagg": 3, "quantization": 1},
    },
}


def score_framework(framework: dict, weights: dict) -> float:
    """按场景权重计算加权分。"""
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0
    return sum(framework[key] * weight for key, weight in weights.items()) / total_weight


def print_matrix():
    """打印能力矩阵。"""
    keys = [
        "ease", "openai_api", "throughput", "prefix_cache", "structured_output",
        "quantization", "multimodal", "pd_disagg", "nvidia_perf", "hf_integration", "community",
    ]
    labels = {
        "ease": "ease",
        "openai_api": "api",
        "throughput": "tput",
        "prefix_cache": "prefix",
        "structured_output": "struct",
        "quantization": "quant",
        "multimodal": "mm",
        "pd_disagg": "pd",
        "nvidia_perf": "nv",
        "hf_integration": "hf",
        "community": "comm",
    }
    print("\nCapability Matrix (1-5, higher is better)")
    print("-" * 108)
    print(f"{'framework':>16} " + " ".join(f"{labels[key]:>7}" for key in keys))
    print("-" * 108)
    for name, scores in FRAMEWORKS.items():
        print(f"{name:>16} " + " ".join(f"{scores[key]:>7}" for key in keys))
    print("-" * 108)
    for name, scores in FRAMEWORKS.items():
        print(f"{name:>16}: {scores['note']}")


def print_recommendations():
    """打印不同场景下的推荐排序。"""
    print("\nScenario Recommendations")
    print("=" * 88)
    for scenario, spec in SCENARIOS.items():
        ranked = sorted(
            FRAMEWORKS.items(),
            key=lambda item: score_framework(item[1], spec["weights"]),
            reverse=True,
        )
        print(f"\n[{scenario}] {spec['desc']}")
        for rank, (name, scores) in enumerate(ranked[:3], start=1):
            score = score_framework(scores, spec["weights"])
            print(f"  {rank}. {name:<16} score={score:.2f}  {scores['note']}")


def main():
    print("=" * 88)
    print("Mainstream LLM Serving Framework Comparison")
    print("=" * 88)
    print("This script uses qualitative scores for course discussion, not benchmark results.")
    print_matrix()
    print_recommendations()
    print("\nRule of thumb:")
    print("- Fast OpenAI-compatible serving: start from vLLM or SGLang.")
    print("- Maximum NVIDIA performance: evaluate TensorRT-LLM with real workloads.")
    print("- Hugging Face native workflow: evaluate TGI.")
    print("- Agent/structured programs: evaluate SGLang first.")


if __name__ == "__main__":
    main()
