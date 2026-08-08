#!/usr/bin/env python3
"""
vLLM Speculative Decoding 示例。

本脚本演示使用 vLLM 的 speculative decoding 功能进行推理。

运行方式：
    PATH=/data/liyangyang/qwen35_env/bin:$PATH python examples/vllm_speculative_decoding.py

说明：
- 脚本已导入 vllm_env_helper 以完成 vLLM 环境初始化（CUDA 13 库预加载、ninja PATH、GPU 选择）。
- 若需指定 GPU，请设置 CUDA_VISIBLE_DEVICES；主进程会自动选择空闲显存最多的 GPU。

环境要求：
    pip install vllm

注意：
- vLLM 对 CUDA 和 PyTorch 版本有严格要求。
- 本脚本默认使用 vLLM 内置的 ngram 投机解码方法，无需额外 Draft 模型。
- 如配置为 draft_model，请提供一个比 target 小的模型作为 Draft。
"""
import sys
import time

import vllm_env_helper  # noqa: F401
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Error: vLLM is not installed.")
    print("Please install it with: pip install vllm")
    sys.exit(1)


TARGET_MODEL_PATH = "/data/liyangyang/models/Qwen3.5-9B"
# 如果配置为 draft_model，请提供一个比 target 小的模型路径；
# 空字符串表示使用 ngram（基于已生成 token 的 n-gram 池进行投机），无需额外模型。
SPECULATIVE_MODEL_PATH = ""
SPEC_METHOD = "ngram"  # 或 "draft_model"（需要配置 SPECULATIVE_MODEL_PATH）

PROMPTS = [
    "请介绍一下机器学习中的梯度下降算法。",
    "什么是注意力机制？它在自然语言处理中有什么作用？",
    "简述大语言模型量化技术的意义。",
    "请用中文解释 Mixture of Experts 架构。",
]


def benchmark(speculative_model: str = None, num_speculative_tokens: int = 5):
    """运行 benchmark。"""
    print(f"\nBenchmark with spec_method={SPEC_METHOD}, spec_tokens={num_speculative_tokens}")
    print("-" * 60)

    kwargs = {
        "model": TARGET_MODEL_PATH,
        "dtype": "float16",
        "max_model_len": 4096,
        # 控制 vLLM 可使用的 GPU 显存比例
        "gpu_memory_utilization": 0.7,
    }

    if SPEC_METHOD and num_speculative_tokens > 0:
        kwargs["spec_method"] = SPEC_METHOD
        kwargs["spec_tokens"] = num_speculative_tokens
        if SPEC_METHOD == "draft_model" and speculative_model:
            kwargs["spec_model"] = speculative_model

    llm = LLM(**kwargs)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=128,
    )

    start = time.time()
    outputs = llm.generate(PROMPTS, sampling_params)
    elapsed = time.time() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Throughput: {total_tokens/elapsed:.2f} tok/s")
    print(f"  Requests/s: {len(PROMPTS)/elapsed:.2f} req/s")

    return total_tokens / elapsed


def main():
    print(f"Target model: {TARGET_MODEL_PATH}")
    print(f"Speculative method: {SPEC_METHOD}")
    print(f"Speculative model: {SPECULATIVE_MODEL_PATH or 'N/A (ngram does not need a separate model)'}")
    print()

    # 1. 普通生成
    throughput_no_spec = benchmark(speculative_model=None, num_speculative_tokens=0)

    # 2. 投机解码生成
    throughput_with_spec = benchmark(
        speculative_model=SPECULATIVE_MODEL_PATH or None,
        num_speculative_tokens=5,
    )

    print("\n" + "=" * 60)
    if throughput_with_spec > 0 and throughput_no_spec > 0:
        speedup = throughput_with_spec / throughput_no_spec
        print(f"Throughput speedup: {speedup:.2f}x")
    print("=" * 60)

    print("\nNote: Speculative decoding speedup depends heavily on the method and workload.")
    print("ngram is a simple built-in method that may not accelerate general open-ended text.")
    print("In production, use a smaller model from the same family as draft (draft_model).")


if __name__ == "__main__":
    main()
