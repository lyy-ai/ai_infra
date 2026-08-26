#!/usr/bin/env python3
"""
vLLM Continuous Batching 示例。

本脚本演示：
1. 使用 vLLM 进行批量推理（vLLM 内部自动使用 Continuous Batching）
2. 通过调整 --max-num-seqs 控制最大并发数
3. 对比不同并发数下的吞吐

运行方式：
    PATH=/data/qwen35_env/bin:$PATH python examples/vllm_continuous_batching.py

说明：
- 脚本已导入 vllm_env_helper 以完成 vLLM 环境初始化（CUDA 13 库预加载、ninja PATH、GPU 选择）。
- 若需指定 GPU，请设置 CUDA_VISIBLE_DEVICES；主进程会自动选择空闲显存最多的 GPU。

环境要求：
    pip install vllm
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


MODEL_PATH = "/data/models/Qwen3.5-9B"


def generate_prompts(n: int) -> list:
    """生成测试 prompt。"""
    base = [
        "请介绍一下机器学习中的梯度下降算法。",
        "什么是注意力机制？它在自然语言处理中有什么作用？",
        "简述大语言模型量化技术的意义。",
        "请用中文解释 Mixture of Experts 架构。",
    ]
    return [base[i % len(base)] for i in range(n)]


def benchmark(max_num_seqs: int, num_prompts: int = 20, max_tokens: int = 64):
    """使用指定 max_num_seqs 运行 benchmark。"""
    print(f"\nBenchmark with max_num_seqs={max_num_seqs}")
    print("-" * 60)

    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",
        max_num_seqs=max_num_seqs,
        # 限制最大长度，避免 OOM
        max_model_len=4096,
        # 控制 vLLM 可使用的 GPU 显存比例
        gpu_memory_utilization=0.7,
    )

    prompts = generate_prompts(num_prompts)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens,
    )

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Throughput: {total_tokens/elapsed:.2f} tok/s")
    print(f"  Requests/s: {num_prompts/elapsed:.2f} req/s")

    return total_tokens / elapsed


def main():
    print(f"Model: {MODEL_PATH}")
    print("vLLM uses Continuous Batching internally.\n")

    # 测试不同并发数下的吞吐
    for max_num_seqs in [1, 2, 4, 8]:
        benchmark(max_num_seqs, num_prompts=20, max_tokens=64)

    print("\n" + "=" * 60)
    print("Observation: As max_num_seqs increases, throughput typically")
    print("increases due to better GPU utilization, but with diminishing returns.")
    print("=" * 60)


if __name__ == "__main__":
    main()
