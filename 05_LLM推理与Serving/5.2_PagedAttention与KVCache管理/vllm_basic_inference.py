#!/usr/bin/env python3
"""
vLLM 基础推理示例。

本脚本演示：
1. 使用 vLLM 进行离线批量推理
2. 使用 vLLM 的 SamplingParams 控制解码
3. 启用 Prefix Caching
4. 使用 KV Cache 量化

环境要求：
    pip install vllm

注意：
- 脚本已导入 vllm_env_helper 以完成 vLLM 环境初始化（CUDA 13 库预加载、ninja PATH、GPU 选择）。
- 建议运行命令：PATH=/data/liyangyang/qwen35_env/bin:$PATH python examples/vllm_basic_inference.py
- 若需指定 GPU，请设置 CUDA_VISIBLE_DEVICES；主进程会自动选择空闲显存最多的 GPU。
"""
import time

import vllm_env_helper  # noqa: F401
from vllm import LLM, SamplingParams


MODEL_PATH = "/data/liyangyang/models/Qwen3.5-9B"
GPU_MEMORY_UTILIZATION = 0.7
MAX_MODEL_LEN = 4096

PROMPTS = [
    "请介绍一下机器学习中的梯度下降算法。",
    "什么是注意力机制？它在自然语言处理中有什么作用？",
    "简述大语言模型量化技术的意义。",
    "请用中文解释 Mixture of Experts 架构。",
]


def common_llm_kwargs():
    """返回通用的 LLM 构造参数。"""
    return {
        "model": MODEL_PATH,
        "dtype": "float16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_model_len": MAX_MODEL_LEN,
    }


def basic_inference():
    """基础批量推理。"""
    print("=" * 60)
    print("Basic vLLM Inference")
    print("=" * 60)

    llm = LLM(**common_llm_kwargs())

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=128,
        repetition_penalty=1.1,
    )

    start = time.time()
    outputs = llm.generate(PROMPTS, sampling_params)
    elapsed = time.time() - start

    total_generated_tokens = 0
    for output in outputs:
        total_generated_tokens += len(output.outputs[0].token_ids)
        print(f"\nPrompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text[:200]}...")

    print(f"\nTotal time: {elapsed:.3f}s")
    print(f"Total generated tokens: {total_generated_tokens}")
    print(f"Throughput: {total_generated_tokens/elapsed:.2f} tok/s")


def inference_with_prefix_caching():
    """演示 Prefix Caching：多个请求共享相同前缀。"""
    print("\n" + "=" * 60)
    print("vLLM Inference with Prefix Caching")
    print("=" * 60)

    kwargs = common_llm_kwargs()
    kwargs["enable_prefix_caching"] = True
    llm = LLM(**kwargs)

    # 多个请求共享相同的 system prompt / 前缀
    prefix = "你是一个专业的人工智能助手。请详细回答以下问题：\n"
    questions = [
        "什么是机器学习？",
        "什么是深度学习？",
        "什么是强化学习？",
    ]
    prompts = [prefix + q for q in questions]

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=64,
    )

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    print(f"\nGenerated {len(outputs)} prompts with shared prefix")
    print(f"Total time: {elapsed:.3f}s")
    for output in outputs:
        print(f"  - {output.prompt[:50]}... -> {output.outputs[0].text[:80]}...")


def inference_with_kv_cache_quantization():
    """演示 KV Cache 量化。"""
    print("\n" + "=" * 60)
    print("vLLM Inference with KV Cache Quantization")
    print("=" * 60)

    # 注意：需要 GPU 支持 FP8 才能使用 kv_cache_dtype="fp8"
    # 对于不支持 FP8 的 GPU，可以改为 "int8" 或注释掉
    kwargs = common_llm_kwargs()
    kwargs["kv_cache_dtype"] = "fp8"  # 或 "int8"
    llm = LLM(**kwargs)

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=64,
    )

    outputs = llm.generate(PROMPTS[:2], sampling_params)
    for output in outputs:
        print(f"\nPrompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text[:150]}...")


if __name__ == "__main__":
    print(f"Model path: {MODEL_PATH}")
    print(f"Number of prompts: {len(PROMPTS)}\n")

    basic_inference()
    inference_with_prefix_caching()
    # inference_with_kv_cache_quantization()  # 需要 GPU 支持相应量化格式
