#!/usr/bin/env python3
"""
transformers 与 vLLM 吞吐对比示例。

本脚本设计：
1. 准备一组不同长度的 prompt
2. 使用 transformers 原生方式推理（顺序或简单静态 batch）
3. 如果 vLLM 可用，使用 vLLM 批量推理
4. 对比两种方式的 Latency 和 Throughput

运行环境：
    PATH=/data/liyangyang/qwen35_env/bin:$PATH python examples/vllm_vs_transformers_benchmark.py

说明：
- 脚本已导入 vllm_env_helper 以完成 vLLM 环境初始化（CUDA 13 库预加载、ninja PATH、GPU 选择）。
- 若需指定 GPU，请设置 CUDA_VISIBLE_DEVICES；主进程会自动选择空闲显存最多的 GPU。
"""
import sys
import time

import vllm_env_helper  # noqa: F401  必须在 import torch 之前，以正确设置 CUDA_VISIBLE_DEVICES


MODEL_PATH = "/data/liyangyang/models/Qwen3.5-9B"
MAX_NEW_TOKENS = 128
NUM_PROMPTS = 16


def generate_prompts(n=16):
    """生成一组 prompt，模拟不同用户请求。"""
    base_prompts = [
        "请介绍一下机器学习中的梯度下降算法。",
        "什么是注意力机制？它在自然语言处理中有什么作用？",
        "简述大语言模型量化技术的意义。",
        "请用中文解释 Mixture of Experts 架构。",
        "如何理解 Transformer 中的自注意力机制？",
        "什么是大模型的涌现能力？",
        "请解释 RLHF 的基本流程。",
        "为什么 LLM 需要做上下文学习？",
    ]
    return [base_prompts[i % len(base_prompts)] for i in range(n)]


def transformers_baseline(prompts, max_new_tokens=128):
    """使用 transformers 顺序推理（最朴素方式）。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n[transformers Baseline - Sequential]")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    total_generated_tokens = 0
    start = time.time()

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        num_input_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        total_generated_tokens += outputs.shape[1] - num_input_tokens

    elapsed = time.time() - start
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Total generated tokens: {total_generated_tokens}")
    print(f"  Throughput: {total_generated_tokens/elapsed:.2f} tok/s")
    print(f"  Requests per second: {len(prompts)/elapsed:.2f} req/s")

    del model, tokenizer
    torch.cuda.empty_cache()
    return elapsed, total_generated_tokens


def transformers_static_batch(prompts, max_new_tokens=128, batch_size=4):
    """使用 transformers 静态 batch 推理。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[transformers Static Batch - batch_size={batch_size}]")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    total_generated_tokens = 0
    start = time.time()

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        num_input_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        total_generated_tokens += (outputs.shape[1] - num_input_tokens) * len(batch)

    elapsed = time.time() - start
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Total generated tokens: {total_generated_tokens}")
    print(f"  Throughput: {total_generated_tokens/elapsed:.2f} tok/s")
    print(f"  Requests per second: {len(prompts)/elapsed:.2f} req/s")

    del model, tokenizer
    torch.cuda.empty_cache()
    return elapsed, total_generated_tokens


def vllm_benchmark(prompts, max_new_tokens=128):
    """使用 vLLM 批量推理。"""
    print("\n[vLLM]")
    try:
        import vllm_env_helper  # noqa: F401
        from vllm import LLM, SamplingParams
    except ImportError:
        print("  vLLM not installed. Skipping vLLM benchmark.")
        print("  Install with: pip install vllm")
        return None, None

    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        max_model_len=4096,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_new_tokens,
    )

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    total_generated_tokens = 0
    for output in outputs:
        total_generated_tokens += len(output.outputs[0].token_ids)

    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Total generated tokens: {total_generated_tokens}")
    print(f"  Throughput: {total_generated_tokens/elapsed:.2f} tok/s")
    print(f"  Requests per second: {len(prompts)/elapsed:.2f} req/s")

    del model, tokenizer
    torch.cuda.empty_cache()
    return elapsed, total_generated_tokens


def main():
    prompts = generate_prompts(NUM_PROMPTS)
    print(f"Model: {MODEL_PATH}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Max new tokens per prompt: {MAX_NEW_TOKENS}")

    results = {}

    # transformers 顺序推理
    t, tok = transformers_baseline(prompts, MAX_NEW_TOKENS)
    results["transformers_sequential"] = (t, tok)

    # transformers 静态 batch
    t, tok = transformers_static_batch(prompts, MAX_NEW_TOKENS, batch_size=4)
    results["transformers_batch4"] = (t, tok)

    # vLLM 推理
    t, tok = vllm_benchmark(prompts, MAX_NEW_TOKENS)
    if t is not None:
        results["vllm"] = (t, tok)

    # 汇总
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    baseline_t, baseline_tok = results["transformers_sequential"]
    for name, (t, tok) in results.items():
        if t is None:
            continue
        speedup = baseline_t / t if t > 0 else 0
        print(f"{name:>30}: {tok/t:.2f} tok/s | {len(prompts)/t:.2f} req/s | speedup vs sequential: {speedup:.2f}x")


if __name__ == "__main__":
    main()
