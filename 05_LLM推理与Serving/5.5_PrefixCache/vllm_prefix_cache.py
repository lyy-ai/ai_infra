#!/usr/bin/env python3
"""
vLLM Prefix Cache 示例。

本脚本演示：
1. 通过 enable_prefix_caching=True 开启 vLLM 的 Automatic Prefix Caching
2. 对一组共享长 system prompt 的请求连续推理多次，观察 warm pass 的 TTFT/总耗时下降
3. 用一组不共享前缀的请求作为对照，理解 Prefix Cache 的收益边界

运行方式：
    PATH=/data/qwen35_env/bin:$PATH python 5.5_PrefixCache/vllm_prefix_cache.py

说明：
- 脚本已导入 vllm_env_helper 以完成 vLLM 环境初始化。
- 若需指定 GPU，请设置 CUDA_VISIBLE_DEVICES；主进程会自动选择空闲显存最多的 GPU。
- 共享 GPU 上如果其他进程临时占用显存，可降低 GPU_MEMORY_UTILIZATION 后重试。
- Qwen3.5 混合架构在较低显存利用率下会减少 Mamba cache blocks；本脚本显式限制 MAX_NUM_SEQS，避免 CUDA graph capture 失败。
"""
import time

import vllm_env_helper  # noqa: F401
from vllm import LLM, SamplingParams


MODEL_PATH = "/data/models/Qwen3.5-9B"
GPU_MEMORY_UTILIZATION = 0.55
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 32
MAX_NEW_TOKENS = 8
SHARED_PREFIX_REPEAT = 48
UNIQUE_PREFIX_REPEAT = SHARED_PREFIX_REPEAT * 2

SHARED_PREFIX = (
    "你是一个严谨的中文 AI 基础设施讲师，请基于 LLM 推理与 Serving 课程材料回答。"
    "回答需要简洁、准确，并优先解释系统层面的原因。"
    * SHARED_PREFIX_REPEAT
    + "\n问题："
)

QUESTIONS = [
    "什么是 Prefix Cache？",
    "Prefix Cache 和 KV Cache 有什么区别？",
    "为什么共享 system prompt 可以降低 TTFT？",
    "PagedAttention 和 Prefix Cache 是什么关系？",
    "什么场景下 Prefix Cache 命中率会下降？",
    "为什么 Prefix Cache 通常按 Block 复用？",
    "Prefix Cache 会改变模型输出分布吗？",
    "RAG 场景为什么适合使用 Prefix Cache？",
]


def build_shared_prefix_prompts():
    """构造一组共享长前缀的请求。"""
    return [SHARED_PREFIX + q for q in QUESTIONS]


def build_unique_prefix_prompts():
    """构造一组长度相近但前缀几乎不共享的对照请求。"""
    prompts = []
    for i, q in enumerate(QUESTIONS):
        unique_header = (
            f"这是第{i+1}个完全独立的评测案例，请不要复用任何其他案例的上下文。"
            * UNIQUE_PREFIX_REPEAT
            + "\n问题："
        )
        prompts.append(unique_header + q)
    return prompts


def count_prompt_tokens(outputs) -> int:
    """统计输出对象中的 prompt token 数。"""
    total = 0
    for output in outputs:
        total += len(getattr(output, "prompt_token_ids", []) or [])
    return total


def run_batch(llm, prompts, sampling_params, label: str):
    """运行一批请求并打印耗时与吞吐。"""
    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    prompt_tokens = count_prompt_tokens(outputs)
    generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

    print(f"requests: {len(outputs)}")
    print(f"prompt tokens: {prompt_tokens}")
    print(f"generated tokens: {generated_tokens}")
    print(f"elapsed: {elapsed:.3f}s")
    if elapsed > 0:
        print(f"prefill throughput: {prompt_tokens / elapsed:.2f} tok/s")
        print(f"total throughput: {(prompt_tokens + generated_tokens) / elapsed:.2f} tok/s")
    return elapsed, prompt_tokens, generated_tokens


def main():
    print(f"Model: {MODEL_PATH}")
    print(f"Shared prefix questions: {len(QUESTIONS)}")
    print(f"max_new_tokens: {MAX_NEW_TOKENS}")

    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        enable_prefix_caching=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
    )

    warmup_prompt = "请用一句话介绍 KV Cache。"
    llm.generate([warmup_prompt], SamplingParams(temperature=0.0, max_tokens=1))

    shared_prompts = build_shared_prefix_prompts()
    cold_elapsed, _, _ = run_batch(
        llm,
        shared_prompts,
        sampling_params,
        "Shared Prefix Batch - Cold (first pass, populate prefix cache)",
    )

    warm_elapsed_list = []
    for warm_round in (1, 2):
        elapsed, _, _ = run_batch(
            llm,
            shared_prompts,
            sampling_params,
            f"Shared Prefix Batch - Warm {warm_round} (reuse prefix cache)",
        )
        warm_elapsed_list.append(elapsed)
    warm_elapsed = min(warm_elapsed_list)

    unique_prompts = build_unique_prefix_prompts()
    unique_elapsed, _, _ = run_batch(
        llm,
        unique_prompts,
        sampling_params,
        "Unique Prefix Batch - Control (similar length, little reusable prefix)",
    )

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    if warm_elapsed > 0:
        print(f"shared-prefix cold/best-warm speedup: {cold_elapsed / warm_elapsed:.2f}x")
    if unique_elapsed > 0 and warm_elapsed > 0:
        print(f"best warm shared-prefix vs unique-prefix elapsed: {warm_elapsed:.3f}s vs {unique_elapsed:.3f}s")
    print("- Cold pass stores the shared prefix blocks in the prefix cache.")
    print("- Warm passes should reuse those blocks and spend less time in prefill.")
    print("- Qwen3.5 on this vLLM build may use a large attention block size; make the shared prefix long enough to form full blocks.")
    print("- Unique prefixes have little to reuse, so Prefix Cache helps much less.")
    print("=" * 72)


if __name__ == "__main__":
    main()
