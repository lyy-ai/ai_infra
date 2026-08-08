#!/usr/bin/env python3
"""
Tensor Parallel / Pipeline Parallel Serving 规划示例。

本脚本用近似估算帮助你理解：
1. TP（Tensor Parallel）把单层权重/KV 切到多张 GPU 上，通信频繁但延迟低。
2. PP（Pipeline Parallel）把不同层切到不同 stage 上，通信少但有 pipeline bubble。
3. 对给定模型大小、GPU 显存、batch/prompt，粗略判断单卡、TP 或 PP 是否合适。

运行方式：
    python examples/parallelism_planner.py

注意：以下是教学估算，不代表 vLLM/TensorRT-LLM 的真实内存占用；上线前必须实测。
"""


MODEL_NAME = "Qwen3.5-9B"
WEIGHT_GB_FP16 = 18.0
NUM_LAYERS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
DTYPE_BYTES = 2

GPU_MEM_GB = 40.0
GPUS_PER_NODE = 8
RUNTIME_OVERHEAD_GB_PER_GPU = 4.0
SAFETY_HEADROOM = 0.85

TARGET_PROMPT_TOKENS = 2048
TARGET_BATCH_SIZE = 8


def kv_bytes_per_token() -> int:
    """估算标准 Transformer 每 token KV 字节数。"""
    return 2 * NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM * DTYPE_BYTES


def fits(total_gb: float) -> bool:
    """判断是否能在安全余量内放下。"""
    return total_gb <= GPU_MEM_GB * SAFETY_HEADROOM


def print_tp_table(total_kv_gb: float):
    """打印不同 TP degree 下的每 GPU 估算。"""
    print("\n[Tensor Parallel] per-GPU estimate")
    print("-" * 88)
    print(f"{'tp':>4} {'weight/gpu':>12} {'kv/gpu':>10} {'overhead':>10} {'total':>10} {'fits':>6}")
    print("-" * 88)
    for tp in [1, 2, 4, 8]:
        weight_per_gpu = WEIGHT_GB_FP16 / tp
        kv_per_gpu = total_kv_gb / tp
        total = weight_per_gpu + kv_per_gpu + RUNTIME_OVERHEAD_GB_PER_GPU
        print(f"{tp:>4} {weight_per_gpu:>12.2f} {kv_per_gpu:>10.2f} {RUNTIME_OVERHEAD_GB_PER_GPU:>10.2f} {total:>10.2f} {str(fits(total)):>6}")
    print("-" * 88)
    print("TP note: every layer is sharded; all-reduce happens inside each layer, so prefer NVLink within one node.")


def print_pp_table(total_kv_gb: float):
    """打印不同 PP stage 下的每 GPU 估算。"""
    print("\n[Pipeline Parallel] per-GPU estimate")
    print("-" * 88)
    print(f"{'pp':>4} {'weight/gpu':>12} {'kv/gpu':>10} {'overhead':>10} {'total':>10} {'fits':>6}")
    print("-" * 88)
    for pp in [1, 2, 4, 8]:
        weight_per_gpu = WEIGHT_GB_FP16 / pp
        kv_per_gpu = total_kv_gb / pp
        total = weight_per_gpu + kv_per_gpu + RUNTIME_OVERHEAD_GB_PER_GPU
        print(f"{pp:>4} {weight_per_gpu:>12.2f} {kv_per_gpu:>10.2f} {RUNTIME_OVERHEAD_GB_PER_GPU:>10.2f} {total:>10.2f} {str(fits(total)):>6}")
    print("-" * 88)
    print("PP note: each stage owns a contiguous block of layers; communication is stage-to-stage, but bubbles hurt utilization.")


def recommend(total_kv_gb: float):
    """给出粗略推荐。"""
    single_total = WEIGHT_GB_FP16 + total_kv_gb + RUNTIME_OVERHEAD_GB_PER_GPU
    print("\nRecommendation")
    print("-" * 88)
    print(f"single-GPU estimate: {single_total:.2f} GiB on a {GPU_MEM_GB:.0f} GiB GPU")
    if fits(single_total):
        print("- Single GPU can hold this workload estimate; prefer simple deployment first.")
    else:
        print("- Single GPU estimate does not fit; consider TP within one NVLink node before PP across nodes.")
    print(f"- For {MODEL_NAME}-scale models, TP=1/2 inside one node is usually simpler than PP.")
    print("- Use PP mainly when the model cannot fit even after TP within a node, or when crossing nodes is unavoidable.")
    print("- Before complex PP, evaluate quantization, smaller max_model_len, prefix cache, PD disaggregation, and batching.")


def main():
    total_kv_gb = kv_bytes_per_token() * TARGET_PROMPT_TOKENS * TARGET_BATCH_SIZE / 1e9
    print("=" * 88)
    print("Tensor/Pipeline Parallelism Planner (teaching estimate)")
    print("=" * 88)
    print(f"model: {MODEL_NAME}, fp16 weight: {WEIGHT_GB_FP16:.1f} GiB")
    print(f"target: prompt={TARGET_PROMPT_TOKENS}, batch={TARGET_BATCH_SIZE}, KV={total_kv_gb:.2f} GiB")
    print(f"gpu: {GPU_MEM_GB:.0f} GiB, gpus/node={GPUS_PER_NODE}, safety={SAFETY_HEADROOM:.0%}")
    print_tp_table(total_kv_gb)
    print_pp_table(total_kv_gb)
    recommend(total_kv_gb)


if __name__ == "__main__":
    main()
