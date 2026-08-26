#!/usr/bin/env python3
"""
KV Cache 显存分析示例。

本脚本演示：
1. 如何根据模型配置估算 KV Cache 显存
2. 如何随 batch_size 和 seq_len 变化分析 KV Cache 增长
3. 比较不同模型架构（标准 Attention vs GQA）的 KV Cache 差异
"""
import torch
from transformers import AutoConfig


MODEL_PATH = "/data/models/Qwen3.5-9B"
BYTES_PER_FP16 = 2
BYTES_PER_FP32 = 4


def estimate_kv_cache_bytes(
    num_layers,
    num_kv_heads,
    head_dim,
    seq_len,
    batch_size=1,
    bytes_per_element=BYTES_PER_FP16,
):
    """估算 KV Cache 显存占用（bytes）。"""
    # K 和 V 各一份
    per_layer_cache = 2 * num_kv_heads * seq_len * head_dim * bytes_per_element
    total_cache = per_layer_cache * num_layers * batch_size
    return total_cache


def get_model_kv_config(model_path):
    """从 config 读取 KV Cache 相关配置。"""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)

    num_layers = getattr(text_config, "num_hidden_layers", None)
    num_kv_heads = getattr(text_config, "num_key_value_heads", None)
    num_heads = getattr(text_config, "num_attention_heads", None)
    hidden_size = getattr(text_config, "hidden_size", None)

    # 部分 config 没有 num_key_value_heads，则退化为 num_attention_heads
    if num_kv_heads is None:
        num_kv_heads = num_heads

    # head_dim 可能没有直接给出，需要计算
    head_dim = getattr(text_config, "head_dim", None)
    if head_dim is None and num_heads is not None and hidden_size is not None:
        head_dim = hidden_size // num_heads

    return {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
    }


def print_kv_cache_table(config, seq_lengths, batch_sizes, bytes_per_element=BYTES_PER_FP16):
    """打印不同 batch_size 和 seq_len 下的 KV Cache 显存占用表。"""
    num_layers = config["num_layers"]
    num_kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]

    print("\n" + "=" * 80)
    print(f"KV Cache Memory Estimation (dtype=FP16, num_layers={num_layers}, num_kv_heads={num_kv_heads}, head_dim={head_dim})")
    print("=" * 80)
    print(f"{'Batch \\ Seq':>12}", end="")
    for seq_len in seq_lengths:
        print(f"{seq_len:>12}", end="")
    print()
    print("-" * 80)

    for batch_size in batch_sizes:
        print(f"{batch_size:>12}", end="")
        for seq_len in seq_lengths:
            mem_bytes = estimate_kv_cache_bytes(
                num_layers, num_kv_heads, head_dim, seq_len,
                batch_size=batch_size, bytes_per_element=bytes_per_element,
            )
            mem_gb = mem_bytes / 1024**3
            print(f"{mem_gb:>11.2f}GB", end="")
        print()
    print("=" * 80 + "\n")


def compare_gqa_vs_mha():
    """对比 GQA 与标准 MHA 在 KV Cache 上的差异。"""
    print("=" * 60)
    print("GQA vs MHA KV Cache Comparison")
    print("=" * 60)

    # 假设一个 7B 规模的模型
    num_layers = 28
    head_dim = 128
    seq_len = 4096
    batch_size = 1

    configs = {
        "MHA (num_kv_heads=28)": {"num_kv_heads": 28},
        "GQA (num_kv_heads=4)": {"num_kv_heads": 4},
        "MQA (num_kv_heads=1)": {"num_kv_heads": 1},
    }

    for name, cfg in configs.items():
        mem_bytes = estimate_kv_cache_bytes(
            num_layers, cfg["num_kv_heads"], head_dim, seq_len,
            batch_size=batch_size, bytes_per_element=BYTES_PER_FP16,
        )
        print(f"{name}: {mem_bytes / 1024**3:.3f} GB")


def main():
    print(f"Loading config from {MODEL_PATH}...")
    config = get_model_kv_config(MODEL_PATH)
    print("Model KV config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # 打印不同 batch_size 和 seq_len 下的 KV Cache 表
    seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
    batch_sizes = [1, 4, 8]
    print_kv_cache_table(config, seq_lengths, batch_sizes)

    # 对比 GQA vs MHA
    compare_gqa_vs_mha()

    # 模型权重显存对比
    print("=" * 60)
    print("Model Weight Memory Estimation")
    print("=" * 60)
    # 这里只估算 7B 参数在几种精度下的显存
    params = 7_000_000_000
    for dtype_name, bytes_per in [("FP32", BYTES_PER_FP32), ("FP16/BF16", BYTES_PER_FP16), ("INT8", 1), ("INT4", 0.5)]:
        mem_gb = params * bytes_per / 1024**3
        print(f"{dtype_name}: {mem_gb:.2f} GB")
    print()


if __name__ == "__main__":
    main()
