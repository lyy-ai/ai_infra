# kvcache_memory_analysis.py
import torch


def compute_kv_cache_bytes(
    batch_size,
    seq_len,
    num_layers,
    num_heads,
    head_dim,
    dtype_bytes=2,
):
    """计算 KV Cache 显存占用（字节）"""
    return 2 * batch_size * num_layers * num_heads * head_dim * seq_len * dtype_bytes


def print_model_kv_cache(name, batch_size, num_layers, num_heads, head_dim, seq_lens):
    """打印不同序列长度、不同精度下的 KV Cache 大小"""
    print(f"\n{name}: batch={batch_size}, layers={num_layers}, heads={num_heads}, head_dim={head_dim}")
    print(f"{'seq_len':>10} {'FP16(GB)':>10} {'INT8(GB)':>10} {'KIVI2(GB)':>10}")
    for seq_len in seq_lens:
        fp16 = compute_kv_cache_bytes(batch_size, seq_len, num_layers, num_heads, head_dim, 2)
        int8 = compute_kv_cache_bytes(batch_size, seq_len, num_layers, num_heads, head_dim, 1)
        kivi2 = compute_kv_cache_bytes(batch_size, seq_len, num_layers, num_heads, head_dim, 0.25)
        print(f"{seq_len:>10} {fp16/1024**3:>10.2f} {int8/1024**3:>10.2f} {kivi2/1024**3:>10.2f}")


def main():
    seq_lens = [2048, 8192, 32768, 100000, 200000]

    # Llama-2-7B 风格配置
    print_model_kv_cache(
        "Llama-2-7B-style",
        batch_size=1,
        num_layers=32,
        num_heads=32,
        head_dim=128,
        seq_lens=seq_lens,
    )

    # Llama-2-70B 风格配置
    print_model_kv_cache(
        "Llama-2-70B-style",
        batch_size=1,
        num_layers=80,
        num_heads=8,
        head_dim=128,
        seq_lens=seq_lens,
    )

    # 展示 batch size 的影响
    print("\n\nBatch size 对 KV Cache 的影响（Llama-2-7B-style, seq_len=32768）")
    print(f"{'batch':>6} {'FP16(GB)':>10} {'INT8(GB)':>10} {'KIVI2(GB)':>10}")
    for bs in [1, 4, 8, 16]:
        fp16 = compute_kv_cache_bytes(bs, 32768, 32, 32, 128, 2)
        int8 = compute_kv_cache_bytes(bs, 32768, 32, 32, 128, 1)
        kivi2 = compute_kv_cache_bytes(bs, 32768, 32, 32, 128, 0.25)
        print(f"{bs:>6} {fp16/1024**3:>10.2f} {int8/1024**3:>10.2f} {kivi2/1024**3:>10.2f}")


if __name__ == "__main__":
    main()
