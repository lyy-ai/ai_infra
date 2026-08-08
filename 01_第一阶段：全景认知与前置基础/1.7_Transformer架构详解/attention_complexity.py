# 2.3 Transformer：Attention 复杂度与 KV Cache 计算
#
# 运行：
#   /data/liyangyang/qwen35_env/bin/python 1.7_Transformer架构详解/attention_complexity.py


def attention_cost(seq_len, head_dim=128, n_heads=32):
    flops = 4 * seq_len * seq_len * head_dim * n_heads
    mem_bytes = seq_len * seq_len * n_heads * 2
    return flops, mem_bytes


def kv_cache_bytes(layers, kv_heads, head_dim, seq, batch=1, dtype_bytes=2):
    return 2 * layers * kv_heads * head_dim * seq * batch * dtype_bytes


def main():
    print("=== Attention O(N^2) 增长 ===")
    print(f"{'seq':>6} | {'FLOPs(G)':>10} | {'Attn矩阵(MB)':>14}")
    for n in [512, 1024, 2048, 4096, 8192]:
        f, m = attention_cost(n)
        print(f"{n:>6} | {f / 1e9:>10.1f} | {m / 1e6:>14.1f}")

    print("\n=== KV Cache：MHA vs GQA vs MQA（32层, head_dim=128, seq=4096, bs=1）===")
    for name, kv_heads in [("MHA(32)", 32), ("GQA(8)", 8), ("MQA(1)", 1)]:
        mb = kv_cache_bytes(32, kv_heads, 128, 4096) / 1e6
        print(f"  {name:<9}: {mb:,.0f} MB")
    print("  结论：KV 头数越少，KV Cache 越小，可支撑更长序列/更大 batch")


if __name__ == "__main__":
    main()
