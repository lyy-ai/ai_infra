#!/usr/bin/env python3
"""
Prefill/Decode 解耦中的 KV 传输成本分析。

本脚本估算：Prefill worker 把 prompt 对应的 KV Cache 传给 Decode worker 时，
不同 prompt 长度、不同互连带宽下的传输数据量与耗时。

无需 GPU，可直接运行。

运行方式：
    python examples/kv_transfer_cost_analysis.py

注意：以下模型结构参数是示例默认值，真实 Qwen3.5/具体模型请以 config.json 为准。
"""


NUM_LAYERS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
DTYPE_BYTES = 2  # fp16/bf16

PROMPT_LENGTHS = [512, 1024, 2048, 4096]
BATCH_SIZES = [1, 4, 8, 16]
REFERENCE_PROMPT = 2048

BANDWIDTHS_GBPS = {
    "PCIe 4.0 x16 (~16 GB/s)": 16,
    "RDMA 400GbE (~50 GB/s)": 50,
    "NVLink/NVSwitch (~100+ GB/s)": 100,
}


def kv_bytes_per_token(num_layers=NUM_LAYERS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, dtype_bytes=DTYPE_BYTES) -> int:
    """计算每个 token 的 KV Cache 字节数。"""
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes


def mib(num_bytes: float) -> float:
    return num_bytes / (1024 ** 2)


def transfer_ms(num_bytes: int, bandwidth_gbps: float) -> float:
    """按十进制 GB/s 估算传输毫秒数。"""
    return num_bytes / (bandwidth_gbps * 1e9) * 1000.0


def print_prompt_table():
    """打印不同 prompt 长度下的 KV 传输成本。"""
    per_token = kv_bytes_per_token()
    print("\nKV transfer cost by prompt length")
    print("-" * 92)
    header = f"{'prompt tokens':>14} {'KV MiB':>12}"
    for name in BANDWIDTHS_GBPS:
        header += f" {name[:18]:>20}"
    print(header)
    print("-" * 92)

    for prompt_tokens in PROMPT_LENGTHS:
        num_bytes = per_token * prompt_tokens
        row = f"{prompt_tokens:>14} {mib(num_bytes):>12.2f}"
        for bandwidth in BANDWIDTHS_GBPS.values():
            row += f" {transfer_ms(num_bytes, bandwidth):>20.3f}"
        print(row)

    print("-" * 92)
    print("time unit: ms")


def print_batch_table():
    """打印固定 prompt 长度下，batch 内多请求同时转移 KV 的总成本。"""
    per_token = kv_bytes_per_token()
    num_bytes = per_token * REFERENCE_PROMPT

    print(f"\nKV transfer cost by batch size (prompt={REFERENCE_PROMPT} tokens/request)")
    print("-" * 92)
    header = f"{'batch size':>12} {'total KV MiB':>14}"
    for name in BANDWIDTHS_GBPS:
        header += f" {name[:18]:>20}"
    print(header)
    print("-" * 92)

    for batch_size in BATCH_SIZES:
        total_bytes = num_bytes * batch_size
        row = f"{batch_size:>12} {mib(total_bytes):>14.2f}"
        for bandwidth in BANDWIDTHS_GBPS.values():
            row += f" {transfer_ms(total_bytes, bandwidth):>20.3f}"
        print(row)

    print("-" * 92)
    print("time unit: ms")


def main():
    per_token = kv_bytes_per_token()
    print("=" * 92)
    print("KV Transfer Cost Analysis for Prefill/Decode Disaggregation")
    print("=" * 92)
    print("Assumed model shape (edit as needed):")
    print(f"  num_layers={NUM_LAYERS}, num_kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}, dtype_bytes={DTYPE_BYTES}")
    print(f"KV bytes per token: {per_token} bytes ({per_token / 1024:.1f} KiB)")

    print_prompt_table()
    print_batch_table()

    print("\nNotes")
    print("-" * 92)
    print("- PD disaggregation moves the prompt KV from prefill workers to decode workers.")
    print("- Long prompts and large batches can make KV transfer a first-class cost.")
    print("- GQA/MQA reduce NUM_KV_HEADS and therefore reduce KV transfer size.")
    print("- Hybrid/Mamba-style models may have additional non-transformer state; this script only estimates standard KV.")
    print("- If transfer time approaches prefill time, PD disaggregation needs faster interconnect or transfer/compute overlap.")


if __name__ == "__main__":
    main()
