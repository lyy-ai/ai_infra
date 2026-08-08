# dequant_overhead.py
import torch
import time


def fake_int8_gemm_with_dequant(a, b, s_a, s_b, dequant_granularity='per-tensor', c_int32=None):
    """
    模拟 INT8 GEMM + Dequant 的完整流程。
    a, b: 已经量化好的 int8 矩阵
    s_a, s_b: scale
    c_int32: 预计算的 INT32 GEMM 结果，避免重复计算
    """
    if c_int32 is None:
        c_int32 = torch.matmul(a.to(torch.int32), b.to(torch.int32))

    s_out = s_a * s_b
    M, N = c_int32.shape

    if dequant_granularity == 'per-tensor':
        # 整个张量一个 scale
        c_fp = s_out * c_int32.to(torch.float32)
    elif dequant_granularity == 'per-channel':
        # 每输出通道（每列）一个 scale
        scales = torch.full((1, N), s_out, dtype=torch.float32)
        c_fp = scales * c_int32.to(torch.float32)
    elif dequant_granularity == 'per-token':
        # 每输入 token（每行）一个 scale
        scales = torch.full((M, 1), s_out, dtype=torch.float32)
        c_fp = scales * c_int32.to(torch.float32)
    else:
        raise ValueError(f"Unknown granularity: {dequant_granularity}")

    return c_fp


def benchmark_dequant_overhead(M, K, N, iterations=100):
    """对比不同 dequant 粒度的开销"""
    # PyTorch 1.10 中 CUDA 不支持 int32 matmul，统一在 CPU 上演示
    device = torch.device('cpu')
    print(f"Running on CPU (PyTorch 1.10 CUDA does not support int32 matmul)")

    torch.manual_seed(42)
    a = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
    b = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)
    s_a = 0.01
    s_b = 0.02

    # 先预计算 INT32 结果，后续 dequant 都基于它
    c_int32 = torch.matmul(a.to(torch.int32), b.to(torch.int32))

    # 纯 INT8 GEMM（无 dequant）
    start = time.time()
    for _ in range(iterations):
        c = torch.matmul(a.to(torch.int32), b.to(torch.int32))
    gemm_time = time.time() - start

    # 分别测量 dequant 本身的时间
    start = time.time()
    for _ in range(iterations):
        c = fake_int8_gemm_with_dequant(a, b, s_a, s_b, 'per-tensor', c_int32)
    per_tensor_time = time.time() - start

    start = time.time()
    for _ in range(iterations):
        c = fake_int8_gemm_with_dequant(a, b, s_a, s_b, 'per-channel', c_int32)
    per_channel_time = time.time() - start

    start = time.time()
    for _ in range(iterations):
        c = fake_int8_gemm_with_dequant(a, b, s_a, s_b, 'per-token', c_int32)
    per_token_time = time.time() - start

    print(f"M={M}, K={K}, N={N}, iterations={iterations}, device={device}")
    print(f"  Pure INT8 GEMM:       {gemm_time * 1000:.2f} ms")
    print(f"  Dequant per-tensor:   {per_tensor_time * 1000:.2f} ms ({(per_tensor_time/gemm_time)*100:.1f}% of GEMM)")
    print(f"  Dequant per-channel:  {per_channel_time * 1000:.2f} ms ({(per_channel_time/gemm_time)*100:.1f}% of GEMM)")
    print(f"  Dequant per-token:    {per_token_time * 1000:.2f} ms ({(per_token_time/gemm_time)*100:.1f}% of GEMM)")
    print(f"  Note: smaller dequant time means faster; overhead is relative to GEMM time")


if __name__ == "__main__":
    benchmark_dequant_overhead(512, 1024, 512)
