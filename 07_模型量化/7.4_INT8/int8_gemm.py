# int8_gemm.py
import torch
import torch.nn as nn
import time


def quantize_symmetric(x, num_bits=8):
    """对称量化到 int8"""
    qmax = 2 ** (num_bits - 1) - 1
    scale = x.abs().max() / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    x_q = torch.round(x / scale).clamp(-qmax - 1, qmax).to(torch.int8)
    return x_q, scale


def int8_gemm_naive(a_fp32, b_fp32):
    """
    手动实现 INT8 GEMM：
    1. 量化 A 和 B
    2. INT8 矩阵乘，累加用 INT32
    3. 反量化为 FP32
    """
    a_q, s_a = quantize_symmetric(a_fp32)
    b_q, s_b = quantize_symmetric(b_fp32)

    # INT8 矩阵乘，结果用 INT32 累加
    c_q = torch.matmul(a_q.to(torch.int32), b_q.to(torch.int32))

    # 反量化
    c_fp32 = s_a * s_b * c_q.to(torch.float32)
    return c_fp32


def benchmark_int8_gemm(M, K, N, iterations=100):
    """对比 FP32、FP16、INT8 GEMM 的精度和速度"""
    torch.manual_seed(42)
    a_fp32 = torch.randn(M, K)
    b_fp32 = torch.randn(K, N)

    # FP32 参考
    c_fp32 = torch.matmul(a_fp32, b_fp32)

    # FP16 参考
    a_fp16 = a_fp32.half()
    b_fp16 = b_fp32.half()
    c_fp16 = torch.matmul(a_fp16, b_fp16).float()

    # INT8 实现
    c_int8 = int8_gemm_naive(a_fp32, b_fp32)

    # 精度对比
    mse_fp16 = torch.mean((c_fp32 - c_fp16) ** 2).item()
    mse_int8 = torch.mean((c_fp32 - c_int8) ** 2).item()
    print(f"M={M}, K={K}, N={N}")
    print(f"  FP16 MSE: {mse_fp16:.6f}")
    print(f"  INT8 MSE: {mse_int8:.6f}")

    # 速度对比（粗略，CPU 上不一定能体现加速）
    if torch.cuda.is_available():
        device = torch.device('cuda')
        a_fp32_d = a_fp32.to(device)
        b_fp32_d = b_fp32.to(device)
        a_fp16_d = a_fp32_d.half()
        b_fp16_d = b_fp32_d.half()

        # 预热
        for _ in range(10):
            torch.matmul(a_fp32_d, b_fp32_d)
            torch.matmul(a_fp16_d, b_fp16_d)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            torch.matmul(a_fp32_d, b_fp32_d)
        torch.cuda.synchronize()
        fp32_time = time.time() - start

        start = time.time()
        for _ in range(iterations):
            torch.matmul(a_fp16_d, b_fp16_d)
        torch.cuda.synchronize()
        fp16_time = time.time() - start

        print(f"  FP32 time: {fp32_time * 1000:.2f} ms")
        print(f"  FP16 time: {fp16_time * 1000:.2f} ms")
        print(f"  FP16 speedup: {fp32_time / fp16_time:.2f}x")
    else:
        print("  CUDA not available, skipping GPU benchmark")


if __name__ == "__main__":
    benchmark_int8_gemm(256, 512, 256)
