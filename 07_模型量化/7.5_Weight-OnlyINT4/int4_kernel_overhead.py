# int4_kernel_overhead.py
# INT4 权重的 unpack + dequant 开销分析

import torch
import time


def pack_int4_to_uint8(int4_weights):
    """每 2 个 int4 打包成一个 uint8"""
    shape = int4_weights.shape
    last_dim = shape[-1]
    assert last_dim % 2 == 0, "最后一维必须是 2 的倍数"
    w = int4_weights.reshape(*shape[:-1], last_dim // 2, 2)
    shifts = torch.tensor([0, 4], device=w.device, dtype=torch.uint8)
    return (w.to(torch.uint8) << shifts).sum(dim=-1)


def unpack_uint8_to_int4(packed_weights):
    """将 uint8 解包成两个 int4"""
    q_low = packed_weights & 0xF
    q_high = (packed_weights >> 4) & 0xF
    q = torch.stack([q_low, q_high], dim=-1)
    return q.reshape(*packed_weights.shape[:-1], -1)


def unpack_dequant_gemm(qweight, scales, x, group_size=128):
    """非融合路径：unpack -> dequant -> FP16 GEMM"""
    out_c, packed_in_c = qweight.shape
    in_c = packed_in_c * 2
    q = unpack_uint8_to_int4(qweight)
    q = q.reshape(out_c, in_c // group_size, group_size)
    w_deq = scales * (q.to(torch.float32) - 7.0)
    w_deq = w_deq.reshape(out_c, in_c).to(torch.float16)
    return torch.matmul(x, w_deq.t())


def benchmark_unpack_overhead(M, K, N, group_size=128, iterations=100):
    """粗略测量 unpack + dequant 开销（CPU 演示）"""
    torch.manual_seed(42)
    w_fp16 = torch.randn(N, K, dtype=torch.float16)
    x = torch.randn(M, K, dtype=torch.float16)

    # 量化权重
    num_groups = K // group_size
    w_groups = w_fp16.reshape(N, num_groups, group_size)
    max_abs = w_groups.abs().max(dim=-1, keepdim=True).values
    max_abs = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs)
    scales = (max_abs / 7.0).to(torch.float16)
    q = torch.round((w_groups / scales + 7).float()).clamp(0, 14).to(torch.uint8)
    q = q.reshape(N, K)
    qweight = pack_int4_to_uint8(q)

    # 1) FP16 参考
    start = time.time()
    for _ in range(iterations):
        y_fp16 = torch.matmul(x, w_fp16.t())
    fp16_time = time.time() - start

    # 2) 非融合 W4A16：unpack + dequant + GEMM
    start = time.time()
    for _ in range(iterations):
        y_w4 = unpack_dequant_gemm(qweight, scales, x, group_size)
    w4_time = time.time() - start

    # 3) 仅 unpack + dequant（不含 GEMM）
    start = time.time()
    for _ in range(iterations):
        q = unpack_uint8_to_int4(qweight)
        q = q.reshape(N, K // group_size, group_size)
        w_deq = scales * (q.to(torch.float32) - 7.0)
        w_deq = w_deq.reshape(N, K).to(torch.float16)
    unpack_time = time.time() - start

    mse = torch.mean((y_fp16 - y_w4) ** 2).item()
    print(f"M={M}, K={K}, N={N}, group_size={group_size}")
    print(f"  FP16 time:       {fp16_time * 1000:.2f} ms")
    print(f"  W4A16 unpack:    {w4_time * 1000:.2f} ms ({(w4_time/fp16_time)*100:.1f}% of FP16)")
    print(f"  Unpack only:     {unpack_time * 1000:.2f} ms ({(unpack_time/w4_time)*100:.1f}% of W4A16)")
    print(f"  MSE vs FP16:     {mse:.6f}")


if __name__ == "__main__":
    benchmark_unpack_overhead(4, 512, 256, group_size=128)
