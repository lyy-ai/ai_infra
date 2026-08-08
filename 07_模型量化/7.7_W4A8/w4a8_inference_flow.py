# w4a8_inference_flow.py
# 演示 W4A8 推理流程：激活量化 -> 4-bit 权重反量化 -> GEMM

import torch


def quantize_activation_per_token(x, num_bits=8):
    """per-token 对称量化到 INT8"""
    qmax = 2 ** (num_bits - 1) - 1
    # x shape: [B, S, I]，沿最后一个维度取 max
    max_abs = x.abs().max(dim=-1, keepdim=True).values
    max_abs = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs)
    scale = max_abs / qmax
    x_q = torch.round((x / scale).float()).clamp(-qmax - 1, qmax).to(torch.int8)
    return x_q, scale


def simple_matmul_w4a8(w_q4, scales, x, group_size=128):
    """
    W4A8 推理流程：
    1. 量化激活 x -> INT8
    2. 解包 4-bit 权重 -> 整数
    3. 按组反量化为 FP16
    4. FP16 矩阵乘法
    """
    # 激活量化（per-token）
    x_q, x_scale = quantize_activation_per_token(x)
    x_deq = x_q.to(torch.float16) * x_scale

    # 权重反量化
    out_c, num_groups, gs = w_q4.shape
    w_deq = scales * (w_q4.to(torch.float32) - 7.0)
    w_deq = w_deq.reshape(out_c, num_groups * group_size).to(torch.float16)

    # GEMM
    return torch.matmul(x_deq, w_deq.t())


def demo_w4a8_flow():
    torch.manual_seed(42)
    in_f, out_f = 256, 128
    group_size = 128
    batch = 4
    seq_len = 8

    w_fp16 = torch.randn(out_f, in_f, dtype=torch.float16)
    num_groups = in_f // group_size
    w_groups = w_fp16.reshape(out_f, num_groups, group_size)
    max_abs = w_groups.abs().max(dim=-1, keepdim=True).values
    max_abs = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs)
    scales = (max_abs / 7.0).to(torch.float16)
    w_q4 = torch.round((w_groups / scales + 7).float()).clamp(0, 14).to(torch.uint8)

    x = torch.randn(batch, seq_len, in_f, dtype=torch.float16)
    y_fp16 = torch.matmul(x, w_fp16.t())
    y_w4a8 = simple_matmul_w4a8(w_q4, scales, x, group_size)

    mse = torch.mean((y_fp16 - y_w4a8) ** 2).item()
    print(f"FP16 output shape: {y_fp16.shape}")
    print(f"W4A8 output shape: {y_w4a8.shape}")
    print(f"MSE between FP16 and W4A8: {mse:.6f}")


if __name__ == "__main__":
    demo_w4a8_flow()
