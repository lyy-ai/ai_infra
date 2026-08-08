# w4a16_inference_flow.py
import torch
import torch.nn as nn


def simple_matmul_fp16(w_fp16, x_fp16):
    """标准 FP16 矩阵乘法"""
    return torch.matmul(x_fp16, w_fp16.t())


def simple_matmul_w4a16(w_q4, scales, x_fp16, group_size=32):
    """
    W4A16 推理流程：
    1. 从存储中解包 4-bit 权重
    2. 按组反量化为 FP16
    3. 与 FP16 激活做矩阵乘法
    """
    out_c, num_groups, gs = w_q4.shape
    # 反量化
    w_deq = scales * (w_q4.to(torch.float32) - 7.0)
    w_deq = w_deq.reshape(out_c, num_groups * group_size)
    w_deq = w_deq.to(torch.float16)
    return torch.matmul(x_fp16, w_deq.t())


def demo_inference_flow():
    torch.manual_seed(42)
    in_f, out_f = 256, 128
    group_size = 32
    batch = 8
    
    # 原始 fp16 权重
    w_fp16 = torch.randn(out_f, in_f, dtype=torch.float16)
    
    # 量化
    num_groups = in_f // group_size
    w_groups = w_fp16.reshape(out_f, num_groups, group_size)
    max_abs = w_groups.abs().max(dim=-1, keepdim=True).values
    max_abs = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs)
    scales = (max_abs / 7.0).to(torch.float16)
    w_q4 = torch.round(w_groups.float() / scales.float() + 7).clamp(0, 14).to(torch.uint8)
    
    # fp16 激活
    x_fp16 = torch.randn(batch, in_f, dtype=torch.float16)
    
    # 标准 FP16 推理
    y_fp16 = simple_matmul_fp16(w_fp16, x_fp16)
    
    # W4A16 推理
    y_w4a16 = simple_matmul_w4a16(w_q4, scales, x_fp16, group_size)
    
    mse = torch.mean((y_fp16 - y_w4a16) ** 2).item()
    print(f"FP16 output shape: {y_fp16.shape}")
    print(f"W4A16 output shape: {y_w4a16.shape}")
    print(f"MSE between FP16 and W4A16: {mse:.6f}")


if __name__ == "__main__":
    demo_inference_flow()
