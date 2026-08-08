# symmetric_vs_asymmetric.py
import torch
import numpy as np


def symmetric_quantize(x, bits=8):
    """对称量化"""
    alpha = torch.max(torch.abs(x.min()), torch.abs(x.max()))
    qmax = 2 ** (bits - 1) - 1
    scale = alpha / qmax
    x_q = torch.round(x / scale).clamp(-qmax - 1, qmax)
    x_deq = x_q * scale
    return x_q, x_deq, scale, 0


def asymmetric_quantize(x, bits=8):
    """非对称量化"""
    x_min, x_max = x.min(), x.max()
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = qmin - torch.round(x_min / scale)
    zero_point = torch.clamp(zero_point, qmin, qmax)
    x_q = torch.round(x / scale + zero_point).clamp(qmin, qmax)
    x_deq = scale * (x_q - zero_point)
    return x_q, x_deq, scale, zero_point


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(10) * 3  # 假设激活值

    print("原始张量:", x)
    print()

    sq, s_deq, s_scale, _ = symmetric_quantize(x)
    print("对称量化:")
    print("  scale:", s_scale.item())
    print("  量化值:", sq)
    print("  反量化:", s_deq)
    print(f"  MSE: {torch.mean((x - s_deq) ** 2).item():.6f}")
    print()

    aq, a_deq, a_scale, a_zp = asymmetric_quantize(x)
    print("非对称量化:")
    print("  scale:", a_scale.item())
    print("  zero_point:", a_zp.item())
    print("  量化值:", aq)
    print("  反量化:", a_deq)
    print(f"  MSE: {torch.mean((x - a_deq) ** 2).item():.6f}")
