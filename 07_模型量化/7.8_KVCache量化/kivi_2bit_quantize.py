# kivi_2bit_quantize.py
import torch


def kivi_quantize_2bit(x, scheme='per-channel'):
    """
    KIVI 风格的 2-bit 非对称量化。

    Args:
        x: 输入张量，形状通常为 [batch, seq_len, num_heads, head_dim]
        scheme: 'per-channel' 用于 Key（沿 head_dim），
                'per-token' 用于 Value（沿 seq_len）
    """
    if scheme == 'per-channel':
        # Key: 每个 head_dim 通道独立量化
        dim = -1
        x_min = x.min(dim=dim, keepdim=True).values
        x_max = x.max(dim=dim, keepdim=True).values
    elif scheme == 'per-token':
        # Value: 每个 token 独立量化
        dim = 1
        x_min = x.min(dim=dim, keepdim=True).values
        x_max = x.max(dim=dim, keepdim=True).values
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    # 2-bit 只有 4 个码点: 0, 1, 2, 3
    qmax = 3
    scale = (x_max - x_min) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zero_point = torch.round(-x_min / scale).clamp(0, qmax)

    x_q = torch.round(x / scale + zero_point).clamp(0, qmax).to(torch.uint8)
    return x_q, scale, zero_point


def kivi_dequantize_2bit(x_q, scale, zero_point):
    """KIVI 2-bit 反量化"""
    return scale * (x_q.to(torch.float32) - zero_point)


def demo_kivi_quantize():
    """演示 KIVI 2-bit 对 Key 和 Value 的量化效果"""
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 1024
    num_heads = 32
    head_dim = 128

    # 模拟 Key 和 Value
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim)

    # Key 使用 per-channel 量化
    k_q, k_scale, k_zp = kivi_quantize_2bit(k, scheme='per-channel')
    k_deq = kivi_dequantize_2bit(k_q, k_scale, k_zp)

    # Value 使用 per-token 量化
    v_q, v_scale, v_zp = kivi_quantize_2bit(v, scheme='per-token')
    v_deq = kivi_dequantize_2bit(v_q, v_scale, v_zp)

    k_mse = torch.mean((k - k_deq) ** 2).item()
    v_mse = torch.mean((v - v_deq) ** 2).item()

    print(f"Key  per-channel 2-bit quantization MSE: {k_mse:.6f}")
    print(f"Value per-token   2-bit quantization MSE: {v_mse:.6f}")
    print(f"Original FP16 size: {k.numel() * 2 / 1024**2:.2f} MB per KV tensor")
    print(f"KIVI 2-bit size:    {k_q.numel() * 1 / 8 / 1024**2:.2f} MB per KV tensor (uint8 packed)")
    print(f"Compression ratio: 8x")


if __name__ == "__main__":
    demo_kivi_quantize()
