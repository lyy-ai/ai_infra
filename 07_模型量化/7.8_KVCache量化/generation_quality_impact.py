# generation_quality_impact.py
import torch
import torch.nn.functional as F


def kivi_quantize_2bit(x, scheme='per-channel'):
    """KIVI 2-bit 量化（复用自 kivi_2bit_quantize.py）"""
    if scheme == 'per-channel':
        dim = -1
    elif scheme == 'per-token':
        dim = 1
    else:
        raise ValueError(scheme)

    x_min = x.min(dim=dim, keepdim=True).values
    x_max = x.max(dim=dim, keepdim=True).values
    qmax = 3
    scale = (x_max - x_min) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zero_point = torch.round(-x_min / scale).clamp(0, qmax)
    x_q = torch.round(x / scale + zero_point).clamp(0, qmax).to(torch.uint8)
    return x_q, scale, zero_point


def kivi_dequantize_2bit(x_q, scale, zero_point):
    return scale * (x_q.to(torch.float32) - zero_point)


def synthetic_attention(q, k, v, k_quant_scheme=None, v_quant_scheme=None):
    """
    简化的单头注意力，可选择对 K 和 V 做 KIVI 量化。
    """
    if k_quant_scheme:
        k_q, k_scale, k_zp = kivi_quantize_2bit(k, k_quant_scheme)
        k = kivi_dequantize_2bit(k_q, k_scale, k_zp)
    if v_quant_scheme:
        v_q, v_scale, v_zp = kivi_quantize_2bit(v, v_quant_scheme)
        v = kivi_dequantize_2bit(v_q, v_scale, v_zp)

    scores = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out


def demo_quality_impact():
    """对比 FP16 KV Cache vs KIVI 2-bit KV Cache 的注意力输出差异"""
    torch.manual_seed(42)
    seq_len = 2048
    head_dim = 128

    # 模拟查询向量和长序列的 K/V
    q = torch.randn(1, 1, head_dim)
    k = torch.randn(1, seq_len, head_dim)
    v = torch.randn(1, seq_len, head_dim)

    # FP16 baseline
    out_fp16 = synthetic_attention(q, k, v, None, None)

    # INT8 量化（对称量化后反量化）
    k_scale_int8 = k.abs().max() / 127
    v_scale_int8 = v.abs().max() / 127
    k_int8 = (torch.round(k / k_scale_int8).clamp(-128, 127) * k_scale_int8)
    v_int8 = (torch.round(v / v_scale_int8).clamp(-128, 127) * v_scale_int8)
    out_int8 = synthetic_attention(q, k_int8, v_int8, None, None)

    # KIVI 2-bit
    out_kivi = synthetic_attention(q, k, v, 'per-channel', 'per-token')

    mse_int8 = torch.mean((out_fp16 - out_int8) ** 2).item()
    mse_kivi = torch.mean((out_fp16 - out_kivi) ** 2).item()
    cos_int8 = F.cosine_similarity(out_fp16, out_int8, dim=-1).mean().item()
    cos_kivi = F.cosine_similarity(out_fp16, out_kivi, dim=-1).mean().item()

    print(f"Attention output comparison (seq_len={seq_len}, head_dim={head_dim})")
    print(f"  INT8  vs FP16: MSE={mse_int8:.6f}, Cosine Similarity={cos_int8:.6f}")
    print(f"  KIVI2 vs FP16: MSE={mse_kivi:.6f}, Cosine Similarity={cos_kivi:.6f}")

    # 模拟“大海捞针”：某个 token 的 K 和 V 都包含关键信息
    needle_idx = seq_len // 2
    k_needle = k.clone()
    v_needle = v.clone()
    # 让 needle 位置的 key 与 query 高度对齐，value 设置强信号
    k_needle[0, needle_idx, :] = q[0, 0, :] * 5.0
    v_needle[0, needle_idx, :] = 10.0
    out_fp16_needle = synthetic_attention(q, k_needle, v_needle, None, None)
    out_kivi_needle = synthetic_attention(q, k_needle, v_needle, 'per-channel', 'per-token')

    # 查看注意力是否仍然聚焦到 needle 位置
    scores_fp16 = torch.matmul(q, k_needle.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_fp16 = F.softmax(scores_fp16, dim=-1)
    scores_kivi = torch.matmul(q, k_needle.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_kivi = F.softmax(scores_kivi, dim=-1)

    print(f"\nNeedle retrieval test (needle at index {needle_idx})")
    print(f"  FP16  attention peak index: {attn_fp16.argmax(dim=-1).item()}")
    print(f"  KIVI2 attention peak index: {attn_kivi.argmax(dim=-1).item()}")
    print(f"  FP16  attention mass at needle: {attn_fp16[0, 0, needle_idx].item():.4f}")
    print(f"  KIVI2 attention mass at needle: {attn_kivi[0, 0, needle_idx].item():.4f}")


if __name__ == "__main__":
    demo_quality_impact()
