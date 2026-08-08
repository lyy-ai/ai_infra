# awq_quantize.py
# 简化版 AWQ（Activation-aware Weight Quantization）4-bit 线性层演示

import torch
import torch.nn as nn


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


class AWQInt4Linear:
    """简化版 AWQ 4-bit 线性层"""

    def __init__(self, in_features, out_features, group_size=128):
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_groups = (in_features + group_size - 1) // group_size

        weight = torch.randn(out_features, in_features, dtype=torch.float16)
        self.quantize_weight(weight)

    def compute_activation_scale(self, x, alpha=0.5):
        """per-channel 激活缩放：s_j = mean(|x_j|^alpha)"""
        s = x.abs().pow(alpha).mean(dim=0)
        s = torch.where(s == 0, torch.ones_like(s), s)
        return s

    def quantize_weight(self, weight, act_scale=None):
        out_c, in_c = weight.shape
        pad_len = (self.group_size - in_c % self.group_size) % self.group_size
        if pad_len > 0:
            weight = torch.nn.functional.pad(weight, (0, pad_len), value=0)

        # 如果没有激活缩放，退化为普通 group-wise 量化
        if act_scale is None:
            act_scale = torch.ones(in_c, device=weight.device, dtype=weight.dtype)

        # 将激活缩放应用于权重（按输入通道）
        weight = weight * act_scale.unsqueeze(0)

        w_groups = weight.reshape(out_c, self.num_groups, self.group_size)
        max_abs = w_groups.abs().max(dim=-1, keepdim=True).values
        max_abs = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs)

        self.scales = (max_abs / 7.0).to(torch.float16)
        q = torch.round((w_groups / self.scales + 7).float()).clamp(0, 14).to(torch.uint8)
        q = q.reshape(out_c, self.num_groups * self.group_size)
        self.qweight = pack_int4_to_uint8(q)

        # 保存 activation scale 的逆，用于反量化时还原
        self.act_scale_inv = (1.0 / act_scale).to(torch.float16)

    def forward(self, x):
        q = unpack_uint8_to_int4(self.qweight)
        q = q[:, :self.in_features]
        q = q.reshape(self.out_features, self.num_groups, self.group_size)

        # 反量化：W_hat = scale * (q - 7) / act_scale
        w_deq = self.scales * (q.to(torch.float32) - 7.0)
        w_deq = w_deq.reshape(self.out_features, -1)[:, :self.in_features]
        w_deq = w_deq * self.act_scale_inv.unsqueeze(0)
        w_deq = w_deq.to(torch.float16)
        return torch.matmul(x, w_deq.t())

    __call__ = forward


def test_awq():
    torch.manual_seed(42)
    in_f, out_f = 256, 128
    group_size = 128
    x = torch.randn(2, 8, in_f, dtype=torch.float16)

    fc = nn.Linear(in_f, out_f, dtype=torch.float16)
    awq_fc = AWQInt4Linear(in_f, out_f, group_size=group_size)

    # 用校准数据计算 activation scale
    calib_data = torch.randn(64, in_f, dtype=torch.float16)
    act_scale = awq_fc.compute_activation_scale(calib_data, alpha=0.5)
    awq_fc.quantize_weight(fc.weight.data, act_scale=act_scale)

    y_fp16 = fc(x)
    y_awq = awq_fc(x)
    mse = torch.mean((y_fp16 - y_awq) ** 2).item()
    print(f"FP16 vs AWQ-Int4 MSE: {mse:.6f}")

    fp16_bytes = fc.weight.numel() * 2
    q_bytes = awq_fc.qweight.numel()
    s_bytes = awq_fc.scales.numel() * 2
    print(f"Compression ratio: {fp16_bytes / (q_bytes + s_bytes):.2f}x")


if __name__ == "__main__":
    test_awq()
