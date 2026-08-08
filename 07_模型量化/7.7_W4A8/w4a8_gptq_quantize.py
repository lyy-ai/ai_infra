# w4a8_gptq_quantize.py
# 手动演示 GPTQ-Int4 分组权重量化 + W4A8 激活量化

import torch
import torch.nn as nn


def pack_int4_to_uint8(int4_weights):
    """
    将 int4 权重打包成 uint8。
    每 2 个 int4 打包成 1 个 uint8（低 4 位 + 高 4 位）。
    """
    shape = int4_weights.shape
    last_dim = shape[-1]
    assert last_dim % 2 == 0, "最后一维必须是 2 的倍数"

    packed_shape = list(shape[:-1]) + [last_dim // 2, 2]
    w = int4_weights.reshape(packed_shape)
    shifts = torch.tensor([0, 4], device=w.device, dtype=torch.uint8)
    packed = (w.to(torch.uint8) << shifts).sum(dim=-1)
    return packed


def unpack_uint8_to_int4(packed_weights):
    """将 uint8 解包成两个 int4"""
    q_low = packed_weights & 0xF
    q_high = (packed_weights >> 4) & 0xF
    q = torch.stack([q_low, q_high], dim=-1)
    return q.reshape(*packed_weights.shape[:-1], -1)


class GPTQInt4Linear:
    """模拟 GPTQ-Int4 线性层（对称、group-wise，不含 GPTQ 优化过程）"""

    def __init__(self, in_features, out_features, group_size=128):
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_groups = (in_features + group_size - 1) // group_size

        weight = torch.randn(out_features, in_features, dtype=torch.float16)
        self.quantize_weight(weight)

    def quantize_weight(self, weight):
        out_c, in_c = weight.shape
        pad_len = (self.group_size - in_c % self.group_size) % self.group_size
        if pad_len > 0:
            weight = torch.nn.functional.pad(weight, (0, pad_len), value=0)

        w_groups = weight.reshape(out_c, self.num_groups, self.group_size)
        max_abs = w_groups.abs().max(dim=-1, keepdim=True).values
        max_abs = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs)

        self.scales = (max_abs / 7.0).to(torch.float16)
        q = torch.round((w_groups / self.scales + 7).float()).clamp(0, 14).to(torch.uint8)
        q = q.reshape(out_c, self.num_groups * self.group_size)
        self.qweight = pack_int4_to_uint8(q)

    def forward(self, x):
        q = unpack_uint8_to_int4(self.qweight)
        q = q[:, :self.in_features]
        q = q.reshape(self.out_features, self.num_groups, self.group_size)
        w_deq = self.scales * (q.to(torch.float32) - 7.0)
        w_deq = w_deq.reshape(self.out_features, -1)[:, :self.in_features]
        w_deq = w_deq.to(torch.float16)
        return torch.matmul(x, w_deq.t())

    __call__ = forward


class W4A8Linear(GPTQInt4Linear):
    """在 GPTQ-Int4 权重基础上，把激活也量化到 INT8"""

    def quantize_activation(self, x):
        # per-tensor 对称量化到 INT8
        max_abs = x.abs().max()
        scale = max_abs / 127.0 if max_abs > 0 else 1.0
        x_q = torch.round((x / scale).float()).clamp(-128, 127).to(torch.int8)
        return x_q, scale

    def forward(self, x):
        x_q, x_scale = self.quantize_activation(x)
        # 真实 W4A8 kernel 会直接做 INT8 x INT4/INT8 GEMM；
        # 这里为了演示数值等价性，先反量化回 FP16 再计算。
        x_deq = x_q.to(torch.float16) * x_scale
        return super().forward(x_deq)

    __call__ = forward


def test_w4a8():
    torch.manual_seed(42)
    in_f, out_f = 256, 128
    group_size = 128
    x = torch.randn(2, 8, in_f, dtype=torch.float16)

    fc = nn.Linear(in_f, out_f, dtype=torch.float16)
    w4a8_fc = W4A8Linear(in_f, out_f, group_size=group_size)
    w4a8_fc.quantize_weight(fc.weight.data)

    y_fp16 = fc(x)
    y_w4a8 = w4a8_fc(x)
    mse = torch.mean((y_fp16 - y_w4a8) ** 2).item()

    print(f"FP16 vs W4A8 MSE: {mse:.6f}")

    fp16_bytes = fc.weight.numel() * 2
    qweight_bytes = w4a8_fc.qweight.numel()
    scales_bytes = w4a8_fc.scales.numel() * 2
    total_bytes = qweight_bytes + scales_bytes
    print(f"FP16 weight size: {fp16_bytes} bytes")
    print(f"W4A8 weight size: {total_bytes} bytes")
    print(f"Compression ratio: {fp16_bytes / total_bytes:.2f}x")


if __name__ == "__main__":
    test_w4a8()
