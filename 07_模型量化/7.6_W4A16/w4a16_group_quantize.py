# w4a16_group_quantize.py
import torch
import torch.nn as nn


def pack_int4_to_uint32(int4_weights):
    """
    将 int4 权重打包成 uint32。
    假设 int4_weights 最后一个维度大小是 8 的倍数。
    每 8 个 4bit 打包成 1 个 uint32。
    """
    shape = int4_weights.shape
    last_dim = shape[-1]
    assert last_dim % 8 == 0, "最后一维必须是 8 的倍数"
    
    # reshape 成 [..., last_dim//8, 8]
    packed_shape = list(shape[:-1]) + [last_dim // 8, 8]
    w = int4_weights.reshape(packed_shape)
    
    # 每个 int4 占据 4 bit，低位到高位
    shifts = torch.arange(0, 32, 4, device=w.device, dtype=torch.int64)
    packed = (w.to(torch.int64) << shifts).sum(dim=-1)
    return packed.to(torch.long)


def unpack_uint32_to_int4(packed_weights, total_int4):
    """
    将 uint32 解包成 int4 权重。
    total_int4 是期望还原的 int4 元素总数。
    """
    shape = packed_weights.shape
    shifts = torch.arange(0, 32, 4, device=packed_weights.device, dtype=torch.int64)
    shifts = shifts.view(1, 1, -1)
    
    w = ((packed_weights.to(torch.int64).unsqueeze(-1) >> shifts) & 0xF).to(torch.uint8)
    w = w.reshape(-1)
    return w[:total_int4]


class W4A16Linear:
    """手动实现 W4A16 线性层"""

    def __init__(self, in_features, out_features, group_size=32):
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_groups = (in_features + group_size - 1) // group_size
        self.padded_in_features = self.num_groups * group_size

    def quantize_weight(self, weight):
        """对权重进行 group-wise 4bit 对称量化"""
        out_c, in_c = weight.shape
        assert in_c == self.in_features, "输入维度不匹配"
        
        # padding 到 group_size 整数倍
        pad_len = self.padded_in_features - in_c
        if pad_len > 0:
            weight = torch.nn.functional.pad(weight, (0, pad_len), value=0)
        
        # reshape: [out_c, num_groups, group_size]
        w_groups = weight.reshape(out_c, self.num_groups, self.group_size)
        
        # 每组求 max_abs
        max_abs = w_groups.abs().max(dim=-1, keepdim=True).values
        max_abs = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs)
        
        # scale = max_abs / 7
        self.scales = (max_abs / 7.0).to(torch.float16)
        
        # 量化：qi = clip(round(Wi / scale + 7), 0, 14)
        # 在 CPU 上 round 可能不支持 fp16，先用 fp32 计算
        q = torch.round(w_groups.float() / self.scales.float() + 7).clamp(0, 14).to(torch.uint8)
        
        # reshape 回 [out_c, padded_in_features]
        q = q.reshape(out_c, self.padded_in_features)
        
        # 打包成 uint32
        self.qweight = pack_int4_to_uint32(q)

    def forward(self, x):
        """x: [B, in_features], fp16"""
        # 解包 int4
        total_int4 = self.out_features * self.padded_in_features
        q = unpack_uint32_to_int4(self.qweight, total_int4)
        q = q.reshape(self.out_features, self.padded_in_features)
        q = q[:, :self.in_features]
        
        # reshape 到 [out_c, num_groups, group_size]
        q = q.reshape(self.out_features, self.num_groups, self.group_size)
        
        # 反量化：W_hat = scale * (q - 7)
        w_deq = self.scales.float() * (q.to(torch.float32) - 7.0)
        w_deq = w_deq.reshape(self.out_features, self.padded_in_features)
        w_deq = w_deq[:, :self.in_features].to(torch.float16)
        
        # 矩阵乘法
        out = torch.matmul(x, w_deq.t())
        return out


def test_w4a16():
    torch.manual_seed(42)
    in_f, out_f = 128, 64
    group_size = 32
    
    # FP16 参考
    fc = nn.Linear(in_f, out_f, dtype=torch.float16)
    
    # W4A16 量化版
    w4a16_fc = W4A16Linear(in_f, out_f, group_size=group_size)
    with torch.no_grad():
        w4a16_fc.quantize_weight(fc.weight.data)
    
    x = torch.randn(4, in_f, dtype=torch.float16)
    y_fp16 = fc(x)
    y_w4a16 = w4a16_fc.forward(x)
    
    mse = torch.mean((y_fp16 - y_w4a16) ** 2).item()
    print(f"FP16 vs W4A16 Linear MSE: {mse:.6f}")
    
    # 计算压缩比
    fp16_bytes = fc.weight.numel() * 2
    qweight_bytes = w4a16_fc.qweight.numel() * 4
    scales_bytes = w4a16_fc.scales.numel() * 2
    total_bytes = qweight_bytes + scales_bytes
    ratio = fp16_bytes / total_bytes
    print(f"FP16 size: {fp16_bytes} bytes")
    print(f"W4A16 size: {total_bytes} bytes")
    print(f"Compression ratio: {ratio:.2f}x")


if __name__ == "__main__":
    test_w4a16()
