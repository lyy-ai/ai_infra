# gptq_quantize.py
# 简化版 GPTQ 逐层量化概念演示：Hessian + 误差补偿

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


def quantize_group_symmetric(w_groups, group_size=128):
    """对分组权重做对称 4-bit 量化，返回 qweight 和 scales"""
    max_abs = w_groups.abs().max(dim=-1, keepdim=True).values
    max_abs = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs)
    scales = (max_abs / 7.0).to(torch.float16)
    q = torch.round((w_groups / scales + 7).float()).clamp(0, 14).to(torch.uint8)
    return q, scales


class SimpleGPTQInt4Linear:
    """
    简化版 GPTQ 4-bit 线性层。
    这里只演示：计算 Hessian、逐列量化、误差补偿的核心思想。
    完整 GPTQ 还需处理 block、顺序、cholesky 等细节。
    """

    def __init__(self, in_features, out_features, group_size=128):
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_groups = (in_features + group_size - 1) // group_size

    def quantize_weight(self, weight, calib_X, damp_percent=0.01):
        """
        weight: [out_features, in_features]
        calib_X: [B, in_features] 校准数据
        """
        out_c, in_c = weight.shape
        pad_len = (self.group_size - in_c % self.group_size) % self.group_size
        if pad_len > 0:
            weight = torch.nn.functional.pad(weight, (0, pad_len), value=0)
            calib_X = torch.nn.functional.pad(calib_X, (0, pad_len), value=0)
            in_c = in_c + pad_len
            self.num_groups = (in_c + self.group_size - 1) // self.group_size

        # 1. 计算 Hessian H = X^T X，并加阻尼
        H = (calib_X.float().T @ calib_X.float()).to(torch.float32)
        H = H + damp_percent * torch.diag(torch.diag(H))
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))

        W = weight.clone().to(torch.float32)
        q_list = []
        for j in range(out_c):
            w_j = W[j].clone()
            for i in range(in_c):
                # 获取当前元素所在分组的 scale
                g = i // self.group_size
                # 这里先用动态 scale 计算，简化演示
                group_w = w_j[g * self.group_size:(g + 1) * self.group_size]
                scale = group_w.abs().max() / 7.0
                scale = max(scale, 1e-8)

                # 2. 量化当前元素
                q_i = torch.round((w_j[i] / scale + 7)).clamp(0, 14)
                delta = w_j[i] - scale * (q_i - 7.0)

                # 3. 将误差分配到剩余未量化元素（OBS-style）
                if i + 1 < in_c:
                    w_j[i + 1:] -= delta * H_inv[i + 1:, i] / H_inv[i, i]
            q_list.append(w_j)

        # 4. 重新按分组量化（把补偿后的权重再量化）
        W_comp = torch.stack(q_list, dim=0)
        W_groups = W_comp.reshape(out_c, self.num_groups, self.group_size)
        q, self.scales = quantize_group_symmetric(W_groups, self.group_size)
        q = q.reshape(out_c, self.num_groups * self.group_size)
        self.qweight = pack_int4_to_uint8(q)

    def forward(self, x):
        """反量化后做 FP16 GEMM"""
        q_low = self.qweight & 0xF
        q_high = (self.qweight >> 4) & 0xF
        q = torch.stack([q_low, q_high], dim=-1).reshape(
            self.out_features, self.num_groups, self.group_size
        )
        q = q[:, :, :self.in_features]
        w_deq = self.scales * (q.to(torch.float32) - 7.0)
        w_deq = w_deq.reshape(self.out_features, -1)[:, :self.in_features]
        w_deq = w_deq.to(torch.float16)
        return torch.matmul(x, w_deq.t())

    __call__ = forward


def test_gptq_concept():
    """演示 GPTQ 核心思想：Hessian + 误差补偿"""
    torch.manual_seed(42)
    in_f, out_f = 64, 32
    group_size = 32

    fc = nn.Linear(in_f, out_f, dtype=torch.float16)
    calib = torch.randn(128, in_f, dtype=torch.float16)
    x = torch.randn(2, 8, in_f, dtype=torch.float16)

    gptq = SimpleGPTQInt4Linear(in_f, out_f, group_size=group_size)
    gptq.quantize_weight(fc.weight.data, calib)

    y_fp16 = fc(x)
    y_gptq = gptq(x)
    mse = torch.mean((y_fp16 - y_gptq) ** 2).item()

    print(f"qweight shape: {gptq.qweight.shape}")
    print(f"scales shape: {gptq.scales.shape}")
    print(f"FP16 vs GPTQ-Int4 MSE: {mse:.6f}")
    print("GPTQ 概念演示完成（完整实现请参考 auto-gptq）")


if __name__ == "__main__":
    test_gptq_concept()
