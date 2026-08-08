# qat_fake_quantize.py
import torch
import torch.nn as nn


class FakeQuantize(nn.Module):
    """
    伪量化节点：前向模拟量化-反量化，反向使用 STE。
    支持对称 / 非对称、per-tensor / per-channel。
    """
    def __init__(self, num_bits=8, symmetric=True, per_channel=False):
        super().__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel

        if symmetric:
            self.qmin = -(2 ** (num_bits - 1))
            self.qmax = 2 ** (num_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** num_bits - 1

    def _compute_scale_zp(self, x):
        """计算 scale 和 zero point"""
        if self.symmetric:
            if self.per_channel:
                max_abs = x.abs().max(dim=-1, keepdim=True).values
            else:
                max_abs = x.abs().max()
            scale = max_abs / (2 ** (self.num_bits - 1) - 1)
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            zp = torch.zeros_like(scale)
        else:
            if self.per_channel:
                x_min = x.min(dim=-1, keepdim=True).values
                x_max = x.max(dim=-1, keepdim=True).values
            else:
                x_min = x.min()
                x_max = x.max()
            scale = (x_max - x_min) / (self.qmax - self.qmin)
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            zp = torch.round(-x_min / scale).clamp(self.qmin, self.qmax)
        return scale, zp

    def forward(self, x):
        scale, zp = self._compute_scale_zp(x)
        x_int = torch.round(x / scale + zp).clamp(self.qmin, self.qmax)
        x_deq = scale * (x_int - zp)
        # STE：前向返回反量化值，反向梯度直接传给 x
        return x + (x_deq - x).detach()


def test_fake_quantize():
    torch.manual_seed(42)
    fq = FakeQuantize(num_bits=8, symmetric=True)
    x = torch.randn(4, 8, requires_grad=True)
    y = fq(x)
    loss = y.sum()
    loss.backward()

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Gradient is all ones:", x.grad.abs().allclose(torch.ones_like(x.grad)))
    print("Original input sample:", x.detach()[0][:5].tolist())
    print("Fake-quantized output sample:", y.detach()[0][:5].tolist())

    # 8-bit 量化后，输出值应该落在量化网格上
    print("\nMSE between input and fake-quantized:", torch.mean((x.detach() - y.detach()) ** 2).item())


if __name__ == "__main__":
    test_fake_quantize()
