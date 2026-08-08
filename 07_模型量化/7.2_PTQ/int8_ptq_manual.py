# int8_ptq_manual.py
import torch
import torch.nn as nn


class ManualInt8Linear:
    """手动实现 INT8 线性层 PTQ"""

    def __init__(self, fp32_linear: nn.Linear):
        self.in_features = fp32_linear.in_features
        self.out_features = fp32_linear.out_features
        self.bias = fp32_linear.bias.data.clone() if fp32_linear.bias is not None else None

        # 先初始化默认 scale，再注册权重
        self.weight_scale = None
        self.weight_zero_point = None
        self.input_scale = None
        self.input_zero_point = None
        self.register_weight(fp32_linear.weight.data)

    def register_weight(self, weight):
        # 权重量化：per-channel，对称量化
        w = weight.detach()
        alpha = torch.max(torch.abs(w.min(dim=1, keepdim=True).values),
                          torch.abs(w.max(dim=1, keepdim=True).values))
        self.weight_scale = alpha / 127.0
        self.weight_scale = torch.where(self.weight_scale == 0,
                                        torch.ones_like(self.weight_scale),
                                        self.weight_scale)
        self.weight_q = torch.round(w / self.weight_scale).clamp(-128, 127).to(torch.int8)

    def calibrate_input(self, x):
        # 激活量化：per-tensor，非对称量化
        x_min, x_max = x.min(), x.max()
        self.input_scale = (x_max - x_min) / 255.0
        if self.input_scale == 0:
            self.input_scale = 1.0
        self.input_zero_point = torch.round(-x_min / self.input_scale).clamp(-128, 127).to(torch.int8)

    def forward(self, x):
        # x: [B, in_features], FP32
        # 量化输入
        x_q = torch.round(x / self.input_scale + self.input_zero_point).clamp(-128, 127).to(torch.float32)
        # 反量化输入
        x_deq = self.input_scale * (x_q - self.input_zero_point)

        # 反量化权重
        w_deq = self.weight_q.to(torch.float32) * self.weight_scale

        # FP32 计算（模拟 INT8 推理）
        out = torch.matmul(x_deq, w_deq.t())
        if self.bias is not None:
            out = out + self.bias
        return out


def test_int8_linear():
    torch.manual_seed(42)
    fc = nn.Linear(64, 32)
    int8_fc = ManualInt8Linear(fc)

    # 模拟校准数据
    x_calib = torch.randn(100, 64)
    int8_fc.calibrate_input(x_calib)

    # 测试
    x_test = torch.randn(10, 64)
    y_fp32 = fc(x_test)
    y_int8 = int8_fc.forward(x_test)

    mse = torch.mean((y_fp32 - y_int8) ** 2).item()
    print(f"FP32 vs INT8 Linear MSE: {mse:.6f}")


if __name__ == "__main__":
    test_int8_linear()
