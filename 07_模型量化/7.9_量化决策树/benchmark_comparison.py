# benchmark_comparison.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class TinyMLP(nn.Module):
    """一个用于演示的小 MLP"""
    def __init__(self, hidden=512, intermediate=1024):
        super().__init__()
        self.gate = nn.Linear(hidden, intermediate)
        self.up = nn.Linear(hidden, intermediate)
        self.down = nn.Linear(intermediate, hidden)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def quantize_symmetric(x, num_bits=8):
    """对称量化"""
    qmax = 2 ** (num_bits - 1) - 1
    scale = x.abs().max() / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (torch.round(x / scale).clamp(-qmax - 1, qmax) * scale, scale)


def quantize_weights_to_8bit(model):
    """把模型权重模拟量化到 8-bit"""
    scales = {}
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            q_param, scale = quantize_symmetric(param.data, 8)
            param.data = q_param
            scales[name] = scale
    return scales


def quantize_weights_to_4bit(model):
    """把模型权重模拟量化到 4-bit"""
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            q_param, _ = quantize_symmetric(param.data, 4)
            param.data = q_param
    return model


def benchmark_model(model, x, iterations=100):
    """测量模型前向时间"""
    model.eval()
    with torch.no_grad():
        # 预热
        for _ in range(10):
            model(x)
        start = time.time()
        for _ in range(iterations):
            model(x)
        elapsed = time.time() - start
    return elapsed / iterations * 1000  # ms


def compare_quantization_methods():
    """对比 FP32 / INT8 / W4A16 模拟量化下的输出差异和速度"""
    torch.manual_seed(42)
    hidden = 512
    intermediate = 1024
    batch = 8
    seq_len = 128

    x = torch.randn(batch, seq_len, hidden)

    # FP32 baseline
    model_fp32 = TinyMLP(hidden, intermediate)
    out_fp32 = model_fp32(x)
    time_fp32 = benchmark_model(model_fp32, x)

    # INT8 权重模拟
    model_int8 = TinyMLP(hidden, intermediate)
    quantize_weights_to_8bit(model_int8)
    out_int8 = model_int8(x)
    time_int8 = benchmark_model(model_int8, x)

    # W4A16 权重模拟（激活 FP32）
    model_w4a16 = TinyMLP(hidden, intermediate)
    quantize_weights_to_4bit(model_w4a16)
    out_w4a16 = model_w4a16(x)
    time_w4a16 = benchmark_model(model_w4a16, x)

    mse_int8 = torch.mean((out_fp32 - out_int8) ** 2).item()
    mse_w4a16 = torch.mean((out_fp32 - out_w4a16) ** 2).item()

    print(f"TinyMLP (hidden={hidden}, intermediate={intermediate})")
    print(f"  FP32:  time={time_fp32:.3f} ms, baseline")
    print(f"  INT8:  time={time_int8:.3f} ms, MSE={mse_int8:.6f}")
    print(f"  W4A16: time={time_w4a16:.3f} ms, MSE={mse_w4a16:.6f}")
    print("\nNote: timing on CPU is for illustration; actual GPU speedup depends on kernel implementation")


if __name__ == "__main__":
    compare_quantization_methods()
