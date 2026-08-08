# qat_model_demo.py
import torch
import torch.nn as nn


class FakeQuantize(nn.Module):
    def __init__(self, num_bits=8):
        super().__init__()
        self.num_bits = num_bits
        self.qmin = -(2 ** (num_bits - 1))
        self.qmax = 2 ** (num_bits - 1) - 1

    def forward(self, x):
        scale = x.abs().max() / (2 ** (self.num_bits - 1) - 1)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        x_int = torch.round(x / scale).clamp(self.qmin, self.qmax)
        x_deq = x_int * scale
        return x + (x_deq - x).detach()


class QATLinear(nn.Module):
    """可替换 nn.Linear 的 QAT 版本"""
    def __init__(self, in_features, out_features, num_bits=8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.weight_quant = FakeQuantize(num_bits)
        self.act_quant = FakeQuantize(num_bits)

    def forward(self, x):
        w = self.weight_quant(self.linear.weight)
        x = self.act_quant(x)
        return torch.nn.functional.linear(x, w, self.linear.bias)


class QATMLP(nn.Module):
    """一个 QAT 版本的 MLP，类似 LLM 中的 FFN"""
    def __init__(self, hidden_size=512, intermediate_size=1024, num_bits=8):
        super().__init__()
        self.gate_proj = QATLinear(hidden_size, intermediate_size, num_bits)
        self.up_proj = QATLinear(hidden_size, intermediate_size, num_bits)
        self.down_proj = QATLinear(intermediate_size, hidden_size, num_bits)
        self.act = nn.SiLU()

    def forward(self, x):
        gate = self.act(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


def replace_linear_with_qat(model, num_bits=8):
    """递归把模型中所有 nn.Linear 替换成 QATLinear"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            qat_module = QATLinear(module.in_features, module.out_features, num_bits)
            qat_module.linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                qat_module.linear.bias.data = module.bias.data.clone()
            setattr(model, name, qat_module)
        else:
            replace_linear_with_qat(module, num_bits)


def demo_qat_model():
    torch.manual_seed(0)
    mlp = QATMLP(hidden_size=256, intermediate_size=512, num_bits=8)
    x = torch.randn(2, 10, 256)
    y = mlp(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")

    total_params = sum(p.numel() for p in mlp.parameters())
    print(f"Total parameters: {total_params}")

    # 模拟 QAT 训练一步
    y.sum().backward()
    print("Backward pass succeeded, all Linear layers have gradients:")
    for name, p in mlp.named_parameters():
        if p.grad is not None:
            print(f"  {name}: grad_norm={p.grad.norm().item():.4f}")


if __name__ == "__main__":
    demo_qat_model()
