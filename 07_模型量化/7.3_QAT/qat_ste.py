# qat_ste.py
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class STEQuantize(torch.autograd.Function):
    """自定义 autograd 函数，显式实现 STE"""

    @staticmethod
    def forward(ctx, x, scale, qmin, qmax):
        x_int = torch.round(x / scale).clamp(qmin, qmax)
        x_deq = x_int * scale
        ctx.save_for_backward(torch.ones_like(x))
        return x_deq

    @staticmethod
    def backward(ctx, grad_output):
        # STE：梯度直接穿过
        return grad_output, None, None, None


def visualize_ste():
    """可视化 STE 前向阶梯 vs 反向直通"""
    x = torch.linspace(-3, 3, 100, requires_grad=True)
    scale = 0.5
    qmin, qmax = -3, 3

    y = STEQuantize.apply(x, scale, qmin, qmax)
    y.sum().backward()

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x.detach().numpy(), y.detach().numpy())
    plt.title("Forward: staircase (round + clip)")
    plt.xlabel("input x")
    plt.ylabel("fake_quant(x)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x.detach().numpy(), x.grad.numpy())
    plt.title("Backward: straight-through (gradient ≈ 1)")
    plt.xlabel("input x")
    plt.ylabel("gradient")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("ste_visualize.png")
    print("Saved STE visualization to ste_visualize.png")


def test_ste_gradient():
    """验证 STE 确实把梯度传回了输入"""
    x = torch.randn(4, 8, requires_grad=True)
    scale = 0.1
    qmin, qmax = -7, 7

    y = STEQuantize.apply(x, scale, qmin, qmax)
    loss = (y ** 2).sum()
    loss.backward()

    print("Input gradient norm:", x.grad.norm().item())
    print("Gradient is non-zero:", x.grad.abs().sum().item() > 0)


if __name__ == "__main__":
    test_ste_gradient()
    visualize_ste()
