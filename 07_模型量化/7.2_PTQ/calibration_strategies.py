# calibration_strategies.py
import torch
import numpy as np


def minmax_calibration(x, bits=8):
    """MinMax 校准"""
    return x.min().item(), x.max().item()


def percentile_calibration(x, bits=8, percentile=99.99):
    """Percentile 校准"""
    lower = torch.quantile(x, (100 - percentile) / 100).item()
    upper = torch.quantile(x, percentile / 100).item()
    return lower, upper


def entropy_calibration(x, bits=8, num_bins=2048):
    """基于 KL 散度的 Entropy 校准"""
    x_np = x.detach().cpu().numpy().flatten()
    abs_max = float(np.abs(x_np).max())

    hist, bin_edges = np.histogram(
        np.clip(x_np, -abs_max, abs_max),
        bins=num_bins,
        range=(-abs_max, abs_max),
    )

    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    num_levels = qmax - qmin + 1
    best_kl = float("inf")
    best_threshold = abs_max

    for i in range(num_levels, num_bins):
        # 原始分布：i 个 bin + 1 个 outlier bin
        P = np.zeros(num_levels)
        bins_per_group = i / (num_levels - 1)
        for j in range(i):
            p_idx = min(int(j / bins_per_group), num_levels - 2)
            P[p_idx] += hist[j]
        P[-1] = np.sum(hist[i:])

        # 量化后分布：假设均匀量化后落在 num_levels 个级上
        Q = np.ones_like(P)

        P = P / (P.sum() + 1e-10)
        Q = Q / (Q.sum() + 1e-10)
        kl = np.sum(P * np.log((P + 1e-10) / (Q + 1e-10)))

        if kl < best_kl:
            best_kl = kl
            best_threshold = bin_edges[i]

    return -best_threshold, best_threshold


def quantize_with_range(x, xmin, xmax, bits=8):
    """用给定的范围做非对称量化并返回 MSE"""
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    scale = max((xmax - xmin) / (qmax - qmin), 1e-8)
    zp = torch.round(torch.tensor(-xmin / scale)).clamp(qmin, qmax)
    x_q = torch.round(x / scale + zp).clamp(qmin, qmax)
    x_deq = scale * (x_q - zp)
    mse = torch.mean((x - x_deq) ** 2).item()
    return mse


if __name__ == "__main__":
    torch.manual_seed(42)

    # 模拟带异常值的激活分布
    x = torch.randn(10000)
    x[0] = 20.0
    x[1] = -18.0

    strategies = {
        "MinMax": minmax_calibration,
        "Percentile": percentile_calibration,
        "Entropy": entropy_calibration,
    }

    print("=" * 50)
    print("不同校准策略对比")
    print("=" * 50)
    for name, fn in strategies.items():
        xmin, xmax = fn(x)
        mse = quantize_with_range(x, xmin, xmax)
        print(f"{name:12s}: range=[{xmin:8.3f}, {xmax:8.3f}], MSE={mse:.6f}")
