# calibration_methods.py
import torch
import numpy as np


def minmax_calibration(x, bits=8):
    return x.min(), x.max()


def percentile_calibration(x, bits=8, percentile=99.99):
    lower = torch.quantile(x, (100 - percentile) / 100)
    upper = torch.quantile(x, percentile / 100)
    return lower, upper


def mse_calibration(x, bits=8, num_candidates=100):
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    abs_max = torch.abs(x).max()
    best_mse = float("inf")
    best_scale = None

    for ratio in torch.linspace(0.5, 1.0, num_candidates):
        alpha = abs_max * ratio
        scale = alpha / qmax
        x_q = torch.round(x / scale).clamp(-qmax - 1, qmax)
        x_deq = x_q * scale
        mse = torch.mean((x - x_deq) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_scale = scale

    alpha = best_scale * qmax
    return -alpha, alpha


def kl_calibration(x, bits=8, num_bins=2048):
    """
    简化的 KL 散度校准，参考 TensorRT 思想。
    用 KL 散度衡量：将原始分布截断到 [-T, T] 并量化到 num_quant_levels 个级别后，
    量化分布与原始截断分布之间的差异。
    """
    x_np = x.detach().cpu().numpy()
    abs_max = float(np.abs(x_np).max())
    hist, bin_edges = np.histogram(
        np.clip(x_np, -abs_max, abs_max),
        bins=num_bins,
        range=(-abs_max, abs_max),
    )

    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    num_quant_levels = qmax - qmin + 1
    best_kl = float("inf")
    best_threshold = abs_max

    # 遍历截断阈值 bin index，保证映射有效
    for i in range(num_quant_levels, num_bins):
        threshold = bin_edges[i]

        # 原始截断分布：正侧 i 个 bin + 1 个 outlier bin
        # 为了与量化后分布同维度，将 i 个 bin 合并成 num_quant_levels-1 个 bin，
        # 最后一个 bin 放 outlier。
        P = np.zeros(num_quant_levels)
        bins_per_group = i / (num_quant_levels - 1)
        for j in range(i):
            p_idx = min(int(j / bins_per_group), num_quant_levels - 2)
            P[p_idx] += hist[j]
        P[-1] = np.sum(hist[i:])  # outlier bin

        # 量化后分布：假设在 [-threshold, threshold] 内均匀量化到 num_quant_levels 级
        # 由于是对称量化，量化后的值会近似均匀地落在 num_quant_levels 个级上，
        # 这里用每个 P 的组对应一个量化级（即 Q 与 P 同维且均匀）。
        Q = np.ones_like(P)  # 均匀分布作为近似

        # 避免 log(0)
        P = P / (P.sum() + 1e-10)
        Q = Q / (Q.sum() + 1e-10)
        kl = np.sum(P * np.log((P + 1e-10) / (Q + 1e-10)))
        if kl < best_kl:
            best_kl = kl
            best_threshold = threshold

    return -best_threshold, best_threshold


if __name__ == "__main__":
    torch.manual_seed(42)
    # 模拟带异常值的激活
    x = torch.randn(10000)
    x[0] = 20.0  # 异常值

    methods = {
        "Min-Max": minmax_calibration,
        "Percentile": percentile_calibration,
        "MSE": mse_calibration,
        "KL": kl_calibration,
    }

    for name, fn in methods.items():
        xmin, xmax = fn(x)
        scale = (xmax - xmin) / 255
        x_q = torch.round((x - xmin) / scale).clamp(-128, 127)
        x_deq = x_q * scale + xmin
        mse = torch.mean((x - x_deq) ** 2).item()
        print(f"{name:12s}: range=[{xmin:7.3f}, {xmax:7.3f}], MSE={mse:.6f}")
