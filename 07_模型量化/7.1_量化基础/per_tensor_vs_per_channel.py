# per_tensor_vs_per_channel.py
import torch


def per_tensor_quantize(w, bits=8):
    w_min, w_max = w.min(), w.max()
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    scale = (w_max - w_min) / (qmax - qmin)
    zp = qmin - torch.round(w_min / scale)
    zp = torch.clamp(zp, qmin, qmax)
    w_q = torch.round(w / scale + zp).clamp(qmin, qmax)
    return scale * (w_q - zp)


def per_channel_quantize(w, bits=8):
    # w shape: [out_channels, in_channels]
    out_c = w.shape[0]
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    w_deq = torch.zeros_like(w)

    for c in range(out_c):
        wc = w[c]
        w_min, w_max = wc.min(), wc.max()
        scale = (w_max - w_min) / (qmax - qmin)
        zp = qmin - torch.round(w_min / scale)
        zp = torch.clamp(zp, qmin, qmax)
        w_q = torch.round(wc / scale + zp).clamp(qmin, qmax)
        w_deq[c] = scale * (w_q - zp)

    return w_deq


if __name__ == "__main__":
    torch.manual_seed(0)
    # 模拟卷积权重，8 个输出通道
    w = torch.randn(8, 16)
    # 让每个通道尺度差异大
    for c in range(8):
        w[c] *= (c + 1) * 0.5

    w_pt = per_tensor_quantize(w)
    w_pc = per_channel_quantize(w)

    mse_pt = torch.mean((w - w_pt) ** 2).item()
    mse_pc = torch.mean((w - w_pc) ** 2).item()

    print(f"Per-tensor MSE: {mse_pt:.6f}")
    print(f"Per-channel MSE: {mse_pc:.6f}")
