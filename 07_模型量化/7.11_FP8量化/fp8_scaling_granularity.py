# fp8_scaling_granularity.py
#
# FP8 量化粒度教学模拟器：
#   构造一个含 outlier 通道的激活张量（模拟 LLM 激活），
#   对比三种 scaling 粒度下 FP8(E4M3) 量化的 SNR：
#     1. per-tensor        ：整个张量共享一个 scale
#     2. per-channel       ：每个输出通道（列）一个 scale
#     3. per-1x128-block   ：每个 token 行内每 128 通道一块（DeepSeek-V3 做法）
#
# 运行：
#   cd /data/liyangyang/ai_infra/07_模型量化/7.11_FP8量化
#   /data/liyangyang/qwen35_env/bin/python fp8_scaling_granularity.py

import numpy as np

E4M3_MAX = 448.0  # E4M3 最大正规格化数


# ---------------------------------------------------------------------------
# E4M3 量化-反量化（算术法，向量化 O(N)）
# ---------------------------------------------------------------------------

def e4m3_cast(x):
    """把已缩放到 E4M3 范围内的浮点张量 cast 到最近的可表示值。

    正规数：尾数按 2^(e-3) 网格四舍五入；次正规（< 2^-6）：按 2^-9 网格取整；
    溢出（含尾数进位到 480 的 NaN 编码）饱和到 448。
    """
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    a = np.minimum(np.abs(x), E4M3_MAX)

    res = np.zeros_like(a)
    sub = a < 2.0 ** -6                        # 次正规区
    res[sub] = np.round(a[sub] * 512.0) / 512.0

    n = ~sub                                   # 正规区
    an = a[n]
    e = np.clip(np.floor(np.log2(an)), -6, 8)  # E4M3 指数范围 [-6, 8]
    step = 2.0 ** (e - 3)
    res[n] = np.minimum(np.round(an / step) * step, E4M3_MAX)
    return sign * res


def fp8_quant_with_scale(x, amax):
    """scale = amax / 448，先缩放再 cast 再反缩放。amax 为 0 的块原样返回。"""
    scale = np.where(amax > 0, amax / E4M3_MAX, 1.0)
    scaled = x / np.where(scale > 0, scale, 1.0)
    return e4m3_cast(scaled) * scale


def snr_db(x, x_q):
    noise = x - x_q
    nz = np.sum(noise ** 2)
    return 10.0 * np.log10(np.sum(x ** 2) / nz) if nz > 0 else float("inf")


# ---------------------------------------------------------------------------
# 三种粒度的量化
# ---------------------------------------------------------------------------

def quant_per_tensor(x):
    """粒度 1：整个张量共享一个 scale。"""
    return fp8_quant_with_scale(x, np.max(np.abs(x)))


def quant_per_channel(x):
    """粒度 2：每个输出通道（列）一个 scale，shape [1, C]。"""
    amax = np.max(np.abs(x), axis=0, keepdims=True)
    return fp8_quant_with_scale(x, amax)


def quant_per_tile_1x128(x, tile=128):
    """粒度 3：DeepSeek-V3 激活做法，每行内每 128 通道一个 tile 独立 scale。"""
    T, C = x.shape
    assert C % tile == 0
    xt = x.reshape(T, C // tile, tile)
    amax = np.max(np.abs(xt), axis=2, keepdims=True)     # [T, C/128, 1]
    qt = fp8_quant_with_scale(xt, amax)
    return qt.reshape(T, C)


def make_activation(n_tokens=2048, n_channels=4096, n_outlier_ch=4,
                    outlier_gain=50.0, seed=0):
    """模拟 LLM 激活：大部分通道 ~N(0,1)，少数 outlier 通道幅值放大约 50 倍。"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_tokens, n_channels))
    outlier_cols = rng.choice(n_channels, size=n_outlier_ch, replace=False)
    x[:, outlier_cols] *= outlier_gain
    return x, outlier_cols


def main():
    x, outlier_cols = make_activation()
    print("=" * 76)
    print("FP8(E4M3) scaling 粒度对比：含 outlier 的激活张量")
    print("=" * 76)
    print(f"激活 shape = {x.shape}  (token 数 x 通道数)")
    print(f"outlier 通道：{sorted(outlier_cols.tolist())}，幅值放大约 50 倍")
    print(f"全局 amax = {np.abs(x).max():.2f}，去除 outlier 后 amax 约 "
          f"{np.abs(np.delete(x, outlier_cols, axis=1)).max():.2f}")
    print()

    results = [
        ("per-tensor（整 tensor 一个 scale）", quant_per_tensor(x),
         "1 个 scale", "outlier 元素自身贡献巨额误差能量"),
        ("per-channel（每列一个 scale）", quant_per_channel(x),
         f"{x.shape[1]} 个 scale", "列内非 amax 的 outlier 元素仍有大误差"),
        ("per-1x128-block（DS-V3 激活做法）", quant_per_tile_1x128(x),
         f"{x.shape[0] * x.shape[1] // 128} 个 scale", "outlier 成为所在块 amax，精确映射到 448"),
    ]

    print(f"{'量化粒度':<34}{'SNR(dB)':>10}{'scale 数量':>14}   行为分析")
    print("-" * 76)
    for name, x_q, n_scale, note in results:
        print(f"{name:<34}{snr_db(x, x_q):>10.2f}{n_scale:>14}   {note}")

    # 正常值（非 outlier 区域）的单独 SNR + 误差能量分解
    print()
    print("误差能量分解（outlier 列 vs 正常列）与正常区域 SNR：")
    mask = np.ones(x.shape[1], dtype=bool)
    mask[outlier_cols] = False
    print(f"  {'量化粒度':<34}{'outlier误差能量':>14}{'正常误差能量':>14}"
          f"{'正常区SNR(dB)':>14}")
    print("  " + "-" * 62)
    x_normal = x[:, mask]
    for name, x_q, _, _ in results:
        e_out = np.sum((x[:, ~mask] - x_q[:, ~mask]) ** 2)
        e_nor = np.sum((x_normal - x_q[:, mask]) ** 2)
        print(f"  {name:<34}{e_out:>14.3e}{e_nor:>14.3e}"
              f"{snr_db(x_normal, x_q[:, mask]):>14.2f}")

    print()
    print("=" * 76)
    print("结论")
    print("=" * 76)
    print("1. FP8 的相对精度近似 scale 不变（指数自适应）：正常数据区域在三种粒度")
    print("   下 SNR 基本持平。这与 INT8 本质不同——INT8 per-tensor 遇到 outlier 时")
    print("   正常值会被压到码本底端全场崩塌（见 fp8_format_demo.py 实验 3）。")
    print("2. FP8 细粒度 scaling 真正要救的不是正常值，而是 outlier 元素自身：")
    print("   per-tensor 下 outlier 幅值大、其量化误差能量反而主导总噪声。")
    print("   当粒度细到 outlier 成为所在块的 amax（1x128 tile 内每行至多一个")
    print("   outlier 通道）时，该元素精确映射到 E4M3 的 448，误差几乎归零，")
    print("   整体 SNR 显著提升。")
    print("3. 反直觉点：per-channel 并不解决问题——列内除 amax 外的 outlier 元素")
    print("   仍带完整相对误差。这也解释了 DeepSeek-V3 为什么对激活选 1x128")
    print("   per-tile 而不是 per-channel。")
    print("4. 工程上 per-1x128-block 还是精度与开销的平衡点：scale 元数据可控，")
    print("   128 的分块与 Tensor Core MMA tile 天然对齐，便于 GEMM kernel 融合；")
    print("   权重侧对应 128x128 per-block scaling，思路相同。")


if __name__ == "__main__":
    main()
