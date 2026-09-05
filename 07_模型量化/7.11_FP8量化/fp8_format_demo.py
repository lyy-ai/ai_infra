# fp8_format_demo.py
#
# FP8 数据格式教学模拟器：
#   1. 用查表法构建 E4M3 / E5M2 全部 256 个编码（展示格式定义），
#      用算术方法实现 O(N) 的量化-反量化（用于大张量实验）
#   2. 打印两种格式的可表示范围与精度（相邻值间距）分布
#   3. 三组量化误差实验（对照 FP16 / INT8 per-tensor）：
#      - 轻尾正态分布：理想校准下 INT8 与 FP8 相当，FP8 不占优
#      - 重尾钟形分布：FP8 的非均匀码本优势明显（"浮点对钟形分布更友好"）
#      - 含 outlier / 校准失配：per-tensor INT8 崩塌，FP8 靠指数范围免疫
#
# 运行：
#   cd /data/liyangyang/ai_infra/07_模型量化/7.11_FP8量化
#   /data/liyangyang/qwen35_env/bin/python fp8_format_demo.py

import numpy as np

E4M3_MAX = 448.0    # E4M3 最大正规格化数
E5M2_MAX = 57344.0  # E5M2 最大正规格化数


# ---------------------------------------------------------------------------
# 1. FP8 格式定义：查表法（完整枚举 256 个位模式，用于格式分析）
# ---------------------------------------------------------------------------

def build_fp8_lut(n_exp, n_man, bias, has_inf):
    """枚举全部 256 个位模式，构建 FP8 解码查找表。

    NaN 用 np.nan 表示；has_inf=True 时指数全 1、尾数全 0 编码为 INF。
    E4M3（OCP 标准）无 INF，指数全 1 的所有编码均为 NaN。
    """
    codes = np.zeros(256, dtype=np.float64)
    for i in range(256):
        sign = -1.0 if (i >> 7) & 1 else 1.0
        exp = (i >> n_man) & ((1 << n_exp) - 1)
        man = i & ((1 << n_man) - 1)
        max_exp = (1 << n_exp) - 1
        if exp == max_exp:
            codes[i] = sign * np.inf if (has_inf and man == 0) else np.nan
        elif exp == 0:
            codes[i] = sign * man * 2.0 ** (1 - bias - n_man)   # 次正规
        else:
            codes[i] = sign * (1.0 + man / 2.0 ** n_man) * 2.0 ** (exp - bias)
    return codes


# E4M3：1 符号 + 4 指数 + 3 尾数，bias=7，无 INF
E4M3_LUT = build_fp8_lut(n_exp=4, n_man=3, bias=7, has_inf=False)
# E5M2：1 符号 + 5 指数 + 2 尾数，bias=15，保留 INF/NaN
E5M2_LUT = build_fp8_lut(n_exp=5, n_man=2, bias=15, has_inf=True)


# ---------------------------------------------------------------------------
# 2. 算术法 FP8 量化（向量化 O(N)，与查表法等价）
# ---------------------------------------------------------------------------

def fp8_cast(x, n_man, bias, max_val):
    """把浮点张量 cast 到最近的 FP8 可表示值（round-to-nearest，溢出饱和）。

    正规数：尾数按 2^(e-n_man) 网格四舍五入；次正规：按最小次正规网格取整。
    """
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    a = np.minimum(np.abs(x), max_val)

    min_normal = 2.0 ** (1 - bias)
    sub_grid = 2.0 ** (1 - bias - n_man)      # 次正规网格 = 最小次正规值

    res = np.zeros_like(a)
    sub = a < min_normal
    res[sub] = np.round(a[sub] / sub_grid) * sub_grid

    n = ~sub
    an = a[n]
    e = np.floor(np.log2(an))
    e = np.clip(e, 1 - bias, bias)            # 指数范围 [1-bias, bias]
    step = 2.0 ** (e - n_man)
    val = np.round(an / step) * step          # 尾数进位由乘回 step 自然处理
    res[n] = np.minimum(val, max_val)         # 进位溢出（如 448->480）饱和截断
    return sign * res


def e4m3_cast(x):
    return fp8_cast(x, n_man=3, bias=7, max_val=E4M3_MAX)


def e5m2_cast(x):
    return fp8_cast(x, n_man=2, bias=15, max_val=E5M2_MAX)


def int8_quant_per_tensor(x, amax=None):
    """对称 per-tensor INT8 量化-反量化。amax 缺省用当前张量最大值（理想校准）。"""
    x = np.asarray(x, dtype=np.float64)
    if amax is None:
        amax = np.max(np.abs(x))
    scale = amax / 127.0 if amax > 0 else 1.0
    q = np.clip(np.round(x / scale), -127, 127)
    return q * scale


def snr_db(x, x_q):
    """信噪比：10*log10(信号能量 / 量化噪声能量)，越大越好。"""
    x = np.asarray(x, dtype=np.float64)
    nz = np.sum((x - x_q) ** 2)
    return 10.0 * np.log10(np.sum(x ** 2) / nz) if nz > 0 else float("inf")


# ---------------------------------------------------------------------------
# 3. 格式范围与精度分布
# ---------------------------------------------------------------------------

def show_format_specs():
    print("=" * 72)
    print("一、FP8 两种格式的可表示范围（OCP 标准）")
    print("=" * 72)
    print(f"{'格式':<8}{'符号/指数/尾数':<14}{'最大正数':>12}{'最小正规格化':>14}"
          f"{'最小次正规':>12}{'INF':>6}")
    for name, bits, lut, bias, inf in [("E4M3", "1 / 4 / 3", E4M3_LUT, 7, "无"),
                                       ("E5M2", "1 / 5 / 2", E5M2_LUT, 15, "有")]:
        pos = lut[np.isfinite(lut) & (lut > 0)]
        print(f"{name:<8}{bits:<14}{pos.max():>12.0f}{2.0 ** (1 - bias):>14.2e}"
              f"{pos.min():>12.2e}{inf:>6}")

    print()
    print("E4M3 精度分布（相邻可表示值的相对间距，按数量级分段）：")
    pos = np.sort(np.unique(E4M3_LUT[np.isfinite(E4M3_LUT) & (E4M3_LUT > 0)]))
    rel_gap = np.diff(pos) / pos[:-1]
    print(f"  {'数值区间':<20}{'相对间距中位数':>12}   说明")
    for lo, hi, note in [(2**-9, 2**-6, "次正规区，间距绝对值最小"),
                         (2**-6, 2**-3, "0 附近：数据最密集"),
                         (2**-3, 2**0, ""), (2**0, 2**3, ""),
                         (2**3, 2**6, "大数值：数据稀疏"),
                         (2**6, 448.0, "大数值：数据稀疏")]:
        mask = (pos[:-1] >= lo) & (pos[:-1] < hi)
        if mask.any():
            print(f"  [{lo:>9.4f}, {hi:>8.1f})   {np.median(rel_gap[mask]):>12.4f}   {note}")
    print()
    print("结论：FP8 的码本是非均匀的——相对间距由尾数位数决定（E4M3 约 1/8 ~ 1/16），")
    print("      0 附近绝对分辨率极高，大数值处稀疏，形状与钟形分布的数据密度对齐。")


# ---------------------------------------------------------------------------
# 4. 量化误差对比实验
# ---------------------------------------------------------------------------

def report(name, x, int8_amax=None, note=""):
    print(f"\n【{name}】  n={x.size}, std={x.std():.3f}, amax={np.abs(x).max():.2f}")
    if note:
        print(f"  ({note})")
    print(f"  {'格式':<22}{'SNR(dB)':>10}{'相对误差中位数':>14}{'最大绝对误差':>14}")
    x64 = np.asarray(x, dtype=np.float64)
    candidates = [
        ("FP16 (对照)", x.astype(np.float16).astype(np.float64)),
        ("FP8-E4M3", e4m3_cast(x64)),
        ("FP8-E5M2", e5m2_cast(x64)),
        ("INT8 per-tensor", int8_quant_per_tensor(x64, amax=int8_amax)),
    ]
    abs_x = np.maximum(np.abs(x64), 1e-12)
    for label, x_q in candidates:
        rel = np.abs(x64 - x_q) / abs_x
        print(f"  {label:<22}{snr_db(x64, x_q):>10.2f}{np.median(rel):>14.4f}"
              f"{np.abs(x64 - x_q).max():>14.4f}")


def main():
    np.random.seed(42)
    n = 65536

    show_format_specs()

    print()
    print("=" * 72)
    print("二、量化误差对比实验")
    print("=" * 72)

    # 实验 1：轻尾正态 + 理想校准——INT8 与 FP8 打平甚至略优（如实呈现）
    x1 = np.random.randn(n)
    report("实验 1：标准正态 N(0,1)，INT8 用理想校准", x1,
           note="轻尾、动态范围小：INT8 的 127 级均匀网格已够用，FP8 不占优")

    # 实验 2：重尾钟形分布——FP8 的非均匀码本开始获胜
    mix = np.random.rand(n) < 0.05
    x2 = np.random.randn(n) * np.where(mix, 30.0, 1.0)
    report("实验 2：重尾钟形分布（95% N(0,1) + 5% N(0,30)）", x2,
           note="绝大多数值很小、少量值很大：均匀量化把码本浪费在大数值区")

    # 实验 3：含 outlier——per-tensor scaling 被破坏
    x3 = np.random.randn(n)
    idx = np.random.choice(n, size=n // 1000, replace=False)
    x3[idx] *= 50.0
    report("实验 3：正态分布 + 0.1% outlier (x50)", x3,
           note="模拟 LLM 激活中的 outlier 通道")

    # 实验 4：校准失配——INT8 用校准集的 amax，FP8 不需要校准
    x4_calib = np.random.randn(n)                    # 校准集 amax ~4.5
    x4 = np.random.randn(n) * 8.0                    # 真实数据 std 漂移到 8
    calib_amax = np.abs(x4_calib).max()
    report("实验 4：INT8 用校准集 scale、真实数据分布漂移", x4,
           int8_amax=calib_amax,
           note=f"校准集 amax={calib_amax:.2f}，真实数据 amax={np.abs(x4).max():.2f}；"
                f"FP8 无 scale 概念，天然免疫")

    print()
    print("=" * 72)
    print("三、结论")
    print("=" * 72)
    print("1. 轻尾分布 + 理想校准下，INT8 与 FP8 精度相当（实验 1）——")
    print("   FP8 的优势从来不是\"上限更高\"，而是\"下限更稳\"。")
    print("2. 数据越重尾（越接近真实 LLM 激活），FP8 非均匀码本优势越大（实验 2）。")
    print("3. outlier 对 per-tensor INT8 是毁灭性的（实验 3）：一个 outlier 撑大")
    print("   scale，全部正常值被压到码本底端；FP8 靠指数范围把损伤限制在局部。")
    print("4. INT8 的 scale 依赖校准集代表性，分布漂移即失效（实验 4）；FP8 的")
    print("   范围由指数自适应（E4M3 覆盖约 18 个数量级），通常免校准直接 cast。")
    print("   这正是 FP8 工程链路远短于 INT8 的根本原因。")


if __name__ == "__main__":
    main()
