# DiT 推理成本估算器（17.1 节配套）
# 运行: python dit_cost_estimator.py
# 只依赖标准库
#
# 估算不同输出规格下的序列长度、attention FLOPs 与相对成本，
# 展示三个成本杠杆：分辨率/帧数、去噪步数、CFG

VAE_STRIDE = 8          # VAE 下采样倍率
PATCH = 2               # DiT patch 边长
HIDDEN = 3072           # DiT hidden 维度（14B 级模型的量级）
N_LAYERS = 40


def seq_len(width, height, frames=1):
    return frames * (width // VAE_STRIDE // PATCH) * (height // VAE_STRIDE // PATCH)


def step_flops(width, height, frames=1):
    """单步 DiT 前向 FLOPs 估算：线性层(≈4×N×d²×layers) + attention(2×N²×d×layers)"""
    n = seq_len(width, height, frames)
    d = HIDDEN
    linear = 4 * n * d * d * N_LAYERS
    attn = 2 * n * n * d * N_LAYERS
    return linear + attn, n


def total_cost(width, height, frames=1, steps=30, cfg=True):
    f, n = step_flops(width, height, frames)
    return f * steps * (2 if cfg else 1), n


def fmt(flops):
    return f"{flops/1e12:.1f} TFLOPs"


def main():
    print("=" * 78)
    print("DiT 推理成本估算（14B 级模型，30 步，CFG 开启）")
    print("=" * 78)
    print(f"{'规格':<22}{'序列长度':>10}{'总计算量':>14}{'相对成本':>10}")
    print("-" * 78)
    base, _ = total_cost(512, 512)
    cases = [
        ("图 512x512", 512, 512, 1),
        ("图 1024x1024", 1024, 1024, 1),
        ("图 2048x2048", 2048, 2048, 1),
        ("视频 480p 16帧", 832, 480, 16),
        ("视频 720p 81帧", 1280, 720, 81),
        ("视频 1080p 81帧", 1920, 1080, 81),
    ]
    for name, w, h, f in cases:
        cost, n = total_cost(w, h, f)
        print(f"{name:<22}{n:>10,}{fmt(cost):>14}{cost/base:>9.1f}x")

    print()
    print("=" * 78)
    print("三大成本杠杆（以 720p 81帧 视频为例）")
    print("=" * 78)
    ref, _ = total_cost(1280, 720, 81)
    levers = [
        ("基准：30步 + CFG", 30, True),
        ("关 CFG（蒸馏后单前向）", 30, False),
        ("步数蒸馏：30 -> 8 步", 8, False),
        ("步数蒸馏 + 半分辨率草稿", 8, False),
    ]
    for name, steps, cfg in levers:
        if "半分辨率" in name:
            cost, _ = total_cost(640, 360, 41, steps, cfg)  # 先低分辨率少帧再超分
            note = "（另需超分模型开销）"
        else:
            cost, _ = total_cost(1280, 720, 81, steps, cfg)
            note = ""
        print(f"{name:<30}{fmt(cost):>14}  节省 {(1-cost/ref)*100:>5.1f}%  {note}")

    print()
    print("读法：")
    print("1. 分辨率/帧数是最猛的成本驱动（序列长度平方进入 attention）")
    print("2. 关 CFG 白省一半（前提是做了引导蒸馏）")
    print("3. 步数蒸馏叠加引导蒸馏（30步+CFG -> 8步单前向）约省 87%——双重杠杆")
    print("4. 级联生成（低分辨率草稿+超分）把平方成本拆成两段小成本")


if __name__ == "__main__":
    main()
