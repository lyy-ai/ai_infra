# 数据配比模拟：DoReMi/RegMix 思路教学演示（16.3 节配套）
# 运行: python data_mixing_simulator.py
# 只依赖标准库 + numpy
#
# 模型假设（教学简化）：每个域提供一份"独特技能"，
# 单域技能随 token 投入饱和（收益递减）：skill_i = B_i * (1 - exp(-tokens_i / C_i))
# 模型总技能 = 各域技能之和；loss = BASE - 总技能（越低越好）。
# 由此可以演示两个真实现象：
#   1. 全投单一域不是最优（单域饱和，预算浪费在边际收益趋零处）
#   2. 均匀配比也不是最优（各域技能上限与饱和速度不同）

import numpy as np

rng = np.random.default_rng(3)

DOMAINS = ["books", "filtered_web", "raw_web"]
B = np.array([0.80, 0.60, 0.30])    # 各域技能上限（数据质量）
C = np.array([300.0, 400.0, 500.0])  # 各域饱和尺度
TOTAL_TOKENS = 1000.0               # 归一化 token 预算
BASE = 3.0


def mixture_loss(weights):
    weights = np.asarray(weights, dtype=float) / np.sum(weights)
    tokens = weights * TOTAL_TOKENS
    skills = B * (1.0 - np.exp(-tokens / C))
    return BASE - float(np.sum(skills))


def main():
    print("=" * 76)
    print("数据配比模拟：3 个域，固定 token 预算，搜索最优配比")
    print("=" * 76)

    candidates = {
        "均匀配比 (1/3,1/3,1/3)": [1, 1, 1],
        "经验配比 (10%,70%,20%)": [1, 7, 2],
        "全投高质量 (100%,0,0)": [1, 0, 0],
    }
    print(f"{'方案':<26}{'loss(越低越好)':>14}")
    print("-" * 76)
    for name, w in candidates.items():
        print(f"{name:<26}{mixture_loss(w):>14.4f}")

    # RegMix 思路：随机采样配比 -> 拟合回归面 -> 在预测面上搜索最优
    print("-" * 76)
    print("RegMix 式搜索：采样 400 组随机配比，拟合回归面并搜索最优点")
    samples = rng.dirichlet([1, 1, 1], size=400)
    losses = np.array([mixture_loss(w) for w in samples])
    X = np.column_stack([samples, samples ** 2, np.ones(len(samples))])
    coef, *_ = np.linalg.lstsq(X, losses, rcond=None)

    def pred(w):
        w = np.asarray(w)
        return float(coef @ np.concatenate([w, w ** 2, [1.0]]))

    grid = rng.dirichlet([1, 1, 1], size=20000)
    best_w, best_l = min(((w, pred(w)) for w in grid), key=lambda t: t[1])
    print(f"回归面预测的最优配比: books={best_w[0]:.1%}, filtered_web={best_w[1]:.1%}, raw_web={best_w[2]:.1%}")
    print(f"真实评估该配比: loss={mixture_loss(best_w):.4f}（对照上面三个方案）")

    print("-" * 76)
    print("读法：")
    print("1. 全投高质量域并非最优 —— 单域技能饱和后，追加 token 的边际收益趋零")
    print("2. 均匀配比几乎从不是最优 —— 各域技能上限(B)与饱和速度(C)不同")
    print("3. 小样本回归 + 面上搜索（RegMix/DoReMi 思路）用有限实验逼近最优配比")
    print("4. 真实场景同理：能力剖面由配比决定，配比要用小模型实验搜索而非拍脑袋")


if __name__ == "__main__":
    main()
