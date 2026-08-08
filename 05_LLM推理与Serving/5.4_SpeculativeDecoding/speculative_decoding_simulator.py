#!/usr/bin/env python3
"""
Speculative Decoding 模拟器。

本脚本用纯 Python 模拟 Speculative Decoding 的核心流程：
1. 用 Draft 分布生成 gamma 个候选 token
2. 用 Target 分布并行验证这些候选 token
3. 使用 Rejection Sampling 保证输出分布无偏
4. 统计平均接受长度和理论加速比

无需 GPU 或 transformers，可直接运行。
"""
import numpy as np
import random


def create_correlated_distributions(vocab_size: int, similarity: float = 0.7):
    """
    创建两个相关的概率分布，模拟 Draft Model 和 Target Model。

    Args:
        vocab_size: 词表大小
        similarity: 两个分布的相似度，0~1，越高 Draft 越接近 Target

    Returns:
        draft_probs, target_probs
    """
    # 生成一个基础分布
    base = np.random.dirichlet(np.ones(vocab_size) * 0.5)

    # Draft 分布 = similarity * base + (1 - similarity) * noise
    draft_noise = np.random.dirichlet(np.ones(vocab_size) * 0.5)
    draft_probs = similarity * base + (1 - similarity) * draft_noise
    draft_probs /= draft_probs.sum()

    # Target 分布 = similarity * base + (1 - similarity) * different_noise
    target_noise = np.random.dirichlet(np.ones(vocab_size) * 0.5)
    target_probs = similarity * base + (1 - similarity) * target_noise
    target_probs /= target_probs.sum()

    return draft_probs, target_probs


def generate_draft_tokens(draft_probs: np.ndarray, gamma: int = 5) -> list:
    """Draft Model 自回归生成 gamma 个候选 token。"""
    vocab_size = len(draft_probs)
    tokens = []
    for _ in range(gamma):
        token = np.random.choice(vocab_size, p=draft_probs)
        tokens.append(token)
    return tokens


def rejection_sampling(
    draft_tokens: list,
    draft_probs: np.ndarray,
    target_probs: np.ndarray,
    gamma: int = 5,
) -> list:
    """
    Target Model 对候选 token 进行 Rejection Sampling 验证。

    返回：被接受的 token 列表（如果中途拒绝，包含重新采样的 token）
    """
    accepted = []

    for i in range(gamma):
        x = draft_tokens[i]
        q_x = draft_probs[x]
        p_x = target_probs[x]

        # 接受概率 alpha = min(1, p(x) / q(x))
        alpha = min(1.0, p_x / q_x) if q_x > 0 else 1.0

        if random.random() < alpha:
            accepted.append(x)
        else:
            # 拒绝后，从修正分布 p'(x) = normalize(max(0, p(x) - q(x))) 中采样
            corrected = np.maximum(target_probs - draft_probs, 0.0)
            corrected_sum = corrected.sum()

            if corrected_sum > 1e-9:
                corrected /= corrected_sum
            else:
                # 退化情况，直接使用 target 分布
                corrected = target_probs

            new_token = np.random.choice(len(corrected), p=corrected)
            accepted.append(new_token)
            break  # 拒绝后停止本轮

    return accepted


def speculative_decoding_round(
    draft_probs: np.ndarray,
    target_probs: np.ndarray,
    gamma: int = 5,
) -> list:
    """执行一轮 Speculative Decoding，返回本轮生成的 token 数。"""
    draft_tokens = generate_draft_tokens(draft_probs, gamma)
    accepted = rejection_sampling(draft_tokens, draft_probs, target_probs, gamma)
    return accepted


def theoretical_accept_rate(draft_probs: np.ndarray, target_probs: np.ndarray) -> float:
    """计算理论上的期望接受率：sum(min(p, q))。"""
    return np.minimum(draft_probs, target_probs).sum()


def simulate(
    vocab_size: int = 10000,
    gamma: int = 5,
    similarity: float = 0.7,
    num_rounds: int = 1000,
):
    """运行模拟并输出统计结果。"""
    print("=" * 70)
    print("Speculative Decoding Simulator")
    print("=" * 70)
    print(f"Vocab size: {vocab_size}")
    print(f"Draft-Target similarity: {similarity}")
    print(f"Speculative tokens (gamma): {gamma}")
    print(f"Simulation rounds: {num_rounds}")
    print()

    draft_probs, target_probs = create_correlated_distributions(vocab_size, similarity)

    accept_lengths = []
    for _ in range(num_rounds):
        accepted = speculative_decoding_round(draft_probs, target_probs, gamma)
        accept_lengths.append(len(accepted))

    avg_accept = np.mean(accept_lengths)
    theoretical_rate = theoretical_accept_rate(draft_probs, target_probs)

    print(f"Theoretical expected accept rate: {theoretical_rate:.4f}")
    print(f"Average accepted tokens per round: {avg_accept:.4f}")
    print(f"Theoretical speedup vs autoregressive: {avg_accept:.2f}x")
    print(f"Max tokens per round: {max(accept_lengths)}")
    print(f"Min tokens per round: {min(accept_lengths)}")

    # 不同接受长度的分布
    from collections import Counter
    dist = Counter(accept_lengths)
    print("\nAccepted tokens distribution:")
    print(f"{'Tokens':>10} {'Count':>10} {'Percentage':>12}")
    print("-" * 35)
    for k in sorted(dist.keys()):
        pct = dist[k] / num_rounds * 100
        print(f"{k:>10} {dist[k]:>10} {pct:>11.2f}%")

    print("=" * 70)


def sweep_similarity():
    """扫描不同 Draft-Target 相似度下的加速比。"""
    print("\n" + "=" * 70)
    print("Sweep: Draft-Target Similarity vs Speedup")
    print("=" * 70)

    vocab_size = 10000
    gamma = 5
    num_rounds = 500

    print(f"{'Similarity':>12} {'Accept Rate':>14} {'Avg Accepted':>14} {'Speedup':>10}")
    print("-" * 55)
    for similarity in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
        draft_probs, target_probs = create_correlated_distributions(vocab_size, similarity)
        accept_lengths = []
        for _ in range(num_rounds):
            accepted = speculative_decoding_round(draft_probs, target_probs, gamma)
            accept_lengths.append(len(accepted))
        avg_accept = np.mean(accept_lengths)
        rate = theoretical_accept_rate(draft_probs, target_probs)
        print(f"{similarity:>12.2f} {rate:>14.4f} {avg_accept:>14.4f} {avg_accept:>10.2f}x")

    print("=" * 70)


def sweep_gamma():
    """扫描不同 gamma 下的加速比。"""
    print("\n" + "=" * 70)
    print("Sweep: Gamma vs Speedup")
    print("=" * 70)

    vocab_size = 10000
    similarity = 0.7
    num_rounds = 500

    print(f"{'Gamma':>10} {'Avg Accepted':>14} {'Speedup':>10}")
    print("-" * 40)
    for gamma in [1, 2, 3, 5, 7, 10]:
        draft_probs, target_probs = create_correlated_distributions(vocab_size, similarity)
        accept_lengths = []
        for _ in range(num_rounds):
            accepted = speculative_decoding_round(draft_probs, target_probs, gamma)
            accept_lengths.append(len(accepted))
        avg_accept = np.mean(accept_lengths)
        print(f"{gamma:>10} {avg_accept:>14.4f} {avg_accept:>10.2f}x")

    print("=" * 70)


if __name__ == "__main__":
    simulate(vocab_size=10000, gamma=5, similarity=0.7, num_rounds=1000)
    sweep_similarity()
    sweep_gamma()

    print("\nKey Takeaways:")
    print("- Higher Draft-Target similarity → higher accept rate → better speedup.")
    print("- Larger gamma helps when accept rate is high, but has diminishing returns.")
    print("- Actual speedup is lower than theoretical due to Draft Model overhead.")
    print("- Speculative Decoding is most beneficial for large Target Models.")
