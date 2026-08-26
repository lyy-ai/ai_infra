# 3.5 Attention 算子：FlashAttention Online Softmax 思想模拟
#
# 运行：
#   cd /data/ai_infra/02_CUDA编程与HPC高性能计算
#   /data/qwen35_env/bin/python 2.5_Attention算子/flash_attention_sim.py
import numpy as np


def standard_attention(Q, K, V):
    """标准 Attention：显式构造 N x N score 矩阵"""
    d = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d)
    P = np.exp(S - S.max(axis=-1, keepdims=True))
    P /= P.sum(axis=-1, keepdims=True)
    return P @ V, P


def online_softmax_update(m_old, l_old, x_new):
    """Online softmax 局部更新"""
    m_new = np.maximum(m_old, x_new.max())
    l_new = np.exp(m_old - m_new) * l_old + np.sum(np.exp(x_new - m_new))
    return m_new, l_new


def flash_attention_tiled(Q, K, V, tile_n=64):
    """模拟 FlashAttention：按 tile 计算，不构造完整 N x N 矩阵"""
    N, d = Q.shape
    O = np.zeros((N, d))
    m = np.full(N, -np.inf)  # 每个 query 当前最大值
    l = np.zeros(N)          # 每个 query 当前指数和

    for q_start in range(0, N, tile_n):
        q_end = min(q_start + tile_n, N)
        Q_tile = Q[q_start:q_end]

        m_tile = np.full(q_end - q_start, -np.inf)
        l_tile = np.zeros(q_end - q_start)
        O_tile = np.zeros((q_end - q_start, d))

        for k_start in range(0, N, tile_n):
            k_end = min(k_start + tile_n, N)
            K_tile = K[k_start:k_end]
            V_tile = V[k_start:k_end]

            S_tile = Q_tile @ K_tile.T / np.sqrt(d)
            m_new = np.maximum(m_tile, S_tile.max(axis=1))

            # 修正旧值
            alpha = np.exp(m_tile - m_new)
            beta = np.exp(S_tile - m_new[:, None])

            l_new = alpha * l_tile + beta.sum(axis=1)
            O_tile = alpha[:, None] * O_tile + beta @ V_tile

            m_tile = m_new
            l_tile = l_new

        O[q_start:q_end] = O_tile / l_tile[:, None]

    return O


def main():
    np.random.seed(0)
    N, d = 256, 64
    Q = np.random.randn(N, d).astype(np.float32)
    K = np.random.randn(N, d).astype(np.float32)
    V = np.random.randn(N, d).astype(np.float32)

    O_std, P = standard_attention(Q, K, V)
    O_flash = flash_attention_tiled(Q, K, V, tile_n=64)

    diff = np.max(np.abs(O_std - O_flash))
    print(f"max diff between standard and tiled attention: {diff:.6f}")
    print(f"peak memory of standard attention score matrix: {N * N * 4 / 1024:.1f} KB")


if __name__ == "__main__":
    main()
