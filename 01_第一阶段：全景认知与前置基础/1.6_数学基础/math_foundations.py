# 2.2 数学基础：工程数学验证
#
# 运行：
#   /data/qwen35_env/bin/python 1.6_数学基础/math_foundations.py

import numpy as np


def gemm_flops(m, n, k):
    return 2 * m * n * k


def block_matmul_check():
    np.random.seed(0)
    A = np.random.randn(4, 4)
    B = np.random.randn(4, 4)
    C = A @ B
    A11, A12 = A[:2, :2], A[:2, 2:]
    A21, A22 = A[2:, :2], A[2:, 2:]
    B11, B12 = B[:2, :2], B[:2, 2:]
    B21, B22 = B[2:, :2], B[2:, 2:]
    C11 = A11 @ B11 + A12 @ B21
    return np.allclose(C11, C[:2, :2])


def stable_softmax(z):
    z = np.array(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def chain_rule_check():
    # y = x^2, z = sin(y) -> dz/dx = cos(x^2) * 2x
    x = 0.7
    analytic = np.cos(x * x) * 2 * x
    eps = 1e-6
    numeric = (np.sin((x + eps) ** 2) - np.sin((x - eps) ** 2)) / (2 * eps)
    return analytic, numeric


def main():
    print(f"GEMM(4096^3) FLOPs: {gemm_flops(4096, 4096, 4096) / 1e9:.1f} GFLOP")
    print(f"分块矩阵 C11 正确: {block_matmul_check()}")
    print(f"稳定 softmax([1000,1001]): {stable_softmax([1000, 1001])}")
    a, n = chain_rule_check()
    print(f"链式法则: 解析 {a:.6f} vs 数值 {n:.6f}")


if __name__ == "__main__":
    main()
