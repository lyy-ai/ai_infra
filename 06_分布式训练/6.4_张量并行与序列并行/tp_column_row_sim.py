# 7.4 张量并行与序列并行：列切分与行切分模拟
#
# 运行：
#   cd /data/liyangyang/ai_infra/06_分布式训练
#   /data/liyangyang/qwen35_env/bin/python 6.4_张量并行与序列并行/tp_column_row_sim.py
import numpy as np


def column_parallel_linear(X, W, world_size):
    """按列切分权重，每张卡计算一部分输出，最后 all-gather"""
    H_out = W.shape[1]
    split = H_out // world_size
    local_outputs = [X @ W[:, i * split:(i + 1) * split] for i in range(world_size)]
    return np.concatenate(local_outputs, axis=-1)


def row_parallel_linear(X, W, world_size):
    """按行切分权重，输入也切分，最后 all-reduce"""
    H_in = W.shape[0]
    split = H_in // world_size
    local_outputs = [
        X[..., i * split:(i + 1) * split] @ W[i * split:(i + 1) * split, :]
        for i in range(world_size)
    ]
    return sum(local_outputs)


def main():
    B, S, H_in, H_out = 2, 8, 16, 32
    X = np.random.randn(B * S, H_in).astype(np.float32)
    W = np.random.randn(H_in, H_out).astype(np.float32)

    Y_ref = X @ W
    Y_col = column_parallel_linear(X, W, world_size=4)
    Y_row = row_parallel_linear(X, W, world_size=4)

    print("column parallel max diff:", np.max(np.abs(Y_ref - Y_col)))
    print("row parallel max diff:", np.max(np.abs(Y_ref - Y_row)))


if __name__ == "__main__":
    main()
