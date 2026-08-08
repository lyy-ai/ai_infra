# 7.8 多卡训练通信优化：Ring AllReduce 模拟
#
# 运行：
#   cd /data/liyangyang/ai_infra/06_分布式训练
#   /data/liyangyang/qwen35_env/bin/python 6.8_多卡训练通信优化/ring_allreduce_sim.py
import numpy as np


def ring_allreduce(data_list):
    """
    data_list: list of arrays, one per rank
    returns: list of arrays, each containing the sum of all input arrays
    """
    n = len(data_list)
    size = len(data_list[0])
    chunk_size = size // n
    result = [d.copy() for d in data_list]

    # Scatter-Reduce: snapshot all sends, then apply receives
    for step in range(n - 1):
        send_values = []
        for rank in range(n):
            send_chunk = (rank - step) % n
            start = send_chunk * chunk_size
            end = start + chunk_size
            send_values.append(result[rank][start:end].copy())
        for rank in range(n):
            recv_chunk = (rank - step - 1) % n
            start = recv_chunk * chunk_size
            end = start + chunk_size
            sender = (rank - 1) % n
            result[rank][start:end] += send_values[sender]

    # AllGather: after scatter-reduce rank r owns chunk (r+1) % n
    for step in range(n - 1):
        send_values = []
        for rank in range(n):
            send_chunk = (rank + 1 - step) % n
            start = send_chunk * chunk_size
            end = start + chunk_size
            send_values.append(result[rank][start:end].copy())
        for rank in range(n):
            recv_chunk = (rank - step) % n
            start = recv_chunk * chunk_size
            end = start + chunk_size
            sender = (rank - 1) % n
            result[rank][start:end] = send_values[sender]

    return result


def main():
    n_ranks = 4
    size = 16
    data_list = [np.ones(size, dtype=np.float32) * (i + 1) for i in range(n_ranks)]

    result = ring_allreduce(data_list)
    expected_sum = sum(i + 1 for i in range(n_ranks))
    for rank, arr in enumerate(result):
        ok = np.allclose(arr, expected_sum)
        print(f"rank {rank}: sum correct = {ok}")


if __name__ == "__main__":
    main()
