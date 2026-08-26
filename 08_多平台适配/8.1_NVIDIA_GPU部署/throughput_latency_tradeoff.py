# 9.1 NVIDIA GPU 部署：吞吐-延迟权衡估算
#
# 运行：
#   cd /data/ai_infra/08_多平台适配
#   /data/qwen35_env/bin/python 8.1_NVIDIA_GPU部署/throughput_latency_tradeoff.py


def estimate(params_b, bandwidth_gbps, batch_sizes, dtype_bytes=2):
    """
    简化模型：decode 阶段为 memory-bound，时间 ≈ 参数读取时间。
    """
    param_bytes = params_b * 1e9 * dtype_bytes
    bw = bandwidth_gbps * 1e9
    print(f"Model: {params_b}B params, dtype bytes={dtype_bytes}, bandwidth={bandwidth_gbps} GB/s")
    print(f"{'Batch':>8} | {'Latency(ms/token)':>18} | {'Throughput(tokens/s)':>22}")
    print("-" * 55)
    for b in batch_sizes:
        # 每个 token 推理需要读取一次全部参数
        seconds = param_bytes / bw
        latency_ms = seconds * 1000
        throughput = b / seconds
        print(f"{b:8d} | {latency_ms:18.2f} | {throughput:22.1f}")


def main():
    estimate(params_b=7, bandwidth_gbps=2000, batch_sizes=[1, 4, 8, 16, 32])


if __name__ == "__main__":
    main()
