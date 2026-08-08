# 9.5 Benchmark 方法论：统一 benchmark 框架
#
# 运行：
#   cd /data/liyangyang/ai_infra/08_多平台适配
#   /data/liyangyang/qwen35_env/bin/python 8.5_Benchmark方法论/benchmark_framework.py

import time
import statistics


class BenchmarkConfig:
    def __init__(self, name, batch_size, seq_len, num_tokens, warmup=5, iterations=20):
        self.name = name
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_tokens = num_tokens
        self.warmup = warmup
        self.iterations = iterations


class BenchmarkRunner:
    def __init__(self, config, infer_fn):
        self.config = config
        self.infer_fn = infer_fn

    def run(self):
        # warmup
        for _ in range(self.config.warmup):
            self.infer_fn(self.config.batch_size, self.config.num_tokens)

        latencies = []
        for _ in range(self.config.iterations):
            t0 = time.perf_counter()
            self.infer_fn(self.config.batch_size, self.config.num_tokens)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        total_time = sum(latencies) / len(latencies)
        throughput = self.config.batch_size * self.config.num_tokens / (total_time / 1000.0)
        return {
            "mean_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "throughput_tokens_per_sec": throughput,
        }


def dummy_infer(batch_size, num_tokens):
    # 模拟推理耗时：batch 越大越慢
    time.sleep(0.001 * batch_size + 0.0001 * num_tokens)


def main():
    configs = [
        BenchmarkConfig("A100-FP16", 1, 512, 128),
        BenchmarkConfig("A100-FP16", 8, 512, 128),
        BenchmarkConfig("Orin-INT4", 1, 512, 128),
    ]
    print(f"{'Config':>12} | {'Batch':>5} | {'Mean Lat(ms)':>14} | {'Throughput(t/s)':>18}")
    print("-" * 58)
    for cfg in configs:
        runner = BenchmarkRunner(cfg, dummy_infer)
        result = runner.run()
        print(f"{cfg.name:>12} | {cfg.batch_size:>5} | {result['mean_latency_ms']:>14.2f} | {result['throughput_tokens_per_sec']:>18.1f}")


if __name__ == "__main__":
    main()
