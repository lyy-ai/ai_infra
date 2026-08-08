# 10.2 性能回归平台：回归门禁判定
#
# 运行：
#   cd /data/liyangyang/ai_infra/09_企业级工程体系
#   /data/liyangyang/qwen35_env/bin/python 9.2_性能回归平台/regression_gate.py


def check_regression(baseline, current, thresholds):
    print(f"{'Metric':>16} | {'Baseline':>10} | {'Current':>10} | {'Change':>10} | {'Threshold':>10} | {'Action':>8}")
    print("-" * 78)
    actions = []
    for metric, base in baseline.items():
        cur = current[metric]
        change = (cur - base) / base
        threshold = thresholds.get(metric, 0.05)
        # latency/memory: higher is worse; throughput: lower is worse
        is_regression = change > threshold if metric in ["ttft_ms", "tbt_ms", "memory_gb"] else -change > threshold
        action = "BLOCK" if is_regression else "PASS"
        actions.append((metric, action))
        print(f"{metric:>16} | {base:>10.2f} | {cur:>10.2f} | {change:>+9.1%} | {threshold:>10.1%} | {action:>8}")
    blocked = any(a == "BLOCK" for _, a in actions)
    print()
    print("Result:", "BLOCK" if blocked else "PASS")


def main():
    baseline = {"ttft_ms": 120.0, "tbt_ms": 45.0, "throughput": 1500.0, "memory_gb": 32.0}
    current = {"ttft_ms": 118.0, "tbt_ms": 48.0, "throughput": 1480.0, "memory_gb": 31.5}
    thresholds = {"ttft_ms": 0.05, "tbt_ms": 0.05, "throughput": 0.05, "memory_gb": 0.10}
    check_regression(baseline, current, thresholds)


if __name__ == "__main__":
    main()
