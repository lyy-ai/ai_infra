# 10.5 监控与告警：SLO 告警规则
#
# 运行：
#   cd /data/ai_infra/09_企业级工程体系
#   /data/qwen35_env/bin/python 9.5_监控与告警/slo_alert_rule.py


def check_slo(metrics, thresholds):
    alerts = []
    if metrics["p99_latency_ms"] > thresholds["p99_latency_ms"]:
        alerts.append(("P99_LATENCY", metrics["p99_latency_ms"], thresholds["p99_latency_ms"]))
    if metrics["error_rate"] > thresholds["error_rate"]:
        alerts.append(("ERROR_RATE", metrics["error_rate"], thresholds["error_rate"]))
    if metrics["gpu_memory_used_ratio"] > thresholds["gpu_memory_used_ratio"]:
        alerts.append(("GPU_MEMORY", metrics["gpu_memory_used_ratio"], thresholds["gpu_memory_used_ratio"]))
    return alerts


def main():
    metrics = {
        "p99_latency_ms": 230.0,
        "error_rate": 0.0005,
        "gpu_memory_used_ratio": 0.92,
    }
    thresholds = {
        "p99_latency_ms": 200.0,
        "error_rate": 0.001,
        "gpu_memory_used_ratio": 0.90,
    }
    alerts = check_slo(metrics, thresholds)
    print("=== SLO Alert Check ===")
    if not alerts:
        print("All SLOs satisfied.")
    else:
        for name, value, threshold in alerts:
            print(f"ALERT {name}: current={value}, threshold={threshold}")


if __name__ == "__main__":
    main()
