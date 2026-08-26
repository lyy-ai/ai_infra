# 10.5 监控与告警：GPU 监控
#
# 运行：
#   cd /data/ai_infra/09_企业级工程体系
#   /data/qwen35_env/bin/python 9.5_监控与告警/gpu_monitor.py

try:
    import pynvml
except ImportError:
    pynvml = None


def get_gpu_metrics():
    if pynvml is None:
        print("pynvml not installed, using mock data.")
        return [
            {"gpu": 0, "util": 45.0, "memory_used_mb": 10240, "memory_total_mb": 40960, "temp": 72, "power": 220},
            {"gpu": 1, "util": 30.0, "memory_used_mb": 8192, "memory_total_mb": 40960, "temp": 68, "power": 180},
        ]
    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    metrics = []
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        metrics.append({
            "gpu": i,
            "util": util,
            "memory_used_mb": mem.used // 1024 // 1024,
            "memory_total_mb": mem.total // 1024 // 1024,
            "temp": temp,
            "power": power,
        })
    return metrics


def main():
    metrics = get_gpu_metrics()
    print(f"{'GPU':>4} | {'Util%':>6} | {'Mem Used':>10} | {'Mem Total':>10} | {'Temp':>5} | {'Power(W)':>9}")
    print("-" * 60)
    for m in metrics:
        print(f"{m['gpu']:>4} | {m['util']:>6.1f} | {m['memory_used_mb']:>10d} | {m['memory_total_mb']:>10d} | {m['temp']:>5d} | {m['power']:>9.1f}")


if __name__ == "__main__":
    main()
