# 10.3 Profiling 平台：Nsight Systems Summary 解析模拟
#
# 运行：
#   cd /data/ai_infra/09_企业级工程体系
#   /data/qwen35_env/bin/python 9.3_Profiling平台/nsys_summary_parser.py


def parse_nsys_summary(text):
    """模拟从 nsys stats 文本中提取关键指标。"""
    metrics = {}
    for line in text.splitlines():
        if "Total Time" in line:
            metrics["total_time_ms"] = float(line.split(":")[1].strip().split()[0])
        elif "GPU Kernel Time" in line:
            metrics["gpu_kernel_time_ms"] = float(line.split(":")[1].strip().split()[0])
        elif "Memcpy Time" in line:
            metrics["memcpy_time_ms"] = float(line.split(":")[1].strip().split()[0])
    if "total_time_ms" in metrics and "gpu_kernel_time_ms" in metrics:
        metrics["gpu_util_ratio"] = metrics["gpu_kernel_time_ms"] / metrics["total_time_ms"]
    return metrics


def main():
    sample = """
Total Time: 120.0 ms
GPU Kernel Time: 90.0 ms
Memcpy Time: 15.0 ms
CPU Time: 15.0 ms
"""
    metrics = parse_nsys_summary(sample)
    print("=== Parsed Nsight Systems Summary ===")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.3f}")


if __name__ == "__main__":
    main()
