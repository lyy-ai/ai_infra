# 9.5 Benchmark 方法论：自动生成对比报告
#
# 运行：
#   cd /data/ai_infra/08_多平台适配
#   /data/qwen35_env/bin/python 8.5_Benchmark方法论/generate_report.py


def generate_markdown_report(results, output_path):
    lines = [
        "# 多平台 Benchmark 对比报告",
        "",
        "| Platform | Batch | Mean Latency(ms) | Throughput(tokens/s) |",
        "|----------|-------|------------------|----------------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['platform']:10s} | {r['batch']:5d} | {r['latency_ms']:16.2f} | {r['throughput']:20.1f} |"
        )
    lines.append("")
    lines.append("## 结论")
    best = max(results, key=lambda x: x["throughput"])
    lines.append(f"- 吞吐最高平台：{best['platform']} (batch={best['batch']})，吞吐 {best['throughput']:.1f} tokens/s")
    fastest = min(results, key=lambda x: x["latency_ms"])
    lines.append(f"- 延迟最低平台：{fastest['platform']}，平均延迟 {fastest['latency_ms']:.2f} ms")

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(content)
    print(f"\nReport saved to {output_path}")


def main():
    results = [
        {"platform": "A100-FP16", "batch": 1, "latency_ms": 45.2, "throughput": 2832.0},
        {"platform": "A100-FP16", "batch": 8, "latency_ms": 98.5, "throughput": 10395.0},
        {"platform": "Orin-INT4", "batch": 1, "latency_ms": 320.0, "throughput": 400.0},
        {"platform": "Ascend-FP16", "batch": 1, "latency_ms": 52.0, "throughput": 2461.0},
    ]
    generate_markdown_report(results, "benchmark_report.md")


if __name__ == "__main__":
    main()
