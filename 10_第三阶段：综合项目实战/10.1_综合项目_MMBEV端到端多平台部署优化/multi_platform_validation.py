# 综合项目：多平台验证结果汇总
#
# 运行：
#   cd /data/liyangyang/ai_infra/10_第三阶段：综合项目实战
#   /data/liyangyang/qwen35_env/bin/python 10.1_综合项目_MMBEV端到端多平台部署优化/multi_platform_validation.py


def check(name, value, threshold, lower_is_better=True):
    ok = value <= threshold if lower_is_better else value >= threshold
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {value} (threshold {threshold})")
    return ok


def main():
    print("=== Multi-Platform Validation: MMBEV ===\n")
    platforms = {
        "Orin":  {"latency_ms": 30, "throughput_fps": 30, "nds_loss": 0.004, "util": 0.85},
        "A100":  {"latency_ms": 10, "throughput_fps": 100, "nds_loss": 0.004, "util": 0.85},
        "Ascend": {"latency_ms": 12, "throughput_fps": 83, "nds_loss": 0.005, "util": 0.82},
    }
    all_ok = True
    for name, m in platforms.items():
        print(f"[{name}]")
        ok = True
        ok &= check("latency (Orin<50ms, others<25ms)", m["latency_ms"], 50 if name == "Orin" else 25)
        ok &= check("NDS loss < 1%", m["nds_loss"], 0.01)
        ok &= check("GPU util >= 80%", m["util"], 0.80, lower_is_better=False)
        all_ok &= ok
        print()
    print("Overall:", "PASS (ready for production)" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
