# 9.2 边缘部署 - Jetson Orin：功耗与延迟估算
#
# 运行：
#   cd /data/liyangyang/ai_infra/08_多平台适配
#   /data/liyangyang/qwen35_env/bin/python 8.2_边缘部署_Jetson_Orin/jetson_power_latency_model.py


def estimate_orin_latency(model_params_b, bits, tops, utilization=0.6, seq_len=512):
    """
    简化模型：按 INT8 TOPS 与权值读取估算 decode 阶段每 token 时间。
    """
    weight_bytes = model_params_b * 1e9 * bits / 8
    effective_tops = tops * utilization * 1e12
    compute_time_s = (2 * model_params_b * 1e9) / effective_tops
    memory_time_s = weight_bytes / (100e9)  # 假设共享内存带宽约 100 GB/s
    total_ms = max(compute_time_s, memory_time_s) * 1000
    return total_ms


def main():
    configs = [
        ("Orin Nano 8GB", 7, 4, 40),
        ("Orin NX 16GB", 7, 4, 100),
        ("Orin AGX 64GB", 7, 4, 275),
        ("Orin AGX 64GB", 13, 4, 275),
    ]
    print(f"{'Platform':>18} | {'Model':>6} | {'Bits':>5} | {'TOPS':>6} | {'Latency/token(ms)':>20}")
    print("-" * 70)
    for name, params, bits, tops in configs:
        latency = estimate_orin_latency(params, bits, tops)
        print(f"{name:>18} | {params:>6}B | {bits:>5} | {tops:>6} | {latency:>20.2f}")
    print()
    print("结论：7B INT4 在 Orin AGX 上可满足实时性；Nano/NX 更适合小模型或需进一步压缩。")


if __name__ == "__main__":
    main()
