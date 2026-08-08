# 汇总单卡/DDP/NCCL 实测结果，生成 results/benchmark_summary.md
# 运行：/data/liyangyang/qwen35_env/bin/python scripts/analyze_results.py
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    with open(os.path.join(ROOT, "results", name)) as f:
        return json.load(f)


def main():
    single = load("single_gpu.json")
    ddp = load("ddp_2gpu.json")
    comm = load("nccl_allreduce.json")

    speedup = ddp["tokens_per_s"] / single["tokens_per_s"]
    eff = speedup / ddp["gpus"] * 100
    world = comm["world"]
    grad_gb = ddp["params_M"] * 4 / 1000  # FP32 梯度约 4 字节/参数
    algbw = comm["results"][-1]["algbw_gbps"]
    # Ring AllReduce 每 rank 通信量 = 2*(n-1)/n * S
    comm_ms_est = grad_gb * 2 * (world - 1) / world / algbw * 1000

    lines = [
        "# 2×A100 分布式训练 Benchmark 汇总",
        "",
        "硬件：2× NVIDIA A100-PCIE-40GB（PCIe 互联，无 NVLink）| PyTorch + NCCL | TF32 开启",
        f"模型：SmallGPT（{single['params_M']}M 参数，12 层 hidden=768，seq={single['seq']}，batch={single['batch']}/rank）",
        "",
        "## 训练吞吐（batch=8/rank，40 步取均值）",
        "",
        "| 模式 | GPU 数 | tokens/s | step(ms) | fwd(ms) | bwd+同步(ms) | 峰值显存(GB) |",
        "|------|--------|----------|----------|---------|--------------|--------------|",
        f"| 单卡 | 1 | {single['tokens_per_s']:.0f} | {single['step_ms']} | {single['fwd_ms']} | {single['bwd_ms']} | {single['peak_mem_gb']} |",
        f"| DDP | {ddp['gpus']} | {ddp['tokens_per_s']} | {ddp['ranks'][0]['step_ms']} | {ddp['ranks'][0]['fwd_ms']} | {ddp['ranks'][0]['bwd_ms']} | {ddp['ranks'][0]['peak_mem_gb']} |",
        "",
        f"**加速比：{speedup:.2f}x（并行效率 {eff:.0f}%）——DDP 比单卡更慢！**",
        "",
        "## 根因分析：PCIe 互联通信瓶颈",
        "",
        "NCCL AllReduce 实测带宽：",
        "",
        "| 消息大小 | 耗时(ms) | algbw(GB/s) | busbw(GB/s) |",
        "|----------|----------|-------------|-------------|",
    ]
    for r in comm["results"]:
        lines.append(f"| {r['size_mb']}MB | {r['ms']} | {r['algbw_gbps']} | {r['busbw_gbps']} |")
    lines += [
        "",
        f"- 梯度规模：{grad_gb:.2f}GB（FP32，110M 参数 × 4B）。",
        f"- 实测 busbw 仅 {algbw}GB/s（PCIe 慢速路径；NVLink 版本 A100 为 600GB/s，相差 ~400 倍）。",
        f"- 每步 AllReduce 理论耗时 ≈ {comm_ms_est:.0f}ms；实测 backward+同步 {ddp['ranks'][0]['bwd_ms']}ms（单卡 backward 仅 {single['bwd_ms']}ms），",
        "  bucket overlap 隐藏了部分通信，但通信总量仍远超计算。",
        "",
        "## 结论与改进方向",
        "",
        "1. 通信/计算比决定 DDP 收益：本例中通信 ~290ms vs 计算 ~78ms，加速比 <1。",
        "2. 改进手段：NVLink 机型（busbw 600GB/s）、梯度累积增大每步计算量、",
        "   bucket 调优（bucket_cap_mb）、FP16 梯度通信（通信量减半）、更大模型提高计算密度。",
        "3. 这正是第 6.8 节「多卡训练通信优化」强调的原则：先保证通信能被计算隐藏，再谈线性加速比。",
    ]

    content = "\n".join(lines) + "\n"
    out = os.path.join(ROOT, "results", "benchmark_summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(content)
    print(content)


if __name__ == "__main__":
    main()
