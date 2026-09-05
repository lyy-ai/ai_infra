# 2.10 Hopper 与 Blackwell 新特性：三代 GPU 规格对比与 decode 吞吐估算
#
# 运行：
#   cd /data/ai_infra/02_CUDA编程与HPC高性能计算
#   /data/qwen35_env/bin/python 2.10_Hopper与Blackwell新特性/gpu_generation_compare.py
#
# 说明：本脚本是教学估算器。decode 阶段每生成 1 个 token 需要把全部
# 模型权重从 HBM 读一遍（batch 较小时），因此理论吞吐上限 ≈
# 显存带宽 ÷ 每 token 激活参数字节数。这是 memory-bound 场景的经典账本。
import unicodedata

# 三代 GPU 规格（约数，来自公开 datasheet 常见口径；B200 不同功耗版本有差异）
GPUS = [
    {
        "name": "A100 SXM", "year": 2020, "tensor_core": "第三代",
        "fp16_tflops": 312, "fp8_tflops": None, "fp4_tflops": None,
        "mem_gb": 80, "mem_bw_tbps": 2.0, "nvlink_gbps": 600, "scaleup": "8 卡 (DGX)",
    },
    {
        "name": "H100 SXM", "year": 2022, "tensor_core": "第四代 (+FP8)",
        "fp16_tflops": 989, "fp8_tflops": 1979, "fp4_tflops": None,
        "mem_gb": 80, "mem_bw_tbps": 3.35, "nvlink_gbps": 900, "scaleup": "8 卡 (HGX)",
    },
    {
        "name": "B200", "year": 2024, "tensor_core": "第五代 (+FP4/MXFP)",
        "fp16_tflops": 2250, "fp8_tflops": 4500, "fp4_tflops": 9000,
        "mem_gb": 192, "mem_bw_tbps": 8.0, "nvlink_gbps": 1800, "scaleup": "72 卡 (NVL72)",
    },
]


def pad(s, width):
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return s + " " * max(0, width - w)


def fmt(v, suffix=""):
    return f"{v}{suffix}" if v is not None else "-"


def print_spec_table():
    print("=" * 88)
    print("三代 GPU 规格对比表（约数，教学口径）")
    print("=" * 88)
    header = ["规格", "A100 SXM", "H100 SXM", "B200"]
    widths = [24, 18, 18, 18]
    print("  " + " | ".join(pad(h, w) for h, w in zip(header, widths)))
    print("  " + "-" * 84)
    rows = [
        ("发布年份", [str(g["year"]) for g in GPUS]),
        ("Tensor Core", [g["tensor_core"] for g in GPUS]),
        ("FP16/BF16 dense", [fmt(g["fp16_tflops"], " TFLOPS") for g in GPUS]),
        ("FP8 dense", [fmt(g["fp8_tflops"], " TFLOPS") for g in GPUS]),
        ("FP4 dense", [fmt(g["fp4_tflops"], " TFLOPS") for g in GPUS]),
        ("显存容量", [fmt(g["mem_gb"], " GB") for g in GPUS]),
        ("显存带宽", [f'{g["mem_bw_tbps"]:.2f} TB/s' for g in GPUS]),
        ("NVLink 带宽", [fmt(g["nvlink_gbps"], " GB/s") for g in GPUS]),
        ("scale-up 域", [g["scaleup"] for g in GPUS]),
    ]
    for label, vals in rows:
        print("  " + pad(label, widths[0]) + " | " + " | ".join(pad(v, w) for v, w in zip(vals, widths[1:])))
    print()


def decode_upper_bound(mem_bw_tbps, params_b, bytes_per_param):
    """decode 理论吞吐上限（tokens/s/卡） = 带宽 / 每 token 需读取的参数字节数。"""
    bytes_per_token = params_b * 1e9 * bytes_per_param
    return mem_bw_tbps * 1e12 / bytes_per_token


def print_decode_table(params_b=70):
    print("=" * 88)
    print(f"{params_b}B 模型 decode 理论吞吐上限估算（单卡，小 batch，假设权重全量读取）")
    print("=" * 88)
    precisions = [("FP16/BF16", 2.0), ("FP8", 1.0), ("FP4", 0.5)]
    header = ["GPU", "带宽 (TB/s)"] + [f"{p} 上限 (tok/s)" for p, _ in precisions]
    widths = [12, 14, 18, 18, 18]
    print("  " + " | ".join(pad(h, w) for h, w in zip(header, widths)))
    print("  " + "-" * 84)
    for g in GPUS:
        vals = [g["name"], f'{g["mem_bw_tbps"]:.2f}']
        for pname, bpp in precisions:
            supported = not ((pname == "FP8" and g["fp8_tflops"] is None)
                             or (pname == "FP4" and g["fp4_tflops"] is None))
            if supported:
                vals.append(f'{decode_upper_bound(g["mem_bw_tbps"], params_b, bpp):.1f}')
            else:
                vals.append("不支持")
        print("  " + " | ".join(pad(v, w) for v, w in zip(vals, widths)))
    print()
    print("  公式：tokens/s = 显存带宽 / (参数量 x 每参数字节数)")
    print(f"  以 FP16 为例：每 token 需读取 {params_b}B x 2 字节 = {params_b * 2:.0f} GB 权重")
    print()
    print("  解读：")
    print("  1. 同一代 GPU 上，FP8/FP4 把每 token 的字节数砍半/再砍半，上限同比翻倍。")
    print("  2. 跨代看，H100 -> B200 的带宽提升（约 2.4 倍）叠加 FP4（4 倍），")
    print("     decode 上限提升近一个数量级——这就是低精度 + 高带宽的商业价值。")
    print("  3. 注意这只是 memory-bound 上限：实际吞吐还受 batch size、KV cache")
    print("     读取、kernel 效率影响；batch 大时逐渐转向 compute-bound。")


def print_capacity_check(params_b=70):
    print()
    print("=" * 88)
    print(f"单卡能否放下 {params_b}B 模型（仅权重，不含 KV cache）")
    print("=" * 88)
    print(f"  {'GPU':<12} | {'显存':>8} | {'FP16 权重':>10} | {'FP8 权重':>10} | {'FP4 权重':>10}")
    print("  " + "-" * 62)
    for g in GPUS:
        cells = []
        for bpp in (2.0, 1.0, 0.5):
            need = params_b * bpp
            cells.append(f"{need:.0f} GB ({'可' if need <= g['mem_gb'] else '否'})")
        print(f"  {g['name']:<12} | {g['mem_gb']:>6} GB | " + " | ".join(f"{c:>10}" for c in cells))
    print()
    print("  解读：量化不仅提速，还直接决定单卡能不能装下模型，进而决定最少需要")
    print("  几张卡、用不用跨 NVLink 域——显存维度与通信维度在这里耦合。")


def main():
    print_spec_table()
    print_decode_table()
    print_capacity_check()


if __name__ == "__main__":
    main()
