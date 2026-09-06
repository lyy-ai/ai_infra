# 2.5 GPU 硬件概论：规格对比与 roofline 拐点
#
# 运行：
#   /data/qwen35_env/bin/python 1.9_GPU硬件概论/gpu_spec_compare.py


GPUS = [
    ("A100", 80, 2.0, 312, False),
    ("H100", 80, 3.35, 989, True),
    ("H200", 141, 4.8, 989, True),
    ("Orin AGX", 64, 0.2, 55, False),
]


def roofline_knee(tflops, tbps):
    return tflops * 1e12 / (tbps * 1e12)


def main():
    print(f"{'GPU':<10} | {'显存(GB)':>8} | {'带宽(TB/s)':>10} | {'FP16 TFLOPS':>11} | {'FP8':>4} | {'拐点(FLOP/B)':>12}")
    print("-" * 72)
    for name, mem, bw, tf, fp8 in GPUS:
        print(f"{name:<10} | {mem:>8} | {bw:>10.2f} | {tf:>11} | {'Y' if fp8 else 'N':>4} | {roofline_knee(tf, bw):>12.0f}")
    print()
    print("算术强度 < 拐点 → memory-bound（省访存）；> 拐点 → compute-bound（上 Tensor Core）")
    print("例：LLM decode 算术强度≈2 FLOP/B，全部平台 memory-bound")


if __name__ == "__main__":
    main()
