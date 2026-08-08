# 3.8 性能分析工具链：Roofline 模型绘制
#
# 运行：
#   cd /data/liyangyang/ai_infra/02_CUDA编程与HPC高性能计算
#   /data/liyangyang/qwen35_env/bin/python 2.8_性能分析工具链/roofline_calc.py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_roofline(peak_tflops, peak_bw_tbs, save_path="roofline.png"):
    I = np.logspace(-2, 4, 500)  # FLOPs/Byte
    memory_roof = peak_bw_tbs * I  # TFLOPs
    compute_roof = np.full_like(I, peak_tflops)
    attainable = np.minimum(memory_roof, compute_roof)

    plt.figure(figsize=(8, 6))
    plt.loglog(I, memory_roof, "--", label=f"Memory Bandwidth {peak_bw_tbs} TB/s")
    plt.loglog(I, compute_roof, "--", label=f"Compute Peak {peak_tflops} TFLOPs")
    plt.loglog(I, attainable, "-", label="Attainable Performance", linewidth=2)
    plt.xlabel("Arithmetic Intensity (FLOPs/Byte)")
    plt.ylabel("Performance (TFLOPs/s)")
    plt.title("Roofline Model")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(save_path)
    print(f"Roofline plot saved to {save_path}")


def main():
    peak_tflops = 312  # A100 FP16/FP32 mixed
    peak_bw_tbs = 2.0  # TB/s
    plot_roofline(peak_tflops, peak_bw_tbs, "2.8_性能分析工具链/roofline.png")

    M = N = K = 4096
    flops = 2 * M * N * K
    bytes_moved = (M * K + K * N + M * N) * 4  # FP32
    intensity = flops / bytes_moved
    print(f"SGEMM {M}x{N}x{K} intensity: {intensity:.2f} FLOPs/Byte")
    print(f"Memory-bound threshold: {peak_tflops / peak_bw_tbs:.2f} FLOPs/Byte")
    if intensity < peak_tflops / peak_bw_tbs:
        print("This kernel is likely MEMORY-BOUND")
    else:
        print("This kernel is likely COMPUTE-BOUND")


if __name__ == "__main__":
    main()
