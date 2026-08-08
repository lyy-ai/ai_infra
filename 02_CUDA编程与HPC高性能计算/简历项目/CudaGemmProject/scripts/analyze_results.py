# 汇总 CUDA benchmark 与 cuBLAS benchmark，生成 results/benchmark_summary.md
# 运行：/data/liyangyang/qwen35_env/bin/python scripts/analyze_results.py
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUDA_JSON = os.path.join(ROOT, "results", "cuda_gemm_benchmark.json")
CUBLAS_JSON = os.path.join(ROOT, "results", "cublas_benchmark.json")
OUT_MD = os.path.join(ROOT, "results", "benchmark_summary.md")


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    cuda = load(CUDA_JSON)
    cublas = load(CUBLAS_JSON) if os.path.exists(CUBLAS_JSON) else []
    rows = {(r["kernel"], r["size"]): r for r in cuda + cublas}

    sizes = sorted({r["size"] for r in cuda})
    kernels = ["naive", "tiled_smem", "tiled_vec4", "wmma_fp16_tc",
               "cublas_fp32", "cublas_fp16_tc", "torch_matmul_fp32", "torch_matmul_fp16"]

    lines = [
        "# A100 CUDA GEMM Benchmark 汇总",
        "",
        "硬件：NVIDIA A100-PCIE-40GB（sm_80）| CUDA 12.8 | 测试：warmup 3 + 20 次迭代取均值",
        "",
        "| Kernel | Size | Latency(ms) | TFLOPS | max_diff(vs cuBLAS) |",
        "|--------|------|-------------|--------|---------------------|",
    ]
    for k in kernels:
        for s in sizes:
            r = rows.get((k, s))
            if not r:
                continue
            diff = f"{r.get('max_diff', 0):.1e}" if r.get("max_diff") is not None else "-"
            lines.append(f"| {k} | {s} | {r['ms']:.3f} | {r['tflops']:.2f} | {diff} |")

    lines += ["", "## 加速比（4096³，相对 naive/相对 cuBLAS）", ""]
    for s in sizes:
        base = rows.get(("naive", s))
        ref32 = rows.get(("cublas_fp32", s))
        ref16 = rows.get(("cublas_fp16_tc", s))
        tiled = rows.get(("tiled_vec4", s))
        wmma = rows.get(("wmma_fp16_tc", s))
        lines.append(f"- size={s}:")
        if base and tiled:
            lines.append(f"  - tiled_vec4 vs naive: {base['ms'] / tiled['ms']:.1f}x")
        if ref32 and tiled:
            lines.append(f"  - tiled_vec4 达到 cuBLAS FP32 的 {tiled['tflops'] / ref32['tflops'] * 100:.0f}%")
        if ref16 and wmma:
            lines.append(f"  - wmma_fp16 达到 cuBLAS FP16 的 {wmma['tflops'] / ref16['tflops'] * 100:.0f}%")

    content = "\n".join(lines) + "\n"
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(content)
    print(content)


if __name__ == "__main__":
    main()
