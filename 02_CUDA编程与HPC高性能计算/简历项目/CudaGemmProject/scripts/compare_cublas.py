# 用 PyTorch（cuBLAS）在相同 size 上 benchmark，作为框架侧参考
# 运行：/data/liyangyang/qwen35_env/bin/python scripts/compare_cublas.py
import json
import os

import torch

SIZES = [1024, 2048, 4096]
WARMUP, ITERS = 3, 20
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "cublas_benchmark.json")


def bench(matmul, warmup=WARMUP, iters=ITERS):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        matmul()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        matmul()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    assert torch.cuda.is_available()
    results = []
    for n in SIZES:
        flops = 2.0 * n * n * n
        for dtype, name in [(torch.float32, "torch_matmul_fp32"),
                            (torch.float16, "torch_matmul_fp16")]:
            a = torch.randn(n, n, dtype=dtype, device="cuda")
            b = torch.randn(n, n, dtype=dtype, device="cuda")
            ms = bench(lambda: a @ b)
            results.append({"kernel": name, "size": n, "ms": round(ms, 4),
                            "tflops": round(flops / ms / 1e9, 2)})
            print(f"{name} {n}: {ms:.3f} ms, {flops / ms / 1e9:.2f} TFLOPS")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    main()
