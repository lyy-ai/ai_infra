# CudaGemmProject：A100 手写 GEMM 多级优化实战

在真实硬件（NVIDIA A100-PCIE-40GB, sm_80, CUDA 12.8）上手写 CUDA GEMM，从 naive 逐级优化到 Tensor Core，并以 cuBLAS 为黄金标准做数值对齐与性能对比。所有结论数字都在 `results/` 中可复现。

## 优化路径

| 版本 | 技术 | 4096³ 实测 |
|------|------|-----------|
| naive | 一线程一元素，直接读 HBM | 1.3 TFLOPS |
| tiled_smem | Shared Memory Tiling（BM=BN=128, BK=16）+ 寄存器分块（TM=TN=8） | 2.7 TFLOPS |
| tiled_vec4 | + float4 向量化全局加载 | 4.2 TFLOPS（达 cuBLAS FP32 ~60-70%） |
| wmma_fp16_tc | WMMA 调用 Tensor Core（128x128 block tile，8 warp，16x16x16 MMA） | ~7 TFLOPS（cuBLAS FP16 为 95 TFLOPS） |
| cublas | FP32 / FP16（GemmEx, COMPUTE_16F） | 7.3 / 95 TFLOPS |

## 目录结构

```
CudaGemmProject/
├── src/
│   ├── gemm_kernels.cuh     # 4 级 kernel：naive / smem tiled / float4 / WMMA
│   └── main.cu              # benchmark + 数值对齐（vs cuBLAS），输出 JSON
├── scripts/
│   ├── compare_cublas.py    # PyTorch matmul（cuBLAS）同 size 对比
│   └── analyze_results.py   # 汇总生成 results/benchmark_summary.md
├── results/                 # 真实测量数据（可复现）
│   ├── cuda_gemm_benchmark.json
│   ├── cublas_benchmark.json
│   └── benchmark_summary.md
├── Makefile
└── README.md
```

## 构建与运行

```bash
cd /data/liyangyang/ai_infra/02_CUDA编程与HPC高性能计算/简历项目/CudaGemmProject
make            # nvcc -O3 -arch=sm_80 -lcublas
make run        # CUDA kernel benchmark -> results/cuda_gemm_benchmark.json
make cublas     # PyTorch/cuBLAS 对比 -> results/cublas_benchmark.json
make report     # 汇总 -> results/benchmark_summary.md
```

## 关键结论（真实测量）

1. **内存层级是第一优化目标**：naive → smem tiling+float4，1024³ 下加速 ~2.3-2.7x，收益几乎全部来自 HBM 访问复用（每元素 A/B 子块复用 128 次）。
2. **手写 FP32 kernel 可达 cuBLAS 的 60-70%**：差距根因是 cuBLAS 有 double buffering、split-k、swizzle 与 auto-tuning。
3. **Tensor Core 是数量级武器**：cuBLAS FP16（95 TFLOPS）是 FP32（7.3 TFLOPS）的 13 倍；手写 WMMA 版本验证了 MMA 编程模型，也量化了与库实现的差距。
4. **数值对齐是工程底线**：FP32 max diff < 1e-4（vs cuBLAS）；FP16 累加在 K=2048 时 max diff 0.5（相对误差 ~1%），面试可展开 FP16 累加器精度问题。

详细数据见 [results/benchmark_summary.md](results/benchmark_summary.md)。
