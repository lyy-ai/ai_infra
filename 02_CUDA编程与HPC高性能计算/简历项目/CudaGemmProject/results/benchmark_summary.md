# A100 CUDA GEMM Benchmark 汇总

硬件：NVIDIA A100-PCIE-40GB（sm_80）| CUDA 12.8 | 测试：warmup 3 + 20 次迭代取均值

| Kernel | Size | Latency(ms) | TFLOPS | max_diff(vs cuBLAS) |
|--------|------|-------------|--------|---------------------|
| naive | 1024 | 1.154 | 1.86 | 7.6e-05 |
| tiled_smem | 1024 | 1.204 | 1.78 | 7.6e-05 |
| tiled_smem | 2048 | 7.669 | 2.24 | 0.0e+00 |
| tiled_smem | 4096 | 51.471 | 2.67 | 0.0e+00 |
| tiled_vec4 | 1024 | 0.496 | 4.33 | 7.6e-05 |
| tiled_vec4 | 2048 | 4.436 | 3.87 | 0.0e+00 |
| tiled_vec4 | 4096 | 32.864 | 4.18 | 0.0e+00 |
| wmma_fp16_tc | 1024 | 0.402 | 5.34 | 0.0e+00 |
| wmma_fp16_tc | 2048 | 1.699 | 10.11 | 5.0e-01 |
| wmma_fp16_tc | 4096 | 19.791 | 6.94 | 0.0e+00 |
| cublas_fp32 | 1024 | 0.314 | 6.85 | 0.0e+00 |
| cublas_fp32 | 2048 | 2.134 | 8.05 | 0.0e+00 |
| cublas_fp32 | 4096 | 23.677 | 5.80 | 0.0e+00 |
| cublas_fp16_tc | 1024 | 0.019 | 113.32 | 0.0e+00 |
| cublas_fp16_tc | 2048 | 0.101 | 169.68 | 0.0e+00 |
| cublas_fp16_tc | 4096 | 1.295 | 106.10 | 0.0e+00 |
| torch_matmul_fp32 | 1024 | 0.248 | 8.65 | - |
| torch_matmul_fp32 | 2048 | 2.619 | 6.56 | - |
| torch_matmul_fp32 | 4096 | 23.582 | 5.83 | - |
| torch_matmul_fp16 | 1024 | 0.017 | 129.93 | - |
| torch_matmul_fp16 | 2048 | 0.080 | 214.34 | - |
| torch_matmul_fp16 | 4096 | 1.380 | 99.58 | - |

## 加速比（4096³，相对 naive/相对 cuBLAS）

- size=1024:
  - tiled_vec4 vs naive: 2.3x
  - tiled_vec4 达到 cuBLAS FP32 的 63%
  - wmma_fp16 达到 cuBLAS FP16 的 5%
- size=2048:
  - tiled_vec4 达到 cuBLAS FP32 的 48%
  - wmma_fp16 达到 cuBLAS FP16 的 6%
- size=4096:
  - tiled_vec4 达到 cuBLAS FP32 的 72%
  - wmma_fp16 达到 cuBLAS FP16 的 7%
