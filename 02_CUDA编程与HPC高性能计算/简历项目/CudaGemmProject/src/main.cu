// GEMM benchmark：naive / tiled / float4 / WMMA FP16 / cuBLAS 对比
// 输出 JSON 行，由 analyze_results.py 汇总
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gemm_kernels.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)

static float max_diff(const float *ref, const float *out, size_t n) {
    float d = 0.f;
    for (size_t i = 0; i < n; ++i) d = fmaxf(d, fabsf(ref[i] - out[i]));
    return d;
}

static float max_diff_h(const half *ref, const half *out, size_t n) {
    float d = 0.f;
    for (size_t i = 0; i < n; ++i)
        d = fmaxf(d, fabsf(__half2float(ref[i]) - __half2float(out[i])));
    return d;
}

template <typename F>
static double bench(F fn, int warmup, int iters) {
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    for (int i = 0; i < warmup; ++i) fn();
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) fn();
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / iters;
}

int main() {
    const int sizes[] = {1024, 2048, 4096};
    cublasHandle_t handle;
    cublasCreate(&handle);

    printf("[\n");
    bool first = true;
    for (int si = 0; si < 3; ++si) {
        int M = sizes[si], N = sizes[si], K = sizes[si];
        size_t bytes_ab_f = (size_t)M * K * sizeof(float);
        size_t bytes_c_f = (size_t)M * N * sizeof(float);
        size_t bytes_ab_h = (size_t)M * K * sizeof(__half);
        size_t bytes_c_h = (size_t)M * N * sizeof(__half);

        float *hA = (float *)malloc(bytes_ab_f), *hB = (float *)malloc(bytes_ab_f);
        srand(42);
        for (size_t i = 0; i < (size_t)M * K; ++i) { hA[i] = (rand() % 100 - 50) / 50.f; hB[i] = (rand() % 100 - 50) / 50.f; }

        float *dA, *dB, *dC, *dRef;
        CK(cudaMalloc(&dA, bytes_ab_f)); CK(cudaMalloc(&dB, bytes_ab_f));
        CK(cudaMalloc(&dC, bytes_c_f)); CK(cudaMalloc(&dRef, bytes_c_f));
        CK(cudaMemcpy(dA, hA, bytes_ab_f, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, hB, bytes_ab_f, cudaMemcpyHostToDevice));

        float alpha = 1.f, beta = 0.f;
        // cuBLAS（列主序接口，用转置技巧算行主序 C = A*B）作为黄金标准
        auto cublas_f32 = [&]() {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dRef, N);
        };
        cublas_f32();
        CK(cudaDeviceSynchronize());
        float *hRef = (float *)malloc(bytes_c_f), *hOut = (float *)malloc(bytes_c_f);
        CK(cudaMemcpy(hRef, dRef, bytes_c_f, cudaMemcpyDeviceToHost));

        double flops = 2.0 * M * N * K;
        auto report = [&](const char *name, double ms, float diff) {
            if (!first) printf(",\n");
            first = false;
            printf("  {\"kernel\":\"%s\",\"size\":%d,\"ms\":%.4f,\"tflops\":%.2f,\"max_diff\":%.2e}",
                   name, M, ms, flops / ms / 1e9, diff);
        };

        // cuBLAS FP32
        {
            double ms = bench(cublas_f32, 3, 20);
            report("cublas_fp32", ms, 0.f);
        }
        // naive（大 size 太慢，只在 1024 上跑数值）
        if (M <= 1024) {
            dim3 blk(16, 16), grd((N + 15) / 16, (M + 15) / 16);
            auto fn = [&]() { gemm_naive<<<grd, blk>>>(dA, dB, dC, M, N, K); };
            double ms = bench(fn, 2, 5);
            CK(cudaMemcpy(hOut, dC, bytes_c_f, cudaMemcpyDeviceToHost));
            report("naive", ms, max_diff(hRef, hOut, (size_t)M * N));
        }
        // tiled
        {
            constexpr int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
            dim3 blk((BM / TM) * (BN / TN)), grd(N / BN, M / BM);
            auto fn = [&]() { gemm_tiled<BM, BN, BK, TM, TN><<<grd, blk>>>(dA, dB, dC, M, N, K); };
            double ms = bench(fn, 3, 20);
            CK(cudaMemcpy(hOut, dC, bytes_c_f, cudaMemcpyDeviceToHost));
            report("tiled_smem", ms, max_diff(hRef, hOut, (size_t)M * N));
        }
        // float4 tiled
        {
            constexpr int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
            dim3 blk((BM / TM) * (BN / TN)), grd(N / BN, M / BM);
            auto fn = [&]() { gemm_tiled_vec4<BM, BN, BK, TM, TN><<<grd, blk>>>(dA, dB, dC, M, N, K); };
            double ms = bench(fn, 3, 20);
            CK(cudaMemcpy(hOut, dC, bytes_c_f, cudaMemcpyDeviceToHost));
            report("tiled_vec4", ms, max_diff(hRef, hOut, (size_t)M * N));
        }
        // WMMA FP16 + cuBLAS FP16
        {
            __half *hAh = (__half *)malloc(bytes_ab_h), *hBh = (__half *)malloc(bytes_ab_h);
            for (size_t i = 0; i < (size_t)M * K; ++i) { hAh[i] = __float2half(hA[i]); hBh[i] = __float2half(hB[i]); }
            __half *dAh, *dBh, *dCh, *dRefH;
            CK(cudaMalloc(&dAh, bytes_ab_h)); CK(cudaMalloc(&dBh, bytes_ab_h));
            CK(cudaMalloc(&dCh, bytes_c_h)); CK(cudaMalloc(&dRefH, bytes_c_h));
            CK(cudaMemcpy(dAh, hAh, bytes_ab_h, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dBh, hBh, bytes_ab_h, cudaMemcpyHostToDevice));

            __half alpha_h = __float2half(1.f), beta_h = __float2half(0.f);
            auto cublas_f16 = [&]() {
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                             &alpha_h, dBh, CUDA_R_16F, N, dAh, CUDA_R_16F, K,
                             &beta_h, dRefH, CUDA_R_16F, N,
                             CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
            };
            cublas_f16();
            CK(cudaDeviceSynchronize());
            __half *hRefH = (__half *)malloc(bytes_c_h), *hOutH = (__half *)malloc(bytes_c_h);
            CK(cudaMemcpy(hRefH, dRefH, bytes_c_h, cudaMemcpyDeviceToHost));

            double ms = bench(cublas_f16, 3, 20);
            report("cublas_fp16_tc", ms, 0.f);

            constexpr int BM = 128, BN = 128, BK = 32;
            dim3 blk(256), grd(N / BN, M / BM);
            auto fn = [&]() { gemm_wmma_fp16<BM, BN, BK><<<grd, blk>>>(dAh, dBh, dCh, M, N, K); };
            ms = bench(fn, 3, 20);
            CK(cudaMemcpy(hOutH, dCh, bytes_c_h, cudaMemcpyDeviceToHost));
            report("wmma_fp16_tc", ms, max_diff_h(hRefH, hOutH, (size_t)M * N));

            free(hAh); free(hBh); free(hRefH); free(hOutH);
            cudaFree(dAh); cudaFree(dBh); cudaFree(dCh); cudaFree(dRefH);
        }

        free(hA); free(hB); free(hRef); free(hOut);
        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    }
    printf("\n]\n");
    cublasDestroy(handle);
    return 0;
}
