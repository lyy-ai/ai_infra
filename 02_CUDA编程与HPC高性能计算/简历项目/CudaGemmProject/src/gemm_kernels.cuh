// CUDA GEMM 多级优化 kernel：naive / smem tiled / float4 / WMMA Tensor Core
// A: row-major MxK, B: row-major KxN, C: row-major MxN
#pragma once

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// ---------------- 1. Naive：一线程一元素 ----------------
__global__ void gemm_naive(const float *A, const float *B, float *C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// ---------------- 2. Shared Memory Tiling + 寄存器分块 ----------------
// BM=BN=128, BK=16, TM=TN=8, blockDim=256
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_tiled(const float *A, const float *B, float *C,
                           int M, int N, int K) {
    __shared__ float As[BM][BK + 4];
    __shared__ float Bs[BK][BN + 4];

    const int tid = threadIdx.x;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int thread_row = (tid / (BN / TN)) * TM;
    const int thread_col = (tid % (BN / TN)) * TN;

    float acc[TM][TN] = {0.f};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // 加载 A/B tile（标量加载）
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int m = i / BK, k = i % BK;
            As[m][k] = A[(block_row + m) * K + k0 + k];
        }
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int k = i / BN, n = i % BN;
            Bs[k][n] = B[(k0 + k) * N + block_col + n];
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float ra[TM], rb[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) ra[i] = As[thread_row + i][k];
#pragma unroll
            for (int j = 0; j < TN; ++j) rb[j] = Bs[k][thread_col + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            C[(block_row + thread_row + i) * N + block_col + thread_col + j] = acc[i][j];
}

// ---------------- 3. Tiling + float4 向量化加载 ----------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_tiled_vec4(const float *A, const float *B, float *C,
                                int M, int N, int K) {
    __shared__ float As[BM][BK + 4];
    __shared__ float Bs[BK][BN + 4];

    const int tid = threadIdx.x;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int thread_row = (tid / (BN / TN)) * TM;
    const int thread_col = (tid % (BN / TN)) * TN;

    float acc[TM][TN] = {0.f};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // float4 向量化加载（要求 K、N 为 4 的倍数）
        for (int i = tid * 4; i < BM * BK; i += blockDim.x * 4) {
            int m = i / BK, k = i % BK;
            float4 v = *reinterpret_cast<const float4 *>(&A[(block_row + m) * K + k0 + k]);
            As[m][k] = v.x; As[m][k + 1] = v.y; As[m][k + 2] = v.z; As[m][k + 3] = v.w;
        }
        for (int i = tid * 4; i < BK * BN; i += blockDim.x * 4) {
            int k = i / BN, n = i % BN;
            float4 v = *reinterpret_cast<const float4 *>(&B[(k0 + k) * N + block_col + n]);
            Bs[k][n] = v.x; Bs[k][n + 1] = v.y; Bs[k][n + 2] = v.z; Bs[k][n + 3] = v.w;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float ra[TM], rb[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) ra[i] = As[thread_row + i][k];
#pragma unroll
            for (int j = 0; j < TN; ++j) rb[j] = Bs[k][thread_col + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            C[(block_row + thread_row + i) * N + block_col + thread_col + j] = acc[i][j];
}

// ---------------- 4. WMMA Tensor Core FP16 ----------------
// BM=BN=128, BK=32；8 个 warp(2x4)，warp tile 64x32 = 4x2 个 16x16 fragment
// 要求 M/N/K 为 128/128/32 的倍数
template <int BM, int BN, int BK>
__global__ void gemm_wmma_fp16(const half *A, const half *B, half *C,
                               int M, int N, int K) {
    constexpr int PAD = 8;
    __shared__ half As[BM][BK + PAD];        // row-major
    __shared__ half Bs[BN][BK + PAD];        // 存 B^T tile，方便按 col_major 读

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_row = (warp_id / 4) * 64;  // 2 行 warp
    const int warp_col = (warp_id % 4) * 32;  // 4 列 warp
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc[4][2];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], __float2half(0.f));

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int m = i / BK, k = i % BK;
            As[m][k] = A[(block_row + m) * K + k0 + k];
        }
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int k = i / BN, n = i % BN;
            Bs[n][k] = B[(k0 + k) * N + block_col + n];  // 转置存储
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[2];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                wmma::load_matrix_sync(a_frag[i], &As[warp_row + i * 16][kk], BK + PAD);
#pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(b_frag[j], &Bs[warp_col + j * 16][kk], BK + PAD);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(
                &C[(block_row + warp_row + i * 16) * N + block_col + warp_col + j * 16],
                acc[i][j], N, wmma::mem_row_major);
}
