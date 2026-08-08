// 3.4 经典算子实现 - GEMM：Shared Memory Tiling SGEMM
//
// 编译运行：
//   cd /data/liyangyang/ai_infra/CUDA编程与HPC高性能计算
//   nvcc -o /tmp/sgemm_tiled 3.4_经典算子实现-GEMM/sgemm_tiled.cu && /tmp/sgemm_tiled
#include <cuda_runtime.h>
#include <stdio.h>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_tiled(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;

    int thread_row = tx / (BN / TN);
    int thread_col = tx % (BN / TN);

    float acc[TM][TN] = {0.0f};

    int num_k_tiles = (K + BK - 1) / BK;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        for (int i = tx; i < BM * BK; i += blockDim.x) {
            int r = i / BK;
            int c = i % BK;
            int a_row = by * BM + r;
            int a_col = kt * BK + c;
            As[r][c] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        }
        for (int i = tx; i < BK * BN; i += blockDim.x) {
            int r = i / BN;
            int c = i % BN;
            int b_row = kt * BK + r;
            int b_col = bx * BN + c;
            Bs[r][c] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        }
        __syncthreads();

        for (int kk = 0; kk < BK; ++kk) {
            float a_frag[TM], b_frag[TN];
            for (int i = 0; i < TM; ++i) a_frag[i] = As[thread_row * TM + i][kk];
            for (int j = 0; j < TN; ++j) b_frag[j] = Bs[kk][thread_col * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int c_row = by * BM + thread_row * TM + i;
            int c_col = bx * BN + thread_col * TN + j;
            if (c_row < M && c_col < N) {
                C[c_row * N + c_col] = acc[i][j];
            }
        }
    }
}

int main() {
    int M = 1024, N = 1024, K = 1024;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    dim3 block((BM / TM) * (BN / TN));
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    printf("SGEMM tiled kernel launched: grid=(%d,%d), block=%d\n", grid.x, grid.y, block.x);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
