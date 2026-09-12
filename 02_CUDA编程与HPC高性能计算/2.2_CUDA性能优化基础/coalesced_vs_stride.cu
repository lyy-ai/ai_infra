// 合并访问 vs 跨步访问性能对比
//
// 编译运行：
//   cd /data/ai_infra/02_CUDA编程与HPC高性能计算
//   nvcc -O3 -o /tmp/coalesced_vs_stride 2.2_CUDA性能优化基础/coalesced_vs_stride.cu
//   /tmp/coalesced_vs_stride
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// N 取奇数，保证 stride=32/1024 与 N 互质；这样取模后仍会遍历整个输入，
// 不会因为 N 是 2 的幂而只访问 N/stride 个元素。
#define N (1024 * 1024 + 1)
#define THREADS_PER_BLOCK 256
#define BLOCKS 256
#define REPEATS 16

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error__ = (call);                                     \
        if (error__ != cudaSuccess) {                                     \
            fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__,         \
                    __LINE__, cudaGetErrorString(error__));              \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

__global__ void initialize_input(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    for (int i = idx; i < N; i += total_threads) {
        // 使用非零、变化的数据，避免把“读全局内存”误测成读零值。
        data[i] = 1.0f + static_cast<float>(i & 31) * 0.001f;
    }
}

__global__ void strided_read(const float* __restrict__ in,
                             float* __restrict__ out, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    float sum = 0.0f;

    // 让一个 kernel 完成足够多的工作，降低 kernel/event 启动开销对结果的影响。
    for (int repeat = 0; repeat < REPEATS; ++repeat) {
        for (int i = idx; i < N; i += total_threads) {
            // 对 stride=1，同一个 warp 的线程访问连续地址；较大的 stride
            // 则让线程访问相距更远的 cache line。
            int access_idx = (i * stride + repeat) % N;
            sum += in[access_idx];
        }
    }
    out[idx] = sum;
}

int main() {
    float* d_in = nullptr;
    float* d_out = nullptr;
    const size_t output_count =
        static_cast<size_t>(BLOCKS) * THREADS_PER_BLOCK;

    CUDA_CHECK(cudaMalloc(&d_in, static_cast<size_t>(N) * sizeof(float)));
    // 每个线程写一个结果，而不是只分配 1024 个 float。
    CUDA_CHECK(cudaMalloc(&d_out, output_count * sizeof(float)));

    const dim3 grid(BLOCKS);
    const dim3 block(THREADS_PER_BLOCK);
    initialize_input<<<grid, block>>>(d_in);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int strides[] = {1, 32, 1024};
    for (int stride : strides) {
        // 预热，避免第一次启动的初始化开销混入测量。
        strided_read<<<grid, block>>>(d_in, d_out, stride);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        strided_read<<<grid, block>>>(d_in, d_out, stride);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("stride=%4d: %.3f ms\n", stride, ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
