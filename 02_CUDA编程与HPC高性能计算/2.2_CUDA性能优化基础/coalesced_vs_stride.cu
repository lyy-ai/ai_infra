// 3.2 CUDA 性能优化基础：合并访问 vs 跨步访问性能对比
//
// 编译运行：
//   cd /data/liyangyang/ai_infra/CUDA编程与HPC高性能计算
//   nvcc -o /tmp/coalesced_vs_stride 3.2_CUDA性能优化基础/coalesced_vs_stride.cu && /tmp/coalesced_vs_stride
#include <cuda_runtime.h>
#include <stdio.h>

#define N (1024 * 1024)

__global__ void coalesced_read(const float* in, float* out, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (int i = idx; i < N; i += total_threads) {
        int access_idx = (stride == 1) ? i : (i * stride) % N;
        sum += in[access_idx];
    }
    out[idx] = sum;
}

int main() {
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, 1024 * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int strides[] = {1, 32, 1024};
    for (int s = 0; s < 3; ++s) {
        int stride = strides[s];
        cudaEventRecord(start);
        coalesced_read<<<(N + 255) / 256, 256>>>(d_in, d_out, stride);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("stride=%4d: %.3f ms\n", stride, ms);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
