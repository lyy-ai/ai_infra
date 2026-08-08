// 3.3 经典算子实现 - Reduce：共享内存 + Warp Shuffle 归约
//
// 编译运行：
//   cd /data/liyangyang/ai_infra/CUDA编程与HPC高性能计算
//   nvcc -o /tmp/reduce 3.3_经典算子实现-Reduce/reduce.cu && /tmp/reduce
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <numeric>

#define N (1024 * 1024)
#define BLOCK 256

__inline__ __device__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    return val;
}

__global__ void reduce_kernel(const float* in, float* out, int n) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float local = 0.0f;
    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        local += in[i];
    }
    sdata[tid] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    local = sdata[tid];
    local = warp_reduce_sum(local);

    if (tid == 0) atomicAdd(out, local);
}

int main() {
    std::vector<float> h_in(N, 1.0f);
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(float));

    reduce_kernel<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_in, d_out, N);

    float h_out = 0;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU sum: %.1f, expected: %d\n", h_out, N);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
