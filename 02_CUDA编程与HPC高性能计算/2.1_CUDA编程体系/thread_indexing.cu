// 3.1 CUDA 编程体系：Grid/Block/Thread 索引计算示例
//
// 编译运行：
//   cd /data/liyangyang/ai_infra/CUDA编程与HPC高性能计算
//   nvcc -o /tmp/thread_indexing 3.1_CUDA编程体系/thread_indexing.cu && /tmp/thread_indexing
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void indexing_kernel(int* out, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        int idx = row * width + col;
        int block_id = blockIdx.y * gridDim.x + blockIdx.x;
        int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
        out[idx] = block_id * 10000 + thread_id;
    }
}

int main() {
    const int W = 32, H = 16;
    int* d_out;
    cudaMalloc(&d_out, W * H * sizeof(int));

    dim3 block(16, 8);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    indexing_kernel<<<grid, block>>>(d_out, W, H);

    int h_out[W * H];
    cudaMemcpy(h_out, d_out, W * H * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < W * H; ++i) {
        if (i % W == 0) printf("\n");
        printf("%6d ", h_out[i]);
    }
    printf("\n");
    cudaFree(d_out);
    return 0;
}
