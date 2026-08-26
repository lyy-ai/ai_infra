// 3.6 多线程并行优化：OpenMP + NUMA + SIMD 示例
//
// 编译运行：
//   cd /data/ai_infra/CUDA编程与HPC高性能计算
//   g++ -O3 -fopenmp -march=native -o /tmp/openmp_numa_simd 3.6_多线程并行优化/openmp_numa_simd.cpp && /tmp/openmp_numa_simd
#include <omp.h>
#include <immintrin.h>
#include <iostream>
#include <vector>

void add_avx2(const float* a, const float* b, float* c, int N) {
    int i = 0;
    for (; i + 8 <= N; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    for (; i < N; ++i) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1024 * 1024;
    std::vector<float> a(N, 1.0f), b(N, 2.0f), c(N);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        c[i] = a[i] + b[i];
    }

    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += c[i];
    }

    add_avx2(a.data(), b.data(), c.data(), N);

    std::cout << "OpenMP threads: " << omp_get_max_threads()
              << ", sum: " << sum << std::endl;
    return 0;
}
