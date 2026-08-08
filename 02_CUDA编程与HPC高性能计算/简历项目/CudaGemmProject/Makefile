NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= sm_80
NVCCFLAGS = -O3 -arch=$(ARCH) -std=c++17

all: build/gemm_bench

build/gemm_bench: src/main.cu src/gemm_kernels.cuh
	mkdir -p build results
	$(NVCC) $(NVCCFLAGS) -o $@ src/main.cu -lcublas

run: build/gemm_bench
	./build/gemm_bench > results/cuda_gemm_benchmark.json

cublas: 
	/data/liyangyang/qwen35_env/bin/python scripts/compare_cublas.py

report:
	/data/liyangyang/qwen35_env/bin/python scripts/analyze_results.py

clean:
	rm -rf build

.PHONY: all run cublas report clean
