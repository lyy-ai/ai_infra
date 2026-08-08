# 综合项目：编译 / Runtime / 量化收益估算
#
# 运行：
#   cd /data/liyangyang/ai_infra/10_第三阶段：综合项目实战
#   /data/liyangyang/qwen35_env/bin/python 10.1_综合项目_MMBEV端到端多平台部署优化/compile_runtime_quant_sim.py


def graph_opt(ops=1000):
    reduced = int(ops * 0.05) + int(ops * 0.15) + int(ops * 0.03)
    return ops, ops - reduced


def memory_pool(no_pool_gb=14.0, pool_gb=9.8):
    return (no_pool_gb - pool_gb) / no_pool_gb


def cuda_graph(kernels=200, overhead_ms=0.04):
    return kernels * overhead_ms


def int8_speedup(fp16_ms):
    return fp16_ms, fp16_ms / 1.8


def main():
    before, after = graph_opt()
    print(f"Graph opt:     {before} ops -> {after} ops (-{(before - after) / before:.0%})")
    print(f"Memory pool:   saved {memory_pool():.0%} peak memory")
    print(f"CUDA Graph:    saved ~{cuda_graph():.1f} ms launch overhead (Orin)")
    fp16, int8 = int8_speedup(55.0)
    print(f"INT8 quant:    Orin {fp16:.0f}ms -> {int8:.0f}ms ({fp16 / int8:.1f}x)")


if __name__ == "__main__":
    main()
