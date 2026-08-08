# 综合项目：HPC 算子优化收益（BEV Pooling + 算子融合）
#
# 运行：
#   cd /data/liyangyang/ai_infra/10_第三阶段：综合项目实战
#   /data/liyangyang/qwen35_env/bin/python 10.1_综合项目_MMBEV端到端多平台部署优化/hpc_kernel_sim.py


def bev_pooling(grid, points, level):
    base = grid[0] * grid[1] * points * 1.5e-8
    speedup = {0: 1.0, 1: 1.8, 2: 2.5, 3: 3.0}[level]
    return base / speedup * 1000


def fusion_traffic(seq, hidden, layers):
    rw = 2 * 2  # read+write, fp16 bytes
    unfused = seq * hidden * layers * rw * 3
    fused = seq * hidden * layers * rw * 1
    return unfused, fused


def main():
    print("=== BEV Pooling Optimization (Orin) ===")
    print(f"{'Level':<22} | {'Latency(ms)':>11}")
    print("-" * 38)
    naive = None
    for name, lv in [("naive scatter", 0), ("shared_mem", 1),
                     ("half2+tensorcore", 2), ("fused+warp reduce", 3)]:
        lat = bev_pooling((200, 200), 64, lv)
        naive = naive or lat
        print(f"{name:<22} | {lat:>10.1f} ({naive / lat:.1f}x)")

    print("\n=== LayerNorm+SiLU Fusion ===")
    ub, fb = fusion_traffic(512, 256, 6)
    print(f"memory traffic: {ub / 1e9:.3f} GB -> {fb / 1e9:.3f} GB ({ub / fb:.1f}x)")


if __name__ == "__main__":
    main()
