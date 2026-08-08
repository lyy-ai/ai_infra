# 7.6 3D 并行与混合训练策略：并行拓扑设计
#
# 运行：
#   cd /data/liyangyang/ai_infra/06_分布式训练
#   /data/liyangyang/qwen35_env/bin/python 6.6_3D并行与混合训练策略/parallel_topo_design.py


def design_3d_topology(total_gpus, layers, hidden, params_b, tp_pref=8, max_layers_per_stage=32):
    tp = min(tp_pref, total_gpus)
    remaining = total_gpus // tp

    # 选择最小的可用 PP（减少气泡），要求 layers 能被 pp 整除，且每阶段不超过 max_layers_per_stage
    pp_candidates = [p for p in range(1, remaining + 1)
                       if remaining % p == 0
                       and layers % p == 0
                       and 1 <= layers // p <= max_layers_per_stage]
    pp = min(pp_candidates) if pp_candidates else 1
    dp = remaining // pp

    return tp, pp, dp


def main():
    total_gpus = 64
    layers = 80
    hidden = 8192
    params_b = 175

    tp, pp, dp = design_3d_topology(total_gpus, layers, hidden, params_b)
    print(f"3D Parallel Design for {total_gpus} GPUs, {params_b}B model:")
    print(f"  TP={tp}, PP={pp}, DP={dp}")
    print(f"  total GPUs = {tp * pp * dp} (check: {tp * pp * dp == total_gpus})")
    print(f"  layers per PP stage = {layers // pp}")


if __name__ == "__main__":
    main()
