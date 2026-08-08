# 1.1 什么是 AI Infra：不可能三角权衡计算
#
# 运行：
#   /data/liyangyang/qwen35_env/bin/python 1.1_什么是AI_Infra/infra_tradeoff_calc.py


def main():
    print("=== 计算 / 通信 / 显存 不可能三角 ===\n")
    rows = [
        ("ZeRO-3 分片", "通信 +AllGather/RS", "显存 16P → 16P/N", "计算 不变"),
        ("Activation Ckpt", "通信 不变", "激活显存 ~sqrt(L)", "计算 +30% 重算"),
        ("TP 张量并行", "通信 每层 AllReduce", "单卡参数 /TP", "计算 不变"),
        ("INT8 量化", "通信 不变", "权重显存 /2", "精度 -0.x%"),
        ("CUDA Graph", "灵活性（shape 固定）", "显存 不变", "launch 开销→0"),
    ]
    print(f"{'技术':<18} | {'牺牲/代价':<26} | {'换取'}")
    print("-" * 78)
    for name, cost, mem, comp in rows:
        print(f"{name:<18} | {cost:<26} | {mem}；{comp}")
    print()
    print("核心思维：每个优化都是三角上的取舍，没有免费午餐。")


if __name__ == "__main__":
    main()
