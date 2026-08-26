# 1.4 学习路线总览：自查清单
#
# 运行：
#   /data/qwen35_env/bin/python 1.4_学习路线总览/learning_path_checklist.py


PATH = [
    ("第一阶段 全景认知与前置基础", "概念清晰，能手算显存/带宽/通信量"),
    ("CUDA 与算子优化", "能手写并调优 kernel，会用 nsys/ncu"),
    ("分布式训练", "能设计 TP/PP/DP 拓扑，算得清显存"),
    ("推理与部署", "能端到端把模型推上线（编译/Runtime/量化/Serving）"),
    ("综合项目实战", "能讲清每个决策的取舍"),
    ("简历与面试冲刺", "STAR + 量化，四岗位模拟通过"),
]

TRADEOFFS = [
    ("ZeRO", "通信", "显存"), ("Activation Ckpt", "重计算", "显存"),
    ("TP", "通信带宽", "单卡显存"), ("INT8", "精度", "显存+速度"),
    ("CUDA Graph", "灵活性", "launch开销"), ("PagedAttention", "管理复杂度", "显存利用率"),
]


def main():
    print("=== 学习路线自查 ===")
    for stage, bar in PATH:
        print(f"  [ ] {stage}：{bar}")
    print("\n=== 取舍表（每学一个技术填一行）===")
    print(f"{'技术':<18} | {'牺牲':<10} | {'换取'}")
    print("-" * 46)
    for name, cost, gain in TRADEOFFS:
        print(f"{name:<18} | {cost:<10} | {gain}")


if __name__ == "__main__":
    main()
