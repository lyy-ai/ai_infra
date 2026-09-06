# 1.2 大厂 AI Infra 组织架构速查
#
# 运行：
#   /data/qwen35_env/bin/python 1.2_大厂AI_Infra组织架构/org_landscape.py


ORGS = [
    ("NVIDIA", "CUDA / TensorRT / Triton", "GPU 架构、CUDA 编程、性能分析"),
    ("字节 AML", "训练 Infra / 推理引擎", "大规模训练、成本与稳定性"),
    ("阿里 PAI", "PAI-DLC / PAI-EAS / Blade", "平台工程、编译优化、资源调度"),
    ("百度", "Paddle / FastDeploy / 昆仑芯", "框架机制、算子、软硬协同"),
    ("自动驾驶", "部署 / HPC 算子 / 车端引擎", "TensorRT、量化、C++、实时性"),
]


def main():
    print("=== 大厂 AI Infra 组织速查 ===\n")
    print(f"{'公司':<10} | {'团队/产品':<28} | {'面试侧重'}")
    print("-" * 72)
    for name, products, focus in ORGS:
        print(f"{name:<10} | {products:<28} | {focus}")


if __name__ == "__main__":
    main()
