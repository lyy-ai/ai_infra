# 1.3 AI Infra 岗位路线图
#
# 运行：
#   /data/qwen35_env/bin/python 1.3_AI_Infra岗位路线图/career_roadmap.py


LEVELS = [
    ("初级", "20k-35k/月", "推理部署 / TensorRT / CUDA 工程师",
     ["会用 TensorRT 转换部署", "能写基础 CUDA kernel", "跑通 DDP 训练"]),
    ("中级", "35k-60k/月", "AI Infra / 编译器 / LLM Serving 工程师",
     ["独立负责一个模块", "讲清 PA/ZeRO/图优化原理", "优化有 benchmark 支撑"]),
    ("高级", "60k-100k/月", "架构师 / 推理平台 / 基础设施负责人",
     ["设计平台架构与技术选型", "全链路协同", "技术指标转化为业务指标"]),
]


def main():
    for level, salary, roles, checks in LEVELS:
        print(f"=== {level}（{salary}）===")
        print(f"代表岗位：{roles}")
        for c in checks:
            print(f"  [ ] {c}")
        print()


if __name__ == "__main__":
    main()
