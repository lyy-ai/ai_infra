# 3. 模拟面试：按岗位抽题模拟
#
# 运行：
#   cd /data/ai_infra/11_第四阶段：简历项目包装与面试冲刺
#   /data/qwen35_env/bin/python 11.3_模拟面试/mock_interview.py [role]
# role: operator | compiler | serving | training（默认随机）

import random
import sys

ROLES = {
    "operator": {
        "name": "CUDA / 算子优化",
        "questions": [
            "自我介绍（5min，突出 HPC kernel 项目）",
            "深挖：你的 BEV Pooling kernel 为什么快 65%？具体优化手段？",
            "手写：画出 GEMM 的 block/thread tiling 结构",
            "基础：shared memory bank conflict 是什么？如何避免？",
            "基础：FP16 累加为什么用 FP32 accumulator？",
        ],
    },
    "compiler": {
        "name": "AI 编译器",
        "questions": [
            "自我介绍（5min，突出编译流水线项目）",
            "深挖：你的编译流水线各阶段？兼容性问题怎么解决的？",
            "问答：Relay 和 Relax 的区别？",
            "问答：动态 shape 的三种处理策略？",
            "场景：一个新算子接入编译器的完整流程？",
        ],
    },
    "serving": {
        "name": "推理框架 / LLM Serving",
        "questions": [
            "自我介绍（5min，突出推理引擎项目）",
            "深挖：continuous batching 在你的引擎里怎么实现的？",
            "问答：PagedAttention 为什么能提升吞吐？",
            "问答：TTFT 和 TBT 冲突时怎么取舍？",
            "场景：设计支持 1000 并发的 LLM serving 系统",
        ],
    },
    "training": {
        "name": "分布式训练",
        "questions": [
            "自我介绍（5min，突出 ZeRO/3D 并行经验）",
            "深挖：你的 TP/PP/DP 拓扑怎么定的？通信优化做了什么？",
            "问答：ZeRO 三个阶段各分片什么？",
            "计算：70B 模型 Adam 混合精度训练显存估算",
            "场景：512 卡训练 70B，设计并行方案",
        ],
    },
}


def main():
    role = sys.argv[1] if len(sys.argv) > 1 else random.choice(list(ROLES))
    spec = ROLES.get(role, ROLES[random.choice(list(ROLES))])
    print(f"=== 模拟面试：{spec['name']} 岗 ===")
    print("规则：每题计时，答完自评 1-5 分\n")
    total = 0
    for i, q in enumerate(spec["questions"], 1):
        print(f"[{i}/{len(spec['questions'])}] {q}")
        input("  （答完回车）")
        score = input("  自评(1-5): ").strip()
        total += int(score) if score.isdigit() else 0
    avg = total / len(spec["questions"])
    print(f"\n面试结束，平均自评 {avg:.1f}/5")
    print("复盘：把 < 4 分的题写入错题本，24h 内补齐答案")


if __name__ == "__main__":
    main()
