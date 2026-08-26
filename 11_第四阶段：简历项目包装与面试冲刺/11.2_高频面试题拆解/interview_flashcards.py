# 2. 高频面试题拆解：自测抽认卡
#
# 运行（交互）：
#   cd /data/ai_infra/11_第四阶段：简历项目包装与面试冲刺
#   /data/qwen35_env/bin/python 11.2_高频面试题拆解/interview_flashcards.py

import random
import sys

CARDS = [
    ("算子", "GEMM 如何优化？", "tiling → shared memory → 寄存器分块 → Tensor Core → float4 向量化"),
    ("算子", "FlashAttention 原理？", "分块 + online softmax，N×N 矩阵不落 HBM，IO 从 O(N²) 降到 O(N)"),
    ("算子", "Bank Conflict 怎么避免？", "padding（A[33]）、调整排布、swizzle；用 NCU bank conflict 指标定位"),
    ("编译器", "常见图优化 Pass？", "常量折叠、死代码消除、算子融合、布局转换、CSE"),
    ("编译器", "TVM 架构？", "Relay/Relax → 图优化 → TE/TIR 调度 → codegen；MetaSchedule 搜索"),
    ("编译器", "动态 Shape 怎么处理？", "分桶编译优先；符号维度 + VM + 动态内存兜底"),
    ("Runtime", "内存池怎么设计？", "shape 分桶缓存 + liveness 复用 + per-stream 池"),
    ("Runtime", "CUDA Graph 限制？", "shape 固定、无 CPU-GPU 同步、无动态分支"),
    ("量化", "PTQ vs QAT？", "PTQ 校准快速精度略差；QAT 模拟量化训练精度好成本高"),
    ("量化", "SmoothQuant 原理？", "激活除以 s、权重乘 s，平滑 outlier，离线融合零开销"),
    ("分布式", "ZeRO 三阶段？", "S1 优化器分片、S2 +梯度分片、S3 +参数分片；通信换显存"),
    ("分布式", "TP/PP/DP 拓扑？", "TP 节点内吃 NVLink，PP 跨节点，DP 填满剩余"),
    ("LLM推理", "PagedAttention？", "KV Cache 分页管理（类虚拟内存），消除碎片，吞吐 2-4 倍"),
    ("LLM推理", "Continuous Batching？", "iteration 级调度，完成即移出、新请求即插入"),
    ("LLM推理", "Speculative Decoding？", "draft 猜 k 个 token，target 并行验证，分布不变，2-3 倍加速"),
]


def main():
    n = 5 if len(sys.argv) < 2 else int(sys.argv[1])
    random.seed()
    picked = random.sample(CARDS, min(n, len(CARDS)))
    score = 0
    for i, (mod, q, a) in enumerate(picked, 1):
        print(f"\n[{i}/{len(picked)}] 【{mod}】{q}")
        input("  （想好后回车看答案）")
        print(f"  答案要点：{a}")
        ok = input("  答对了吗？(y/n): ").strip().lower()
        if ok == "y":
            score += 1
    print(f"\n自测得分：{score}/{len(picked)}")


if __name__ == "__main__":
    main()
