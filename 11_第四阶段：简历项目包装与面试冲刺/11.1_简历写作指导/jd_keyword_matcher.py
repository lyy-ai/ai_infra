# 1. 简历写作指导：简历-JD 关键词匹配
#
# 运行：
#   cd /data/ai_infra/11_第四阶段：简历项目包装与面试冲刺
#   /data/qwen35_env/bin/python 11.1_简历写作指导/jd_keyword_matcher.py


RESUME_TEXT = """
CUDA kernel, TVM Relax, TensorRT, INT8 quantization, memory pool, CUDA Graph,
BEV pooling, warp reduce, shared memory, Laser compiler, Ascend CANN, benchmark
"""

JD_KEYWORDS = {
    "CUDA / 算子优化": ["CUDA", "kernel", "shared memory", "Tensor Core", "Nsight", "warp", "HPC"],
    "推理框架 / LLM Serving": ["vLLM", "PagedAttention", "KV Cache", "Continuous Batching", "TensorRT"],
    "AI 编译器": ["TVM", "Relax", "compiler", "graph optimization", "codegen", "dynamic shape"],
    "推理部署（车企/边缘）": ["TensorRT", "quantization", "CUDA", "INT8", "real-time", "C++"],
}


def match(resume, keywords):
    low = resume.lower()
    hit = [k for k in keywords if k.lower() in low]
    return hit, len(hit) / len(keywords)


def main():
    print("=== Resume vs JD Keyword Match ===\n")
    for role, kws in JD_KEYWORDS.items():
        hit, ratio = match(RESUME_TEXT, kws)
        miss = [k for k in kws if k not in hit]
        print(f"[{role}] 匹配度 {ratio:.0%}")
        print(f"  命中: {', '.join(hit) if hit else '(无)'}")
        print(f"  缺失: {', '.join(miss) if miss else '(无)'}\n")


if __name__ == "__main__":
    main()
