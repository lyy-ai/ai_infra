MODEL_PATH = "/data/liyangyang/models/Qwen2-0.5B-Instruct"
SERVED_MODEL_NAME = "qwen2-0.5b-instruct"
RESULTS_DIR = "results"

# vLLM offline benchmark 配置：为了在当前共享 GPU 上可复现，默认使用较小显存与 eager 模式。
GPU_MEMORY_UTILIZATION = 0.50
MAX_MODEL_LEN = 1024
MAX_NEW_TOKENS = 16
ENFORCE_EAGER = True
ENABLE_PREFIX_CACHING = True
TEMPERATURE = 0.0
TOP_P = 1.0

# 压测批量：用于展示 batch/concurrency 对吞吐的影响。
BATCH_SIZES = [1, 8, 32]
USE_SHARED_PREFIX = True
SHARED_PREFIX_REPEAT = 6

SAMPLE_PROMPTS = [
    "请介绍一下机器学习中的梯度下降算法。",
    "什么是注意力机制？它在自然语言处理中有什么作用？",
    "简述大语言模型量化技术的意义。",
    "请用中文解释 Mixture of Experts 架构。",
    "什么是 KV Cache？为什么它能加速自回归生成？",
    "PagedAttention 解决了传统 KV Cache 管理的什么问题？",
    "Continuous Batching 相比 Static Batching 有什么优势？",
    "Prefix Cache 适合什么业务场景？",
]
