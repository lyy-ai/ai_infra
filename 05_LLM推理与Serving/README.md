# 第五章：LLM 推理与 Serving

本目录包含《LLM 推理与 Serving》专题的课程资料与代码示例。

## 目录结构

```
05_LLM推理与Serving/
├── 5.1_LLM推理基础/                              # 5.1 LLM 推理基础
│   ├── 5.1_LLM推理基础.md
│   ├── basic_inference.py
│   └── kv_cache_analysis.py
├── 5.2_PagedAttention与KVCache管理/               # 5.2 PagedAttention 与 KV Cache 管理
│   ├── 5.2_PagedAttention与KVCache管理.md
│   ├── pagedattention_simulator.py
│   ├── vllm_basic_inference.py
│   ├── vllm_vs_transformers_benchmark.py
│   └── vllm_env_helper.py
├── 5.3_ContinuousBatching/                       # 5.3 Continuous Batching
│   ├── 5.3_ContinuousBatching.md
│   ├── continuous_batching_simulator.py
│   ├── vllm_continuous_batching.py
│   └── vllm_env_helper.py
├── 5.4_SpeculativeDecoding/                      # 5.4 Speculative Decoding（投机解码）
│   ├── 5.4_SpeculativeDecoding.md
│   ├── speculative_decoding_simulator.py
│   ├── transformers_speculative_decoding.py
│   ├── vllm_speculative_decoding.py
│   └── vllm_env_helper.py
├── 5.5_PrefixCache/                              # 5.5 Prefix Cache（前缀缓存）
│   ├── 5.5_PrefixCache.md
│   ├── prefix_cache_simulator.py
│   ├── vllm_prefix_cache.py
│   └── vllm_env_helper.py
├── 5.6_PrefillDecode解耦/                         # 5.6 Prefill/Decode 解耦（PD Disaggregation）
│   ├── 5.6_PrefillDecode解耦.md
│   ├── pd_disaggregation_simulator.py
│   └── kv_transfer_cost_analysis.py
├── 5.7_主流推理框架/                              # 5.7 主流推理框架
│   ├── 5.7_主流推理框架.md
│   ├── serving_framework_comparison.py
│   └── openai_compatible_smoke_test.py
├── 5.8_多机多卡Serving/                           # 5.8 多机多卡 Serving
│   ├── 5.8_多机多卡Serving.md
│   ├── parallelism_planner.py
│   └── serving_autoscaling_simulator.py
├── 5.9_业务模型推理引擎开发/                      # 5.9 业务模型推理引擎开发
│   ├── 5.9_业务模型推理引擎开发.md
│   ├── bev_memory_compute_estimator.py
│   ├── operator_fusion_roofline.py
│   └── ad_pipeline_parallel_simulator.py
├── 5.10_端侧与多模态推理/                          # 5.10 端侧与多模态推理
│   ├── 5.10_端侧与多模态推理.md
│   ├── edge_multimodal_memory_estimator.py
│   ├── vla_action_budget_estimator.py
│   └── edge_colocated_scheduler.py
├── 简历项目/                                  # 项目：简历写法与 Qwen2-0.5B vLLM 集群实战
│   ├── 简历项目.md
│   ├── start_qwen2_vllm_server.sh
│   ├── start_qwen2_cluster.sh
│   ├── router_nginx.conf.example
│   ├── benchmark_qwen2_concurrency.py
│   └── Qwen2VLLMClusterProject/               # 工程化子项目：config/src/scripts/results + 真实 benchmark 结果
├── README.md                                  # 本文件
└── examples/                                  # 原始平铺示例（保留作兼容/备份）
```

> 建议按主题文件夹阅读与运行：每个文件夹内都是“课程讲义 md + 对应代码”。vLLM 相关文件夹内已放入 `vllm_env_helper.py`，因此在该脚本所在目录运行示例时可直接完成 vLLM 环境初始化。

## 示例与课时对应关系

| 示例脚本 | 对应课时 | 对应章节 | 说明 |
|---------|---------|---------|------|
| `5.1_LLM推理基础/kv_cache_analysis.py` | 5.1 LLM 推理基础 | 4. KV Cache 详解 | KV Cache 显存估算、batch/seq_len 增长表、MHA/GQA/MQA 对比 |
| `5.1_LLM推理基础/basic_inference.py` | 5.1 LLM 推理基础 | 8. 代码实践：使用 transformers 进行推理 | 基础生成、Latency/TPS 测量、TTFT/TPOT 分离、解码策略对比 |
| `5.2_PagedAttention与KVCache管理/pagedattention_simulator.py` | 5.2 PagedAttention 与 KV Cache 管理 | 2. PagedAttention 核心思想 | 块分配模拟、按需分配、前缀共享、连续 vs 分页分配对比 |
| `5.2_PagedAttention与KVCache管理/vllm_basic_inference.py` | 5.2 PagedAttention 与 KV Cache 管理 | 5. 代码实践：vLLM 推理与对比 | vLLM 离线批量推理、Prefix Caching、KV Cache 量化 |
| `5.2_PagedAttention与KVCache管理/vllm_vs_transformers_benchmark.py` | 5.2 PagedAttention 与 KV Cache 管理 | 6. 性能对比：transformers vs vLLM | transformers 顺序/静态 batch 与 vLLM 的吞吐对比 |
| `5.3_ContinuousBatching/continuous_batching_simulator.py` | 5.3 Continuous Batching | 7. 代码实践：Continuous Batching 模拟器 | Static vs Continuous Batching 对比、FCFS、Preemption |
| `5.3_ContinuousBatching/vllm_continuous_batching.py` | 5.3 Continuous Batching | 8. 代码实践：vLLM 中的 Continuous Batching | vLLM 内部 Continuous Batching、不同并发数吞吐对比 |
| `5.4_SpeculativeDecoding/speculative_decoding_simulator.py` | 5.4 Speculative Decoding | 7. 代码实践：Speculative Decoding 模拟器 | Draft + Target 模拟、Rejection Sampling、接受率与加速比 |
| `5.4_SpeculativeDecoding/transformers_speculative_decoding.py` | 5.4 Speculative Decoding | 8. 代码实践：使用 transformers 进行 Speculative Decoding | transformers assistant_model 投机解码示例 |
| `5.4_SpeculativeDecoding/vllm_speculative_decoding.py` | 5.4 Speculative Decoding | 8. 代码实践：vLLM 中的 Speculative Decoding | vLLM speculative_model 投机解码示例 |
| `5.5_PrefixCache/prefix_cache_simulator.py` | 5.5 Prefix Cache | 6. 代码实践：Prefix Cache 模拟器 | Block 级前缀命中、system prompt 共享、多轮对话、Block 边界、LRU 淘汰 |
| `5.5_PrefixCache/vllm_prefix_cache.py` | 5.5 Prefix Cache | 7. 代码实践：vLLM Prefix Cache | `enable_prefix_caching=True`、cold/warm shared prefix 对比、unique prefix 对照 |
| `5.6_PrefillDecode解耦/pd_disaggregation_simulator.py` | 5.6 Prefill/Decode 解耦 | 7. 代码实践：PD 调度模拟器 | Coupled vs Disaggregated 的 TTFT/TPOT/decode stall 对比 |
| `5.6_PrefillDecode解耦/kv_transfer_cost_analysis.py` | 5.6 Prefill/Decode 解耦 | 8. 代码实践：KV Transfer 成本估算 | KV 传输数据量、不同带宽下传输时间、batch 放大效应 |
| `5.7_主流推理框架/serving_framework_comparison.py` | 5.7 主流推理框架 | 3. 定性对比与选型建议 | 框架能力矩阵、场景加权推荐 |
| `5.7_主流推理框架/openai_compatible_smoke_test.py` | 5.7 主流推理框架 | 4. 用 OpenAI-compatible API 做统一冒烟测试 | 标准库调用 `/v1/chat/completions`，默认不请求服务 |
| `5.8_多机多卡Serving/parallelism_planner.py` | 5.8 多机多卡 Serving | 4. TP/PP 估算与部署建议 | 估算 TP/PP 下每 GPU 权重与 KV 占用，给出部署建议 |
| `5.8_多机多卡Serving/serving_autoscaling_simulator.py` | 5.8 多机多卡 Serving | 6. 弹性扩缩容 | round-robin/least-queue、突发流量、autoscaler warmup 与 cooldown |
| `5.9_业务模型推理引擎开发/bev_memory_compute_estimator.py` | 5.9 业务模型推理引擎开发 | 2. BEV/PlanNN 的计算热点 | BEV 显存、dense/deformable attention FLOPs、结构优化收益估算 |
| `5.9_业务模型推理引擎开发/operator_fusion_roofline.py` | 5.9 业务模型推理引擎开发 | 4. 模型结构优化与算子定制 | roofline 估算、memory-bound 识别、算子融合收益 |
| `5.9_业务模型推理引擎开发/ad_pipeline_parallel_simulator.py` | 5.9 业务模型推理引擎开发 | 5. 推理流水线并行优化 | Sequential vs Pipelined 的延迟/FPS 对比 |
| `5.10_端侧与多模态推理/edge_multimodal_memory_estimator.py` | 5.10 端侧与多模态推理 | 2. 开源大模型适配：Qwen 系列 | VLM 权重、KV Cache、image tokens、workspace 内存估算 |
| `5.10_端侧与多模态推理/vla_action_budget_estimator.py` | 5.10 端侧与多模态推理 | 3. 开源大模型适配：VLA 系列 | action chunk token 数、控制频率、decode token/s 预算 |
| `5.10_端侧与多模态推理/edge_colocated_scheduler.py` | 5.10 端侧与多模态推理 | 5. 端侧共线多模态模型部署 | safety/planner pinned、VLM 内存准入、拒绝/降级策略模拟 |
| `简历项目/benchmark_qwen2_concurrency.py` | 项目：简历项目 | 3. Qwen2-0.5B vLLM 集群实战 | OpenAI-compatible 并发压测、吞吐/延迟/成功率统计、简历 bullet 模板 |
| `简历项目/Qwen2VLLMClusterProject/scripts/run_offline_benchmark.py` | 项目：简历项目 | 工程化离线 benchmark | vLLM offline batch 吞吐压测，输出 `results/offline_throughput.json` |

## 环境要求

- Python 3.10+
- PyTorch 2.0+
- transformers 4.35+
- 1 张 NVIDIA GPU（显存 24GB+，用于运行基础推理示例）
- vLLM（可选，用于 vLLM 相关示例）

安装 vLLM：

```bash
pip install vllm
```

> 注意：vLLM 对 CUDA 和 PyTorch 版本有要求，请确保与当前环境兼容。当前环境中，vLLM 0.25.0 需要预加载 `nvidia-cu13` 提供的动态库，并确保 `ninja` 在 PATH 中。所有 vLLM 示例脚本已统一通过 `import vllm_env_helper` 完成这些环境设置；主进程会自动选择空闲显存最多的 GPU。

## 运行示例

```bash
source /data/qwen35_env/bin/activate

cd /data/ai_infra/05_LLM推理与Serving

# 非 vLLM 示例，直接运行即可
python 5.1_LLM推理基础/kv_cache_analysis.py
python 5.2_PagedAttention与KVCache管理/pagedattention_simulator.py
python 5.3_ContinuousBatching/continuous_batching_simulator.py
python 5.4_SpeculativeDecoding/speculative_decoding_simulator.py
python 5.5_PrefixCache/prefix_cache_simulator.py
python 5.6_PrefillDecode解耦/pd_disaggregation_simulator.py
python 5.6_PrefillDecode解耦/kv_transfer_cost_analysis.py
python 5.7_主流推理框架/serving_framework_comparison.py
python 5.7_主流推理框架/openai_compatible_smoke_test.py
python 5.8_多机多卡Serving/parallelism_planner.py
python 5.8_多机多卡Serving/serving_autoscaling_simulator.py
python 5.9_业务模型推理引擎开发/bev_memory_compute_estimator.py
python 5.9_业务模型推理引擎开发/operator_fusion_roofline.py
python 5.9_业务模型推理引擎开发/ad_pipeline_parallel_simulator.py
python 5.10_端侧与多模态推理/edge_multimodal_memory_estimator.py
python 5.10_端侧与多模态推理/vla_action_budget_estimator.py
python 5.10_端侧与多模态推理/edge_colocated_scheduler.py

# vLLM 相关示例，建议显式设置 PATH 以使用 venv 内的 ninja
PATH=/data/qwen35_env/bin:$PATH python 5.2_PagedAttention与KVCache管理/vllm_basic_inference.py
PATH=/data/qwen35_env/bin:$PATH python 5.2_PagedAttention与KVCache管理/vllm_vs_transformers_benchmark.py
PATH=/data/qwen35_env/bin:$PATH python 5.3_ContinuousBatching/vllm_continuous_batching.py
PATH=/data/qwen35_env/bin:$PATH python 5.4_SpeculativeDecoding/vllm_speculative_decoding.py
PATH=/data/qwen35_env/bin:$PATH python 5.5_PrefixCache/vllm_prefix_cache.py

# 项目：Qwen2-0.5B vLLM 集群（需先启动服务）
# bash 简历项目/start_qwen2_vllm_server.sh
# GPU_IDS="0 1" PORTS="8000 8001" bash 简历项目/start_qwen2_cluster.sh
# python 简历项目/benchmark_qwen2_concurrency.py --base-url http://localhost:8080/v1 --model qwen2-0.5b-instruct --requests 128 --concurrency 128

# 项目：Qwen2-0.5B vLLM 工程化离线 benchmark（已生成 results/offline_throughput.json）
# cd 简历项目/Qwen2VLLMClusterProject
# PATH=/data/qwen35_env/bin:$PATH python scripts/run_offline_benchmark.py
# PATH=/data/qwen35_env/bin:$PATH python scripts/analyze_results.py

# 如需固定 GPU，可设置 CUDA_VISIBLE_DEVICES；vllm_env_helper 会在主进程选择空闲显存最多的 GPU
# CUDA_VISIBLE_DEVICES=1 PATH=/data/qwen35_env/bin:$PATH python 5.2_PagedAttention与KVCache管理/vllm_basic_inference.py
```

## 课程目标

通过本专题，你将系统掌握：

1. LLM 推理的核心流程与阶段划分（Prefill / Decode）。
2. 自回归生成、KV Cache、解码策略等关键概念。
3. LLM 推理性能指标及其测量方法。
4. 显存占用分析与优化方向。
5. 从基础推理到生产级 Serving 框架的演进路径。

## 后续课程预告

- 本专题 5.1-5.10 与「简历项目」实战已完结；后续可扩展真实硬件 benchmark、推理系统性能调优与压测实践。

---

*本课程基于 Qwen3.5-9B 模型与 transformers 库进行示例讲解。*
