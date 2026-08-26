# Qwen2-0.5B vLLM 集群服务实战项目

## 项目概述

本项目以 **Qwen2-0.5B-Instruct**（`/data/models/Qwen2-0.5B-Instruct`）为对象，搭建一套可复现的 vLLM 推理服务实验闭环：

- 单实例 vLLM OpenAI-compatible 服务启动脚本。
- 多实例集群启动脚本与 Nginx `least_conn` 路由示例。
- vLLM 离线 batch 吞吐 benchmark，输出 JSON 结果。
- 结果分析脚本，直接生成可写进简历的指标模板。

项目目标不是“把 0.5B 跑起来”，而是沉淀一套可迁移到更大模型的 **Serving 优化方法论**：baseline 定义、batch/concurrency 压测、PagedAttention/Prefix Cache/Continuous Batching 参数调优、集群路由与结果复现。

---

## 1. 项目结构

```
简历项目/Qwen2VLLMClusterProject/
├── README.md                          # 项目说明（本文件）
├── requirements.txt                   # 最小依赖
├── config.py                          # 模型路径、benchmark 参数、prompt 配置
├── src/
│   ├── vllm_env_helper.py             # vLLM 环境初始化（CUDA 13 库、PATH、GPU 选择）
│   ├── prompts.py                     # benchmark prompt 构造，支持 shared prefix
│   ├── metrics.py                     # 吞吐/延迟指标计算
│   ├── offline_benchmark.py           # vLLM 离线 batch benchmark 核心逻辑
│   └── utils.py                       # JSON 保存、表格打印
├── scripts/
│   ├── run_offline_benchmark.py       # 运行离线吞吐 benchmark，写 results/
│   ├── analyze_results.py             # 分析 results 并生成简历 bullet
│   ├── chat_demo.py                   # 可选：离线 chat demo（--offline）
│   ├── start_qwen2_vllm_server.sh     # 启动单实例 OpenAI-compatible 服务
│   ├── start_qwen2_cluster.sh         # 启动多实例集群
│   └── router_nginx.conf.example      # Nginx least_conn 路由示例
└── results/
    └── offline_throughput.json        # benchmark 输出（运行后生成）
```

---

## 2. 快速开始

### 2.1 离线吞吐 benchmark

```bash
cd /data/ai_infra/05_LLM推理与Serving/简历项目/Qwen2VLLMClusterProject
PATH=/data/qwen35_env/bin:$PATH python scripts/run_offline_benchmark.py
PATH=/data/qwen35_env/bin:$PATH python scripts/analyze_results.py
```

说明：

- 默认使用 `ENFORCE_EAGER=True`，便于在共享 GPU 上快速复现；追求极限性能时可改为 `False`。
- 默认 `ENABLE_PREFIX_CACHING=True`，并使用较长 shared prefix 模拟固定 system prompt 场景。
- 默认 batch 为 `[1, 8, 32]`，用于观察 Continuous Batching 下吞吐随 batch 的变化。

### 2.2 单实例服务

```bash
cd /data/ai_infra/05_LLM推理与Serving/简历项目/Qwen2VLLMClusterProject
bash scripts/start_qwen2_vllm_server.sh
```

关键环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `/data/models/Qwen2-0.5B-Instruct` | 模型路径 |
| `PORT` | `8000` | 服务端口 |
| `GPU_ID` | `0` | 使用哪张卡 |
| `MAX_NUM_SEQS` | `128` | vLLM 最大并发序列数 |
| `GPU_MEM_UTIL` | `0.85` | vLLM 显存利用率 |
| `MAX_MODEL_LEN` | `4096` | 最大上下文长度 |
| `ENABLE_PREFIX_CACHING` | `1` | 是否开启 Prefix Cache |

### 2.3 多实例集群

```bash
cd /data/ai_infra/05_LLM推理与Serving/简历项目/Qwen2VLLMClusterProject
GPU_IDS="0 1" PORTS="8000 8001" bash scripts/start_qwen2_cluster.sh
```

Nginx 路由示例见 `scripts/router_nginx.conf.example`。启动后可用项目根目录上一级提供的并发压测脚本，或用任何 OpenAI-compatible client 访问网关。

---

## 3. 指标口径：怎么证明“吞吐提升”

简历里写“吞吐提升 4 倍”必须能回答：baseline 是什么、负载是什么、延迟有没有恶化。

建议至少保留三组结果：

| 实验 | 配置 | 说明 |
|------|------|------|
| baseline | batch=1，短 prompt，`max_new_tokens=16` | 近似串行/低并发下吞吐 |
| batched | batch=8/32，同 prompt 分布 | Continuous Batching 带来的吞吐提升 |
| shared prefix | 固定 system prompt，多请求共享前缀 | Prefix Cache 对 prefill 的节省 |

本项目的 `offline_throughput.json` 会记录：

- `elapsed_s`：该 batch 总耗时。
- `prompt_tokens` / `completion_tokens`：输入/输出 token 数。
- `tok_per_s`：总吞吐。
- `req_per_s`：请求级吞吐。
- `prefill_tok_per_s` / `decode_tok_per_s`：用于判断瓶颈在 prefill 还是 decode。

---

## 4. 简历写法模板

> 用 `scripts/analyze_results.py` 输出的真实数值替换下面的占位符。

**中文版本：**

```text
基于 vLLM 部署 Qwen2-0.5B-Instruct 推理服务，构建单实例 + 多实例 Nginx 集群两套形态；
通过 Continuous Batching、PagedAttention、Prefix Caching 与 max_num_seqs/gpu_memory_utilization/max_model_len 调优，
在 <GPU>、batch=<B>、max_new_tokens=<N> 的离线压测下，吞吐从 batch=1 的 <base> tok/s 提升至 batch=<B> 的 <best> tok/s（<X>x）；
项目包含启动脚本、集群路由示例、压测脚本与结果分析脚本，可复现并支持迁移到更大模型。
```

**更偏生产的版本：**

```text
设计并实现基于 vLLM 的 Qwen2-0.5B 推理服务实验框架：单实例 OpenAI-compatible API、多实例 Nginx least_conn 集群、
离线 batch benchmark 与结果分析闭环；针对固定 system prompt 场景启用 Prefix Caching，针对不同 batch 对比吞吐与延迟，
形成 baseline/batched/shared-prefix 三组可复现实验，为后续 7B/9B 模型 Serving 调优提供方法论。
```

---

## 5. 面试可能追问

- **为什么 0.5B 还要集群？** 0.5B 是低成本复现 Serving 方法论的载体；真正价值在于参数、压测与路由闭环可迁移到更大模型。
- **吞吐提升来自哪里？** 主要来自 Continuous Batching 提高 GPU 利用率；固定前缀场景还来自 Prefix Cache 减少重复 prefill；多实例来自水平扩容与路由分流。
- **为什么默认 eager？** 当前 GPU 被其他任务共享，eager 更快复现；生产压测应关闭 eager 并固定 GPU、CUDA graph、版本与随机种子。
- **如果并发继续上升会怎样？** 需要观察 KV Cache block 使用率、queue wait、p95/p99 TTFT/TPOT，而不是只看 tok/s。

---

## 6. 与更大项目的关系

本项目是简历项目里的“最小可复现闭环”。要让它更像生产项目，可以继续加：

1. 真实 OpenAI-compatible 并发压测，而不仅是 offline batch。
2. baseline/tuned/cluster 三组结果，明确吞吐提升倍数。
3. Prometheus 指标：TTFT、TPOT、queue wait、KV usage、prefix hit rate。
4. 模型升级路径：同一套脚本切到 Qwen2.5-7B/Qwen3.5-9B，验证方法是否仍成立。

---

## 7. 当前结果

已在当前环境运行 `scripts/run_offline_benchmark.py`，结果保存到 `results/offline_throughput.json`：

| batch | elapsed_s | tok/s | req/s |
|------:|----------:|------:|------:|
| 1 | 0.590 | 281.51 | 1.70 |
| 8 | 0.504 | 2543.30 | 15.87 |
| 32 | 0.558 | 9186.30 | 57.32 |

对应简历 bullet（由 `scripts/analyze_results.py` 生成）：

> 基于 vLLM 部署 Qwen2-0.5B-Instruct 推理服务，离线 batch 压测显示：batch=1 时 281.5 tok/s，batch=32 时 9186.3 tok/s（32.63x）；max_model_len=1024，prefix_caching=True，enforce_eager=True。

注意：该结果在当前共享 GPU、`enforce_eager=True`、`max_model_len=1024`、`max_new_tokens=16` 的轻量配置下得到，主要用于复现实验流程；生产简历指标应在固定 GPU、关闭 eager、明确 prompt 分布与并发模型后重新压测。
