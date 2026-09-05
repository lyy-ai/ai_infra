# 第十二章：最新论文分享

AI Infra 领域最新与奠基性论文的精选解读（共 31 篇）。每篇解读按统一模板组织：背景与问题 → 核心设计 → 关键数据 → 计算/通信/显存三角取舍 → 与仓库章节的关联 → 局限与启发 → 面试视角。

选文标准：2024 年后对工业界有实际影响的工作（被主流框架/产品采用或代表明确的技术趋势），兼顾 2025-2026 年的最新进展。所有论文均通过 arXiv 镜像核对过标题与摘要，文内数字以"论文报告"口径标注。

## 目录结构

```
12_最新论文分享/
├── 12.1_推理引擎与Serving/
│   ├── 2312.07104_SGLang.md
│   ├── 2401.09670_DistServe.md
│   ├── 2407.00079_Mooncake.md
│   ├── 2506.13585_MiniMax-M1.md
│   └── 2507.20534_Kimi-K2.md
├── 12.2_训练系统与并行/
│   ├── 2402.15627_MegaScale.md
│   ├── 2412.19437_DeepSeek-V3.md
│   ├── 2508.09591_HierMoE.md
│   ├── 2509.03047_FlashRecovery.md
│   └── 2602.11543_SPES.md
├── 12.3_MoE系统/
│   ├── 2405.04434_DeepSeek-V2.md
│   ├── 2409.02060_OLMoE.md
│   ├── 2508.06471_GLM-4.5.md
│   └── 2601.21420_ConceptMoE.md
├── 12.4_长上下文与KVCache/
│   ├── 2406.10774_Quest.md
│   ├── 2502.11089_NSA.md
│   ├── 2510.26692_Kimi-Linear.md
│   └── 2602.11761_MiniCPM-SALA.md
├── 12.5_量化与低精度/
│   ├── 2410.19313_COAT.md
│   ├── 2411.10958_SageAttention2.md
│   ├── 2505.11594_SageAttention3.md
│   ├── 2506.08027_MXFP8_Pretraining_Recipes.md
│   ├── 2509.25149_NVFP4_Pretraining.md
│   ├── 2603.10444_FP4-MeanBias.md
│   └── 2609.04105_FP4_FlashAttention4.md
└── 12.6_RLHF与PostTraining系统/
    ├── 2409.19256_HybridFlow-verl.md
    ├── 2504.13914_Seed1.5-Thinking.md
    ├── 2505.24034_LlamaRL.md
    ├── 2505.24298_AReaL.md
    ├── 2506.06122_ROLL.md
    └── 2505.07291_INTELLECT-2.md
```

## 论文速览表

### 12.1 推理引擎与 Serving

| 论文 | 时间 | 一句话 |
|------|------|--------|
| SGLang (2312.07104) | 2023-12 | RadixAttention 前缀树复用 + 结构化输出状态机，吞吐最高 6.4 倍 |
| DistServe (2401.09670) | 2024-01 | PD 分离开山作：prefill/decode 拆卡部署，按 TTFT/TPOT SLO 各自优化 |
| Mooncake (2407.00079) | 2024-07 | Kimi 生产级架构：KVCache 中心化 + 分布式分层缓存 + 预测式拒载 |
| MiniMax-M1 (2506.13585) | 2025-06 | 首个开源大规模混合注意力推理模型，Lightning Attention 支撑百万级上下文 |
| Kimi K2 (2507.20534) | 2025-07 | 1T/32B MoE，MuonClip 零 loss spike 训练，agentic 后训练工厂 |

### 12.2 训练系统与并行

| 论文 | 时间 | 一句话 |
|------|------|--------|
| MegaScale (2402.15627) | 2024-02 | 字节万卡训练系统：全栈协同 + 深度可观测性，12288 卡 55.2% MFU |
| DeepSeek-V3 (2412.19437) | 2024-12 | FP8 训练 + DualPipe + aux-loss-free，2.788M H800 小时训完 671B |
| HierMoE (2508.09591) | 2025-08 | 拓扑感知 token 去重 + 专家交换，削减 MoE 训练 all-to-all 通信与负载不均 |
| FlashRecovery (2509.03047) | 2025-09 | 万卡训练秒级故障恢复：与规模无关的重启、单步内无 checkpoint 恢复 |
| SPES (2602.11543) | 2026-02 | 去中心化 MoE 预训练：专家按节点切分，16 张 48GB 卡互联网互联训练 |

### 12.3 MoE 系统

| 论文 | 时间 | 一句话 |
|------|------|--------|
| DeepSeek-V2 (2405.04434) | 2024-05 | DeepSeekMoE 细粒度专家 + MLA 出处：KV cache 降 93.3%，成本降 42.5% |
| OLMoE (2409.02060) | 2024-09 | 完全开源的 MoE 研究样本：1B 激活超 Llama2-13B，路由行为系统分析 |
| GLM-4.5 (2508.06471) | 2025-08 | 355B/32B MoE，专家模型迭代 + 多阶段 RL 的后训练工厂范式 |
| ConceptMoE (2601.21420) | 2026-01 | token 合并成"概念"做隐式算力分配：attention 降 R^2、KV cache 降 R |

### 12.4 长上下文与 KV Cache

| 论文 | 时间 | 一句话 |
|------|------|--------|
| Quest (2406.10774) | 2024-06 | 推理时 query 感知稀疏：页级 min/max 元数据选关键页，注意力提速约 7 倍 |
| NSA (2502.11089) | 2025-02 | 可训练稀疏注意力：压缩+选择+滑窗三分支，硬件对齐设计，效果不降 |
| Kimi Linear (2510.26692) | 2025-10 | 线性注意力首次全面超全注意力：KDA + MLA 混层，KV cache 降 75% |
| MiniCPM-SALA (2602.11761) | 2026-02 | 稀疏+线性 1:3 混层，持续训练转换省 75% 成本，单卡跑 1M 上下文 |

### 12.5 量化与低精度

| 论文 | 时间 | 一句话 |
|------|------|--------|
| COAT (2410.19313) | 2024-10 | FP8 训练压优化器状态与激活：端到端显存降 1.54 倍 |
| SageAttention2 (2411.10958) | 2024-11 | 注意力 INT4(per-thread)+FP8 量化，4090 上比 FA2 快约 3 倍 |
| SageAttention3 (2505.11594) | 2025-05 | Blackwell FP4 注意力 1038 TOPS（5090 上 FA 的 5 倍），首探 8bit 训练 |
| MXFP8 Recipes (2506.08027) | 2025-06 | NVIDIA MXFP8 预训练配方：微缩放格式对齐 BF16 曲线，验证至 8B/15T |
| NVFP4 Pretraining (2509.25149) | 2025-09 | 首次公开 FP4 大规模预训练：12B 参数 10T token 对齐 FP8 基线 |
| FP4 Mean Bias (2603.10444) | 2026-03 | FP4 训练不稳定的主因是秩一均值偏置，减均值即可廉价消除 |
| FP4 FlashAttention-4 (2609.04105) | 2026-09 | Blackwell FP4 注意力 kernel：Direct-P 设计，非因果前向达 BF16 的 2.13 倍 |

### 12.6 RLHF 与 Post-Training 系统

| 论文 | 时间 | 一句话 |
|------|------|--------|
| HybridFlow/verl (2409.19256) | 2024-09 | 单+多控制器混合架构 + 3D-HybridEngine 权重重分片，RLHF 框架事实标准 |
| Seed1.5-Thinking (2504.13914) | 2025-04 | 200B/20B 推理模型，混合奖励 + RL 基础设施的工业参照 |
| LlamaRL (2505.24034) | 2025-05 | Meta 全异步分布式 RL 框架，支撑 Llama 3 后训练，405B 上提速最高 10.7 倍 |
| AReaL (2505.24298) | 2025-05 | 全异步 RL：rollout/训练彻底解耦 + 陈旧度修正，吞吐 2.77 倍 |
| ROLL (2506.06122) | 2025-06 | 阿里 agentic RL 训练库：单控制器 + parallel worker + 环境/奖励 worker |
| INTELLECT-2 (2505.07291) | 2025-05 | 首个全球去中心化无许可 RL 训练（32B），TOPLOC 可验证 rollout |

## 阅读建议

1. **先读与已有章节强关联的论文**：每篇解读的第 5 节标出了对应的仓库章节，学完某章后读对应论文效果最好（如学完 5.6 读 DistServe/Mooncake，学完 6.11 读 HybridFlow/AReaL）。
2. **按演进线读**：同一主题内按时间顺序读，能看到清晰的技术演进，例如：
   - 稀疏注意力：Quest（推理时稀疏）→ NSA（训练时稀疏）→ Kimi Linear / SALA（线性/混合架构）
   - 低精度：COAT（FP8 显存）→ SageAttention2/3（注意力量化 INT4→FP4）→ MXFP8/NVFP4（FP8→FP4 预训练配方）
   - RL 系统：verl（同步优化）→ LlamaRL/AReaL/ROLL（异步化与 agentic 化）
   - 训练容错：MegaScale（可观测性）→ FlashRecovery（秒级恢复）
3. **带着三角取舍读**：每篇第 4 节都分析了"牺牲什么换什么"——这是把论文读成工程直觉的关键步骤。
4. **面试冲刺用**：每篇第 7 节是高频面试问题与一句话答案，可配合 11 章面试部分使用。

## 如何持续跟进新论文

- arXiv 每日列表：cs.DC（系统）、cs.LG（训练）、cs.CL（模型与推理）。
- HuggingFace Daily Papers：社区筛选的热度榜。
- 关注系统会议：OSDI、SOSP、NSDI、MLSys、EuroSys、ATC；以及大厂技术报告（DeepSeek、字节 Seed、月之暗面、智谱、阿里 Qwen、Meta、NVIDIA）。
- 判断一篇论文值不值得深读的标准：是否被主流框架（vLLM/SGLang/verl/Megatron）采纳，是否改变了某个关键指标的数量级。
