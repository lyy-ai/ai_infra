# 第十三章：开源社区与生态

AI Infra 领域关键开源项目的全景地图与参与指南。讲义之外的知识主要在开源社区里迭代——本章回答三个问题：**格局是什么（有哪些项目、各自生态位）、怎么选（场景选型）、怎么用（读源码与参与社区）**。

## 目录结构

```
13_开源社区与生态/
├── 13.1_推理与Serving引擎/      # vLLM / SGLang / TensorRT-LLM / LMDeploy / Mooncake / LMCache / llama.cpp
│   └── 13.1_推理与Serving引擎.md
├── 13.2_训练框架/               # Megatron-LM / DeepSpeed / FSDP2 / torchtitan / NCCL / DeepEP / DeepGEMM
│   └── 13.2_训练框架.md
├── 13.3_RL与后训练框架/          # verl / OpenRLHF / slime / NeMo-RL / TRL / ROLL / AReaL
│   └── 13.3_RL与后训练框架.md
├── 13.4_Kernel与底层库/          # FlashAttention / FlashInfer / Triton / CUTLASS / DeepGEMM / DeepEP
│   └── 13.4_Kernel与底层库.md
└── 13.5_源码阅读与社区参与实战/   # 四步读码法、社区参与阶梯、简历转化
    └── 13.5_源码阅读与社区参与实战.md
```

## 生态地图（一张图看懂分层）

```text
应用层   │ vLLM  SGLang  TensorRT-LLM  LMDeploy  llama.cpp   （推理引擎，13.1）
         │ verl  OpenRLHF  slime  TRL  ROLL  AReaL           （后训练，13.3）
框架层   │ Megatron-LM  DeepSpeed  FSDP2  torchtitan         （训练框架，13.2）
算子层   │ FlashAttention  FlashInfer  Triton  CUTLASS       （kernel，13.4）
通信层   │ NCCL  DeepEP  NVSHMEM                             （通信，13.2/13.4）
存储/缓存│ Mooncake  LMCache                                 （KV 系统，13.1）
```

阅读建议：每节都给了"读源码从哪个文件/目录开始"的入口，配合本仓库对应章节（讲义）与第 12 章（论文）三层联动：**讲义学原理 → 论文读思想 → 源码看实现**。

## 学习路径

1. 先按 13.1-13.4 建立生态认知（每个项目知道"是什么、解决什么、选它的理由"）。
2. 选一个与自己目标岗位最匹配的项目深入（推理岗 → vLLM/SGLang；训练岗 → Megatron/verl；算子岗 → FlashAttention/Triton/CUTLASS）。
3. 按 13.5 的四步法读源码，产出源码分析笔记（可公开）。
4. 从 Level 1-2（高质量 issue、文档/测试 PR）开始参与社区，把成果写进简历（对接第 11 章）。

## 与前面章节的关系

- 第 2-7 章的每个技术点，本章都标注了对应的开源实现项目——学完原理立即去看实现。
- 第 12 章的论文解读与本章项目一一对应（如 Mooncake/LMCache ↔ 12.1，verl/AReaL/ROLL ↔ 12.6）。
- 第 11 章的简历包装以本章的社区参与成果为素材来源。
