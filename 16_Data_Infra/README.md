# 第十六章：Data Infra（大模型数据基础设施）

Data Infra 是大模型竞争的隐形主战场：模型架构趋同、算力可以购买，而数据配方（什么数据、什么质量、什么配比）是各家的核心差异化。本章覆盖完整版图：预训练数据处理、开源数据集与配比、后训练与合成数据、向量数据与 RAG、数据飞轮与治理。

## 目录结构

```
16_Data_Infra/
├── 16.1_Data_Infra全景与数据管线/       # 为什么数据是主战场；生命周期全景；数量/质量/成本三角；岗位画像
│   └── 16.1_Data_Infra全景与数据管线.md
├── 16.2_预训练数据处理/                # 文本提取/去重(MinHash+LSH)/质量过滤/去污/PII；五大开源框架
│   ├── 16.2_预训练数据处理.md
│   └── minhash_dedup_demo.py          # MinHash+LSH 模糊去重完整实现
├── 16.3_开源数据集与数据配比/           # FineWeb/DCLM/RedPajama/Dolma/Nemotron-CC；DoReMi/RegMix 配比方法论
│   ├── 16.3_开源数据集与数据配比.md
│   └── data_mixing_simulator.py       # 配比搜索教学模拟（单域饱和现象 + 回归寻优）
├── 16.4_后训练数据与合成数据/           # SFT/偏好/RLVR 三形态；蒸馏/Magpie/persona/拒绝采样；模型坍缩
│   └── 16.4_后训练数据与合成数据.md
├── 16.5_向量数据与RAG基础设施/          # chunking/嵌入/混合召回/rerank；Milvus/FAISS/Qdrant/pgvector 选型
│   └── 16.5_向量数据与RAG基础设施.md
└── 16.6_数据飞轮与数据治理/             # 线上日志飞轮；DVC/lakeFS 版本管理；版权/隐私合规工程化
    └── 16.6_数据飞轮与数据治理.md
```

## 各节速览

| 节 | 核心内容 | 一句话收获 |
|----|---------|-----------|
| 16.1 全景 | 数据生命周期六环节、放大器效应、岗位能力画像 | 模型大战的本质是数据大战 |
| 16.2 预训练处理 | 去重（MinHash+LSH）、规则+分类器双过滤、去污、PII；datatrove/NeMo Curator/Data-Juicer/Dolma/DCLM 框架对比 | 去重是收益最大的单一环节；过滤阈值是配比问题 |
| 16.3 数据集与配比 | 第一梯队开源数据集档案、三类配比方法、超训与数据墙 | FineWeb-Edu 快速起步，DCLM 做研究，配比要用小模型实验搜索 |
| 16.4 后训练与合成 | SFT 质量>>数量、偏好对设计、RLVR 难度分布、五种合成方法、坍缩对策 | 合成数据每步过滤都要有对照实验 |
| 16.5 向量与 RAG | chunking 四策略、混合召回、向量库选型、三分归因法 | 没有评测集的 RAG 调优是盲人摸象 |
| 16.6 飞轮与治理 | 日志→筛选→训练闭环、版本与血缘、合规管线化 | 飞轮每圈必须有可测量改善 |

## 开源可复用框架清单

| 层 | 项目 | 用途 |
|----|------|------|
| LLM 数据框架 | huggingface/datatrove | FineWeb 生产工艺，学原理首选 |
| | NVIDIA/NeMo-Curator | GPU 加速处理、语义去重 |
| | modelscope/data-juicer | 算子最全，多模态/后训练数据 |
| 数据集 | FineWeb/FineWeb-Edu、DCLM、RedPajama、Dolma、Nemotron-CC | 直接可训练的第一梯队语料 |
| 合成数据 | argilla-io/distilabel、Magpie 管线 | 合成数据生产线 |
| 向量检索 | FAISS、Milvus、Qdrant、pgvector、LanceDB | RAG 检索层 |
| 版本管理 | DVC、lakeFS、HuggingFace Datasets | 数据的 Git |

## 运行示例

```bash
source /data/qwen35_env/bin/activate
cd /data/ai_infra/16_Data_Infra

python 16.2_预训练数据处理/minhash_dedup_demo.py      # 去重三件套：MinHash签名/LSH分桶/Jaccard验证
python 16.3_开源数据集与数据配比/data_mixing_simulator.py  # 配比现象与 RegMix 式搜索
```

## 阅读建议

1. 按顺序读，16.1 的三角取舍（数量/质量/成本）贯穿全章。
2. 两个脚本必跑：去重和配比是 Data Infra 面试的必考题，跑过一遍胜过读十遍。
3. 与 12 章联动：DeepSeek-V3/Llama-3 报告中的数据章节是本章知识的工业界印证。
4. 与 15 章联动：agentic 轨迹数据（16.4 节）是 Agent 与 Data Infra 的交汇点，也是当前最热的方向。
