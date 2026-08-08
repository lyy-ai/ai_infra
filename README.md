# AI Infra 课程总目录

从全景认知到综合实战的完整 AI Infra 学习路径，按编号顺序学习。

## 学习路线

| 顺序 | 目录 | 内容 |
|------|------|------|
| 01 | `01_第一阶段：全景认知与前置基础` | 第 1 章（1.1-1.11）：AI Infra 全景、岗位路线、编程/数学/Transformer/PyTorch/GPU/集合通信/工具链 |
| 02 | `02_CUDA编程与HPC高性能计算` | 第 2 章：CUDA 编程、性能优化、Reduce/GEMM/Attention 算子、HPC 实战 |
| 03 | `03_AI编译器` | 第 3 章：TVM 架构、编译前中后端、图优化 Pass、Laser 框架 |
| 04 | `04_Runtime推理引擎` | 第 4 章：Runtime 架构、内存池、CUDA Graph、多流并发、动态调度 |
| 05 | `05_LLM推理与Serving` | 第 5 章：PagedAttention、KV Cache、Continuous Batching、Speculative Decoding、多机 Serving |
| 06 | `06_分布式训练` | 第 6 章：DP/DDP/FSDP、ZeRO、TP/SP、PP、3D 并行、训练框架、通信优化 |
| 07 | `07_模型量化` | 第 7 章：量化基础、PTQ/QAT、INT8、Weight-Only INT4、W4A16/W4A8、KV Cache 量化、选型决策树 |
| 08 | `08_多平台适配` | 第 8 章：NVIDIA GPU、Jetson Orin、Ascend NPU、自研芯片、Benchmark 方法论 |
| 09 | `09_企业级工程体系` | 第 9 章：CI/CD、性能回归平台、Profiling 平台、质量保障、监控告警 |
| 10 | `10_第三阶段：综合项目实战` | MMBEV 端到端多平台部署优化（串起全部知识点） |
| 11 | `11_第四阶段：简历项目包装与面试冲刺` | STAR 简历写作、高频面试题、分岗位模拟面试、系统设计 |

## 建议学习顺序

```text
01 全景认知与前置基础
  ↓
02 CUDA 与算子优化 ──→ 03 AI 编译器 ──→ 04 Runtime ──→ 05 LLM 推理
  ↓                                                        ↓
06 分布式训练 ──→ 07 模型量化 ──→ 08 多平台适配 ──→ 09 企业级工程体系
  ↓
10 综合项目实战（MMBEV 端到端部署）
  ↓
11 简历包装与面试冲刺
```

## 核心思维

> 每个技术都是在 **计算 / 通信 / 显存** 不可能三角上做取舍：学习任何优化手段时，先问"牺牲了什么，换取了什么"。

## 使用说明

- 每个专题内按子目录编号（如 3.1 → 3.8）顺序学习。
- 每个子目录 = 讲义 md + 可运行代码 + images 演示图。
- 运行环境：除特别说明外，统一使用 `/data/liyangyang/qwen35_env/bin/python`。
