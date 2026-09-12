# 第十五章：Agent 系统与工程

Agent（智能体）是 2024 年后 AI 落地的核心形态，也是 AI Infra 需求的新主体。本章从原理到工程完整覆盖：Agent 核心机制、Harness（运行时外壳）架构解剖、主流框架与协议生态、Coding Agent 专题（Claude Code/OpenCode/Aider/Cursor 原理）、应用场景与评估、以及 Agent 与 AI Infra 的深度结合。

## 目录结构

```
15_Agent系统与工程/
├── 15.1_Agent原理与核心机制/          # Agent=模型+循环；ReAct等四模式；工具调用/记忆；三大工程问题
│   ├── 15.1_Agent原理与核心机制.md
│   └── react_loop_simulator.py        # ReAct 循环的上下文增长与管理策略模拟
├── 15.2_Agent_Harness架构解剖/         # Harness 六组件：循环/工具/上下文/权限/状态/子代理与协议
│   └── 15.2_Agent_Harness架构解剖.md
├── 15.3_主流Agent框架与生态/           # LangGraph/CrewAI/AutoGen/官方SDK/Dify；MCP/A2A 协议
│   └── 15.3_主流Agent框架与生态.md
├── 15.4_Coding_Agent专题/             # Claude Code/OpenCode/Aider/Cursor/云端派原理与设计模式
│   └── 15.4_Coding_Agent专题.md
├── 15.5_Agent应用场景与评估/           # 六大场景负载特征；结果/过程/成本三维评估；失败模式
│   └── 15.5_Agent应用场景与评估.md
└── 15.6_Agent与AI_Infra的结合/         # Agent 负载特征/前缀复用/Agentic RL/沙箱基础设施/成本模型
    ├── 15.6_Agent与AI_Infra的结合.md
    └── agent_cost_model.py            # Agent 任务成本模型（前缀命中率杠杆）
```

## 各节速览

| 节 | 核心内容 | 一句话收获 |
|----|---------|-----------|
| 15.1 原理与核心机制 | Agent 循环、四种模式、工具调用、三层记忆、上下文爆炸/错误累积/循环失控 | Agent = 模型 + 循环；可靠性是第一工程问题 |
| 15.2 Harness 架构解剖 | 循环驱动、工具系统、上下文管理、权限沙箱、状态持久化、子代理与 MCP/A2A | 同一模型配不同 Harness，表现天差地别 |
| 15.3 框架与生态 | LangGraph/CrewAI/AutoGen/OpenAI SDK/smolagents/Dify；MCP 详解；选型决策树 | 框架越来越薄，先学原理再选框架 |
| 15.4 Coding Agent 专题 | Claude Code 组件解剖、OpenCode 开源对照、Aider 编辑格式、IDE 派 vs 终端派 vs 云端派 | 强模型+简单工具+上下文工程 > 复杂编排 |
| 15.5 场景与评估 | 编程/深研/数据/客服/GUI/具身；SWE-bench 等；七大失败模式；上线检查清单 | 评估看成功率-成本帕累托前沿 |
| 15.6 与 AI Infra 结合 | Agent 负载画像、prefix cache 生命线、Agentic RL 环境工程、沙箱 infra、成本模型 | Agent 时代，prefix 命中率就是利润率 |

## 运行示例

```bash
source /data/qwen35_env/bin/activate
cd /data/ai_infra/15_Agent系统与工程

python 15.1_Agent原理与核心机制/react_loop_simulator.py   # 上下文增长与四种管理策略对比
python 15.6_Agent与AI_Infra的结合/agent_cost_model.py     # Agent 任务成本与前缀命中率杠杆
```

## 阅读建议

1. **按顺序读**：15.1（原理）→ 15.2（组件）→ 15.3/15.4（生态与代表实现）→ 15.5（落地）→ 15.6（回到本仓库主线）。
2. **15.4 与 15.2 对照读**：Claude Code/OpenCode 的每个设计都能在 15.2 的组件模型里找到位置——这是"理论→实现"的对照练习。
3. **15.6 是本章对本仓库读者最重要的一节**：Agent 不是独立于 AI Infra 的新领域，而是 serving/训练/调度所有既有知识的新负载形态。
4. 面试相关：Agent 负载特征、MCP、Coding Agent 原理已是 2025 后大模型公司面试的高频题（可配合 14.5 章使用）。
