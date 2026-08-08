# 第六章：分布式训练

![分布式训练封面](images/distributed_training_cover.png)

本专题从分布式训练总论到具体并行策略与框架实战，覆盖数据并行、ZeRO、张量并行、序列并行、流水线并行、3D 并行、混合训练策略、训练框架（Megatron/DeepSpeed/FSDP）以及多卡通信优化。

## 目录结构

```
分布式训练/
├── 6.1_分布式训练总论/
│   ├── 6.1_分布式训练总论.md
│   └── memory_estimator.py
├── 6.2_数据并行/
│   ├── 6.2_数据并行.md
│   └── train_ddp.py
├── 6.3_ZeRO系列/
│   ├── 6.3_ZeRO系列.md
│   └── zero_memory_estimator.py
├── 6.4_张量并行与序列并行/
│   ├── 6.4_张量并行与序列并行.md
│   └── tp_column_row_sim.py
├── 6.5_流水线并行/
│   ├── 6.5_流水线并行.md
│   └── pipeline_bubble_calc.py
├── 6.6_3D并行与混合训练策略/
│   ├── 6.6_3D并行与混合训练策略.md
│   └── parallel_topo_design.py
├── 6.7_训练框架实战/
│   ├── 6.7_训练框架实战.md
│   ├── deepspeed_config_example.json
│   └── fsdp_example.py
├── 6.8_多卡训练通信优化/
│   ├── 6.8_多卡训练通信优化.md
│   └── ring_allreduce_sim.py
├── README.md
├── tools/
│   └── generate_distributed_training_diagrams.py
└── 简历项目/
    └── 简历项目.md
```

每个子专题目录下都有 `images/` 演示图；如需重新生成或改配色，运行：

```bash
source /data/liyangyang/qwen35_env/bin/activate
python /data/liyangyang/ai_infra/06_分布式训练/tools/generate_distributed_training_diagrams.py
```

## 运行环境

已在 `qwen35_env` 中安装/验证：

- `torch`
- `matplotlib==3.10.9`
- `numpy==2.2.6`

激活环境：

```bash
source /data/liyangyang/qwen35_env/bin/activate
```

## 运行示例

```bash
source /data/liyangyang/qwen35_env/bin/activate
cd /data/liyangyang/ai_infra/06_分布式训练

python 6.1_分布式训练总论/memory_estimator.py
python 6.2_数据并行/train_ddp.py
python 6.3_ZeRO系列/zero_memory_estimator.py
python 6.4_张量并行与序列并行/tp_column_row_sim.py
python 6.5_流水线并行/pipeline_bubble_calc.py
python 6.6_3D并行与混合训练策略/parallel_topo_design.py
cat 6.7_训练框架实战/deepspeed_config_example.json
# fsdp_example.py 需要多卡环境：
# python -m torchrun --nproc_per_node=2 6.7_训练框架实战/fsdp_example.py
python 6.8_多卡训练通信优化/ring_allreduce_sim.py

# 重新生成图片
python tools/generate_distributed_training_diagrams.py
```

## 课程目标

1. 理解分布式训练的驱动力：单卡显存和算力限制。
2. 估算训练显存：参数 + 梯度 + 优化器状态 + 激活值。
3. 掌握数据并行：DP、DDP、FSDP 的原理与实现。
4. 理解 ZeRO 系列：ZeRO-1/2/3/Offload 的切分对象与通信代价。
5. 理解张量并行与序列并行：TP 切分、SP 激活分片、GQA/MQA 策略。
6. 理解流水线并行：GPipe、1F1B、气泡率分析。
7. 设计 3D 并行拓扑：TP + PP + DP 的组合。
8. 掌握混合训练策略：混合精度、梯度累积、Activation Checkpointing。
9. 使用 Megatron-LM、DeepSpeed、PyTorch FSDP 进行训练。
10. 优化多卡通信：NCCL、Ring AllReduce、通信与计算重叠、梯度压缩。
11. 能把分布式训练项目写成有分层、有指标、有证据的简历 bullet。
