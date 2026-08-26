# DistributedTrainProject：2×A100 DDP 训练与通信瓶颈实测

在真实双卡环境（2× NVIDIA A100-PCIE-40GB，PCIe 互联）上完成的分布式训练实验项目：单卡 baseline → DDP 多卡 → NCCL AllReduce 带宽实测 → 通信瓶颈根因分析。所有数字在 `results/` 中可复现。

## 项目价值

大多数简历写"4 卡加速比 3.8x"，但说不清为什么。本项目给出一个**反直觉的真实案例**：在 PCIe 互联的 A100 上，110M 模型的 DDP 反而比单卡慢（0.63x），并用实测数据完整定位根因。这比一个虚高的加速比数字更能体现分布式训练功底。

## 目录结构

```
DistributedTrainProject/
├── src/
│   ├── model.py           # SmallGPT（110M 参数，12 层 hidden=768）+ 合成数据
│   ├── train_single.py    # 单卡 baseline：吞吐/step 分解/峰值显存
│   ├── train_ddp.py       # DDP 双卡：每 rank 显存/step 分解（NCCL backend）
│   └── comm_bench.py      # NCCL AllReduce 1MB-512MB 带宽实测（algbw/busbw）
├── scripts/
│   └── analyze_results.py # 汇总生成 results/benchmark_summary.md
├── results/               # 真实测量数据
│   ├── single_gpu.json
│   ├── ddp_2gpu.json
│   ├── nccl_allreduce.json
│   └── benchmark_summary.md
├── Makefile               # make single / ddp / comm / report
└── README.md
```

## 运行

```bash
cd /data/ai_infra/06_分布式训练/简历项目/DistributedTrainProject
make single   # 单卡 baseline
make ddp      # torchrun --nproc_per_node=2 DDP 双卡
make comm     # NCCL AllReduce 带宽实测
make report   # 汇总报告
```

## 核心实测结论

| 指标 | 单卡 | DDP 2 卡 |
|------|------|----------|
| tokens/s | 18757 | 11881（**0.63x**） |
| step 时间 | 97.9ms | 331.2ms |
| backward+同步 | 53.1ms | 286.4ms |
| 峰值显存/rank | 3.46GB | 3.92GB |

根因：

1. **NCCL busbw 实测仅 1.5GB/s**（PCIe 慢速路径；NVLink 版 A100 为 600GB/s，差 ~400 倍）。
2. 梯度 0.44GB（FP32），每步 AllReduce 理论 ~293ms，远超计算（fwd+bwd ≈ 78ms）。
3. 结论：**通信/计算比决定 DDP 收益**；改进方向为 NVLink 机型、梯度累积、bucket 调优、FP16 梯度通信、更大模型。

详细数据见 [results/benchmark_summary.md](results/benchmark_summary.md)。
