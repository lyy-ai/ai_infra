# 自研推理 Runtime 简历项目（C++ Skeleton）

## 项目概述

这个小项目把第五章的 Runtime 概念落成一个可编译的 C++ skeleton：ArenaAllocator、SizeClassPool、Graph Executor 估算、CUDA Graph 收益估算、Dynamic Batch Scheduler 模拟。它不是生产 Runtime，而是简历项目中用于讲清架构与优化路径的最小闭环。

## 构建与运行

```bash
cd /data/ai_infra/04_Runtime推理引擎/简历项目/RuntimeProject
cmake -S . -B build
cmake --build build -j
./build/runtime_demo
```

输出包含：

- ArenaAllocator 使用量。
- SizeClassPool 的 malloc 次数与复用命中。
- 无 CUDA Graph vs CUDA Graph 不同 replay 次数下的 per-iter 时间与加速比。
- Dynamic Batch 的 batch 数、平均等待与吞吐估算。

## 对应简历写法

**自研推理 Runtime：**

> 设计 C++ 推理 Runtime skeleton，覆盖 Session/Graph Executor/Tensor 生命周期、ArenaAllocator 与 SizeClassPool；支持异步执行抽象、内存池复用与动态 batch 调度，通过 Graph Executor 关键路径与 CUDA Graph replay 收益模型指导优化。

**CUDA Graph 加速服务：**

> 针对小 kernel 密集导致的 launch overhead，设计 CUDA Graph capture/replay 与 shape bucket fallback；用收益模型评估 capture 摊销，在线上推理中目标将 P99 延迟降低 22%（需以真实服务压测替换）。

**Relax Runtime 内存池与 CUDA Graph 优化：**

> 实现动态内存池减少显存碎片约 30%（目标值，需用 pool hit rate/fragmentation 验证），集成 CUDA Graph 固定 decode 子图，目标端到端推理延迟降低 40%；通过 IO info、内存地址固定与 eager fallback 保证正确性。

## 下一步

- 把估算模型替换为真实 kernel timing（CUPTI/Nsight Systems）。
- 将 SizeClassPool 换成 device memory pool，并接入 Session Reset。
- 给 Dynamic Batch Scheduler 加 token budget、KV block、aging 与抢占。
