# 第七章：模型量化

本专题覆盖大模型量化的完整知识体系：从量化基础到 PTQ/QAT、INT8 kernel、Weight-Only INT4、W4A16/W4A8、KV Cache 量化，最后是量化选型决策树与 Qwen 实战项目。

## 目录结构

```
07_模型量化/
├── 7.1_量化基础/                # 对称/非对称、per-tensor vs per-channel、校准方法
├── 7.2_PTQ/                     # 训练后量化：校准策略、混合精度
├── 7.3_QAT/                     # 量化感知训练：STE、fake quant、训练流程
├── 7.4_INT8/                    # INT8 GEMM、Tensor Core、dequant 开销
├── 7.5_Weight-OnlyINT4/         # AWQ/GPTQ、INT4 kernel 效率
├── 7.6_W4A16/                   # 4-bit 权重 + 16-bit 激活（MLC-LLM）
├── 7.7_W4A8/                    # W4A8 / GPTQ-Int4（MLC-LLM 导出与推理）
├── 7.8_KVCache量化/              # KV Cache 量化、长上下文、KIVI
├── 7.9_量化决策树/               # 量化方案选型决策树与权衡分析
└── 7.10_QwenQuantizationProject/ # Qwen3.5-9B 多精度部署实战项目
```

每个子目录 = 讲义 md（`8.x_*.md`）+ 可运行代码示例。

## 学习建议

1. 按 8.1 → 8.9 顺序学习，最后做 8.10 实战项目。
2. 量化前先用 1.9 节的 roofline 判断模型是 memory-bound 还是 compute-bound。
3. 学完每节回答取舍问题：这个量化方案牺牲了什么精度，换取了什么显存/速度？

## 运行示例

```bash
source /data/liyangyang/qwen35_env/bin/activate
cd /data/liyangyang/ai_infra/07_模型量化

python 7.1_量化基础/pytorch_ptq_demo.py
python 7.2_PTQ/int8_ptq_manual.py
python 7.3_QAT/qat_ste.py
python 7.4_INT8/int8_gemm.py
python 7.8_KVCache量化/kvcache_memory_analysis.py
python 7.9_量化决策树/quantization_decision_tree.py
```
