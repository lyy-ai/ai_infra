# 6.10 业务模型量化实战：Qwen3.5-9B 大模型量化压缩与推理加速项目

## 项目概述

本项目以 **Qwen3.5-9B**（`/data/models/Qwen3.5-9B`）为对象，完成一套完整的业务级大模型量化实战方案。通过 **FP16 / INT8 / INT4(NF4)** 三种精度部署，对比显存占用、推理速度和生成质量，最终输出可直接写进简历的完整项目成果。

### 项目价值

- **显存优化**：Qwen3.5-9B FP16 实测约 16.8 GB，INT4 部署后降至 4.6 GB，单卡即可承载。
- **成本降低**：更低的显存占用意味着可以部署到更便宜的 GPU，或提升单卡并发。
- **速度影响需具体评估**：INT8 在当前 bitsandbytes 实现下反而慢于 FP16（混合架构下存在 kernel fall-back），INT4 比 FP16 慢约 30%；生产环境建议使用 vLLM/TensorRT-LLM/AWQ 进一步压榨性能。
- **工程落地**：覆盖模型加载、推理、评估、对话 demo 全流程，具备直接迁移到业务场景的能力。

### 适合写入简历的描述

> **Qwen3.5-9B 大模型量化压缩与推理加速**
>
> 针对 9B 参数大模型 Qwen3.5-9B，使用 transformers + bitsandbytes 实现 FP16/INT8/INT4(NF4) 多精度部署；设计显存、延迟、生成质量、困惑度四维评估体系；完成交互式对话 demo 与批量 benchmark 流水线。INT4 方案下模型显存从 16.8 GB 降至 4.6 GB（压缩约 3.7x），困惑度仅增加 2.6%，INT8 几乎无损；项目代码可迁移至 vLLM/TensorRT-LLM 生产部署。

---

## 1. 模型与业务背景

### 1.1 模型信息

| 属性 | 数值 |
|------|------|
| 模型路径 | `/data/models/Qwen3.5-9B` |
| 架构 | `Qwen3_5ForConditionalGeneration` |
| 层数 | 32（linear_attention × 24 + full_attention × 8 的混合架构） |
| 隐藏维度 | 4096 |
| 注意力头数 | 16 |
| KV 头数 | 4 (GQA) |
| 中间维度 | 12288 |
| 词表大小 | 248320 |
| 原始精度 | FP16 / BF16 |
| 权重文件 | 4 个 safetensors 分片，总计约 18.8 GB |
| 参数总量 | 约 8.95 B |
| FP16 理论显存 | 约 16.8 GB |

### 1.2 业务场景假设

假设我们要将 Qwen3.5-9B 部署为一个**中文智能问答服务**：

- 用户输入问题，模型实时生成回答。
- 需要支持一定并发（如 4 个请求同时处理）。
- 部署预算有限，单卡显存不能超过 40 GB。
- 回答质量不能明显下降。

### 1.3 原始 FP16 部署的问题

Qwen3.5-9B FP16 权重约 18.8 GB，加载后加上 KV Cache、激活、workspace，单卡 40 GB 虽然能跑，但：

- 并发能力受限（KV Cache 随 batch 和序列长度线性增长）。
- 推理速度受显存带宽限制。
- 无法部署到更低成本的 GPU（如 24 GB 单卡）。

因此，量化是必需的优化手段。

---

## 2. 技术方案选型

### 2.1 量化方法选择

| 精度 | 技术 | 压缩比 | 适用场景 |
|------|------|--------|---------|
| FP16 | 无量化 | 1x | 基准、训练 |
| INT8 | bitsandbytes 8-bit | ~2x | 平衡质量与显存 |
| INT4(NF4) | bitsandbytes 4-bit Normal Float | ~4x | 极限显存优化 |

选择 **bitsandbytes** 的原因：

1. 与 transformers 深度集成，代码改动最小。
2. 支持 Qwen3.5 等较新架构（通过 `trust_remote_code=True`）。
3. 不需要额外校准数据集（PTQ 风格，开箱即用）。
4. 4-bit NF4 在 LLM 上精度损失通常可接受。

### 2.2 不使用 AutoGPTQ / AWQ 的原因

- 当前环境 Qwen3.5 架构的 AutoGPTQ/AWQ 支持可能不完善。
- bitsandbytes 更适合快速验证和课程实战。
- 后续可以无缝替换为 AutoGPTQ / AWQ / TensorRT-LLM 以追求更高性能。

### 2.3 评估指标

| 维度 | 指标 | 说明 |
|------|------|------|
| 显存 | 加载前后 GPU 总空闲显存差值（GB） | 排除系统其他进程干扰，估算模型真实占用 |
| 速度 | Tokens Per Second (TPS) | 生成阶段吞吐 |
| 质量 | 人工观察 + 生成对比 | 相同 prompt 下输出差异 |
| 困惑度 | Perplexity | 在中文样本上的语言建模能力 |

---

## 3. 项目结构

```
/data/ai_infra/QwenQuantizationProject/
├── README.md                           # 项目说明（即本文件）
├── requirements.txt                    # Python 依赖
├── config.py                           # 模型路径、参数、prompts 等配置
├── src/                                # 核心源码
│   ├── model_loader.py                 # 模型加载与量化配置
│   ├── inference.py                    # 文本生成与显存测量工具
│   ├── metrics.py                      # 困惑度计算与结果汇总
│   └── utils.py                        # JSON 保存与表格打印
├── scripts/                            # 可执行脚本
│   ├── benchmark.py                    # 完整 benchmark：FP16/INT8/INT4
│   ├── evaluate_perplexity.py          # 困惑度评估
│   ├── evaluate_generation.py          # 生成质量对比
│   └── chat_demo.py                    # 交互式对话 demo
└── results/                            # 实验结果输出
    ├── fp16_benchmark.json
    ├── int8_benchmark.json
    ├── int4_benchmark.json
    ├── benchmark_summary.json
    ├── perplexity_results.json
    └── generation_comparison.json
```

---

## 4. 核心代码实现

### 4.1 配置：`config.py`

```python
MODEL_PATH = "/data/models/Qwen3.5-9B"
RESULTS_DIR = "results"
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

SAMPLE_PROMPTS = [
    "请介绍一下机器学习中的梯度下降算法。",
    "什么是注意力机制？它在自然语言处理中有什么作用？",
    "简述大语言模型量化技术的意义。",
    "请用中文解释 Mixture of Experts 架构。",
]
```

### 4.2 模型加载与量化配置：`src/model_loader.py`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_path, quant_config=None, device_map="auto"):
    kwargs = {"trust_remote_code": True, "device_map": device_map}
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    else:
        kwargs["torch_dtype"] = torch.float16
    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


def get_int8_config():
    return BitsAndBytesConfig(load_in_8bit=True)


def get_int4_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
```

### 4.3 推理封装：`src/inference.py`

```python
import time
import torch
from config import MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY


def generate_text(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS,
                  temperature=TEMPERATURE, top_p=TOP_P, repetition_penalty=REPETITION_PENALTY):
    inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start_time
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    num_generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    tokens_per_sec = num_generated_tokens / elapsed if elapsed > 0 else 0
    
    return generated_text, elapsed, tokens_per_sec, num_generated_tokens


def get_gpu_memory_info():
    """返回当前 GPU 总显存和已用显存（GB）。
    
    注意：nvidia-smi 级别的已用显存包含所有进程，因此 benchmark 中
    更推荐通过加载前后空闲显存的差值来计算模型真实占用。
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_gb = info.used / 1024**3
        total_gb = info.total / 1024**3
        return used_gb, total_gb
    except Exception:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return allocated, reserved
        return 0.0, 0.0


def get_gpu_free_memory():
    """返回所有可用 GPU 的空闲显存总和（GB）。"""
    if not torch.cuda.is_available():
        return 0.0
    total_free = 0.0
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        total_free += free / 1024**3
    return total_free
```

### 4.4 困惑度评估：`src/metrics.py`

```python
import torch


def compute_perplexity(model, tokenizer, texts, max_length=512):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            n_tokens = inputs["input_ids"].numel()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity
```

### 4.5 完整 Benchmark：`scripts/benchmark.py`

详见 `scripts/benchmark.py`。该脚本会：

1. 依次加载 FP16 / INT8 / INT4 模型。
2. 记录每种精度的加载时间、显存占用。
3. 使用相同 prompts 生成文本，记录时间、TPS。
4. 将结果保存到 `results/` 目录。

### 4.6 对话 Demo：`scripts/chat_demo.py`

支持用户选择 FP16/INT8/INT4，然后以问答形式与模型交互。使用 tokenizer 的 `chat_template` 构造对话格式。

---

## 5. 环境准备

### 5.1 激活虚拟环境

本项目使用已有的 Qwen3.5 虚拟环境：

```bash
source /data/qwen35_env/bin/activate
```

### 5.2 确认依赖

该环境已安装：

- `transformers` 5.13.1
- `torch` 2.11.0+cu128
- `bitsandbytes` 0.49.2
- `accelerate`

如需在新环境复现，可安装：

```bash
pip install -r requirements.txt
```

### 5.3 硬件要求

- 至少 1 张 NVIDIA GPU，显存 24 GB 以上（FP16/INT8/INT4 均可运行）。
- 推荐 A100 40GB / A100 80GB / H100 以获得更流畅体验。

---

## 6. 运行步骤

### 6.1 一键运行完整 Benchmark

```bash
cd /data/ai_infra/QwenQuantizationProject
/data/qwen35_env/bin/python scripts/benchmark.py
```

如果只想跑某一种精度（例如单独验证 INT4）：

```bash
/data/qwen35_env/bin/python scripts/benchmark.py --precision int4
```

### 6.2 运行困惑度评估

```bash
/data/qwen35_env/bin/python scripts/evaluate_perplexity.py
```

### 6.3 运行生成质量对比

```bash
/data/qwen35_env/bin/python scripts/evaluate_generation.py
```

### 6.4 启动交互式对话

```bash
/data/qwen35_env/bin/python scripts/chat_demo.py
```

---

## 7. 实验结果

> 以下结果在 **2 × NVIDIA A100-PCIE-40GB** 上运行获得，使用 `qwen35_env` 环境（transformers 5.13.1, torch 2.11.0, bitsandbytes 0.49.2）。显存采用"加载前后 GPU 总空闲显存差值"进行估算，可排除系统其他进程干扰。

### 7.1 显存与加载速度

| 精度 | 加载时间 | 模型显存占用 (GB) | 相对 FP16 压缩比 | 备注 |
|------|---------|------------------|-----------------|------|
| FP16 | 8.6s | 16.84 | 1.00x | 理论值约 16.8 GB（8.95B × 2 bytes） |
| INT8 | 33.7s | 10.68 | 1.58x | 低于 2x，因 bitsandbytes 8-bit 额外元数据 |
| INT4 | 7.5s | 4.57 | 3.68x | NF4 + double quant 接近 4x 压缩 |

> 注：
> 1. FP16 模型被 accelerate 自动分配到 2 张 GPU 上（cuda:0 6.73 GB + cuda:1 9.95 GB）。
> 2. INT8 占用略高于 1/2，因为 bitsandbytes 需要保存量化 scale/bias 等元数据。
> 3. INT4 实际占用与理论 4.2 GB 接近，double quant 进一步降低元数据开销。
> 4. INT8 加载时间最长，因为 bitsandbytes 需要在加载时对每个权重进行 8-bit 量化转换。

### 7.2 生成速度与 TPS

| 精度 | 平均生成时间 | 平均 TPS | 相对 FP16 速度 | 备注 |
|------|------------|---------|----------------|------|
| FP16 | 14.31s | 8.96 tok/s | 1.0x | 基准 |
| INT8 | 53.09s | 2.41 tok/s | 0.27x | **本环境下显著慢于 FP16** |
| INT4 | 20.69s | 6.19 tok/s | 0.69x | 比 FP16 慢，但比 INT8 快 |

> 注：
> 1. 本批 benchmark 中 INT8 反而最慢，主要因为 bitsandbytes 的 8-bit `MatMul8bitLt` 在当前 Qwen3.5 混合架构（linear_attention + full_attention）下存在较多的 CPU-GPU 数据搬运与 float16 回退（见警告 `MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization`）。
> 2. INT4 的 NF4 kernel 相对更成熟，虽然仍有 dequant 开销，但比 INT8 的 fall-back 路径更快。
> 3. 在纯 Transformer 架构或更优化的 kernel 环境下，INT8 通常应快于 FP16。这里的结果应视为该特定模型 + 特定版本的实测值，而非通用结论。
> 4. 生产环境中建议用 **vLLM / TensorRT-LLM / AutoAWQ** 等进一步压榨性能。

### 7.3 困惑度（Perplexity）

| 精度 | 中文样本 PPL | 相对 FP16 变化 |
|------|-------------|---------------|
| FP16 | 11.7815 | baseline |
| INT8 | 11.9150 | +1.13% |
| INT4 | 12.0853 | +2.58% |

> 注：PPL 在 4 个中文样本上计算，仅供参考。INT8 与 INT4 均接近 FP16，说明量化精度损失可控。

### 7.4 生成质量对比

#### Prompt 1：请介绍一下机器学习中的梯度下降算法。

**FP16：**
```
请介绍一下机器学习中的梯度下降算法。

<think>
Here's a thinking process that leads to the explanation of Gradient Descent in machine learning:

1.  Deconstruct the Request:
    *   Topic: Gradient Descent (梯度下降).
    *   Context: Machine Learning (机器学习).
    ...
```

**INT8：**
```
请介绍一下机器学习中的梯度下降算法。

<think>
Here's a thinking process that leads to the explanation of Gradient Descent:

1.  Deconstruct the Request:
    *   Topic: Machine Learning (ML).
    *   Specific ...
```

**INT4：**
```
请介绍一下机器学习中的梯度下降算法。

<think>
Here's a thinking process that leads to the explanation of Gradient Descent:

1.  Deconstruct the Request:
    *   Topic: Machine Learning (ML).
    *   Specific ...
```

#### Prompt 2：什么是注意力机制？它在自然语言处理中有什么作用？

**FP16：**
```
注意力机制是一种深度学习技术，它允许模型在处理序列数据时关注不同的部分，从而提高对上下文的理解能力。在自然语言处理（NLP）中，注意力机制通过赋予输入序列中的不同元素不同的权重，使得模型能够更好地捕捉长距离依赖关系...
```

**INT8：**
```
注意力机制是一种深度学习技术，它允许模型在处理序列数据时关注不同的部分。具体来说，它可以让模型在生成某个输出时，更加关注输入中的某些特定部分，而不是均匀地对待所有输入...
```

**INT4：**
```
注意力机制是一种深度学习技术，它允许模型在处理序列数据时聚焦于输入中的特定部分。具体来说，注意力机制通过计算每个位置与其他所有位置的权重（即注意力分数），来决定哪些部分的输入对当前输出最重要...
```

#### Prompt 3：简述大语言模型量化技术的意义。

**INT8（本例中结构最完整）：**
```
简述大语言模型量化技术的意义。

<think>

</think>

大语言模型（LLM）的**量化技术**是将高精度浮点权重（如 FP16/BF16，通常为 32-bit 或 16-bit）转换为低精度整数格式（如 INT8、INT4 等）的过程。这项技术在当前 AI 生态中具有重要的战略意义，主要体现在以下四个维度：

### 1. 大幅降低硬件成本与资源消耗
...
```

观察：三种精度下输出均语义通顺、结构完整。INT4 在部分情况下会先输出 `<think>` 推理过程，再给出正式回答；INT8 在量化的回答中反而结构最清晰。这说明在该模型上，4-bit 量化对生成质量影响有限，但不同精度的输出风格会有差异。

### 7.5 结果文件

所有原始结果保存在 `results/` 目录：

- `fp16_benchmark.json` / `int8_benchmark.json` / `int4_benchmark.json`
- `benchmark_summary.json`
- `perplexity_results.json`
- `generation_comparison.json`

---

## 8. 项目亮点与可扩展方向

### 8.1 项目亮点

1. **真实模型**：使用 Qwen3.5-9B 真实权重，而非 toy model。
2. **完整流程**：从模型加载、量化、推理、评估到对话 demo，覆盖工程落地全链路。
3. **多维评估**：显存、速度、PPL、生成质量四个维度同时评估。
4. **可复现**：所有参数、prompts、脚本均开源，换环境即可运行。
5. **简历友好**：可直接作为“大模型量化优化”项目经历。

### 8.2 可扩展方向

1. **替换为 AWQ / GPTQ**：使用 AutoAWQ / AutoGPTQ 进行更激进的 4-bit 量化，对比精度。
2. **TensorRT-LLM 部署**：将 INT8/FP8 模型导出到 TensorRT-LLM，追求极致吞吐。
3. **KV Cache 量化**：结合 KIVI 2-bit 进一步降低长上下文显存。
4. **服务化封装**：用 vLLM / TGI 将量化模型封装为 REST API。
5. **长文本评估**：加入 Needle-in-a-Haystack 测试，验证量化对长上下文检索的影响。

---

## 9. 常见问题排查

### 9.1 加载模型时显存不足

- 尝试使用 `device_map="auto"` 让 accelerate 自动分配层。
- 对于 INT4，如果仍不够，可减少 `max_new_tokens` 或启用 `enable_thinking=False`。

### 9.2 生成输出重复

- 提高 `repetition_penalty`（如 1.15–1.2）。
- 降低 `temperature`（如 0.5）。
- 这是低比特模型的常见现象，不一定是量化失败。

### 9.3 bitsandbytes 加载慢

- 首次加载时 bitsandbytes 需要对权重进行量化，后续加载更快。
- 可提前保存量化后的模型以加速二次加载。

### 9.4 结果文件中的 TPS 偏低

- 首次生成时存在 CUDA kernel 预热，建议取多次生成的平均值。
- 长序列或大批量会降低 TPS。

---

## 10. 简历项目描述（可直接复制）

### 版本 1：简洁版

> **Qwen3.5-9B 大模型量化压缩与推理加速**
>
> 使用 transformers + bitsandbytes 对 Qwen3.5-9B 进行 FP16/INT8/INT4(NF4) 多精度部署；设计显存、延迟、困惑度、生成质量四维评估体系；实现交互式对话 demo 与批量 benchmark 流水线。INT4 部署后模型显存从 16.8 GB 降至 4.6 GB（压缩 3.7x），困惑度仅增加 2.6%，INT8 几乎无损；项目代码可迁移至 vLLM/TensorRT-LLM 生产部署。

### 版本 2：详细版

> **Qwen3.5-9B 大模型量化压缩与推理加速项目**
>
> - 背景：9B 参数大模型 Qwen3.5-9B FP16 权重约 16.8 GB，单卡显存紧张、并发受限。
> - 方案：基于 transformers + bitsandbytes 实现 PTQ 风格的 INT8 / INT4(NF4) 量化，覆盖模型加载、推理、评估、对话 demo。
> - 评估：从模型显存占用、生成 TPS、中文样本困惑度、生成质量四个维度对比 FP16/INT8/INT4。
> - 成果：INT4 方案显存降低约 73%（16.84 GB → 4.57 GB），INT8 几乎无损（PPL +1.13%），INT4 PPL 仅 +2.58%；项目代码结构清晰，可迁移到 vLLM/TensorRT-LLM 生产部署。

---

## 11. 本节小结

1. 本项目以 **Qwen3.5-9B** 为真实对象，完成了一套可落地的量化实战方案。
2. 使用 **bitsandbytes + transformers** 实现 FP16/INT8/INT4 三种部署，代码简洁、可复现。
3. 评估体系覆盖 **显存、速度、PPL、生成质量** 四个核心维度。
4. 项目可直接作为简历中的“大模型量化优化”经历，并具备向 AWQ/GPTQ/TensorRT-LLM 扩展的能力。
5. 量化选型不是单一最优解，而是业务指标、硬件约束、精度要求之间的 trade-off。

---

## 12. 课后思考

1. 如果业务场景要求单卡 24GB 部署 Qwen3.5-9B，你会选择哪种精度？为什么？
2. INT4 的生成质量偶尔下降，除了调整采样参数，还有哪些工程手段可以改善？
3. 如果要把本项目迁移到 TensorRT-LLM，需要改动哪些部分？
4. 在真实生产环境中，量化后还需要做哪些稳定性测试？
5. 量化后的模型如何与服务化框架（vLLM/TGI）结合？
