#!/usr/bin/env python3
"""
LLM 推理基础示例：加载模型、生成文本、测量关键指标。

本脚本演示：
1. 使用 transformers 加载模型并生成文本
2. 测量 TTFT（首 token 延迟）、TPOT（每 token 延迟）、Latency、TPS
3. 对比 Greedy / Sampling / Beam Search 三种解码策略

运行环境：
    source /data/liyangyang/qwen35_env/bin/activate
    python examples/basic_inference.py
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "/data/liyangyang/models/Qwen3.5-9B"
PROMPT = "请介绍一下机器学习中的梯度下降算法。"
MAX_NEW_TOKENS = 128


def load_model_and_tokenizer(model_path):
    """加载 tokenizer 和模型。"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model, tokenizer


def measure_basic_inference(model, tokenizer, prompt, max_new_tokens=128, **gen_kwargs):
    """测量一次完整生成的 latency 和 TPS。"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    num_input_tokens = inputs["input_ids"].shape[1]

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **gen_kwargs,
        )
    end = time.time()

    num_generated_tokens = outputs.shape[1] - num_input_tokens
    latency = end - start
    tps = num_generated_tokens / latency if latency > 0 else 0

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text, latency, tps, num_generated_tokens


def measure_ttft_and_tpot(model, tokenizer, prompt, max_new_tokens=128, **gen_kwargs):
    """
    分离测量 TTFT 和 TPOT。

    TTFT: 从输入到第一个输出 token 的时间。
    TPOT: 后续每个 token 的平均生成时间。
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Prefill + 生成第一个 token
    start = time.time()
    with torch.no_grad():
        first_outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **gen_kwargs,
        )
    ttft = time.time() - start

    # Decode 后续 token
    start = time.time()
    with torch.no_grad():
        full_outputs = model.generate(
            first_outputs,
            max_new_tokens=max_new_tokens - 1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **gen_kwargs,
        )
    decode_time = time.time() - start

    num_generated_tokens = full_outputs.shape[1] - inputs["input_ids"].shape[1]
    num_decode_tokens = num_generated_tokens - 1
    tpot = (decode_time / num_decode_tokens) * 1000 if num_decode_tokens > 0 else 0
    tps = 1000 / tpot if tpot > 0 else 0

    generated_text = tokenizer.decode(full_outputs[0], skip_special_tokens=True)
    return generated_text, ttft, tpot, tps, num_generated_tokens


def compare_decoding_strategies(model, tokenizer, prompt, max_new_tokens=64):
    """对比不同解码策略的输出。"""
    strategies = {
        "Greedy": {"do_sample": False},
        "Sampling (T=0.7, top_p=0.9)": {"do_sample": True, "temperature": 0.7, "top_p": 0.9},
        "Sampling (T=1.2, top_p=0.95)": {"do_sample": True, "temperature": 1.2, "top_p": 0.95},
    }

    print("\n" + "=" * 60)
    print("Decoding Strategy Comparison")
    print("=" * 60)
    for name, kwargs in strategies.items():
        text, latency, tps, n_tokens = measure_basic_inference(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens, **kwargs
        )
        print(f"\n[{name}]")
        print(f"  Latency: {latency:.3f}s | TPS: {tps:.2f} | Tokens: {n_tokens}")
        print(f"  Output: {text[:150]}...")


def main():
    print(f"Loading model from {MODEL_PATH}...")
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    print("Model loaded.\n")

    # 1. 基础推理
    print("=" * 60)
    print("Basic Inference")
    print("=" * 60)
    text, latency, tps, n_tokens = measure_basic_inference(
        model, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1,
    )
    print(f"Latency: {latency:.3f}s")
    print(f"Generated tokens: {n_tokens}")
    print(f"TPS: {tps:.2f}")
    print(f"Output:\n{text[:300]}...\n")

    # 2. TTFT / TPOT 测量
    print("=" * 60)
    print("TTFT / TPOT Measurement")
    print("=" * 60)
    text, ttft, tpot, tps, n_tokens = measure_ttft_and_tpot(
        model, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1,
    )
    print(f"TTFT: {ttft*1000:.2f} ms")
    print(f"TPOT: {tpot:.2f} ms")
    print(f"TPS:  {tps:.2f}")
    print(f"Total tokens: {n_tokens}\n")

    # 3. 解码策略对比
    compare_decoding_strategies(model, tokenizer, PROMPT, max_new_tokens=64)


if __name__ == "__main__":
    main()
