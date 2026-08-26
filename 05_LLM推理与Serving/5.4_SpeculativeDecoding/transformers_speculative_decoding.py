#!/usr/bin/env python3
"""
使用 transformers 进行 Speculative Decoding 示例。

本脚本演示：
1. 加载 Target Model 和 Draft Model
2. 使用 transformers 的 assistant_model 参数进行投机解码
3. 对比不使用投机解码时的生成时间

运行方式：
    python examples/transformers_speculative_decoding.py

注意：
- 默认使用 /data/models/Qwen3.5-9B 作为 Target。
- 如果没有指定 Draft Model，会复用 Target 作为 Draft（用于演示机制）。
- 实际生产环境中应使用一个更小的同系列模型作为 Draft。
- 运行需要 GPU，耗时较长。
"""
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TARGET_MODEL_PATH = "/data/models/Qwen3.5-9B"
# 如果为空，则复用 Target Model 作为 Draft（演示用）
DRAFT_MODEL_PATH = ""

PROMPT = "请简要介绍一下机器学习中梯度下降算法的基本原理。"
MAX_NEW_TOKENS = 128
NUM_ASSISTANT_TOKENS = 5


def load_model(path: str, dtype=torch.float16):
    """加载模型。"""
    print(f"Loading model from {path}...")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    print("Model loaded.")
    return model


def measure_generate_time(model, tokenizer, prompt, use_assistant=False, assistant_model=None):
    """测量生成时间。"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    num_input_tokens = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if use_assistant:
        gen_kwargs["assistant_model"] = assistant_model
        gen_kwargs["num_assistant_tokens"] = NUM_ASSISTANT_TOKENS

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - start

    num_output_tokens = outputs.shape[1] - num_input_tokens
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text, elapsed, num_output_tokens


def main():
    print(f"Target model: {TARGET_MODEL_PATH}")
    print(f"Draft model: {DRAFT_MODEL_PATH or TARGET_MODEL_PATH} (same as target for demo)")
    print(f"Prompt: {PROMPT}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Num assistant tokens: {NUM_ASSISTANT_TOKENS}\n")

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_model = load_model(TARGET_MODEL_PATH)

    # 如果没有指定 Draft Model，复用 Target 作为 Draft（演示用）
    if DRAFT_MODEL_PATH and DRAFT_MODEL_PATH != TARGET_MODEL_PATH:
        draft_model = load_model(DRAFT_MODEL_PATH)
    else:
        draft_model = target_model
        print("\n[Note] Using target model as draft model for demonstration.")
        print("In production, use a smaller model from the same family as draft.\n")

    # 1. 普通生成
    print("=" * 60)
    print("Standard Autoregressive Generation")
    print("=" * 60)
    text_std, time_std, tokens_std = measure_generate_time(target_model, tokenizer, PROMPT)
    print(f"Time: {time_std:.3f}s")
    print(f"Tokens: {tokens_std}")
    print(f"TPS: {tokens_std/time_std:.2f}")
    print(f"Output: {text_std[:200]}...\n")

    # 2. 投机解码生成
    print("=" * 60)
    print("Speculative Decoding")
    print("=" * 60)
    text_spec, time_spec, tokens_spec = measure_generate_time(
        target_model, tokenizer, PROMPT, use_assistant=True, assistant_model=draft_model
    )
    print(f"Time: {time_spec:.3f}s")
    print(f"Tokens: {tokens_spec}")
    print(f"TPS: {tokens_spec/time_spec:.2f}")
    print(f"Output: {text_spec[:200]}...\n")

    # 3. 对比
    print("=" * 60)
    print("Comparison")
    print("=" * 60)
    if time_spec > 0:
        speedup = time_std / time_spec
        print(f"Speedup: {speedup:.2f}x")
    else:
        print("Speculative decoding time too small to measure.")

    # 验证输出一致性（在 same model 作为 draft 时，结果应该相同或高度相似）
    print(f"\nOutputs match: {text_std == text_spec}")
    if text_std != text_spec:
        print("Note: Different outputs are expected when using sampling, but distributions are identical.")


if __name__ == "__main__":
    main()
