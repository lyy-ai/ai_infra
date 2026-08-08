#!/usr/bin/env python3
"""
Qwen3.5-9B 量化模型交互式对话 Demo
支持 FP16 / INT8 / INT4 三种模式。
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MODEL_PATH, MAX_NEW_TOKENS
from src.model_loader import load_tokenizer, load_model, get_int8_config, get_int4_config


def select_mode():
    print("\n=== Qwen3.5-9B 量化对话 Demo ===")
    print("1. FP16 (高质量，高显存)")
    print("2. INT8 (平衡方案)")
    print("3. INT4 (低显存，可能质量下降)")
    choice = input("请选择量化模式 (1/2/3): ").strip()
    if choice == "2":
        return "INT8", get_int8_config()
    elif choice == "3":
        return "INT4", get_int4_config()
    else:
        return "FP16", None


def build_chat_input(tokenizer, history, user_input):
    """使用 tokenizer 的 chat_template 构造对话输入"""
    messages = []
    for role, content in history:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_input})
    
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return text


def main():
    mode_name, config = select_mode()
    print(f"\n正在加载 {mode_name} 模型...")
    tokenizer = load_tokenizer(MODEL_PATH)
    model = load_model(MODEL_PATH, config)
    print(f"{mode_name} 模型加载完成。输入 'exit' 退出。\n")
    
    history = []
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "quit", "退出"]:
            break
        
        prompt = build_chat_input(tokenizer, history, user_input)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 只保留新增的回复部分
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        print(f"Assistant: {response}\n")
        history.append(("user", user_input))
        history.append(("assistant", response))


if __name__ == "__main__":
    main()
