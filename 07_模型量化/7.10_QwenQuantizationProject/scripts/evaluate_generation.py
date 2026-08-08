#!/usr/bin/env python3
"""
Qwen3.5-9B 量化生成质量对比
使用相同 prompts 在 FP16 / INT8 / INT4 下生成，保存结果供人工对比。
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MODEL_PATH, RESULTS_DIR, SAMPLE_PROMPTS
from src.model_loader import load_tokenizer, load_model, get_int8_config, get_int4_config
from src.inference import generate_text
from src.utils import save_json


PRECISION_CONFIGS = {
    "fp16": None,
    "int8": get_int8_config(),
    "int4": get_int4_config(),
}


def main():
    tokenizer = load_tokenizer(MODEL_PATH)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    all_results = {}
    for precision, config in PRECISION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Generation Quality: {precision.upper()}")
        print(f"{'='*60}")
        
        model = load_model(MODEL_PATH, config)
        outputs = []
        for prompt in SAMPLE_PROMPTS:
            text, elapsed, tps, n_tokens = generate_text(model, tokenizer, prompt)
            outputs.append({
                "prompt": prompt,
                "output": text,
                "time_sec": elapsed,
                "tokens_per_sec": tps,
                "num_generated_tokens": n_tokens,
            })
            print(f"Prompt: {prompt[:40]}...")
            print(f"Output: {text[:200]}...\n")
        
        all_results[precision] = outputs
        
        del model
        torch.cuda.empty_cache()
    
    output_path = os.path.join(RESULTS_DIR, "generation_comparison.json")
    save_json(all_results, output_path)
    print(f"Saved generation comparison to {output_path}")


if __name__ == "__main__":
    main()
