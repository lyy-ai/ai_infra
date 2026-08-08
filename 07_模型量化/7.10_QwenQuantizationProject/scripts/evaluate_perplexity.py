#!/usr/bin/env python3
"""
Qwen3.5-9B 量化困惑度评估
对比 FP16 / INT8 / INT4 在中文样本上的 Perplexity。
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MODEL_PATH, RESULTS_DIR, PPL_SAMPLE_TEXTS
from src.model_loader import load_tokenizer, load_model, get_int8_config, get_int4_config
from src.metrics import compute_perplexity
from src.utils import save_json


PRECISION_CONFIGS = {
    "fp16": None,
    "int8": get_int8_config(),
    "int4": get_int4_config(),
}


def main():
    tokenizer = load_tokenizer(MODEL_PATH)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results = {}
    for precision, config in PRECISION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating Perplexity: {precision.upper()}")
        print(f"{'='*60}")
        
        model = load_model(MODEL_PATH, config)
        ppl = compute_perplexity(model, tokenizer, PPL_SAMPLE_TEXTS, max_length=512)
        results[precision] = {
            "perplexity": ppl,
            "num_samples": len(PPL_SAMPLE_TEXTS),
        }
        print(f"{precision.upper()} Perplexity: {ppl:.4f}")
        
        del model
        torch.cuda.empty_cache()
    
    output_path = os.path.join(RESULTS_DIR, "perplexity_results.json")
    save_json(results, output_path)
    print(f"\nSaved perplexity results to {output_path}")
    
    print("\n" + "=" * 60)
    print(f"{'Precision':>12} {'Perplexity':>16}")
    print("=" * 60)
    for precision, data in results.items():
        print(f"{precision:>12} {data['perplexity']:>16.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
