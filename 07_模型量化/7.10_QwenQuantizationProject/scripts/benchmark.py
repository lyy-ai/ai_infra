#!/usr/bin/env python3
"""
Qwen3.5-9B 量化对比 Benchmark
加载 FP16 / INT8 / INT4 三种精度的模型，测量显存、生成速度和输出质量。
"""
import argparse
import json
import os
import time
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MODEL_PATH, RESULTS_DIR, SAMPLE_PROMPTS
from src.model_loader import load_tokenizer, load_model, get_int8_config, get_int4_config
from src.inference import generate_text, get_gpu_memory_info, get_gpu_free_memory
from src.utils import save_json, print_summary_table


PRECISION_CONFIGS = {
    "fp16": None,
    "int8": get_int8_config(),
    "int4": get_int4_config(),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.5-9B quantization benchmark")
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=list(PRECISION_CONFIGS.keys()),
        help="Run benchmark for a single precision. If not set, run all precisions.",
    )
    return parser.parse_args()


def benchmark_precision(precision_name, quant_config, tokenizer, prompts):
    """对单一精度进行 benchmark"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {precision_name.upper()}")
    print(f"{'='*60}")
    
    # 记录加载前空闲显存，用于后续计算模型真实占用
    free_before = get_gpu_free_memory()
    print(f"Free GPU memory before load: {free_before:.2f} GB")
    
    print(f"Loading model with {precision_name}...")
    start_load = time.time()
    model = load_model(MODEL_PATH, quant_config)
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")
    
    free_after = get_gpu_free_memory()
    model_memory_gb = free_before - free_after
    used_gb, total_gb = get_gpu_memory_info()
    print(f"GPU Memory - Used: {used_gb:.2f} GB / Total: {total_gb:.2f} GB")
    print(f"Estimated model memory footprint: {model_memory_gb:.2f} GB")
    
    generations = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}/{len(prompts)}: {prompt[:40]}...")
        text, elapsed, tps, n_tokens = generate_text(model, tokenizer, prompt)
        print(f"  Time: {elapsed:.3f}s | Tokens: {n_tokens} | TPS: {tps:.2f}")
        # 只保存输出前 300 字符，避免 JSON 过大
        generations.append({
            "prompt": prompt,
            "output": text[:300],
            "time_sec": elapsed,
            "tokens_per_sec": tps,
            "num_generated_tokens": n_tokens,
        })
    
    result = {
        "precision": precision_name,
        "load_time_sec": load_time,
        "memory": {
            "used_gb": used_gb,
            "total_gb": total_gb,
            "free_before_gb": free_before,
            "free_after_gb": free_after,
            "model_memory_gb": model_memory_gb,
        },
        "generations": generations,
    }
    
    del model
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    tokenizer = load_tokenizer(MODEL_PATH)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    precisions = [args.precision] if args.precision else list(PRECISION_CONFIGS.keys())
    all_results = {}
    
    for precision in precisions:
        config = PRECISION_CONFIGS[precision]
        result = benchmark_precision(precision, config, tokenizer, SAMPLE_PROMPTS)
        all_results[precision] = result
        
        output_path = os.path.join(RESULTS_DIR, f"{precision}_benchmark.json")
        save_json(result, output_path)
        print(f"Saved {precision} results to {output_path}")
    
    # 如果运行全部精度，保存汇总结果
    if len(precisions) == len(PRECISION_CONFIGS):
        summary_path = os.path.join(RESULTS_DIR, "benchmark_summary.json")
        save_json(all_results, summary_path)
        print(f"\nSaved full benchmark summary to {summary_path}")
        print_summary_table(all_results)
    else:
        print_summary_table(all_results)


if __name__ == "__main__":
    main()
