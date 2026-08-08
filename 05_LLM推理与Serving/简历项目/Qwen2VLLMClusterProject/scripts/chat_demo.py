#!/usr/bin/env python3
"""Qwen2-0.5B vLLM chat demo。

默认打印用法；使用 --offline 时会加载 vLLM 离线引擎进行简单对话。
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as cfg  # noqa: E402


def run_offline_chat():
    import src.vllm_env_helper  # noqa: F401
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=cfg.MODEL_PATH,
        dtype="float16",
        gpu_memory_utilization=cfg.GPU_MEMORY_UTILIZATION,
        max_model_len=cfg.MAX_MODEL_LEN,
        enable_prefix_caching=cfg.ENABLE_PREFIX_CACHING,
        enforce_eager=cfg.ENFORCE_EAGER,
    )
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)

    print("Qwen2-0.5B offline chat demo. Type 'exit' to quit.")
    while True:
        prompt = input("\nUser: ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break
        outputs = llm.generate([prompt], sampling_params)
        print("Assistant:", outputs[0].outputs[0].text)


def main():
    parser = argparse.ArgumentParser(description="Qwen2-0.5B vLLM chat demo.")
    parser.add_argument("--offline", action="store_true", help="run offline vLLM chat")
    args = parser.parse_args()

    if not args.offline:
        print("Run with --offline to start the local vLLM chat demo.")
        print("For cluster mode, start scripts/start_qwen2_vllm_server.sh first and use an OpenAI client.")
        return 0
    run_offline_chat()
    return 0


if __name__ == "__main__":
    sys.exit(main())
