#!/usr/bin/env python3
"""
OpenAI 兼容接口冒烟测试示例。

很多主流推理框架（vLLM、TGI、SGLang、LMDeploy、部分 TensorRT-LLM serving 封装）
都可以暴露 OpenAI-compatible API。本脚本用 Python 标准库调用
`/v1/chat/completions`，用于验证服务是否可用。

默认不真正请求服务；如需冒烟测试，请显式设置：

    RUN_SMOKE=1 \
    OPENAI_BASE_URL=http://localhost:8000/v1 \
    MODEL_NAME=/data/liyangyang/models/Qwen3.5-9B \
    python openai_compatible_smoke_test.py

可选环境变量：
    PROMPT                 默认 "请用一句话介绍 KV Cache。"
    TIMEOUT_SECONDS        默认 30
"""
import json
import os
import sys
import urllib.error
import urllib.request


def build_request_body(model_name: str, prompt: str) -> bytes:
    """构造 OpenAI chat.completions 请求体。"""
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 16,
    }
    return json.dumps(body).encode("utf-8")


def run_smoke_test() -> int:
    """调用 OpenAI-compatible endpoint 并打印返回。"""
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    model_name = os.environ.get("MODEL_NAME", "/data/liyangyang/models/Qwen3.5-9B")
    prompt = os.environ.get("PROMPT", "请用一句话介绍 KV Cache。")
    timeout = float(os.environ.get("TIMEOUT_SECONDS", "30"))

    url = f"{base_url}/chat/completions"
    request = urllib.request.Request(
        url,
        data=build_request_body(model_name, prompt),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"POST {url}")
    print(f"model: {model_name}")
    print(f"prompt: {prompt}")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        print(f"smoke test failed: {exc}")
        return 2

    choice = payload.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content", "")
    usage = payload.get("usage", {})

    print("\nresponse content:")
    print(content)
    print("\nusage:")
    print(json.dumps(usage, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    """默认只打印用法；设置 RUN_SMOKE=1 后才真正请求。"""
    if os.environ.get("RUN_SMOKE") != "1":
        print("Set RUN_SMOKE=1 to run the smoke test against a running OpenAI-compatible server.")
        print("Example:")
        print("  RUN_SMOKE=1 OPENAI_BASE_URL=http://localhost:8000/v1 \\")
        print("  MODEL_NAME=/data/liyangyang/models/Qwen3.5-9B \\")
        print("  python examples/openai_compatible_smoke_test.py")
        return 0
    return run_smoke_test()


if __name__ == "__main__":
    sys.exit(main())
