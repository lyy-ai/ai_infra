#!/usr/bin/env python3
"""
Qwen2-0.5B vLLM 集群并发压测脚本。

用于简历项目量化指标：吞吐、req/s、成功率、p50/p95 延迟。

示例：
    python benchmark_qwen2_concurrency.py \
      --base-url http://localhost:8080/v1 \
      --model qwen2-0.5b-instruct \
      --requests 128 \
      --concurrency 128 \
      --max-tokens 64

说明：
- 使用 OpenAI-compatible `/chat/completions`。
- 如果服务返回 usage，则按 usage 统计 token；否则 completion token 记为 0。
- 压测结果会随 GPU、并发、prompt 长度、max_tokens、网络而变化。
"""
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(round(pct / 100.0 * len(values))) - 1))
    return values[idx]


def post_chat(base_url, model, prompt, max_tokens, timeout):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        elapsed = time.perf_counter() - start
        usage = payload.get("usage", {}) or {}
        return {
            "ok": True,
            "latency": elapsed,
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "error": "",
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "latency": time.perf_counter() - start,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": str(exc),
        }


def build_prompts(num_requests, same_prompt):
    base = "请用一句话介绍大语言模型推理中的 KV Cache。"
    if same_prompt:
        return [base for _ in range(num_requests)]
    return [f"[case-{i}] {base}" for i in range(num_requests)]


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen2-0.5B vLLM OpenAI-compatible serving.")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--model", default="qwen2-0.5b-instruct")
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--same-prompt", action="store_true", help="use identical prompt to test prefix/cache friendly traffic")
    args = parser.parse_args()

    prompts = build_prompts(args.requests, args.same_prompt)
    wall_start = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(post_chat, args.base_url, args.model, prompt, args.max_tokens, args.timeout)
            for prompt in prompts
        ]
        for future in as_completed(futures):
            results.append(future.result())

    wall_time = time.perf_counter() - wall_start
    ok_results = [r for r in results if r["ok"]]
    errors = [r["error"] for r in results if not r["ok"]]
    latencies = [r["latency"] for r in ok_results]
    prompt_tokens = sum(r["prompt_tokens"] for r in ok_results)
    completion_tokens = sum(r["completion_tokens"] for r in ok_results)
    total_tokens = prompt_tokens + completion_tokens

    success_rate = len(ok_results) / len(results) if results else 0.0
    req_s = len(ok_results) / wall_time if wall_time > 0 else 0.0
    tok_s = total_tokens / wall_time if wall_time > 0 else 0.0
    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    avg_latency = statistics.mean(latencies) if latencies else 0.0

    print("=" * 84)
    print("Qwen2-0.5B vLLM Concurrency Benchmark")
    print("=" * 84)
    print(f"base_url: {args.base_url}")
    print(f"model: {args.model}")
    print(f"requests: {args.requests}, concurrency: {args.concurrency}, max_tokens: {args.max_tokens}")
    print(f"wall time: {wall_time:.3f}s")
    print(f"success: {len(ok_results)}/{len(results)} ({success_rate:.1%})")
    print(f"prompt tokens: {prompt_tokens}, completion tokens: {completion_tokens}")
    print(f"throughput: {tok_s:.2f} tok/s, {req_s:.2f} req/s")
    print(f"latency avg/p50/p95: {avg_latency:.3f}s / {p50:.3f}s / {p95:.3f}s")
    if errors:
        print("errors (first 3):")
        for err in errors[:3]:
            print(f"  - {err}")

    print("\nResume bullet template (replace with your measured values):")
    print(
        f"- 基于 vLLM 部署 Qwen2-0.5B-Instruct 推理集群，OpenAI-compatible API + 多实例路由；"
        f"在 {args.concurrency} 并发、max_tokens={args.max_tokens} 压测下达到 {tok_s:.1f} tok/s、"
        f"{req_s:.2f} req/s，成功率 {success_rate:.1%}，p95 延迟 {p95:.3f}s。"
    )


if __name__ == "__main__":
    main()
