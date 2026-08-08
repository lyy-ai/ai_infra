"""指标计算工具。"""


def percentile(values, pct):
    """简单 percentile。"""
    if not values:
        return 0.0
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(round(pct / 100.0 * len(values))) - 1))
    return values[idx]


def summarize_batch(elapsed_s: float, num_requests: int, prompt_tokens: int, completion_tokens: int) -> dict:
    """汇总一个 batch 的吞吐指标。"""
    total_tokens = prompt_tokens + completion_tokens
    return {
        "elapsed_s": elapsed_s,
        "num_requests": num_requests,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "req_per_s": num_requests / elapsed_s if elapsed_s > 0 else 0.0,
        "tok_per_s": total_tokens / elapsed_s if elapsed_s > 0 else 0.0,
        "prefill_tok_per_s": prompt_tokens / elapsed_s if elapsed_s > 0 else 0.0,
        "decode_tok_per_s": completion_tokens / elapsed_s if elapsed_s > 0 else 0.0,
    }
