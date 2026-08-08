# 9.2 边缘部署 - Jetson Orin：流式 decode 延迟模拟
#
# 运行：
#   cd /data/liyangyang/ai_infra/08_多平台适配
#   /data/liyangyang/qwen35_env/bin/python 8.2_边缘部署_Jetson_Orin/stream_decode_sim.py

import time
import random


def stream_decode(num_tokens, ttft_ms, avg_tbt_ms, jitter_ms=5.0):
    """
    模拟流式 decode：先产生 TTFT，再逐 token 输出。
    """
    delays = []
    # prefill
    time.sleep(ttft_ms / 1000.0)
    delays.append(ttft_ms)

    # decode loop
    for _ in range(num_tokens - 1):
        tbt = max(1.0, avg_tbt_ms + random.uniform(-jitter_ms, jitter_ms))
        time.sleep(tbt / 1000.0)
        delays.append(tbt)

    total = sum(delays)
    avg_tbt = sum(delays[1:]) / max(1, len(delays) - 1)
    print(f"TTFT={delays[0]:.1f}ms, avg TBT={avg_tbt:.1f}ms, total={total:.1f}ms for {num_tokens} tokens")
    return delays


def main():
    random.seed(42)
    print("=== Stream decode simulation ===")
    stream_decode(num_tokens=32, ttft_ms=300, avg_tbt_ms=60)
    stream_decode(num_tokens=128, ttft_ms=300, avg_tbt_ms=60)


if __name__ == "__main__":
    main()
