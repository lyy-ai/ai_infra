# 模拟 ReAct 循环中的上下文增长与三种管理策略（15.1 节配套）
# 运行: python react_loop_simulator.py
# 只依赖标准库 + numpy

import numpy as np

rng = np.random.default_rng(42)

SYSTEM_PROMPT = 2000        # 系统提示 + 工具说明 token
THOUGHT_PER_TURN = 200      # 每轮模型思考+调用请求
TOOL_RESULT_MEAN = 1500     # 工具结果平均 token（文件/日志等）
TOOL_RESULT_STD = 800
MAX_CTX = 128_000           # 上下文窗口


def simulate(n_turns=100, strategy="truncate"):
    """模拟一个 agent 任务的上下文占用轨迹。

    strategy:
      none      - 不管理，记录何时爆窗
      truncate  - 硬截断最老消息（保 system prompt）
      compact   - 超 80% 窗口时把旧历史压缩为 1/10 摘要
      subagent  - 30% 的脏活轮次交给子代理，工具结果只回传 10% 结论
    """
    ctx = SYSTEM_PROMPT
    history = []          # 每轮的 ctx
    compact_events = 0
    overflow_turn = None
    for t in range(1, n_turns + 1):
        result = max(100, rng.normal(TOOL_RESULT_MEAN, TOOL_RESULT_STD))
        if strategy == "subagent" and rng.random() < 0.3:
            result *= 0.1  # 子代理只回传结论
        ctx += THOUGHT_PER_TURN + result
        if strategy == "truncate" and ctx > MAX_CTX:
            ctx = SYSTEM_PROMPT + (ctx - SYSTEM_PROMPT) * 0.5  # 丢弃一半旧历史
        elif strategy == "compact" and ctx > 0.8 * MAX_CTX:
            ctx = SYSTEM_PROMPT + (ctx - SYSTEM_PROMPT) * 0.1  # 摘要为 1/10
            compact_events += 1
        elif strategy == "none" and ctx > MAX_CTX and overflow_turn is None:
            overflow_turn = t
        history.append(ctx)
    return {
        "final_ctx": int(history[-1]),
        "peak_ctx": int(max(history)),
        "overflow_turn": overflow_turn,
        "compact_events": compact_events,
        "total_input_tokens": int(sum(history)),  # 每轮全量重发 = 总 prefill 量
    }


def main():
    print("=" * 78)
    print("ReAct 循环上下文增长模拟（100 轮任务，窗口 128k）")
    print("=" * 78)
    print(f"{'策略':<12}{'最终上下文':>12}{'峰值':>12}{'总 prefill token':>18}{'备注':>20}")
    print("-" * 78)
    for name, kw in [
        ("none", "none"),
        ("truncate", "truncate"),
        ("compact", "compact"),
        ("subagent", "subagent"),
    ]:
        r = simulate(strategy=kw)
        note = ""
        if r["overflow_turn"]:
            note = f"第{r['overflow_turn']}轮爆窗"
        elif r["compact_events"]:
            note = f"压缩{r['compact_events']}次"
        elif kw == "truncate":
            note = "信息有损"
        elif kw == "subagent":
            note = "主上下文最干净"
        print(f"{name:<12}{r['final_ctx']:>12,}{r['peak_ctx']:>12,}"
              f"{r['total_input_tokens']:>18,}{note:>20}")

    print("-" * 78)
    print("读法：")
    print("1. none 在几十轮后必然爆窗 —— 不管理的 agent 无法完成长任务")
    print("2. truncate 能跑完但丢失早期历史（目标漂移的根源）")
    print("3. compact 用摘要换连续性；subagent 让主上下文只见结论")
    print("4. 总 prefill token 就是成本 —— 前缀 cache 命中决定其中多少要重算")


if __name__ == "__main__":
    main()
