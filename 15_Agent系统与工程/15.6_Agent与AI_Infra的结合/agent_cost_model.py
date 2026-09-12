# Agent 任务推理成本模型（15.6 节配套）
# 运行: python agent_cost_model.py
# 只依赖标准库
#
# 成本模型：
#   单任务成本 = Σ_轮次 [ 当前上下文 × prefill单价 × (1-前缀命中率) + 输出 × decode单价 ]
# 对照：单轮问答成本 = 一次 prefill + decode

SYSTEM_PROMPT = 3000       # 系统提示+工具说明
CTX_GROWTH = 1700          # 每轮上下文增长（思考+工具结果）
OUTPUT_PER_TURN = 300      # 每轮输出 token（一次工具调用很短）
PREFILL_PRICE = 1.0        # 归一化单价（每 token）
DECODE_PRICE = 3.0         # decode 约为 prefill 的 3-5 倍（见 14.2 Q18）


def task_cost(n_turns, prefix_hit_rate):
    ctx = SYSTEM_PROMPT
    total = 0.0
    for _ in range(n_turns):
        ctx += CTX_GROWTH
        miss_tokens = ctx * (1 - prefix_hit_rate)
        total += miss_tokens * PREFILL_PRICE + OUTPUT_PER_TURN * DECODE_PRICE
    return total


def main():
    single_turn = SYSTEM_PROMPT * PREFILL_PRICE + 800 * DECODE_PRICE
    print("=" * 76)
    print("Agent 任务成本模型（归一化价格：prefill=1, decode=3）")
    print("=" * 76)
    print(f"基准：单轮问答成本 ≈ {single_turn:,.0f}\n")

    print(f"{'轮数':>6}{'前缀命中率':>10}{'任务成本':>14}{'相对单轮倍数':>14}")
    print("-" * 76)
    for turns in (10, 30, 50, 100):
        for hit in (0.0, 0.9, 0.99):
            c = task_cost(turns, hit)
            print(f"{turns:>6}{hit:>10.0%}{c:>14,.0f}{c/single_turn:>13.1f}x")
        print("-" * 76)

    print("\n不同前缀命中率下 50 轮任务的成本构成：")
    for hit in (0.0, 0.5, 0.9, 0.95, 0.99):
        c = task_cost(50, hit)
        print(f"  命中率 {hit:>5.0%} -> 成本 {c:>10,.0f}  ({c/single_turn:.1f}x 单轮)")

    print("\n读法：")
    print("1. Agent 任务成本是单轮问答的几十到上百倍（轮数 × 上下文累积）")
    print("2. 前缀命中率 0.9 -> 0.99，50 轮任务成本接近减半：")
    print("   Agent 时代，prefix 命中率就是利润率")
    print("3. harness 的上下文压缩（compaction）同时降轮次与上下文两项，是第二杠杆")


if __name__ == "__main__":
    main()
