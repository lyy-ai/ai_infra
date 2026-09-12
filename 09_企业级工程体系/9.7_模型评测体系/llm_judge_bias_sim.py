# LLM-as-judge 位置偏差与纠偏模拟（9.7 节配套）
# 运行: python llm_judge_bias_sim.py
# 只依赖标准库 + numpy
#
# 模拟一个带位置偏差的 judge：
#   真实质量: A=0.8, B=0.75（A 略好）
#   judge 判定 = 质量分 + 位置加成(先出现+0.06) + 噪声
# 对比三种协议的错误率：单次判定 / 交换位置取一致 / 交换位置+多数投票

import numpy as np

rng = np.random.default_rng(11)

QUALITY = {"A": 0.75, "B": 0.74}
POS_BONUS = 0.06      # 先出现位置加成（位置偏差，大于真实质量差）
NOISE = 0.05          # 判定噪声 std


def judge_once(first, second):
    """judge 评价 (first, second) 顺序下的两个答案，返回胜者。"""
    score = {}
    for i, name in enumerate((first, second)):
        s = QUALITY[name] + (POS_BONUS if i == 0 else 0.0) + rng.normal(0, NOISE)
        score[name] = s
    return max(score, key=score.get)


def main():
    n = 20000
    print("=" * 72)
    print("LLM-as-judge 位置偏差模拟（真实质量 A > B，judge 带 +6% 位置加成）")
    print("=" * 72)

    # 协议 1：单次判定，固定 A 在前
    wrong = sum(judge_once("A", "B") != "A" for _ in range(n))
    print(f"协议1 单次判定(A 固定在前): 错误率 {wrong/n:.1%}")

    # 协议 1b：单次判定，固定 B 在前（位置加成给 B）
    wrong = sum(judge_once("B", "A") != "A" for _ in range(n))
    print(f"协议1b 单次判定(B 固定在前): 错误率 {wrong/n:.1%}  <- 位置偏差放大误判")

    # 协议 2：交换位置各评一次，取一致；不一致按平局计 0.5
    err = 0
    for _ in range(n):
        r1 = judge_once("A", "B") == "A"
        r2 = judge_once("B", "A") == "A"
        if r1 and r2:
            pass
        elif not r1 and not r2:
            err += 1
        else:
            err += 0.5
    print(f"协议2 交换位置(平局计0.5):  错误率 {err/n:.1%}")

    # 协议 3：交换位置，两次一致则采信；不一致则第三轮随机顺序决胜
    err = 0
    for _ in range(n):
        r1 = judge_once("A", "B") == "A"
        r2 = judge_once("B", "A") == "A"
        if r1 == r2:
            if not r1:
                err += 1
        else:
            pair = ("A", "B") if rng.random() < 0.5 else ("B", "A")
            if judge_once(*pair) != "A":
                err += 1
    print(f"协议3 交换位置+第三轮决胜: 错误率 {err/n:.1%}")

    print("-" * 72)
    print("读法：")
    print("1. 真实质量差 < 位置加成时，单次判定的结论由摆放位置决定（协议1 vs 1b 翻转）")
    print("2. 交换位置把'自信地判错'变成'暴露不一致'，是消除系统性偏差的最便宜手段")
    print("3. 不一致时加赛（随机顺序）给出无偏结论——judge 协议设计比 judge 模型更影响可信度")


if __name__ == "__main__":
    main()
