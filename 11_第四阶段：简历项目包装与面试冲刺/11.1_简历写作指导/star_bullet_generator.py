# 1. 简历写作指导：STAR bullet 生成器
#
# 运行：
#   cd /data/ai_infra/11_第四阶段：简历项目包装与面试冲刺
#   /data/qwen35_env/bin/python 11.1_简历写作指导/star_bullet_generator.py


def make_star_bullet(situation, actions, results):
    lines = [f"- 背景：{situation}"]
    lines.append(f"- 职责：{'；'.join(actions)}。")
    lines.append(f"- 成果：{'，'.join(results)}。")
    return "\n".join(lines)


def main():
    bullet = make_star_bullet(
        situation="自动驾驶 MMBEV 模型在 Orin 上延迟 100ms，无法满足实时性要求",
        actions=[
            "自定义 BEV Pooling CUDA kernel（warp reduce + shared memory）",
            "基于 TVM Relax 实现内存池与 CUDA Graph",
            "INT8 分层量化（per-channel + 敏感层保护）",
        ],
        results=[
            "端到端延迟 100ms → 30ms",
            "吞吐提升 3 倍",
            "显存减少 30%",
            "通过量产验证",
        ],
    )
    print("=== Generated STAR Bullet ===\n")
    print(bullet)
    print()
    print("自查：量化指标 4 个 ✓ | 技术关键词 5 个 ✓ | 业务价值（量产）✓ | 三段分层 ✓")


if __name__ == "__main__":
    main()
