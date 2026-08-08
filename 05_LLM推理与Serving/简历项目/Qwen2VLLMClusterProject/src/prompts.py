"""Prompt 构造工具。"""
from config import SAMPLE_PROMPTS, SHARED_PREFIX_REPEAT, USE_SHARED_PREFIX


SHARED_PREFIX = (
    "你是一个严谨的中文 AI 基础设施助教，请用简洁、工程化的语言回答问题。"
    * SHARED_PREFIX_REPEAT
    + "\n问题："
)


def build_prompts(batch_size: int, shared_prefix: bool = USE_SHARED_PREFIX) -> list[str]:
    """生成 benchmark prompts；shared_prefix=True 时模拟固定 system prompt 场景。"""
    prompts = []
    for i in range(batch_size):
        question = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
        if shared_prefix:
            prompts.append(SHARED_PREFIX + question)
        else:
            prompts.append(question)
    return prompts
