"""vLLM 离线吞吐 benchmark 核心逻辑。"""
import time

import config as cfg
from src.metrics import summarize_batch
from src.prompts import build_prompts
from src.utils import now_iso


def run_offline_benchmark():
    """加载一次 vLLM，并按多个 batch size 测量吞吐。"""
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

    sampling_params = SamplingParams(
        temperature=cfg.TEMPERATURE,
        top_p=cfg.TOP_P,
        max_tokens=cfg.MAX_NEW_TOKENS,
    )

    warmup = SamplingParams(temperature=0.0, max_tokens=1)
    llm.generate(["请用一句话介绍 KV Cache。"], warmup)

    records = []
    for batch_size in cfg.BATCH_SIZES:
        prompts = build_prompts(batch_size)
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start

        prompt_tokens = sum(len(getattr(output, "prompt_token_ids", []) or []) for output in outputs)
        completion_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        records.append(
            {
                "batch_size": batch_size,
                **summarize_batch(elapsed, batch_size, prompt_tokens, completion_tokens),
            }
        )

    return {
        "metadata": {
            "created_at": now_iso(),
            "model_path": cfg.MODEL_PATH,
            "served_model_name": cfg.SERVED_MODEL_NAME,
            "gpu_memory_utilization": cfg.GPU_MEMORY_UTILIZATION,
            "max_model_len": cfg.MAX_MODEL_LEN,
            "max_new_tokens": cfg.MAX_NEW_TOKENS,
            "enforce_eager": cfg.ENFORCE_EAGER,
            "enable_prefix_caching": cfg.ENABLE_PREFIX_CACHING,
            "use_shared_prefix": cfg.USE_SHARED_PREFIX,
        },
        "records": records,
    }
