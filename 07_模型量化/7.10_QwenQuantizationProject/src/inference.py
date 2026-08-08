import time
import torch
from config import MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY


def generate_text(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS,
                  temperature=TEMPERATURE, top_p=TOP_P, repetition_penalty=REPETITION_PENALTY):
    """
    使用给定模型生成文本，返回生成文本、耗时、每秒 token 数和生成 token 数。
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start_time
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    num_input_tokens = inputs["input_ids"].shape[1]
    num_generated_tokens = outputs.shape[1] - num_input_tokens
    tokens_per_sec = num_generated_tokens / elapsed if elapsed > 0 else 0
    
    return generated_text, elapsed, tokens_per_sec, num_generated_tokens


def get_gpu_free_memory():
    """返回所有可用 GPU 的空闲显存总和（GB）。"""
    if not torch.cuda.is_available():
        return 0.0
    total_free = 0.0
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        total_free += free / 1024**3
    return total_free


def get_gpu_memory_info():
    """返回当前 GPU 总显存和已用显存（GB）。
    
    注意：nvidia-smi 级别的已用显存包含所有进程，无法单独隔离本进程；
    因此 benchmark 中更推荐通过 get_gpu_free_memory() 的差值计算模型真实占用。
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_gb = info.used / 1024**3
        total_gb = info.total / 1024**3
        return used_gb, total_gb
    except Exception:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return allocated, reserved
        return 0.0, 0.0
