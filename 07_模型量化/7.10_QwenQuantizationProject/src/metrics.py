import torch


def compute_perplexity(model, tokenizer, texts, max_length=512):
    """
    计算给定文本集合上的平均困惑度（PPL）。
    使用模型对每个文本计算 cross-entropy loss，再求平均。
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            # loss 是平均每个 token 的 loss
            loss = outputs.loss
            n_tokens = inputs["input_ids"].numel()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def summarize_results(results_dict):
    """对 benchmark 结果做简单汇总"""
    summary = {}
    for precision, data in results_dict.items():
        memory = data.get("memory", {})
        generations = data.get("generations", [])
        avg_tps = sum(g["tokens_per_sec"] for g in generations) / len(generations) if generations else 0
        avg_time = sum(g["time_sec"] for g in generations) / len(generations) if generations else 0
        summary[precision] = {
            "memory_allocated_gb": memory.get("allocated_gb", 0),
            "memory_reserved_gb": memory.get("reserved_gb", 0),
            "avg_tokens_per_sec": avg_tps,
            "avg_generation_time_sec": avg_time,
        }
    return summary
