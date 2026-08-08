# 单卡 baseline：吞吐 / 显存 / step 分解
# 运行：/data/liyangyang/qwen35_env/bin/python src/train_single.py
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from model import SmallGPT, make_batch, num_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"

    model = SmallGPT(seq=args.seq).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    n_params = num_params(model)
    print(f"params: {n_params / 1e6:.1f}M")

    torch.cuda.reset_peak_memory_stats()
    fwd_ms, bwd_ms, step_ms = 0.0, 0.0, 0.0
    t_start = time.perf_counter()

    for step in range(args.steps):
        x, y = make_batch(args.batch, args.seq, 16000, device, seed=step)
        t0 = time.perf_counter()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        opt.step()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        if step >= args.warmup:
            fwd_ms += (t1 - t0) * 1e3
            bwd_ms += (t2 - t1) * 1e3
            step_ms += (t3 - t0) * 1e3
        if step % 10 == 0:
            print(f"step {step}: loss {loss.item():.3f}")

    n = args.steps - args.warmup
    total_s = time.perf_counter() - t_start
    tokens = args.batch * args.seq * args.steps
    result = {
        "mode": "single_gpu", "gpus": 1, "params_M": round(n_params / 1e6, 1),
        "batch": args.batch, "seq": args.seq, "steps": args.steps,
        "fwd_ms": round(fwd_ms / n, 2), "bwd_ms": round(bwd_ms / n, 2),
        "step_ms": round(step_ms / n, 2),
        "tokens_per_s": round(tokens / total_s, 0),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "single_gpu.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
