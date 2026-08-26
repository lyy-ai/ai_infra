# DDP 多卡训练：吞吐 / 每 rank 显存 / step 分解
# 运行：/data/qwen35_env/bin/torchrun --nproc_per_node=2 src/train_ddp.py
import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from model import SmallGPT, make_batch, num_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4, help="每 rank batch")
    ap.add_argument("--seq", type=int, default=256)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(42)  # 各 rank 同参数初始化
    model = SmallGPT(seq=args.seq).to(device)
    model = DDP(model, device_ids=[rank])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    n_params = num_params(model)

    torch.cuda.reset_peak_memory_stats()
    fwd_ms, bwd_ms, step_ms = 0.0, 0.0, 0.0
    t_start = time.perf_counter()

    for step in range(args.steps):
        # 各 rank 不同数据（seed 含 rank）
        x, y = make_batch(args.batch, args.seq, 16000, device, seed=step * 100 + rank)
        t0 = time.perf_counter()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()   # DDP 在此触发梯度 AllReduce（bucket 化 + overlap）
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        opt.step()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        if step >= args.warmup:
            fwd_ms += (t1 - t0) * 1e3
            bwd_ms += (t2 - t1) * 1e3
            step_ms += (t3 - t0) * 1e3
        if step % 10 == 0 and rank == 0:
            print(f"step {step}: loss {loss.item():.3f}")

    n = args.steps - args.warmup
    total_s = time.perf_counter() - t_start
    tokens_global = args.batch * args.seq * world * args.steps

    local = {
        "rank": rank,
        "fwd_ms": round(fwd_ms / n, 2), "bwd_ms": round(bwd_ms / n, 2),
        "step_ms": round(step_ms / n, 2),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "wall_s": round(total_s, 2),
    }
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)

    if rank == 0:
        result = {
            "mode": "ddp", "gpus": world, "params_M": round(n_params / 1e6, 1),
            "batch_per_rank": args.batch, "seq": args.seq, "steps": args.steps,
            "ranks": gathered,
            "tokens_per_s": round(tokens_global / max(r["wall_s"] for r in gathered), 0),
        }
        out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "ddp_2gpu.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
