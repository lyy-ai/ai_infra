# NCCL AllReduce 带宽实测：1MB - 512MB，输出 algbw / busbw
# 运行：/data/qwen35_env/bin/torchrun --nproc_per_node=2 src/comm_bench.py
import json
import os
import time

import torch
import torch.distributed as dist

SIZES_MB = [1, 16, 64, 256, 512]


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)

    results = []
    for mb in SIZES_MB:
        n = mb * 1024 * 1024 // 4
        x = torch.randn(n, device=f"cuda:{rank}")
        # warmup
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 20
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1e3
        size_gb = mb / 1024
        algbw = size_gb / (ms / 1e3)
        busbw = algbw * 2 * (world - 1) / world
        results.append({"size_mb": mb, "ms": round(ms, 3),
                        "algbw_gbps": round(algbw, 1), "busbw_gbps": round(busbw, 1)})
        if rank == 0:
            print(f"{mb:>4}MB: {ms:.3f} ms, algbw {algbw:.1f} GB/s, busbw {busbw:.1f} GB/s")

    if rank == 0:
        out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "nccl_allreduce.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump({"world": world, "results": results}, f, indent=2)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
