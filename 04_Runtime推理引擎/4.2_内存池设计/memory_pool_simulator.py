#!/usr/bin/env python3
"""
内存池设计模拟器：Arena Allocator、Zero Copy、动态内存池与显存碎片。

运行方式：
    python 4.2_内存池设计/memory_pool_simulator.py
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple


ARENA_MB = 1024
COPY_BW_GBPS = 16.0
PINNED_BW_GBPS = 50.0


@dataclass
class Alloc:
    name: str
    size_mb: int
    start: int
    end: int


def alloc_sequence() -> List[Alloc]:
    """构造会产生碎片的分配序列：大小交替、生命周期交错。"""
    return [
        Alloc("a", 128, 0, 5),
        Alloc("b", 256, 1, 3),
        Alloc("c", 64, 2, 6),
        Alloc("d", 256, 3, 5),
        Alloc("e", 128, 4, 7),
        Alloc("f", 512, 5, 6),
        Alloc("g", 64, 6, 8),
    ]


def first_fit(blocks: List[Tuple[int, int]], size: int) -> Optional[int]:
    """在空闲区间中找第一个能放下的位置。"""
    for i, (start, end) in enumerate(blocks):
        if end - start >= size:
            if end - start == size:
                blocks.pop(i)
            else:
                blocks[i] = (start + size, end)
            return start
    return None


def free_interval(blocks: List[Tuple[int, int]], start: int, size: int):
    """释放区间并合并相邻空洞。"""
    blocks.append((start, start + size))
    blocks.sort()
    merged = []
    for s, e in blocks:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    blocks[:] = merged


def simulate_contiguous(allocs: List[Alloc]):
    """模拟连续显存 first-fit 分配，统计峰值、失败与碎片。"""
    free_blocks = [(0, ARENA_MB)]
    active = {}
    peak_used = 0
    max_frag = 0.0
    failed = []
    for t in range(0, 9):
        for alloc in allocs:
            if alloc.end == t and alloc.name in active:
                start, size = active.pop(alloc.name)
                free_interval(free_blocks, start, size)
        for alloc in allocs:
            if alloc.start == t:
                pos = first_fit(free_blocks, alloc.size_mb)
                if pos is None:
                    failed.append(alloc.name)
                else:
                    active[alloc.name] = (pos, alloc.size_mb)
        used = ARENA_MB - sum(e - s for s, e in free_blocks)
        peak_used = max(peak_used, used)
        free_total = sum(e - s for s, e in free_blocks)
        largest = max((e - s for s, e in free_blocks), default=0)
        frag = 1 - (largest / free_total) if free_total > 0 else 0.0
        max_frag = max(max_frag, frag)
    return {"peak_used": peak_used, "failed": failed, "max_fragmentation": max_frag}


def size_class(size: int) -> int:
    """简单 2 的幂 size class。"""
    cls = 64
    while cls < size:
        cls *= 2
    return cls


def simulate_pool(allocs: List[Alloc]):
    """模拟 size-class arena pool：复用空闲块，统计 cudaMalloc 次数。"""
    pools = {}
    active = {}
    cuda_malloc_calls = 0
    peak_used = 0
    for t in range(0, 9):
        for alloc in allocs:
            if alloc.end == t and alloc.name in active:
                cls = active.pop(alloc.name)
                pools.setdefault(cls, []).append(cls)
        for alloc in allocs:
            if alloc.start == t:
                cls = size_class(alloc.size_mb)
                if pools.get(cls):
                    pools[cls].pop()
                else:
                    cuda_malloc_calls += 1
                active[alloc.name] = cls
        used = sum(cls for cls in active.values())
        peak_used = max(peak_used, used)
    return {"cuda_malloc_calls": cuda_malloc_calls, "peak_used": peak_used}


def zero_copy_table(mb: int = 256):
    """对比 pageable copy、pinned copy、zero-copy 的传输时间。"""
    bytes_ = mb * 1e6
    pageable_ms = bytes_ / (COPY_BW_GBPS * 1e9) * 1000
    pinned_ms = bytes_ / (PINNED_BW_GBPS * 1e9) * 1000
    print("\nZero copy / pinned comparison")
    print("-" * 72)
    print(f"{'mode':>18} {'bandwidth':>14} {'time_ms':>12}")
    print("-" * 72)
    print(f"{'pageable H2D':>18} {COPY_BW_GBPS:>12.1f}GB/s {pageable_ms:>12.3f}")
    print(f"{'pinned H2D':>18} {PINNED_BW_GBPS:>12.1f}GB/s {pinned_ms:>12.3f}")
    print(f"{'zero-copy mapped':>18} {'no explicit copy':>14} {'~0 + access cost':>12}")


def main():
    allocs = alloc_sequence()
    contiguous = simulate_contiguous(allocs)
    pool = simulate_pool(allocs)

    print("=" * 72)
    print("Memory Pool Simulator")
    print("=" * 72)
    print(f"arena: {ARENA_MB} MB")
    print("alloc sequence:")
    for alloc in allocs:
        print(f"  {alloc.name}: size={alloc.size_mb}MB life=[{alloc.start},{alloc.end})")

    print("\n[Contiguous first-fit]")
    print(f"  peak used: {contiguous['peak_used']} MB")
    print(f"  failed allocs: {contiguous['failed'] or 'none'}")
    print(f"  max external fragmentation: {contiguous['max_fragmentation']:.1%}")

    print("\n[Size-class pool]")
    print(f"  cudaMalloc calls: {pool['cuda_malloc_calls']}")
    print(f"  peak used (rounded): {pool['peak_used']} MB")

    zero_copy_table(256)

    print("\nTakeaway:")
    print("- Arena pool turns hot-path cudaMalloc/cudaFree into pool lookup.")
    print("- Size classes trade some internal waste for much lower fragmentation risk.")
    print("- Zero-copy avoids explicit H2D copy for small/frequent host accesses, but large one-shot copies may prefer pinned/async copy.")


if __name__ == "__main__":
    main()
