#!/usr/bin/env python3
"""
Prefix Cache 模拟器。

本脚本用纯 Python 模拟 Prefix Cache 的核心机制：
1. 将 prompt 按固定大小切分成 Block
2. 对完全相同的 Block 进行缓存与复用
3. 对比开启 / 关闭 Prefix Cache 时的 prefill 计算量与 KV Block 数量
4. 演示 system prompt 共享、多轮对话、Block 边界与 LRU 淘汰

无需 GPU 或 vLLM，可直接运行。

运行方式：
    python 5.5_PrefixCache/prefix_cache_simulator.py
"""
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple


BLOCK_SIZE = 16
PREFILL_COST_PER_TOKEN = 1.0
PREFIX_LOOKUP_COST_PER_TOKEN = 0.05
DECODE_COST_PER_TOKEN = 1.0


def tokenize(text: str) -> List[int]:
    """把文本映射成伪 token id 序列，仅用于模拟。"""
    return [ord(ch) for ch in text]


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def make_text(tag: str, length: int) -> str:
    """生成指定长度的伪 prompt 文本。"""
    if not tag:
        tag = "x"
    repeats = (length // len(tag)) + 1
    return (tag * repeats)[:length]


@dataclass
class CacheBlock:
    """缓存中的一个 KV Block。key = (block_start, block_tokens)。"""

    block_id: int
    key: Tuple[int, Tuple[int, ...]]
    ref_count: int = 1


class PrefixCache:
    """简化版 Prefix Cache：按 (位置, Block 内容) 做精确匹配，LRU 淘汰。"""

    def __init__(self, capacity_blocks: int, block_size: int = BLOCK_SIZE):
        self.capacity_blocks = capacity_blocks
        self.block_size = block_size
        self.cache: "OrderedDict[Tuple[int, Tuple[int, ...]], CacheBlock]" = OrderedDict()
        self._next_block_id = 0

    def _touch(self, key: Tuple[int, Tuple[int, ...]]):
        self.cache.move_to_end(key)

    def _evict_if_needed(self):
        while len(self.cache) > self.capacity_blocks:
            self.cache.popitem(last=False)

    def lookup_prefix(self, tokens: List[int]) -> Tuple[List[int], int]:
        """
        从 prompt 起始位置开始，匹配最长的连续完整 Block 前缀。

        返回 (matched_block_ids, matched_tokens)。
        注意：只匹配完整 Block；最后一个不满 block_size 的 Block 不参与共享。
        """
        full_prefix_tokens = (len(tokens) // self.block_size) * self.block_size
        matched_block_ids: List[int] = []

        for start in range(0, full_prefix_tokens, self.block_size):
            key = (start, tuple(tokens[start:start + self.block_size]))
            block = self.cache.get(key)
            if block is None:
                break
            self._touch(key)
            matched_block_ids.append(block.block_id)

        return matched_block_ids, len(matched_block_ids) * self.block_size

    def store_from(self, tokens: List[int], start_token: int) -> List[int]:
        """把 start_token 之后的 Block 写入缓存，返回这些 Block 的 id。"""
        stored_block_ids: List[int] = []
        for start in range(start_token, len(tokens), self.block_size):
            chunk = tokens[start:start + self.block_size]
            key = (start, tuple(chunk))
            block = self.cache.get(key)
            if block is None:
                block = CacheBlock(self._next_block_id, key)
                self._next_block_id += 1
                self.cache[key] = block
            else:
                block.ref_count += 1
                self._touch(key)
            stored_block_ids.append(block.block_id)
        self._evict_if_needed()
        return stored_block_ids

    def stats(self) -> Dict[str, int]:
        return {
            "cached_blocks": len(self.cache),
            "capacity_blocks": self.capacity_blocks,
            "cached_tokens": sum(len(block.key[1]) for block in self.cache.values()),
        }


def simulate_without_prefix_cache(requests: List[Tuple[str, List[int]]], decode_tokens: int = 16) -> Dict[str, float]:
    """关闭 Prefix Cache：每个请求都重新计算完整 prompt，并独立占用 KV Block。"""
    total_prompt_tokens = sum(len(tokens) for _, tokens in requests)
    total_blocks = sum(ceil_div(len(tokens), BLOCK_SIZE) for _, tokens in requests)
    total_prefill_cost = total_prompt_tokens * PREFILL_COST_PER_TOKEN
    total_decode_cost = len(requests) * decode_tokens * DECODE_COST_PER_TOKEN

    return {
        "total_prompt_tokens": total_prompt_tokens,
        "computed_prompt_tokens": total_prompt_tokens,
        "matched_prefix_tokens": 0,
        "hit_rate": 0.0,
        "kv_blocks": total_blocks,
        "total_cost": total_prefill_cost + total_decode_cost,
    }


def simulate_with_prefix_cache(
    requests: List[Tuple[str, List[int]]],
    capacity_blocks: int = 64,
    decode_tokens: int = 16,
    verbose: bool = True,
) -> Dict[str, float]:
    """开启 Prefix Cache：相同前缀 Block 只做 lookup，不重复 prefill。"""
    cache = PrefixCache(capacity_blocks=capacity_blocks, block_size=BLOCK_SIZE)

    total_prompt_tokens = 0
    total_computed_tokens = 0
    total_matched_tokens = 0
    total_cost = 0.0
    peak_cached_blocks = 0

    if verbose:
        print(f"{'request':>12} {'prompt':>8} {'matched':>8} {'computed':>9} {'cached_blocks':>14}")

    for name, tokens in requests:
        matched_block_ids, matched_tokens = cache.lookup_prefix(tokens)
        computed_tokens = len(tokens) - matched_tokens
        cache.store_from(tokens, matched_tokens)

        total_prompt_tokens += len(tokens)
        total_matched_tokens += matched_tokens
        total_computed_tokens += computed_tokens
        total_cost += (
            matched_tokens * PREFIX_LOOKUP_COST_PER_TOKEN
            + computed_tokens * PREFILL_COST_PER_TOKEN
            + decode_tokens * DECODE_COST_PER_TOKEN
        )
        peak_cached_blocks = max(peak_cached_blocks, len(cache.cache))

        if verbose:
            print(f"{name:>12} {len(tokens):>8} {matched_tokens:>8} {computed_tokens:>9} {len(cache.cache):>14}")

    stats = cache.stats()
    return {
        "total_prompt_tokens": total_prompt_tokens,
        "computed_prompt_tokens": total_computed_tokens,
        "matched_prefix_tokens": total_matched_tokens,
        "hit_rate": total_matched_tokens / total_prompt_tokens if total_prompt_tokens else 0.0,
        "kv_blocks": stats["cached_blocks"],
        "peak_cached_blocks": peak_cached_blocks,
        "total_cost": total_cost,
    }


def print_comparison(title: str, no_cache: Dict[str, float], with_cache: Dict[str, float]):
    """打印对比结果。"""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"{'metric':>28} {'no cache':>14} {'prefix cache':>14}")
    print("-" * 72)
    print(f"{'prompt tokens':>28} {no_cache['total_prompt_tokens']:>14.0f} {with_cache['total_prompt_tokens']:>14.0f}")
    print(f"{'computed prompt tokens':>28} {no_cache['computed_prompt_tokens']:>14.0f} {with_cache['computed_prompt_tokens']:>14.0f}")
    print(f"{'matched prefix tokens':>28} {no_cache['matched_prefix_tokens']:>14.0f} {with_cache['matched_prefix_tokens']:>14.0f}")
    print(f"{'prefix hit rate':>28} {no_cache['hit_rate']:>14.2%} {with_cache['hit_rate']:>14.2%}")
    print(f"{'kv blocks':>28} {no_cache['kv_blocks']:>14.0f} {with_cache['kv_blocks']:>14.0f}")
    print(f"{'simulated total cost':>28} {no_cache['total_cost']:>14.2f} {with_cache['total_cost']:>14.2f}")

    if with_cache["total_cost"] > 0:
        speedup = no_cache["total_cost"] / with_cache["total_cost"]
        saved_compute = 1 - (with_cache["computed_prompt_tokens"] / no_cache["computed_prompt_tokens"])
        print("-" * 72)
        print(f"prefill compute saved: {saved_compute:.2%}")
        print(f"simulated cost speedup: {speedup:.2f}x")


def scenario_shared_system_prompt():
    """场景 1：多个请求共享同一个 system prompt。"""
    print("\n" + "#" * 72)
    print("Scenario 1: Shared System Prompt")
    print("#" * 72)

    shared_prefix = make_text("S", 64)
    requests = []
    for i in range(6):
        user_prompt = make_text(f"U{i}", 22)
        requests.append((f"req_{i+1}", tokenize(shared_prefix + user_prompt)))

    no_cache = simulate_without_prefix_cache(requests)
    print("\n[With Prefix Cache]")
    with_cache = simulate_with_prefix_cache(requests, capacity_blocks=64)
    print_comparison("Shared System Prompt: No Cache vs Prefix Cache", no_cache, with_cache)


def scenario_multi_turn_chat():
    """场景 2：多轮对话中，后续 turn 复用历史对话前缀。"""
    print("\n" + "#" * 72)
    print("Scenario 2: Multi-turn Conversation")
    print("#" * 72)

    turn_1 = make_text("A", 48)
    turn_2 = turn_1 + make_text("B", 16) + make_text("C", 32)
    turn_3 = turn_2 + make_text("D", 16) + make_text("E", 48)

    requests = [
        ("turn_1", tokenize(turn_1)),
        ("turn_2", tokenize(turn_2)),
        ("turn_3", tokenize(turn_3)),
    ]

    no_cache = simulate_without_prefix_cache(requests)
    print("\n[With Prefix Cache]")
    with_cache = simulate_with_prefix_cache(requests, capacity_blocks=64)
    print_comparison("Multi-turn Conversation: No Cache vs Prefix Cache", no_cache, with_cache)


def scenario_block_boundary():
    """场景 3：前缀差异落在 Block 内部时，只能共享差异前的完整 Block。"""
    print("\n" + "#" * 72)
    print("Scenario 3: Block Boundary Effect")
    print("#" * 72)

    base = make_text("P", 64)
    diff_inside_block_2 = make_text("P", 31) + "X" + make_text("Q", 32)
    diff_after_block_2 = make_text("P", 32) + "X" + make_text("Q", 31)

    requests = [
        ("base", tokenize(base)),
        ("diff@token31", tokenize(diff_inside_block_2)),
        ("diff@token32", tokenize(diff_after_block_2)),
    ]

    print(f"Block size: {BLOCK_SIZE} tokens")
    print("base prompt is cached first; then two variants reuse only the matched full blocks.")
    simulate_with_prefix_cache(requests, capacity_blocks=64)

    print("\nInterpretation:")
    print("- diff@token31: token 31 位于第 2 个 Block 内部，只能复用第 1 个完整 Block（16 tokens）。")
    print("- diff@token32: 前 2 个 Block 完全相同，可以复用 32 tokens。")


def scenario_lru_eviction():
    """场景 4：缓存容量有限时，LRU 淘汰会降低后续命中率。"""
    print("\n" + "#" * 72)
    print("Scenario 4: LRU Eviction")
    print("#" * 72)

    requests = []
    for tag in ["A", "B", "C", "A"]:
        prefix = make_text(tag, 64)
        suffix = make_text(tag.lower(), 16)
        requests.append((f"prefix_{tag}", tokenize(prefix + suffix)))

    print("Capacity: 6 blocks; each request needs 5 blocks (4 shared prefix + 1 suffix).")
    simulate_with_prefix_cache(requests, capacity_blocks=6)
    print("\nObservation: after B and C are inserted, A's blocks are evicted; when A returns, hit rate drops.")


if __name__ == "__main__":
    scenario_shared_system_prompt()
    scenario_multi_turn_chat()
    scenario_block_boundary()
    scenario_lru_eviction()

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print("- Prefix Cache reuses KV blocks for identical prompt prefixes.")
    print("- It saves prefill compute and reduces TTFT, especially for shared system prompts and multi-turn chat.")
    print("- Sharing is block-granular: prefixes that differ inside a block can only share earlier full blocks.")
    print("- A finite cache needs an eviction policy; LRU is a common default.")
    print("=" * 72)
