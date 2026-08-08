#!/usr/bin/env python3
"""
PagedAttention 块分配模拟器。

本脚本用纯 Python 模拟 PagedAttention 的核心机制：
1. 将显存划分为固定大小的 Block
2. 每个请求通过 Block Table 维护 logical → physical 的映射
3. 演示按需分配、共享前缀、释放回收
4. 对比连续分配与分页分配的显存利用率

无需 GPU 或 vLLM，可直接运行。
"""
import random
from typing import Dict, List, Optional, Set


class PhysicalBlock:
    """物理块：实际存储 KV Cache 的单元。"""

    def __init__(self, block_id: int, block_size: int = 16):
        self.block_id = block_id
        self.block_size = block_size
        self.occupied_tokens = 0  # 当前块中实际占用的 token 数
        self.ref_count = 0        # 引用计数（用于 copy-on-write）
        self.is_shared = False

    def __repr__(self):
        return f"PBlock({self.block_id}, used={self.occupied_tokens}/{self.block_size}, ref={self.ref_count})"


class PagedAttentionKVCache:
    """PagedAttention KV Cache 管理器。"""

    def __init__(self, total_blocks: int, block_size: int = 16):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.physical_blocks: Dict[int, PhysicalBlock] = {}
        self.free_blocks: Set[int] = set(range(total_blocks))
        self.request_block_tables: Dict[str, List[int]] = {}
        self._next_block_id = total_blocks

    def allocate_block(self) -> Optional[int]:
        """分配一个空闲物理块。"""
        if not self.free_blocks:
            return None
        block_id = self.free_blocks.pop()
        self.physical_blocks[block_id] = PhysicalBlock(block_id, self.block_size)
        return block_id

    def free_block(self, block_id: int):
        """释放一个物理块。"""
        if block_id in self.physical_blocks:
            del self.physical_blocks[block_id]
            self.free_blocks.add(block_id)

    def allocate_for_request(self, request_id: str, num_tokens: int) -> bool:
        """为某个请求分配 num_tokens 个 token 所需的 KV Cache。"""
        if request_id not in self.request_block_tables:
            self.request_block_tables[request_id] = []

        block_table = self.request_block_tables[request_id]

        # 如果已有块，先填满最后一个块
        if block_table:
            last_block_id = block_table[-1]
            last_block = self.physical_blocks[last_block_id]
            remaining = self.block_size - last_block.occupied_tokens
            if remaining > 0:
                use = min(remaining, num_tokens)
                last_block.occupied_tokens += use
                num_tokens -= use
                if num_tokens == 0:
                    return True

        # 按需分配新块
        while num_tokens > 0:
            block_id = self.allocate_block()
            if block_id is None:
                return False  # OOM
            block_table.append(block_id)
            block = self.physical_blocks[block_id]
            use = min(self.block_size, num_tokens)
            block.occupied_tokens = use
            block.ref_count = 1
            num_tokens -= use

        return True

    def free_request(self, request_id: str):
        """释放某个请求占用的所有块。"""
        if request_id not in self.request_block_tables:
            return
        for block_id in self.request_block_tables[request_id]:
            self.free_block(block_id)
        del self.request_block_tables[request_id]

    def share_prefix(self, donor_request_id: str, recipient_request_id: str, prefix_blocks: int) -> bool:
        """模拟前缀共享：recipient 复用 donor 的前几个块。"""
        if donor_request_id not in self.request_block_tables:
            return False

        donor_table = self.request_block_tables[donor_request_id]
        if recipient_request_id not in self.request_block_tables:
            self.request_block_tables[recipient_request_id] = []

        recipient_table = self.request_block_tables[recipient_request_id]

        for i in range(min(prefix_blocks, len(donor_table))):
            block_id = donor_table[i]
            if block_id not in self.physical_blocks:
                continue
            self.physical_blocks[block_id].ref_count += 1
            self.physical_blocks[block_id].is_shared = True
            recipient_table.append(block_id)

        return True

    def get_memory_usage(self) -> Dict[str, float]:
        """返回显存使用情况统计。"""
        used_blocks = len(self.physical_blocks)
        total_capacity = self.total_blocks * self.block_size
        used_tokens = sum(b.occupied_tokens for b in self.physical_blocks.values())
        wasted_capacity = sum(self.block_size - b.occupied_tokens for b in self.physical_blocks.values())

        return {
            "total_blocks": self.total_blocks,
            "used_blocks": used_blocks,
            "free_blocks": len(self.free_blocks),
            "total_capacity_tokens": total_capacity,
            "used_tokens": used_tokens,
            "wasted_tokens": wasted_capacity,
            "utilization_rate": used_tokens / total_capacity if total_capacity > 0 else 0,
        }

    def print_state(self):
        """打印当前状态。"""
        print("\n--- PagedAttention State ---")
        print(f"Total physical blocks: {self.total_blocks}")
        print(f"Used blocks: {len(self.physical_blocks)}")
        print(f"Free blocks: {len(self.free_blocks)}")
        print("Request block tables:")
        for req_id, blocks in self.request_block_tables.items():
            print(f"  {req_id}: {blocks}")
        print("Physical blocks:")
        for block in self.physical_blocks.values():
            print(f"  {block}")
        print("-" * 40)


class ContinuousAllocator:
    """朴素连续分配器：每个请求预分配固定最大长度的 KV Cache。"""

    def __init__(self, total_capacity: int, max_seq_len: int = 128):
        self.total_capacity = total_capacity
        self.max_seq_len = max_seq_len
        self.used_capacity = 0

    def allocate(self, num_tokens: int) -> bool:
        """为每个请求预分配 max_seq_len，无论实际使用多少。"""
        if self.used_capacity + self.max_seq_len > self.total_capacity:
            return False
        self.used_capacity += self.max_seq_len
        return True

    def free(self):
        """释放一个请求的容量。"""
        self.used_capacity = max(0, self.used_capacity - self.max_seq_len)


def simulate_scenario_a():
    """场景 A：多个请求生成不同长度，对比 PagedAttention 与连续分配。"""
    print("=" * 60)
    print("Scenario A: Variable-length Requests")
    print("=" * 60)

    total_blocks = 20
    block_size = 16
    max_seq_len = 128
    total_capacity = total_blocks * block_size

    requests = [
        ("req_1", 10), ("req_2", 45), ("req_3", 80), ("req_4", 25), ("req_5", 60)
    ]

    # 1. 连续分配
    print("\n[Continuous Allocator]")
    cont = ContinuousAllocator(total_capacity, max_seq_len)
    success = 0
    for req_id, num_tokens in requests:
        if cont.allocate(num_tokens):
            success += 1
            print(f"  {req_id}: allocated {max_seq_len} tokens (actual need: {num_tokens})")
        else:
            print(f"  {req_id}: FAILED (used: {cont.used_capacity}/{total_capacity})")
    print(f"  Success: {success}/{len(requests)}")
    print(f"  Wasted capacity: {success * max_seq_len - sum(t for _, t in requests[:success])} tokens")

    # 2. PagedAttention 分配
    print("\n[PagedAttention Allocator]")
    pa = PagedAttentionKVCache(total_blocks, block_size)
    success = 0
    for req_id, num_tokens in requests:
        if pa.allocate_for_request(req_id, num_tokens):
            success += 1
            print(f"  {req_id}: allocated {num_tokens} tokens in blocks")
        else:
            print(f"  {req_id}: FAILED (OOM)")
    pa.print_state()
    usage = pa.get_memory_usage()
    print(f"  Success: {success}/{len(requests)}")
    print(f"  Utilization rate: {usage['utilization_rate']:.2%}")
    print(f"  Wasted tokens: {usage['wasted_tokens']} tokens")


def simulate_scenario_b():
    """场景 B：共享 system prompt 前缀。"""
    print("\n" + "=" * 60)
    print("Scenario B: Prefix Sharing")
    print("=" * 60)

    total_blocks = 20
    block_size = 16

    system_prompt_len = 32  # 例如 system prompt 占 32 个 token
    user_prompt_len = 20

    pa = PagedAttentionKVCache(total_blocks, block_size)

    # 第一个请求：计算 system prompt + user prompt
    req1_total = system_prompt_len + user_prompt_len
    pa.allocate_for_request("req_1", req1_total)
    print(f"\nreq_1 allocated {req1_total} tokens (system: {system_prompt_len}, user: {user_prompt_len})")

    # 第二个请求：共享 system prompt（32 tokens = 2 blocks）
    prefix_blocks = 2
    pa.share_prefix("req_1", "req_2", prefix_blocks)
    # 再为 req_2 分配自己的 user prompt
    pa.allocate_for_request("req_2", user_prompt_len)
    print(f"req_2 shared {prefix_blocks} blocks of system prompt, then allocated {user_prompt_len} tokens")

    pa.print_state()
    usage = pa.get_memory_usage()
    print(f"Utilization rate: {usage['utilization_rate']:.2%}")

    # 计算共享节省的显存
    shared_blocks = [b for b in pa.physical_blocks.values() if b.is_shared]
    saved_tokens = sum(b.occupied_tokens for b in shared_blocks) * (2 - 1)  # 2 请求共享，少复制 1 份
    print(f"Saved tokens by sharing: {saved_tokens}")


if __name__ == "__main__":
    simulate_scenario_a()
    simulate_scenario_b()

    print("\n" + "=" * 60)
    print("Summary:")
    print("- PagedAttention allocates blocks on-demand, reducing waste.")
    print("- Shared blocks via reference counting save memory for common prefixes.")
    print("- External fragmentation is eliminated because all blocks are same size.")
    print("=" * 60)
