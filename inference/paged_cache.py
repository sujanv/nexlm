"""
Enhancement: Paged Attention & Block-Based KV Cache Manager (vLLM Architecture)

Why Paged Attention?
Kwon et al. (SOSP 2023) identified that in traditional LLM serving:
1. Over 60%-80% of GPU memory is wasted because KV caches are allocated contiguously
   for the maximum possible sequence length before generation even finishes.
2. Memory fragmentation prevents concurrent request batching.

Solution:
Inspired by virtual memory paging in operating systems, Paged Attention splits the KV cache
into fixed-size memory blocks (e.g., 16 tokens per block).
The BlockManager dynamically allocates physical blocks on demand and maps logical sequence
tokens to non-contiguous physical blocks via a block table.
"""

from typing import List, Dict, Tuple, Optional
import torch


class PhysicalTokenBlock:
    """Represents a single allocated fixed-size physical memory block."""

    def __init__(self, block_id: int, block_size: int = 16):
        self.block_id = block_id
        self.block_size = block_size
        self.ref_count = 0

    def is_full(self, num_tokens_in_block: int) -> bool:
        return num_tokens_in_block >= self.block_size


class BlockTable:
    """Maintains logical-to-physical block mappings for a single generation sequence."""

    def __init__(self, seq_id: str, block_size: int = 16):
        self.seq_id = seq_id
        self.block_size = block_size
        self.physical_block_ids: List[int] = []
        self.num_tokens = 0

    def append_token(self, allocator: "BlockAllocator") -> None:
        """Appends one token to the sequence, allocating a new physical block if current block is full."""
        if self.num_tokens % self.block_size == 0:
            # Current block is full or sequence is brand new; allocate a new physical block
            new_block_id = allocator.allocate()
            self.physical_block_ids.append(new_block_id)
        self.num_tokens += 1

    def get_physical_location(self, logical_pos: int) -> Tuple[int, int]:
        """
        Translates a logical token index to (physical_block_id, offset_within_block).
        """
        assert 0 <= logical_pos < self.num_tokens, f"Logical position {logical_pos} out of range"
        block_idx = logical_pos // self.block_size
        offset = logical_pos % self.block_size
        return self.physical_block_ids[block_idx], offset


class BlockAllocator:
    """Manages pool of free and allocated physical memory blocks."""

    def __init__(self, total_blocks: int = 64, block_size: int = 16):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.free_block_ids: List[int] = list(range(total_blocks))
        self.allocated_blocks: Dict[int, PhysicalTokenBlock] = {}

    def allocate(self) -> int:
        if not self.free_block_ids:
            raise MemoryError("Out of physical KV blocks in block pool!")
        block_id = self.free_block_ids.pop(0)
        block = PhysicalTokenBlock(block_id=block_id, block_size=self.block_size)
        block.ref_count = 1
        self.allocated_blocks[block_id] = block
        return block_id

    def free(self, block_id: int) -> None:
        if block_id in self.allocated_blocks:
            self.allocated_blocks[block_id].ref_count -= 1
            if self.allocated_blocks[block_id].ref_count <= 0:
                del self.allocated_blocks[block_id]
                self.free_block_ids.append(block_id)

    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def memory_utilization(self) -> float:
        """Fraction of total physical blocks currently in use."""
        used = self.total_blocks - len(self.free_block_ids)
        return used / self.total_blocks


class PagedKVCacheManager:
    """
    High-level orchestrator managing block tables and memory allocation
    for multiple concurrent serving sequences.
    """

    def __init__(self, total_blocks: int = 64, block_size: int = 16):
        self.block_size = block_size
        self.allocator = BlockAllocator(total_blocks=total_blocks, block_size=block_size)
        self.sequences: Dict[str, BlockTable] = {}

    def start_sequence(self, seq_id: str, prompt_tokens: int) -> BlockTable:
        table = BlockTable(seq_id=seq_id, block_size=self.block_size)
        for _ in range(prompt_tokens):
            table.append_token(self.allocator)
        self.sequences[seq_id] = table
        return table

    def append_generated_token(self, seq_id: str) -> Tuple[int, int]:
        table = self.sequences[seq_id]
        table.append_token(self.allocator)
        return table.get_physical_location(table.num_tokens - 1)

    def finish_sequence(self, seq_id: str) -> None:
        if seq_id in self.sequences:
            table = self.sequences.pop(seq_id)
            for block_id in table.physical_block_ids:
                self.allocator.free(block_id)
