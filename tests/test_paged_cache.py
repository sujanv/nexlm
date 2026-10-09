import pytest
from inference.paged_cache import PagedKVCacheManager, BlockAllocator, BlockTable


def test_block_allocation_and_paging():
    allocator = BlockAllocator(total_blocks=10, block_size=4)
    table = BlockTable(seq_id="req-1", block_size=4)

    # Add 4 tokens -> should take 1 block
    for _ in range(4):
        table.append_token(allocator)
    assert len(table.physical_block_ids) == 1
    assert allocator.num_free_blocks() == 9

    # Add 5th token -> crosses block boundary, must allocate 2nd block
    table.append_token(allocator)
    assert len(table.physical_block_ids) == 2
    assert allocator.num_free_blocks() == 8

    # Verify physical address translation
    block_id, offset = table.get_physical_location(logical_pos=4)
    assert block_id == table.physical_block_ids[1]
    assert offset == 0


def test_paged_kv_manager_lifecycle():
    manager = PagedKVCacheManager(total_blocks=16, block_size=16)

    # Start sequence with 20 tokens -> requires 2 blocks (16 + 4)
    table = manager.start_sequence(seq_id="seq-42", prompt_tokens=20)
    assert len(table.physical_block_ids) == 2
    assert manager.allocator.num_free_blocks() == 14

    # Append 1 token
    block_id, offset = manager.append_generated_token(seq_id="seq-42")
    assert offset == 4  # 21st token is at index 20 (block 1, offset 4)

    # Finish sequence -> must reclaim both blocks
    manager.finish_sequence(seq_id="seq-42")
    assert manager.allocator.num_free_blocks() == 16
    assert manager.allocator.memory_utilization() == 0.0
