"""
Tests for Milestone 47: Tree Attention & Medusa Multi-Head Speculative Decoding.
"""

import pytest
import torch

from model.gpt import GPT, GPTConfig
from inference.tree_attention import (
    CandidateTree,
    MedusaHead,
    MedusaModel,
    verify_tree_candidates,
)


def test_candidate_tree_mask_and_ancestry():
    # Construct a small candidate tree:
    # Root: token 10
    #  ├── child 1: token 20
    #  │     └── child 11: token 30
    #  └── child 2: token 40
    tree = CandidateTree(root_token_id=10)
    c1 = tree.add_child(tree.root_id, 20)
    c11 = tree.add_child(c1, 30)
    c2 = tree.add_child(tree.root_id, 40)

    tokens, mask = tree.flatten()
    assert tokens.tolist() == [10, 20, 30, 40]
    assert mask.shape == (4, 4)

    # Root can only see root
    assert mask[0, 0].item() is True
    assert mask[0, 1].item() is False
    assert mask[0, 2].item() is False
    assert mask[0, 3].item() is False

    # c1 (node 1) can see root (0) and c1 (1)
    assert mask[1, 0].item() is True
    assert mask[1, 1].item() is True
    assert mask[1, 2].item() is False  # cannot see child 11
    assert mask[1, 3].item() is False  # cannot see sibling 2

    # c11 (node 2) can see root (0), c1 (1), and c11 (2)
    assert mask[2, 0].item() is True
    assert mask[2, 1].item() is True
    assert mask[2, 2].item() is True
    assert mask[2, 3].item() is False  # cannot see c2

    # c2 (node 3) can see root (0) and c2 (3)
    assert mask[3, 0].item() is True
    assert mask[3, 1].item() is False
    assert mask[3, 2].item() is False
    assert mask[3, 3].item() is True


def test_tree_branches():
    tree = CandidateTree(root_token_id=1)
    c1 = tree.add_child(tree.root_id, 2)
    tree.add_child(c1, 4)
    tree.add_child(c1, 5)
    c2 = tree.add_child(tree.root_id, 3)

    branches = tree.find_branches()
    # Expect 3 branches: [0, 1, 2], [0, 1, 3], [0, 4]
    assert len(branches) == 3


def test_medusa_model_generate_tree():
    config = GPTConfig(vocab_size=50, max_seq_len=64, d_model=32, n_heads=2, n_layers=2)
    base_gpt = GPT(config)
    medusa = MedusaModel(base_gpt, num_heads=2, top_k_per_head=2)

    hidden = torch.randn(1, 1, 32)
    tree = medusa.generate_candidate_tree(hidden, last_token_id=5)

    tokens, mask = tree.flatten()
    # Root: 1
    # Layer 1: 2 children
    # Layer 2: 2 * 2 = 4 children
    # Total nodes = 1 + 2 + 4 = 7
    assert len(tokens) == 7
    assert mask.shape == (7, 7)


def test_verify_tree_candidates():
    tree = CandidateTree(root_token_id=100)
    c1 = tree.add_child(tree.root_id, 101)  # Correct branch
    c11 = tree.add_child(c1, 102)          # Correct branch
    c2 = tree.add_child(tree.root_id, 999)  # Incorrect branch

    tokens, mask = tree.flatten()
    N = len(tokens)

    # Mock target logits where:
    # At root (idx 0), argmax is 101
    # At c1 (idx 1), argmax is 102
    # At c11 (idx 2), argmax is 103
    target_logits = torch.zeros((N, 200))
    target_logits[0, 101] = 10.0
    target_logits[1, 102] = 10.0
    target_logits[2, 103] = 10.0
    target_logits[3, 50] = 10.0

    best_branch, accepted = verify_tree_candidates(tree, target_logits)
    # Root + c1 (101) + c11 (102) accepted, plus target's next token (103)
    assert best_branch == [100, 101, 102, 103]
    assert accepted == 3
