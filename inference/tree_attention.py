"""
Milestone 47: Tree Attention & Medusa Multi-Head Speculative Decoding

Unlike standard linear speculative decoding (which guesses a single sequence of draft tokens),
Tree Attention (Cai et al., Medusa; Li et al., Eagle) constructs a 2D candidate tree of tokens.

Key Components:
1. Multi-Head Medusa Heads: Multiple lightweight linear projection heads predicting
   candidate tokens at offsets t+1, t+2, t+3...
2. Tree Topology & Flattening: Candidates are structured into a directed tree of alternative paths.
3. 2D Tree Attention Mask: Allows tree nodes to only attend to their causal ancestors:
       M_{i, j} = 0 if j is ancestor of i else -inf
4. Parallel Tree Verification: The target model verifies all candidate branches simultaneously
   in a single forward pass, accepting the longest valid path.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gpt import GPT


@dataclass
class TreeNode:
    node_id: int
    token_id: int
    parent_id: Optional[int]
    depth: int
    children: List[int] = field(default_factory=list)


class CandidateTree:
    """
    Constructs, flattens, and builds 2D attention masks for a speculative candidate tree.
    """

    def __init__(self, root_token_id: int):
        self.nodes: Dict[int, TreeNode] = {}
        self.root_id = 0
        self.nodes[0] = TreeNode(
            node_id=0,
            token_id=root_token_id,
            parent_id=None,
            depth=0,
        )
        self._next_id = 1

    def add_child(self, parent_id: int, token_id: int) -> int:
        child_id = self._next_id
        self._next_id += 1
        node = TreeNode(
            node_id=child_id,
            token_id=token_id,
            parent_id=parent_id,
            depth=self.nodes[parent_id].depth + 1,
        )
        self.nodes[child_id] = node
        self.nodes[parent_id].children.append(child_id)
        return child_id

    def get_ancestors(self, node_id: int) -> Set[int]:
        ancestors = set()
        curr = self.nodes.get(node_id)
        while curr is not None:
            ancestors.add(curr.node_id)
            if curr.parent_id is None:
                break
            curr = self.nodes.get(curr.parent_id)
        return ancestors

    def flatten(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Flattens the candidate tree into:
        1. token_ids: 1D tensor of all candidate tokens
        2. tree_mask: (N, N) bool tensor where tree_mask[i, j] = True if node j is ancestor of node i.
        """
        ordered_ids = sorted(self.nodes.keys())
        N = len(ordered_ids)
        token_list = [self.nodes[nid].token_id for nid in ordered_ids]
        tokens_tensor = torch.tensor(token_list, dtype=torch.long)

        # 2D Tree causal mask
        # tree_mask[i, j] = True if j is an ancestor of i (or j == i)
        id_to_idx = {nid: idx for idx, nid in enumerate(ordered_ids)}
        mask = torch.zeros((N, N), dtype=torch.bool)

        for nid, idx_i in id_to_idx.items():
            ancestors = self.get_ancestors(nid)
            for anc in ancestors:
                idx_j = id_to_idx[anc]
                mask[idx_i, idx_j] = True

        return tokens_tensor, mask

    def find_branches(self) -> List[List[int]]:
        """Returns all root-to-leaf paths as lists of node IDs."""
        branches: List[List[int]] = []

        def dfs(curr_id: int, current_path: List[int]):
            curr_node = self.nodes[curr_id]
            new_path = current_path + [curr_id]
            if not curr_node.children:
                branches.append(new_path)
                return
            for child_id in curr_node.children:
                dfs(child_id, new_path)

        dfs(self.root_id, [])
        return branches


class MedusaHead(nn.Module):
    """
    Auxiliary decoding head predicting candidates for offset t + k.
    Consists of residual block + linear head.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        res = self.act(self.linear(h))
        return self.head(h + res)


class MedusaModel(nn.Module):
    """
    Wraps base GPT model with multiple Medusa heads for tree speculative generation.
    """

    def __init__(self, base_model: GPT, num_heads: int = 3, top_k_per_head: int = 2):
        super().__init__()
        self.base_model = base_model
        self.num_heads = num_heads
        self.top_k_per_head = top_k_per_head

        self.medusa_heads = nn.ModuleList([
            MedusaHead(
                d_model=base_model.config.d_model,
                vocab_size=base_model.config.vocab_size,
            )
            for _ in range(num_heads)
        ])

    def generate_candidate_tree(self, last_hidden: torch.Tensor, last_token_id: int) -> CandidateTree:
        """
        Generates a 2D candidate tree from Medusa head predictions on the last hidden state.
        last_hidden: (1, 1, d_model)
        """
        tree = CandidateTree(root_token_id=last_token_id)
        current_layer_nodes = [tree.root_id]

        for head in self.medusa_heads:
            logits = head(last_hidden).squeeze(0).squeeze(0)  # (vocab_size,)
            topk_tokens = torch.topk(logits, min(self.top_k_per_head, logits.size(-1))).indices.tolist()

            next_layer_nodes = []
            for parent_node_id in current_layer_nodes:
                for token_id in topk_tokens:
                    child_id = tree.add_child(parent_node_id, token_id)
                    next_layer_nodes.append(child_id)
            current_layer_nodes = next_layer_nodes

        return tree


def verify_tree_candidates(
    tree: CandidateTree,
    target_logits: torch.Tensor,
) -> Tuple[List[int], int]:
    """
    Verifies tree candidate tokens against target model greedy predictions.
    Args:
        tree: CandidateTree containing paths
        target_logits: (N, vocab_size) where target_logits[i] is the target model's
                       next-token prediction after seeing node i's prefix.
    Returns:
        best_branch_tokens: The longest accepted list of token IDs along a valid branch.
        accepted_length: Number of accepted speculative tokens (excluding root).
    """
    branches = tree.find_branches()
    ordered_ids = sorted(tree.nodes.keys())
    id_to_idx = {nid: idx for idx, nid in enumerate(ordered_ids)}

    best_branch: List[int] = [tree.nodes[tree.root_id].token_id]
    max_accepted = 0

    for branch in branches:
        # branch: [root_id, c1_id, c2_id, ...]
        curr_accepted: List[int] = [tree.nodes[tree.root_id].token_id]
        all_matched = True
        for i in range(len(branch) - 1):
            parent_id = branch[i]
            child_id = branch[i + 1]

            parent_idx = id_to_idx[parent_id]
            pred_token = torch.argmax(target_logits[parent_idx]).item()
            expected_token = tree.nodes[child_id].token_id

            if pred_token == expected_token:
                curr_accepted.append(pred_token)
            else:
                # Add the corrected token predicted by target model and stop branch
                curr_accepted.append(pred_token)
                all_matched = False
                break

        if all_matched:
            # Append next prediction from the leaf node
            leaf_id = branch[-1]
            leaf_idx = id_to_idx[leaf_id]
            next_token = torch.argmax(target_logits[leaf_idx]).item()
            curr_accepted.append(next_token)

        accepted_tokens_count = len(curr_accepted) - 1
        if accepted_tokens_count > max_accepted:
            max_accepted = accepted_tokens_count
            best_branch = curr_accepted

    return best_branch, max_accepted
