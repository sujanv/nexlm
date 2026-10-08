import math
import torch
import pytest

from model.rope import RotaryEmbedding, apply_rotary_emb, rotate_half


def test_rope_shapes():
    B, n_heads, T, head_dim = 2, 4, 16, 32
    q = torch.randn(B, n_heads, T, head_dim)
    k = torch.randn(B, n_heads, T, head_dim)

    rope = RotaryEmbedding(dim=head_dim, max_seq_len=64)
    cos, sin = rope(q, start_pos=0)

    assert cos.shape == (1, 1, T, head_dim)
    assert sin.shape == (1, 1, T, head_dim)

    q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_norm_preservation():
    """Rotary embedding is an orthogonal transformation: it must preserve vector norms."""
    B, n_heads, T, head_dim = 1, 1, 5, 16
    q = torch.randn(B, n_heads, T, head_dim)
    k = torch.randn(B, n_heads, T, head_dim)

    rope = RotaryEmbedding(dim=head_dim, max_seq_len=32)
    cos, sin = rope(q)
    q_rot, _ = apply_rotary_emb(q, k, cos, sin)

    norm_orig = torch.norm(q, dim=-1)
    norm_rot = torch.norm(q_rot, dim=-1)
    assert torch.allclose(norm_orig, norm_rot, atol=1e-5), "RoPE altered vector Euclidean norms!"


def test_rope_relative_shift_invariance():
    """
    Key RoPE mathematical theorem:
    Inner product <R_m q, R_n k> depends strictly on relative offset (m - n).
    """
    head_dim = 16
    rope = RotaryEmbedding(dim=head_dim, max_seq_len=32)

    v1 = torch.randn(1, 1, 1, head_dim)
    v2 = torch.randn(1, 1, 1, head_dim)

    # Offset pair 1: positions (0, 3) -> distance 3
    cos0, sin0 = rope(v1, start_pos=0)
    cos3, sin3 = rope(v2, start_pos=3)
    v1_rot_0, _ = apply_rotary_emb(v1, v1, cos0, sin0)
    _, v2_rot_3 = apply_rotary_emb(v2, v2, cos3, sin3)
    dot_0_3 = (v1_rot_0 * v2_rot_3).sum()

    # Offset pair 2: positions (5, 8) -> distance 3
    cos5, sin5 = rope(v1, start_pos=5)
    cos8, sin8 = rope(v2, start_pos=8)
    v1_rot_5, _ = apply_rotary_emb(v1, v1, cos5, sin5)
    _, v2_rot_8 = apply_rotary_emb(v2, v2, cos8, sin8)
    dot_5_8 = (v1_rot_5 * v2_rot_8).sum()

    assert torch.allclose(dot_0_3, dot_5_8, atol=1e-4), "RoPE failed relative distance invariance!"
