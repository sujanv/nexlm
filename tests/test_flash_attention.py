import torch
import pytest

from model.flash_attention import FlashCausalSelfAttention


def test_flash_attention_shapes():
    B, T, d_model = 2, 8, 32
    attn = FlashCausalSelfAttention(d_model=d_model, n_heads=4, max_seq_len=16)
    x = torch.randn(B, T, d_model)

    out = attn(x)
    assert out.shape == (B, T, d_model)


def test_sdpa_vs_manual_backend_equivalence():
    """Verify that PyTorch hardware SDPA kernel matches manual matrix multiplication."""
    torch.manual_seed(42)
    B, T, d_model = 2, 8, 32
    attn = FlashCausalSelfAttention(d_model=d_model, n_heads=4, max_seq_len=16, dropout=0.0)
    attn.eval()

    x = torch.randn(B, T, d_model)

    out_sdpa = attn(x, backend_override="sdpa")
    out_manual = attn(x, backend_override="manual")

    assert torch.allclose(out_sdpa, out_manual, atol=1e-5), "SDPA output diverged from manual attention!"
