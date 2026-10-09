import torch
import torch.nn as nn
import pytest

from model.lora import LoRALinear, apply_lora_to_model, mark_only_lora_as_trainable
from model.gpt import GPT, GPTConfig


def test_lora_initial_output_identity():
    """At initialization, B is all zeros, so LoRA output must strictly equal base output."""
    torch.manual_seed(42)
    in_feat, out_feat = 32, 16
    base_linear = nn.Linear(in_feat, out_feat)
    x = torch.randn(4, 8, in_feat)

    base_out = base_linear(x)

    lora_linear = LoRALinear.from_linear(base_linear, r=4, alpha=8.0)
    lora_out = lora_linear(x)

    assert torch.allclose(base_out, lora_out, atol=1e-6), "Initial LoRA output did not match base linear!"


def test_lora_parameter_freezing():
    config = GPTConfig(vocab_size=100, max_seq_len=32, d_model=64, n_heads=2, n_layers=2)
    model = GPT(config)

    # Apply LoRA to attention projections
    replaced = apply_lora_to_model(model, target_modules=("c_attn",), r=4, alpha=8.0)
    assert len(replaced) > 0

    trainable_p, total_p = mark_only_lora_as_trainable(model)
    # LoRA trainable parameters should be a small fraction of the total parameters
    assert trainable_p < total_p * 0.1, f"Trainable params ({trainable_p}) should be < 10% of total ({total_p})"

    # Verify only LoRA params have requires_grad
    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad, f"{name} should require grad"
        else:
            assert not param.requires_grad, f"{name} should be frozen"


def test_lora_merge_and_unmerge():
    in_feat, out_feat = 16, 8
    linear = nn.Linear(in_feat, out_feat)
    lora = LoRALinear.from_linear(linear, r=2, alpha=4.0)

    # Assign non-zero values to B to simulate training
    with torch.no_grad():
        lora.lora_B.fill_(0.5)

    x = torch.randn(2, 4, in_feat)
    out_unmerged = lora(x)

    # Merge weights
    lora.merge()
    assert lora.merged
    out_merged = lora(x)
    assert torch.allclose(out_unmerged, out_merged, atol=1e-5), "Merged output differed from unmerged output!"

    # Unmerge weights
    lora.unmerge()
    assert not lora.merged
    out_restored = lora(x)
    assert torch.allclose(out_unmerged, out_restored, atol=1e-5), "Unmerging failed to restore state!"
