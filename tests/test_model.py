import torch
import pytest

from model.gpt import GPT, GPTConfig


def test_gpt_forward_pass_and_shapes():
    config = GPTConfig(
        vocab_size=100,
        max_seq_len=32,
        d_model=32,
        n_heads=2,
        n_layers=2,
        dropout=0.0,
    )
    model = GPT(config)

    B, T = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    logits, loss, _ = model(input_ids)

    assert logits.shape == (B, T, config.vocab_size)
    assert loss is None


def test_gpt_loss_and_backward():
    config = GPTConfig(
        vocab_size=50,
        max_seq_len=16,
        d_model=16,
        n_heads=2,
        n_layers=2,
    )
    model = GPT(config)

    B, T = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    logits, loss, _ = model(input_ids, targets=targets)
    assert loss is not None
    assert loss.item() > 0.0

    loss.backward()
    assert model.lm_head.weight.grad is not None


def test_gpt_single_batch_overfitting():
    """
    Sanity check (Milestone 8 preview):
    A working model must be able to drive loss to near zero on a single repeated sequence.
    """
    torch.manual_seed(42)
    config = GPTConfig(
        vocab_size=20,
        max_seq_len=8,
        d_model=32,
        n_heads=2,
        n_layers=2,
        dropout=0.0,
    )
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # Toy sequence: [1, 2, 3, 4, 5] -> target [2, 3, 4, 5, 6]
    x = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    y = torch.tensor([[2, 3, 4, 5, 6]], dtype=torch.long)

    initial_loss = float("inf")
    final_loss = float("inf")

    for step in range(50):
        optimizer.zero_grad()
        _, loss, _ = model(x, targets=y)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    # The loss should drop dramatically
    assert final_loss < 0.1, f"Model failed to overfit simple batch: final_loss={final_loss}"
    assert final_loss < initial_loss
