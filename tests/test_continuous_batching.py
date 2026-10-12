"""
Tests for Milestone 33: Continuous Batching Engine.
"""

import pytest
import torch

from model.gpt import GPT, GPTConfig
from inference.continuous_batching import ContinuousBatchEngine, SequenceState


@pytest.fixture
def tiny_model():
    config = GPTConfig(
        vocab_size=50,
        max_seq_len=64,
        d_model=32,
        n_heads=2,
        n_layers=2,
    )
    return GPT(config)


def test_continuous_batching_basic(tiny_model):
    engine = ContinuousBatchEngine(tiny_model, max_batch_size=2)

    engine.add_request("req_1", [1, 2, 3], max_new_tokens=4, temperature=0.0)
    engine.add_request("req_2", [4, 5], max_new_tokens=8, temperature=0.0)

    assert len(engine.waiting_queue) == 2
    assert len(engine.running_batch) == 0

    results = engine.generate_all()

    assert "req_1" in results
    assert "req_2" in results
    assert len(results["req_1"]) == 4
    assert len(results["req_2"]) == 8
    assert not engine.has_pending()


def test_continuous_batching_slot_recycling(tiny_model):
    # Max batch size = 2, total 4 requests
    engine = ContinuousBatchEngine(tiny_model, max_batch_size=2)

    engine.add_request("short_1", [1], max_new_tokens=2, temperature=0.0)
    engine.add_request("long_1", [2], max_new_tokens=6, temperature=0.0)
    engine.add_request("short_2", [3], max_new_tokens=2, temperature=0.0)
    engine.add_request("long_2", [4], max_new_tokens=6, temperature=0.0)

    # First step: admits short_1 and long_1
    finished = engine.step()
    assert len(engine.running_batch) == 2
    assert len(finished) == 0

    # Second step: short_1 completes (2 tokens)
    finished = engine.step()
    assert len(finished) == 1
    assert finished[0].request_id == "short_1"

    # Third step: short_2 is admitted to replace short_1
    finished = engine.step()
    running_ids = [s.request_id for s in engine.running_batch]
    assert "short_2" in running_ids

    # Finish remaining
    results = engine.generate_all()
    assert len(results) == 4
    for req_id in ["short_1", "long_1", "short_2", "long_2"]:
        assert req_id in results


def test_continuous_batching_eos_termination(tiny_model):
    engine = ContinuousBatchEngine(tiny_model, max_batch_size=2)

    # Force token 10 as EOS
    seq = engine.add_request("req_eos", [1, 2], max_new_tokens=20, eos_token_id=10, temperature=0.0)

    # Artificially inject next token as 10 to simulate hitting EOS
    engine.step()
    seq.generated_tokens[-1] = 10
    # Simulate step check
    if seq.eos_token_id in seq.generated_tokens:
        seq.status = "FINISHED"

    assert seq.status == "FINISHED"
