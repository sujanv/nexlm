"""
Tests for Milestone 36: Rejection Sampling & Best-of-N Alignment Engine.
"""

import pytest
import torch

from model.gpt import GPT, GPTConfig
from tokenizer.tokenizer import CharTokenizer
from training.rejection_sampling import (
    HeuristicRewardEvaluator,
    RejectionSamplingEngine,
)


def test_heuristic_reward_evaluator():
    evaluator = HeuristicRewardEvaluator(
        target_length=5,
        positive_keywords=["helpful", "awesome"],
        negative_keywords=["toxic", "bad"],
    )

    good_text = "This is helpful and awesome"
    bad_text = "This is toxic and bad bad bad"

    score_good, meta_good = evaluator.score("prompt", good_text)
    score_bad, meta_bad = evaluator.score("prompt", bad_text)

    assert score_good > score_bad
    assert meta_good["keyword_bonus"] > meta_bad["keyword_bonus"]


def test_rejection_sampling_engine():
    config = GPTConfig(
        vocab_size=60,
        max_seq_len=64,
        d_model=32,
        n_heads=2,
        n_layers=2,
    )
    model = GPT(config)
    tokenizer = CharTokenizer()
    tokenizer.train("abcdefghijklmnopqrstuvwxyz 0123456789.!?")

    evaluator = HeuristicRewardEvaluator(
        target_length=4,
        positive_keywords=["a", "b"],
    )
    engine = RejectionSamplingEngine(
        model=model,
        tokenizer=tokenizer,
        reward_evaluator=evaluator,
    )

    best, candidates = engine.sample_best_of_n(
        prompt="hello",
        num_candidates=3,
        max_new_tokens=6,
        temperature=0.8,
    )

    assert len(candidates) == 3
    assert best == candidates[0]
    # Check descending order of reward
    assert candidates[0].reward_score >= candidates[1].reward_score
    assert candidates[1].reward_score >= candidates[2].reward_score


def test_build_aligned_pairs():
    config = GPTConfig(
        vocab_size=60,
        max_seq_len=64,
        d_model=32,
        n_heads=2,
        n_layers=2,
    )
    model = GPT(config)
    tokenizer = CharTokenizer()
    tokenizer.train("abcdefghijklmnopqrstuvwxyz 0123456789.!?")

    engine = RejectionSamplingEngine(model=model, tokenizer=tokenizer)
    prompts = ["cat", "dog"]

    pairs = engine.build_aligned_pairs(prompts, num_candidates=2, max_new_tokens=4)
    assert len(pairs) == 2
    for item in pairs:
        assert "prompt" in item
        assert "chosen" in item
        assert "rejected" in item
        assert item["chosen_reward"] >= item["rejected_reward"]
