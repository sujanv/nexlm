import pytest
from benchmarks.needle_in_haystack import construct_haystack, insert_needle, HAYSTACK_FILLER
from tokenizer.tokenizer import ByteLevelBPETokenizer


def test_haystack_construction_and_insertion():
    tokenizer = ByteLevelBPETokenizer(vocab_size=300)
    tokenizer.train("".join(HAYSTACK_FILLER * 5), verbose=False)

    haystack = construct_haystack(target_tokens=50, tokenizer=tokenizer)
    assert len(tokenizer.encode(haystack)) >= 50

    needle = "The secret code is 12345."
    # Insert at 50% depth
    modified = insert_needle(haystack, needle, depth_fraction=0.5)
    assert needle in modified
