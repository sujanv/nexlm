import os
import tempfile
import pytest

from tokenizer.tokenizer import ByteLevelBPETokenizer, CharTokenizer


def test_bpe_tokenizer_roundtrip():
    text = "Hello world! 🌍 This is a test of byte-level BPE tokenization. The cat sat on the mat."
    tokenizer = ByteLevelBPETokenizer(vocab_size=300)
    tokenizer.train(text, verbose=False)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text
    assert len(encoded) > 0


def test_bpe_save_load():
    text = "The quick brown fox jumps over the lazy dog."
    tok1 = ByteLevelBPETokenizer(vocab_size=280)
    tok1.train(text, verbose=False)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name

    try:
        tok1.save(path)
        tok2 = ByteLevelBPETokenizer.load(path)

        assert tok1.vocab_size == tok2.vocab_size
        assert tok1.encode(text) == tok2.encode(text)
        assert tok2.decode(tok2.encode(text)) == text
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_char_tokenizer():
    text = "abc def 123"
    tok = CharTokenizer(text)
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text
