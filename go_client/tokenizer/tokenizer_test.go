package tokenizer

import (
	"reflect"
	"testing"
)

func TestTokenizerRoundtrip(t *testing.T) {
	tok := NewTokenizer()
	tok.BuildFromText("abcdefghijklmnopqrstuvwxyz 0123456789")

	input := "hello world 123"
	tokens := tok.Encode(input, false)

	if len(tokens) != len(input) {
		t.Fatalf("expected token length %d, got %d", len(input), len(tokens))
	}

	decoded := tok.Decode(tokens)
	if decoded != input {
		t.Fatalf("expected decoded text '%s', got '%s'", input, decoded)
	}
}

func TestSpecialTokens(t *testing.T) {
	tok := NewTokenizer()
	tok.BuildFromText("abc")

	tokens := tok.Encode("ab", true)
	if len(tokens) != 3 {
		t.Fatalf("expected 3 tokens with EOS, got %d", len(tokens))
	}

	if tokens[2] != tok.EosTokenID {
		t.Fatalf("expected last token to be EOS (%d), got %d", tok.EosTokenID, tokens[2])
	}
}

func TestConcurrentBatchEncoding(t *testing.T) {
	tok := NewTokenizer()
	tok.BuildFromText("abcdefghijklmnopqrstuvwxyz")

	inputs := []string{
		"apple",
		"banana",
		"cherry",
		"date",
		"elderberry",
	}

	batchTokens := tok.EncodeBatchConcurrently(inputs, 2)
	if len(batchTokens) != len(inputs) {
		t.Fatalf("expected %d results, got %d", len(inputs), len(batchTokens))
	}

	for i, in := range inputs {
		seq := tok.Encode(in, false)
		if !reflect.DeepEqual(seq, batchTokens[i]) {
			t.Fatalf("batch result mismatch for '%s': expected %v, got %v", in, seq, batchTokens[i])
		}
	}
}
