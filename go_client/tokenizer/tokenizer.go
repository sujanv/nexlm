package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
)

// Tokenizer represents a high-performance Go Byte-Level / Character tokenizer.
type Tokenizer struct {
	mu         sync.RWMutex
	Vocab      map[string]int `json:"vocab"`
	InverseVocab map[int]string `json:"inverse_vocab"`
	EosTokenID int            `json:"eos_token_id"`
	PadTokenID int            `json:"pad_token_id"`
}

// NewTokenizer initializes an empty tokenizer with default special tokens.
func NewTokenizer() *Tokenizer {
	t := &Tokenizer{
		Vocab:        make(map[string]int),
		InverseVocab: make(map[int]string),
		EosTokenID:   0,
		PadTokenID:   1,
	}
	t.Vocab["<|pad|>"] = 0
	t.InverseVocab[0] = "<|pad|>"
	t.Vocab["<|eos|>"] = 1
	t.InverseVocab[1] = "<|eos|>"
	return t
}

// BuildFromText builds a character-level vocabulary from training text.
func (t *Tokenizer) BuildFromText(text string) {
	t.mu.Lock()
	defer t.mu.Unlock()

	idx := len(t.Vocab)
	for _, r := range text {
		s := string(r)
		if _, exists := t.Vocab[s]; !exists {
			t.Vocab[s] = idx
			t.InverseVocab[idx] = s
			idx++
		}
	}
}

// LoadFromJSON loads tokenizer configuration and vocabulary from JSON file.
func LoadFromJSON(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read tokenizer file: %w", err)
	}

	var raw struct {
		Vocab      map[string]int `json:"vocab"`
		EosTokenID int            `json:"eos_token_id"`
		PadTokenID int            `json:"pad_token_id"`
	}

	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("failed to parse tokenizer JSON: %w", err)
	}

	t := &Tokenizer{
		Vocab:        raw.Vocab,
		InverseVocab: make(map[int]string, len(raw.Vocab)),
		EosTokenID:   raw.EosTokenID,
		PadTokenID:   raw.PadTokenID,
	}

	for k, v := range t.Vocab {
		t.InverseVocab[v] = k
	}

	return t, nil
}

// Encode converts raw text to integer token IDs.
func (t *Tokenizer) Encode(text string, addSpecial bool) []int {
	t.mu.RLock()
	defer t.mu.RUnlock()

	var tokens []int
	for _, r := range text {
		s := string(r)
		if id, ok := t.Vocab[s]; ok {
			tokens = append(tokens, id)
		} else {
			tokens = append(tokens, t.PadTokenID)
		}
	}

	if addSpecial {
		tokens = append(tokens, t.EosTokenID)
	}

	return tokens
}

// Decode reconstructs text from a slice of token IDs.
func (t *Tokenizer) Decode(tokens []int) string {
	t.mu.RLock()
	defer t.mu.RUnlock()

	var result string
	for _, id := range tokens {
		if id == t.EosTokenID || id == t.PadTokenID {
			continue
		}
		if s, ok := t.InverseVocab[id]; ok {
			result += s
		}
	}
	return result
}

// EncodeBatchConcurrently encodes an array of strings in parallel using a goroutine worker pool.
func (t *Tokenizer) EncodeBatchConcurrently(texts []string, numWorkers int) [][]int {
	if numWorkers <= 0 {
		numWorkers = 4
	}

	results := make([][]int, len(texts))
	jobs := make(chan int, len(texts))
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				results[idx] = t.Encode(texts[idx], false)
			}
		}()
	}

	for i := range texts {
		jobs <- i
	}
	close(jobs)

	wg.Wait()
	return results
}
