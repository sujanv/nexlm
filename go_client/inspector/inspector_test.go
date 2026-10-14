package inspector

import (
	"os"
	"path/filepath"
	"testing"
)

func TestComputeDiagnostics(t *testing.T) {
	cfg := &ModelConfig{
		VocabSize:  5000,
		MaxSeqLen:  256,
		DModel:     256,
		NHeads:     4,
		NLayers:    4,
		Dropout:    0.1,
		TieWeights: true,
	}

	diag := ComputeDiagnostics(cfg)

	if diag.TotalParameters <= 0 {
		t.Fatalf("expected positive total parameters, got %d", diag.TotalParameters)
	}

	if diag.EmbeddingParams <= 0 {
		t.Fatalf("expected positive embedding parameters, got %d", diag.EmbeddingParams)
	}

	if diag.MemoryFP16MB <= 0 || diag.MemoryINT4MB <= 0 {
		t.Fatalf("expected positive memory estimates")
	}

	if diag.MemoryINT4MB >= diag.MemoryFP16MB {
		t.Fatalf("INT4 memory (%f) should be smaller than FP16 (%f)", diag.MemoryINT4MB, diag.MemoryFP16MB)
	}

	if diag.KVCachePerTokenKB <= 0 {
		t.Fatalf("expected positive KV cache per token")
	}
}

func TestInspectConfigFile(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.json")

	content := `{
		"vocab_size": 1000,
		"max_seq_len": 128,
		"d_model": 64,
		"n_heads": 2,
		"n_layers": 2,
		"dropout": 0.0,
		"tie_weights": true
	}`

	if err := os.WriteFile(configPath, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write temp config: %v", err)
	}

	cfg, err := InspectConfig(configPath)
	if err != nil {
		t.Fatalf("InspectConfig failed: %v", err)
	}

	if cfg.VocabSize != 1000 || cfg.DModel != 64 {
		t.Fatalf("parsed values mismatch: %+v", cfg)
	}
}
