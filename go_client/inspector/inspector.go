package inspector

import (
	"encoding/json"
	"fmt"
	"os"
)

// ModelConfig represents architectural parameters of a NexLM model.
type ModelConfig struct {
	VocabSize   int     `json:"vocab_size"`
	MaxSeqLen   int     `json:"max_seq_len"`
	DModel      int     `json:"d_model"`
	NHeads      int     `json:"n_heads"`
	NLayers     int     `json:"n_layers"`
	Dropout     float64 `json:"dropout"`
	TieWeights  bool    `json:"tie_weights"`
	NormType    string  `json:"norm_type"`
}

// ModelDiagnostics summarizes parameter counts and memory estimates.
type ModelDiagnostics struct {
	TotalParameters    int64   `json:"total_parameters"`
	EmbeddingParams    int64   `json:"embedding_params"`
	AttentionParams    int64   `json:"attention_params"`
	FeedForwardParams  int64   `json:"feed_forward_params"`
	MemoryFP32MB       float64 `json:"memory_fp32_mb"`
	MemoryFP16MB       float64 `json:"memory_fp16_mb"`
	MemoryINT8MB       float64 `json:"memory_int8_mb"`
	MemoryINT4MB       float64 `json:"memory_int4_mb"`
	KVCachePerTokenKB  float64 `json:"kv_cache_per_token_kb"`
}

// InspectConfig loads and validates a JSON model configuration.
func InspectConfig(path string) (*ModelConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("unable to read config file: %w", err)
	}

	var cfg ModelConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("invalid json in config: %w", err)
	}

	if cfg.VocabSize <= 0 || cfg.DModel <= 0 || cfg.NLayers <= 0 {
		return nil, fmt.Errorf("invalid model config dimensions")
	}

	return &cfg, nil
}

// ComputeDiagnostics calculates theoretical parameter counts and memory requirements.
func ComputeDiagnostics(cfg *ModelConfig) *ModelDiagnostics {
	// 1. Embedding parameters: token emb + pos emb
	tokEmb := int64(cfg.VocabSize) * int64(cfg.DModel)
	posEmb := int64(cfg.MaxSeqLen) * int64(cfg.DModel)
	embParams := tokEmb + posEmb

	// 2. Transformer Layer components:
	// Attention: Q, K, V projections + Out projection = 4 * (d_model * d_model)
	layerAttn := 4 * int64(cfg.DModel) * int64(cfg.DModel)

	// FeedForward: c_fc (d_model -> 4*d_model) + c_proj (4*d_model -> d_model) = 8 * d_model^2
	layerFFN := 8 * int64(cfg.DModel) * int64(cfg.DModel)

	// LayerNorm / RMSNorm parameters per block
	layerNorm := 4 * int64(cfg.DModel)

	totalLayerParams := int64(cfg.NLayers) * (layerAttn + layerFFN + layerNorm)

	// 3. Final head:
	var headParams int64
	if !cfg.TieWeights {
		headParams = int64(cfg.DModel) * int64(cfg.VocabSize)
	}

	totalParams := embParams + totalLayerParams + headParams

	// Bytes per param: FP32 = 4, FP16 = 2, INT8 = 1, INT4 = 0.5
	memFP32 := float64(totalParams*4) / (1024 * 1024)
	memFP16 := float64(totalParams*2) / (1024 * 1024)
	memINT8 := float64(totalParams*1) / (1024 * 1024)
	memINT4 := float64(totalParams) * 0.5 / (1024 * 1024)

	// KV Cache per token: 2 * n_layers * d_model * sizeof(fp16)
	kvCachePerToken := float64(2 * cfg.NLayers * cfg.DModel * 2) / 1024.0

	return &ModelDiagnostics{
		TotalParameters:   totalParams,
		EmbeddingParams:   embParams,
		AttentionParams:   int64(cfg.NLayers) * layerAttn,
		FeedForwardParams: int64(cfg.NLayers) * layerFFN,
		MemoryFP32MB:      memFP32,
		MemoryFP16MB:      memFP16,
		MemoryINT8MB:      memINT8,
		MemoryINT4MB:      memINT4,
		KVCachePerTokenKB: kvCachePerToken,
	}
}
