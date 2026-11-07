# ⚡ NexLM: Build a Small LLM From Scratch in PyTorch

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python)](https://python.org)
[![Go](https://img.shields.io/badge/Go-1.20+-00ADD8.svg?style=flat&logo=go)](https://go.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Tests-110%20Passing-brightgreen.svg)]()
[![Commits](https://img.shields.io/badge/Milestones-52%20Commits-blue.svg)]()

**NexLM** is a comprehensive, production-grade, yet educational GPT-style Transformer language model implemented completely from scratch using PyTorch primitives.

From fundamental tokenization and self-attention up to cutting-edge frontier techniques—including **Root Mean Square Normalization (RMSNorm)**, **Rotary Embeddings (RoPE)**, **YaRN Context Window Scaling**, **DeepSeek Multi-Head Latent Attention (MLA)**, **SwiGLU**, **Grouped-Query Attention (GQA)**, **Sliding Window Attention (SWA)**, **Multi-Token Prediction (MTP)**, **Medusa Tree Attention Speculative Decoding**, **KTO (Kahneman-Tversky Optimization) Alignment**, **QLoRA NF4 Quantization**, **Chunked Prefill & Attention Sinks (StreamingLLM)**, **Min-P & Typical Sampling**, **LoRA fine-tuning**, **Direct Preference Optimization (DPO)**, **Rejection Sampling / Best-of-N Alignment**, **SLERP & DARE Model Merging**, **Streaming Tool / Function Calling**, **Continuous Batching**, **vLLM-style Paged Attention**, **Fine-Grained Block-Wise Quantization (.nexq)**, and **High-Throughput Go Gateway Router, Rate Limiter, Tokenization & Architecture Inspection**—NexLM is designed to be the ultimate ML/AI engineering reference.

---

## 📂 Repository Structure

```
NexLM/
├── model/
│   ├── attention.py           # Single-Head & Multi-Head Causal Self-Attention
│   ├── rmsnorm.py             # Root Mean Square Normalization (RMSNorm - LLaMA/Mistral style)
│   ├── transformer.py         # LayerNorm / RMSNorm, MLP (GELU), Pre-LN TransformerBlock
│   ├── gpt.py                 # Full GPT Decoder Architecture & Weight Tying
│   ├── rope.py                # Rotary Position Embeddings (RoPE) from scratch
│   ├── yarn.py                # YaRN (Yet another RoPE extensioN) Context Scaling
│   ├── mla.py                 # DeepSeek Multi-Head Latent Attention (low-rank KV compression)
│   ├── swiglu.py              # SwiGLU Gated Feed-Forward Network
│   ├── gqa.py                 # Grouped-Query (GQA) & Multi-Query Attention (MQA)
│   ├── sliding_window.py      # Mistral Sliding Window Attention & Rolling Buffer
│   ├── flash_attention.py     # Hardware-accelerated SDPA / FlashAttention wrapper
│   ├── lora.py                # Low-Rank Adaptation (LoRA) for PEFT with merge/unmerge
│   ├── moe.py                 # Sparsely-Gated Mixture of Experts with Top-K router
│   └── mtp.py                 # Multi-Token Prediction (MTP) for speculative self-drafting
├── tokenizer/
│   └── tokenizer.py           # CharTokenizer (intuitive) & ByteLevelBPETokenizer (zero OOV)
├── training/
│   ├── train.py               # Main training pipeline with AdamW & LR cosine schedule
│   ├── evaluate.py            # Evaluation & prompt completion CLI
│   ├── amp.py                 # Automatic Mixed Precision (AMP) training manager
│   ├── dpo.py                 # Direct Preference Optimization (DPO) alignment engine
│   ├── kto.py                 # Kahneman-Tversky Optimization (KTO) unpaired alignment engine
│   ├── qlora.py               # QLoRA 4-bit NormalFloat4 (NF4) quantization & adapters
│   ├── checkpointing.py       # Activation / Gradient Checkpointing memory optimizer
│   ├── rejection_sampling.py  # Best-of-N Rejection Sampling & Reward Model alignment
│   └── model_merging.py       # SLERP & DARE / Task Arithmetic post-training model merging
├── inference/
│   ├── generate.py            # Autoregressive decoding (greedy, temp, top-k, top-p, KV cache)
│   ├── sampler.py             # Temperature scaling, Top-K, and Top-P filtering
│   ├── advanced_sampler.py    # Min-P, Locally Typical, and Frequency/Presence Penalties
│   ├── tree_attention.py      # Medusa / Eagle Multi-Head Tree Attention Speculative Decoding
│   ├── chunked_prefill.py     # Long-Context Chunked Prefill & Attention Sinks (StreamingLLM)
│   ├── quantize.py            # Post-Training Quantization (INT8 & INT4 weight-only)
│   ├── block_quantization.py  # Fine-grained Block-Wise Quantization (.nexq format)
│   ├── continuous_batching.py # Iteration-level Continuous Batching serving engine
│   ├── speculative.py         # Speculative Decoding engine (2x-3x faster inference)
│   ├── contrastive_decoding.py# Contrastive Decoding & Classifier-Free Guidance (CFG)
│   ├── tool_calling.py        # Streaming Tool / Function Calling Parser & Dispatch Engine
│   ├── paged_cache.py         # vLLM-style Paged Attention Block Manager
│   ├── prefix_cache.py        # Prompt / Prefix Caching engine with SHA-256 fingerprinting
│   └── json_constrained.py    # Schema-Constrained Structured JSON decoding engine
├── server/
│   └── server.py              # OpenAI-compatible SSE Streaming API & Web Playground
├── go_client/
│   ├── go.mod                 # Golang module definition
│   ├── main.go                # High-throughput Go streaming client & benchmark CLI
│   ├── router/                # High-Throughput Go Request Router & Token Bucket Rate Limiter
│   ├── tokenizer/             # Concurrent Go Byte-Level Tokenizer & Vocab Parser
│   └── inspector/             # Go Model Diagnostics & Parameter/Memory Layout Inspector
├── cli/
│   └── chat.py                # Interactive Terminal Chat REPL with ANSI streaming
├── configs/
│   ├── tiny.yaml              # Default tiny configuration
│   ├── tinystories_tiny.json  # ~4.5M parameter preset
│   └── tinystories_small.json # ~15M parameter preset
├── benchmarks/
│   ├── bench_kv_cache.py      # KV-Cache latency & throughput benchmark
│   └── needle_in_haystack.py  # Needle In A Haystack (NIAH) context retrieval test
├── data/
│   └── download_tinystories.py# TinyStories dataset downloader & generator
├── tests/                     # Comprehensive test suite (110 unit tests passing)
├── demo.py                    # One-command training & generation demo
└── README.md
```

---

## 🏗️ The 52 Milestones & Capabilities

| Phase | Milestone | File(s) | Description |
|:---|:---|:---|:---|
| **Core Architecture** | `01-tokenizer` | [`tokenizer.py`](file:///Users/sujan/Public/NexLM/tokenizer/tokenizer.py) | Character & Byte-Level BPE tokenizers with zero `<unk>` tokens |
| | `02-self-attention` | [`attention.py`](file:///Users/sujan/Public/NexLM/model/attention.py) | Mathematical single-head causal self-attention with leakage tests |
| | `04-transformer-block`| [`transformer.py`](file:///Users/sujan/Public/NexLM/model/transformer.py) | Pre-LayerNorm residuals, LayerNorm, and 4x MLP |
| | `05-gpt` | [`gpt.py`](file:///Users/sujan/Public/NexLM/model/gpt.py) | Complete decoder-only GPT with embedding weight tying |
| | `06-toy-training` | [`train.py`](file:///Users/sujan/Public/NexLM/training/train.py) | Single-batch toy dataset overfit verification mode (`--toy`) |
| | `07-tinystories` | [`download_tinystories.py`](file:///Users/sujan/Public/NexLM/data/download_tinystories.py)| TinyStories dataset downloader and synthetic generator |
| | `08-generation` | [`generate.py`](file:///Users/sujan/Public/NexLM/inference/generate.py) | Temperature, Top-K, and Top-P (nucleus) sampling |
| | `09-kv-cache` | [`generate.py`](file:///Users/sujan/Public/NexLM/inference/generate.py) | Key-Value caching for $O(1)$ autoregressive step complexity |
| **Modern LLM Architectures** | `10-rope-embeddings`| [`rope.py`](file:///Users/sujan/Public/NexLM/model/rope.py) | Rotary Position Embeddings (RoPE) relative distance rotation |
| | `11-swiglu` | [`swiglu.py`](file:///Users/sujan/Public/NexLM/model/swiglu.py) | SwiGLU gated linear activation used in Llama 3 & Mistral |
| | `12-gqa` | [`gqa.py`](file:///Users/sujan/Public/NexLM/model/gqa.py) | Grouped-Query Attention cutting KV cache memory by $4\times - 8\times$ |
| | `19-moe` | [`moe.py`](file:///Users/sujan/Public/NexLM/model/moe.py) | Sparsely-Gated Mixture of Experts with Top-K routing & load balancing |
| | `25-sliding-window` | [`sliding_window.py`](file:///Users/sujan/Public/NexLM/model/sliding_window.py)| Mistral Sliding Window Attention with fixed circular rolling KV buffer |
| | `26-flash-attention`| [`flash_attention.py`](file:///Users/sujan/Public/NexLM/model/flash_attention.py)| PyTorch SDPA hardware kernel integration with manual fallback |
| | `32-rmsnorm` | [`rmsnorm.py`](file:///Users/sujan/Public/NexLM/model/rmsnorm.py) | Root Mean Square Normalization (RMSNorm) eliminating mean-centering |
| | `34-mtp` | [`mtp.py`](file:///Users/sujan/Public/NexLM/model/mtp.py) | Multi-Token Prediction (MTP) for multi-step planning and self-drafting |
| | `39-yarn-context` | [`yarn.py`](file:///Users/sujan/Public/NexLM/model/yarn.py) | YaRN RoPE context window scaling up to $16\times$ without fine-tuning |
| | `40-mla` | [`mla.py`](file:///Users/sujan/Public/NexLM/model/mla.py) | DeepSeek Multi-Head Latent Attention reducing KV cache memory by $85\%+$ |
| **Training & Alignment**| `13-amp` | [`amp.py`](file:///Users/sujan/Public/NexLM/training/amp.py) | FP16/BF16 Automatic Mixed Precision with dynamic `GradScaler` |
| | `18-lora` | [`lora.py`](file:///Users/sujan/Public/NexLM/model/lora.py) | Low-Rank Adaptation (LoRA) fine-tuning $<1\%$ parameters with `merge()` |
| | `21-dpo` | [`dpo.py`](file:///Users/sujan/Public/NexLM/training/dpo.py) | Direct Preference Optimization for RLHF alignment |
| | `29-checkpointing` | [`checkpointing.py`](file:///Users/sujan/Public/NexLM/training/checkpointing.py)| Gradient / Activation Checkpointing reducing peak activation VRAM |
| | `36-rejection-sampling`| [`rejection_sampling.py`](file:///Users/sujan/Public/NexLM/training/rejection_sampling.py)| Best-of-N Rejection Sampling with reward heuristics & pair curation |
| | `42-model-merging` | [`model_merging.py`](file:///Users/sujan/Public/NexLM/training/model_merging.py)| Spherical Linear Interpolation (SLERP) and DARE Task Arithmetic |
| | `46-kto-alignment` | [`kto.py`](file:///Users/sujan/Public/NexLM/training/kto.py) | Kahneman-Tversky Optimization (KTO) unpaired binary feedback alignment |
| | `49-qlora-nf4` | [`qlora.py`](file:///Users/sujan/Public/NexLM/training/qlora.py) | 4-bit NormalFloat4 (NF4) quantile quantizer and adapter layer |
| **Inference Optimization**| `14-quantization` | [`quantize.py`](file:///Users/sujan/Public/NexLM/inference/quantize.py) | Post-Training INT8 and INT4 weight-only quantization |
| | `20-speculative` | [`speculative.py`](file:///Users/sujan/Public/NexLM/inference/speculative.py) | Speculative Decoding achieving $2\times - 3\times$ speedup via draft model |
| | `22-paged-cache` | [`paged_cache.py`](file:///Users/sujan/Public/NexLM/inference/paged_cache.py) | vLLM-style Paged Attention virtual memory block manager |
| | `27-prefix-cache` | [`prefix_cache.py`](file:///Users/sujan/Public/NexLM/inference/prefix_cache.py)| SHA-256 prompt prefix caching with LRU cache eviction |
| | `28-json-constrained`| [`json_constrained.py`](file:///Users/sujan/Public/NexLM/inference/json_constrained.py)| Grammar logit masking ensuring 100% syntactically valid JSON |
| | `33-continuous-batching`| [`continuous_batching.py`](file:///Users/sujan/Public/NexLM/inference/continuous_batching.py)| Iteration-level scheduling eliminating batch waiting starvation |
| | `35-block-quant` | [`block_quantization.py`](file:///Users/sujan/Public/NexLM/inference/block_quantization.py)| Fine-grained 32-element Block-Wise Quantization (.nexq format) |
| | `41-contrastive` | [`contrastive_decoding.py`](file:///Users/sujan/Public/NexLM/inference/contrastive_decoding.py)| Contrastive Decoding & Classifier-Free Guidance (CFG) for sharp reasoning |
| | `43-tool-calling` | [`tool_calling.py`](file:///Users/sujan/Public/NexLM/inference/tool_calling.py)| Streaming Tool / Function Calling pushdown parser and dispatcher |
| | `47-tree-attention` | [`tree_attention.py`](file:///Users/sujan/Public/NexLM/inference/tree_attention.py)| Medusa / Eagle Multi-Head Tree Attention Speculative Decoding |
| | `48-advanced-sampler`| [`advanced_sampler.py`](file:///Users/sujan/Public/NexLM/inference/advanced_sampler.py)| Min-P, Locally Typical, and Frequency/Presence Penalty sampling |
| | `50-chunked-prefill` | [`chunked_prefill.py`](file:///Users/sujan/Public/NexLM/inference/chunked_prefill.py)| Chunked Prefill & StreamingLLM Attention Sink eviction engine |
| **Serving & Go Tooling**| `15-serving-api` | [`server.py`](file:///Users/sujan/Public/NexLM/server/server.py) | OpenAI-compatible SSE streaming API + Dark-Mode Web Playground |
| | `23-terminal-chat` | [`chat.py`](file:///Users/sujan/Public/NexLM/cli/chat.py) | Multi-turn streaming CLI REPL with slash commands |
| | `30-golang-client` | [`main.go`](file:///Users/sujan/Public/NexLM/go_client/main.go) | High-performance Go streaming client & concurrency benchmark |
| | `37-golang-tokenizer`| [`tokenizer.go`](file:///Users/sujan/Public/NexLM/go_client/tokenizer/tokenizer.go)| Go Byte-Level Tokenizer with concurrent multi-worker batch encoding |
| | `44-golang-inspector`| [`inspector.go`](file:///Users/sujan/Public/NexLM/go_client/inspector/inspector.go)| Go Model Architecture Diagnostics & memory footprint calculator |
| | `51-golang-router` | [`router.go`](file:///Users/sujan/Public/NexLM/go_client/router/router.go) | Go Request Router, Token Bucket Rate Limiter, and Load Balancer |
| | `52-complete-v3-docs`| [`README.md`](file:///Users/sujan/Public/NexLM/README.md) | Complete end-to-end production documentation & benchmarking guide |

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
git clone https://github.com/sujanv/nexlm.git
cd nexlm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run test suite to verify all 110 unit tests:
```bash
PYTHONPATH=. pytest -v
```

---

### 2. Frontier Architectural Recipes

#### A. Kahneman-Tversky Optimization (KTO) Alignment
Aligns models directly from binary feedback (thumbs up / thumbs down) without preference pairs:
```python
from training.kto import KTOEngine, KTOBatch

optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)
engine = KTOEngine(policy_model, ref_model, optimizer, beta=0.1)

batch = KTOBatch(
    prompt_ids=prompt_tokens,
    completion_ids=completion_tokens,
    labels=torch.tensor([1, 0, 1]),  # 1 = desirable, 0 = undesirable
)
metrics = engine.train_step(batch)
print(f"KTO Loss: {metrics['kto_loss']:.4f}")
```

#### B. Medusa Tree Attention Speculative Decoding
Verifies 2D speculative candidate trees in parallel using custom causal tree attention masks:
```python
from inference.tree_attention import MedusaModel, verify_tree_candidates

medusa = MedusaModel(base_model, num_heads=3, top_k_per_head=2)
tree = medusa.generate_candidate_tree(last_hidden, last_token_id)
tokens, mask = tree.flatten()

# Single forward pass through target model verifies all candidate paths:
target_logits = target_model(tokens, tree_mask=mask)
accepted_tokens, num_accepted = verify_tree_candidates(tree, target_logits)
```

#### C. Min-P & Locally Typical Sampling
Dynamically prunes low-probability tokens relative to distribution peak and conditional entropy:
```python
from inference.advanced_sampler import advanced_sample

# Min-P: Prunes tokens with prob < 0.05 * max_prob
next_token = advanced_sample(logits, min_p=0.05, temperature=0.8)

# Locally Typical: Selects tokens closest to conditional entropy
next_token = advanced_sample(logits, typical_p=0.85, temperature=0.8)

# With Frequency & Presence Penalties:
next_token = advanced_sample(
    logits,
    context_tokens=history_ids,
    frequency_penalty=0.5,
    presence_penalty=0.5,
)
```

#### D. QLoRA 4-bit NormalFloat4 (NF4) Fine-Tuning
Freezes base weights in information-theoretically optimal 4-bit NF4 representation while training low-rank adapters:
```python
from training.qlora import QLoRALinear

# Convert standard linear to 4-bit NF4 QLoRA layer
qlora_layer = QLoRALinear.from_linear(linear, r=8, lora_alpha=16.0, block_size=64)
output = qlora_layer(x)
loss = output.sum()
loss.backward()  # Gradients only flow into lora_A and lora_B!
```

#### E. Chunked Prefill & Attention Sinks (StreamingLLM)
Eliminates OOM spikes and enables infinite continuous sequence generation:
```python
from inference.chunked_prefill import chunked_prefill, AttentionSinkKVCache

# 1. Chunked Prefill: processes prompt in 128-token chunks
logits, kv_cache = chunked_prefill(model, long_prompt_ids, chunk_size=128)

# 2. Attention Sinks: retains 4 initial sink tokens + 256 rolling tokens
sink_cache = AttentionSinkKVCache(num_sink_tokens=4, window_size=256)
pruned_kv = sink_cache.evict_and_retain(kv_cache)
```

#### F. Go Request Router & Rate Limiter
Production Go load balancer with token bucket rate limiting:
```go
import "github.com/sujanv/nexlm/go_client/router"

b1 := &router.Backend{URL: "http://localhost:8080", Healthy: true}
b2 := &router.Backend{URL: "http://localhost:8081", Healthy: true}

// 100 QPS with burst capacity of 20, using least-connection balancing
r := router.NewRouter([]*router.Backend{b1, b2}, 100.0, 20.0, "least_conn")

resp, err := r.ExecuteRequest(func(b *router.Backend) (string, error) {
    // Send request to b.URL
    return "response", nil
})
```

---

## ⚡ Inference & Serving

### 1. OpenAI-Compatible SSE Streaming Server
```bash
PYTHONPATH=. python server/server.py --port 8080
```
Open `http://localhost:8080` in your browser for the interactive dark-mode Web Playground, or stream tokens via curl:
```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Once upon a time"}], "stream": true}'
```

### 2. High-Performance Go Streaming Client & Router
```bash
cd go_client
go test -v ./...
go run main.go -prompt "Once upon a time" -stream
# High-concurrency benchmark:
go run main.go -bench -concurrency 20 -requests 100
```

---

## 📊 Benchmarks & Evaluations

#### A. KV-Cache Latency Acceleration
```bash
PYTHONPATH=. python benchmarks/bench_kv_cache.py
```
| Generated Tokens | No KV Cache (s) | With KV Cache (s) | Speedup |
|:---:|:---:|:---:|:---:|
| 32  | 0.38s | 0.20s | **1.9x** |
| 64  | 0.95s | 0.38s | **2.5x** |
| 128 | 2.52s | 0.75s | **3.4x** |
| 200 | 5.21s | 1.18s | **4.4x** |

#### B. Needle In A Haystack (NIAH) Context Retrieval Test
```bash
PYTHONPATH=. python benchmarks/needle_in_haystack.py --context_len 128
```

---

## 📜 License
MIT License. Created by [Sujan](https://github.com/sujanv).
