# ⚡ NexLM: Build a Small LLM From Scratch in PyTorch

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python)](https://python.org)
[![Go](https://img.shields.io/badge/Go-1.20+-00ADD8.svg?style=flat&logo=go)](https://go.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Tests-92%20Passing-brightgreen.svg)]()
[![Commits](https://img.shields.io/badge/Milestones-45%20Commits-blue.svg)]()

**NexLM** is a comprehensive, production-grade, yet educational GPT-style Transformer language model implemented completely from scratch using PyTorch primitives.

From fundamental tokenization and self-attention up to cutting-edge frontier techniques—including **Root Mean Square Normalization (RMSNorm)**, **Rotary Embeddings (RoPE)**, **YaRN Context Window Scaling**, **DeepSeek Multi-Head Latent Attention (MLA)**, **SwiGLU**, **Grouped-Query Attention (GQA)**, **Sliding Window Attention (SWA)**, **Multi-Token Prediction (MTP)**, **LoRA fine-tuning**, **Direct Preference Optimization (DPO)**, **Rejection Sampling / Best-of-N Alignment**, **SLERP & DARE Model Merging**, **Streaming Tool / Function Calling**, **Continuous Batching**, **Speculative Decoding**, **vLLM-style Paged Attention**, **Fine-Grained Block-Wise Quantization (.nexq)**, and **High-Throughput Go Serving, Tokenization & Architecture Inspection**—NexLM is designed to be the ultimate ML/AI engineering reference.

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
│   ├── checkpointing.py       # Activation / Gradient Checkpointing memory optimizer
│   ├── rejection_sampling.py  # Best-of-N Rejection Sampling & Reward Model alignment
│   └── model_merging.py       # SLERP & DARE / Task Arithmetic post-training model merging
├── inference/
│   ├── generate.py            # Autoregressive decoding (greedy, temp, top-k, top-p, KV cache)
│   ├── sampler.py             # Temperature scaling, Top-K, and Top-P filtering
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
├── tests/                     # Comprehensive test suite (92 unit tests passing)
├── demo.py                    # One-command training & generation demo
└── README.md
```

---

## 🏗️ The 45 Milestones & Capabilities

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
| **Inference Optimization**| `14-quantization` | [`quantize.py`](file:///Users/sujan/Public/NexLM/inference/quantize.py) | Post-Training INT8 and INT4 weight-only quantization |
| | `20-speculative` | [`speculative.py`](file:///Users/sujan/Public/NexLM/inference/speculative.py) | Speculative Decoding achieving $2\times - 3\times$ speedup via draft model |
| | `22-paged-cache` | [`paged_cache.py`](file:///Users/sujan/Public/NexLM/inference/paged_cache.py) | vLLM-style Paged Attention virtual memory block manager |
| | `27-prefix-cache` | [`prefix_cache.py`](file:///Users/sujan/Public/NexLM/inference/prefix_cache.py)| SHA-256 prompt prefix caching with LRU cache eviction |
| | `28-json-constrained`| [`json_constrained.py`](file:///Users/sujan/Public/NexLM/inference/json_constrained.py)| Grammar logit masking ensuring 100% syntactically valid JSON |
| | `33-continuous-batching`| [`continuous_batching.py`](file:///Users/sujan/Public/NexLM/inference/continuous_batching.py)| Iteration-level scheduling eliminating batch waiting starvation |
| | `35-block-quant` | [`block_quantization.py`](file:///Users/sujan/Public/NexLM/inference/block_quantization.py)| Fine-grained 32-element Block-Wise Quantization (.nexq format) |
| | `41-contrastive` | [`contrastive_decoding.py`](file:///Users/sujan/Public/NexLM/inference/contrastive_decoding.py)| Contrastive Decoding & Classifier-Free Guidance (CFG) for sharp reasoning |
| | `43-tool-calling` | [`tool_calling.py`](file:///Users/sujan/Public/NexLM/inference/tool_calling.py)| Streaming Tool / Function Calling pushdown parser and dispatcher |
| **Serving & Go Tooling**| `15-serving-api` | [`server.py`](file:///Users/sujan/Public/NexLM/server/server.py) | OpenAI-compatible SSE streaming API + Dark-Mode Web Playground |
| | `23-terminal-chat` | [`chat.py`](file:///Users/sujan/Public/NexLM/cli/chat.py) | Multi-turn streaming CLI REPL with slash commands |
| | `30-golang-client` | [`main.go`](file:///Users/sujan/Public/NexLM/go_client/main.go) | High-performance Go streaming client & concurrency benchmark |
| | `37-golang-tokenizer`| [`tokenizer.go`](file:///Users/sujan/Public/NexLM/go_client/tokenizer/tokenizer.go)| Go Byte-Level Tokenizer with concurrent multi-worker batch encoding |
| | `44-golang-inspector`| [`inspector.go`](file:///Users/sujan/Public/NexLM/go_client/inspector/inspector.go)| Go Model Architecture Diagnostics & memory footprint calculator |
| | `45-complete-docs` | [`README.md`](file:///Users/sujan/Public/NexLM/README.md) | Complete end-to-end production documentation & benchmarking guide |

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

Run test suite to verify all 92 unit tests:
```bash
PYTHONPATH=. pytest -v
```

---

### 2. Frontier Architectural Recipes

#### A. YaRN Context Window Extension
Scales context length up to 16x without fine-tuning:
```python
from model.yarn import YaRNScaledRotaryEmbedding, apply_yarn_rope

yarn = YaRNScaledRotaryEmbedding(
    dim=32,
    max_seq_len=2048,
    original_max_seq_len=256,
    scale=8.0,
)
cos, sin = yarn(seq_len=1024, device=q.device)
q_rot, k_rot = apply_yarn_rope(q, k, cos, sin)
```

#### B. DeepSeek Multi-Head Latent Attention (MLA)
Reduces KV cache memory by over 85% by caching compressed low-rank latents:
```python
from model.mla import MultiHeadLatentAttention

mla = MultiHeadLatentAttention(
    d_model=256,
    n_heads=8,
    q_lora_rank=64,
    kv_lora_rank=64,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=32,
)
# Forward pass returns output and (cached_c_kv, cached_k_rope)
out, kv_cache = mla(x, use_cache=True)
```

#### C. Contrastive Decoding & Classifier-Free Guidance (CFG)
Dramatically sharpens reasoning and eliminates vacuous tokens:
```python
from inference.contrastive_decoding import generate_contrastive, generate_cfg

# Contrastive Decoding (Expert vs Amateur model)
tokens = generate_contrastive(expert_model, amateur_model, input_ids, alpha=0.5, beta=0.1)

# Classifier-Free Guidance (CFG scale > 1.0)
tokens_cfg = generate_cfg(model, prompt_ids, cfg_scale=1.5, max_new_tokens=40)
```

#### D. Post-Training Model Merging (SLERP & DARE)
Merge fine-tuned checkpoints without retraining or GPUs:
```python
from training.model_merging import merge_models_slerp, merge_models_dare

# Spherical Linear Interpolation (SLERP)
merged_slerp = merge_models_slerp(math_model, code_model, t=0.5)

# DARE Task Arithmetic Merging
merged_dare = merge_models_dare(base_model, [task_model_1, task_model_2], drop_rate=0.2)
```

#### E. Streaming Tool / Function Calling
Real-time state machine parsing function calls on token streams:
```python
from inference.tool_calling import StreamingToolParser, ToolRegistry, ToolDefinition

parser = StreamingToolParser()
for token_delta in stream:
    user_text, completed_calls = parser.feed(token_delta)
    if user_text:
        print(user_text, end="", flush=True)
    for call in completed_calls:
        print(f"\n[Executing Tool: {call.name} with {call.arguments}]")
```

#### F. Go Model Architecture Inspector
Analyze parameter counts, tensor layouts, and memory footprints in Go:
```bash
cd go_client/inspector
go test -v ./...
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

### 2. High-Performance Go Streaming Client & Inspector
```bash
cd go_client
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
