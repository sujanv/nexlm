# ⚡ NexLM: Build a Small LLM From Scratch in PyTorch

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Tests-47%20Passing-brightgreen.svg)]()
[![Commits](https://img.shields.io/badge/Milestones-31%20Commits-blue.svg)]()

**NexLM** is a comprehensive, production-grade, yet educational GPT-style Transformer language model implemented completely from scratch using PyTorch primitives.

From fundamental tokenization and self-attention up to frontier techniques—including **Rotary Embeddings (RoPE)**, **SwiGLU**, **Grouped-Query Attention (GQA)**, **Sliding Window Attention (SWA)**, **LoRA fine-tuning**, **Direct Preference Optimization (DPO)**, **Speculative Decoding**, **vLLM-style Paged Attention**, **INT8/INT4 Quantization**, and **Streaming HTTP/Go serving**—NexLM is designed to be the ultimate ML/AI engineering portfolio reference.

---

## 📂 Repository Structure

```
NexLM/
├── model/
│   ├── attention.py           # Single-Head & Multi-Head Causal Self-Attention
│   ├── transformer.py         # LayerNorm, MLP (GELU), Pre-LN TransformerBlock
│   ├── gpt.py                 # Full GPT Decoder Architecture & Weight Tying
│   ├── rope.py                # Rotary Position Embeddings (RoPE) from scratch
│   ├── swiglu.py              # SwiGLU Gated Feed-Forward Network (LLaMA style)
│   ├── gqa.py                 # Grouped-Query (GQA) & Multi-Query Attention (MQA)
│   ├── sliding_window.py      # Mistral Sliding Window Attention & Rolling Buffer
│   ├── flash_attention.py     # Hardware-accelerated SDPA / FlashAttention wrapper
│   ├── lora.py                # Low-Rank Adaptation (LoRA) for PEFT with merge/unmerge
│   └── moe.py                 # Sparsely-Gated Mixture of Experts with Top-K router
├── tokenizer/
│   └── tokenizer.py           # CharTokenizer (intuitive) & ByteLevelBPETokenizer (zero OOV)
├── training/
│   ├── train.py               # Main training pipeline with AdamW & LR cosine schedule
│   ├── evaluate.py            # Evaluation & prompt completion CLI
│   ├── amp.py                 # Automatic Mixed Precision (AMP) training manager
│   ├── dpo.py                 # Direct Preference Optimization (DPO) alignment engine
│   └── checkpointing.py       # Activation / Gradient Checkpointing memory optimizer
├── inference/
│   ├── generate.py            # Autoregressive decoding (greedy, temp, top-k, top-p, KV cache)
│   ├── sampler.py             # Temperature scaling, Top-K, and Top-P filtering
│   ├── quantize.py            # Post-Training Quantization (INT8 & INT4 weight-only)
│   ├── speculative.py         # Speculative Decoding engine (2x-3x faster inference)
│   ├── paged_cache.py         # vLLM-style Paged Attention Block Manager
│   ├── prefix_cache.py        # Prompt / Prefix Caching engine with SHA-256 fingerprinting
│   └── json_constrained.py    # Schema-Constrained Structured JSON decoding engine
├── server/
│   └── server.py              # OpenAI-compatible SSE Streaming API & Web Playground
├── go_client/
│   ├── go.mod                 # Golang module definition
│   └── main.go                # High-throughput Go streaming client & benchmark CLI
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
├── tests/                     # Comprehensive test suite (47 unit tests passing)
├── demo.py                    # One-command training & generation demo
└── README.md
```

---

## 🏗️ The 31 Milestones & Capabilities

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
| **Training & Alignment**| `13-amp` | [`amp.py`](file:///Users/sujan/Public/NexLM/training/amp.py) | FP16/BF16 Automatic Mixed Precision with dynamic `GradScaler` |
| | `18-lora` | [`lora.py`](file:///Users/sujan/Public/NexLM/model/lora.py) | Low-Rank Adaptation (LoRA) fine-tuning $<1\%$ parameters with `merge()` |
| | `21-dpo` | [`dpo.py`](file:///Users/sujan/Public/NexLM/training/dpo.py) | Direct Preference Optimization for RLHF alignment |
| | `29-checkpointing` | [`checkpointing.py`](file:///Users/sujan/Public/NexLM/training/checkpointing.py)| Gradient / Activation Checkpointing reducing peak activation VRAM |
| **Inference Optimization**| `14-quantization` | [`quantize.py`](file:///Users/sujan/Public/NexLM/inference/quantize.py) | Post-Training INT8 and INT4 weight-only quantization |
| | `20-speculative` | [`speculative.py`](file:///Users/sujan/Public/NexLM/inference/speculative.py) | Speculative Decoding achieving $2\times - 3\times$ speedup via draft model |
| | `22-paged-cache` | [`paged_cache.py`](file:///Users/sujan/Public/NexLM/inference/paged_cache.py) | vLLM-style Paged Attention virtual memory block manager |
| | `27-prefix-cache` | [`prefix_cache.py`](file:///Users/sujan/Public/NexLM/inference/prefix_cache.py)| SHA-256 prompt prefix caching with LRU cache eviction |
| | `28-json-constrained`| [`json_constrained.py`](file:///Users/sujan/Public/NexLM/inference/json_constrained.py)| Grammar logit masking ensuring 100% syntactically valid JSON |
| **Serving & Interfaces**| `15-serving-api` | [`server.py`](file:///Users/sujan/Public/NexLM/server/server.py) | OpenAI-compatible SSE streaming API + Dark-Mode Web Playground |
| | `23-terminal-chat` | [`chat.py`](file:///Users/sujan/Public/NexLM/cli/chat.py) | Multi-turn streaming CLI REPL with slash commands |
| | `30-golang-client` | [`main.go`](file:///Users/sujan/Public/NexLM/go_client/main.go) | High-performance Go streaming client & concurrency benchmark |

---

## 🚀 Quickstart & Detailed Instructions

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/sujanv/nexlm.git
cd nexlm

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and local package
pip install -r requirements.txt
pip install --no-build-isolation --no-deps -e .
```

### 2. Verify with Unit Tests
Execute the comprehensive test suite covering all modules:
```bash
PYTHONPATH=. pytest -v
```
*(47 unit tests pass in ~2.5 seconds).*

---

### 3. Training Recipes

#### A. Overfit Verification on Toy Data (Milestone 6)
Before running large datasets, verify that your model and optimizer can drive loss to near zero:
```bash
PYTHONPATH=. python training/train.py --toy --steps 60
```

#### B. Train on TinyStories (Milestone 7)
```bash
# 1. Download or synthetically generate TinyStories dataset
python data/download_tinystories.py --synthetic --num_stories 3000 --output data/tinystories.txt

# 2. Train model with AdamW and Cosine LR schedule
PYTHONPATH=. python training/train.py \
    --data data/tinystories.txt \
    --steps 500 \
    --batch_size 16 \
    --lr 6e-4 \
    --checkpoint_dir checkpoints/tinystories

# 3. Resume training seamlessly from checkpoint
PYTHONPATH=. python training/train.py \
    --resume checkpoints/tinystories/checkpoint.pt \
    --steps 800
```

#### C. Fine-Tune with LoRA (Milestone 18)
Fine-tune less than 1% of the model's parameters using Low-Rank Adaptation:
```python
from model.gpt import GPT, GPTConfig
from model.lora import apply_lora_to_model, mark_only_lora_as_trainable

model = GPT(GPTConfig())
apply_lora_to_model(model, r=8, alpha=16.0, target_modules=("c_attn",))
trainable_params, total_params = mark_only_lora_as_trainable(model)
print(f"Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params / total_params:.2%})")

# Merge adapter weights into base weights for zero-overhead inference
for module in model.modules():
    if hasattr(module, "merge"):
        module.merge()
```

#### D. Align with Direct Preference Optimization (Milestone 21)
Post-train on pairwise preference data without training a reward model:
```python
from model.gpt import GPT, GPTConfig
from training.dpo import DPOTrainer

policy_model = GPT(GPTConfig())
reference_model = GPT(GPTConfig())
optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)

trainer = DPOTrainer(policy_model, reference_model, optimizer, beta=0.1)
# metrics = trainer.train_step(chosen_ids, chosen_labels, rejected_ids, rejected_labels)
```

---

### 4. Inference & Serving Recipes

#### A. Interactive Terminal Chat REPL (Milestone 23)
Chat interactively with token-by-token colorized streaming:
```bash
PYTHONPATH=. python cli/chat.py --temperature 0.7
```
In-chat commands:
- `/clear`: Reset conversation context
- `/tokens`: Display current token count
- `/temp 0.9`: Adjust temperature on the fly
- `/exit`: Quit session

#### B. Web Playground & Streaming REST Server (Milestone 15)
Start the zero-dependency streaming server:
```bash
PYTHONPATH=. python server/server.py --port 8080
```
- Open **`http://localhost:8080/`** in your browser for the dark-mode playground.
- Connect programmatically via OpenAI-compatible API:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time, Lily found", "max_tokens": 50, "stream": true}'
```

#### C. High-Performance Go (Golang) Client & Benchmark (Milestone 30)
Run high-concurrency streaming inference using the native Go client:
```bash
cd go_client

# Single streaming prompt with Time-To-First-Token (TTFT) metrics
go run main.go --prompt "Once upon a time" --tokens 60

# Concurrency benchmark (20 requests across 4 workers)
go run main.go --bench -c 4 -n 20
```

#### D. Speculative Decoding (Milestone 20)
Generate tokens $2\times - 3\times$ faster using a lightweight draft model to speculate candidates:
```python
from inference.speculative import speculative_generate
# out_tokens, accept_rate = speculative_generate(target_model, draft_model, prompt, gamma=4)
```

#### E. Post-Training INT8 / INT4 Quantization (Milestone 14)
Halve memory consumption while retaining $>0.999$ output cosine similarity:
```python
from inference.quantize import quantize_model, compute_model_size_mb

print(f"FP32 Size: {compute_model_size_mb(model):.2f} MB")
q_model = quantize_model(model, bits=8)
print(f"INT8 Size: {compute_model_size_mb(q_model):.2f} MB")
```

---

### 5. Benchmarking

#### A. Key-Value (KV) Cache Benchmark
Compare latency and speedup across generation lengths:
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
Test retrieval accuracy at multiple document depths (10%, 50%, 90%):
```bash
PYTHONPATH=. python benchmarks/needle_in_haystack.py --context_len 128
```

---

## 📜 License
MIT License. Created by [Sujan](https://github.com/sujanv).
