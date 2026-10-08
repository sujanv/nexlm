# ⚡ NexLM: Build a Small LLM From Scratch in PyTorch

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

**NexLM** is a clean, educational, and high-performance GPT-style Transformer language model implemented completely from scratch using PyTorch primitives.

It covers the complete lifecycle of a modern language model: **Byte-level BPE tokenization**, **embeddings**, **causal multi-head self-attention**, **Pre-LN transformer blocks**, **causal masking**, **training with AdamW and cosine schedules**, **checkpointing**, **advanced sampling strategies**, and **Key-Value (KV) caching for accelerated inference**.

---

## 📂 Repository Structure

```
nexlm/
├── model/
│   ├── __init__.py
│   ├── normalization.py       # LayerNorm & RMSNorm from scratch
│   ├── embeddings.py          # Token & Learned Positional Embeddings
│   ├── attention.py           # Multi-Head Causal Self-Attention + KV Cache
│   ├── transformer.py         # MLP & Pre-LN Transformer Blocks
│   └── gpt.py                 # Full NexLM GPT Architecture & Weight Tying
├── tokenizer/
│   ├── __init__.py
│   ├── base.py               # BaseTokenizer abstract interface
│   └── bpe.py                # Byte-level BPE & Char tokenizers
├── training/
│   ├── __init__.py
│   ├── lr_schedule.py        # Cosine schedule with linear warmup
│   ├── dataset.py            # Language modeling next-token datasets
│   ├── trainer.py            # Gradient accumulation, clipping & eval
│   ├── train.py              # CLI training script
│   └── evaluate.py           # Evaluation & prompt testing CLI
├── inference/
│   ├── __init__.py
│   ├── sampler.py            # Temperature, Top-K, and Top-P (nucleus) sampling
│   ├── kv_cache.py           # KV Cache state container
│   └── generate.py           # Autoregressive generation & streaming
├── configs/
│   ├── tinystories_tiny.json  # ~4.5M parameter preset
│   └── tinystories_small.json # ~15M parameter preset
├── benchmarks/
│   └── bench_kv_cache.py     # Inference latency & throughput comparison
├── data/
│   └── download_tinystories.py # TinyStories downloader & generator
├── tests/                    # Comprehensive unit tests (16/16 passing)
│   ├── test_normalization.py
│   ├── test_attention.py
│   ├── test_transformer.py
│   ├── test_gpt.py
│   ├── test_tokenizer.py
│   ├── test_sampler.py
│   └── test_kv_cache.py
├── demo.py                   # One-command interactive demo
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🧠 Architectural Overview & Core Concepts

### 1. Tokenization: Byte-Level BPE from Scratch
Traditional word-level tokenizers suffer from Out-Of-Vocabulary (`<unk>`) errors. NexLM implements a pure Python **Byte-Level Byte-Pair Encoding (BPE)** tokenizer (GPT-2 style):
- Starts with 256 byte tokens (`0x00` through `0xFF`), plus special tokens (`<|endoftext|>`, `<|pad|>`).
- Iteratively finds the most frequent pair of tokens in the corpus and merges them into a new token.
- **Guarantee**: Any UTF-8 string (including emojis, non-Latin scripts, and code) can be encoded and decoded without loss or `<unk>` tokens.

### 2. Embeddings & Positional Encoding
- **Token Embeddings**: Projects token index $t \in [0, V-1]$ into a continuous vector space $\mathbb{R}^{d}$.
- **Positional Embeddings**: Learned positional embeddings $P \in \mathbb{R}^{T \times d}$ representing positions $0, 1, \dots, T-1$.
- Combined input: $x = \text{Embedding}(t) + \text{PositionalEmbedding}(pos)$.

### 3. Causal Multi-Head Self-Attention
Given hidden representations $X \in \mathbb{R}^{B \times T \times d}$:
1. Project into Queries, Keys, and Values using a single linear layer:
   $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
2. Split each into $h$ heads of dimension $d_k = d / h$.
3. Compute scaled dot-product attention with **causal masking**:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
   where $M_{i, j} = 0$ for $j \le i$, and $-\infty$ for $j > i$.
   This guarantees position $i$ never attends to future tokens $j > i$.

### 4. Layer Normalization
Implemented from scratch using PyTorch primitives:
- **LayerNorm**:
  $$\mu = \frac{1}{d}\sum_{k=1}^d x_k, \quad \sigma^2 = \frac{1}{d}\sum_{k=1}^d (x_k - \mu)^2, \quad y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
- **RMSNorm**: Root-mean-square normalization (popular in modern LLMs like Llama):
  $$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{k=1}^d x_k^2 + \epsilon}, \quad y = \frac{x}{\text{RMS}(x)} \odot \gamma$$

### 5. Pre-LN Transformer Blocks
NexLM uses the stable Pre-LayerNorm formulation:
$$x^{(1)} = x + \text{Attention}(\text{LN}_1(x))$$
$$x^{(2)} = x^{(1)} + \text{MLP}(\text{LN}_2(x^{(1)}))$$
where the MLP is:
$$\text{MLP}(u) = W_2 \cdot \text{GELU}(W_1 u + b_1) + b_2$$
with expansion ratio $4\times$.

### 6. Extra Credit: Key-Value (KV) Caching
In standard autoregressive generation, generating token $t+1$ requires recomputing attention over all prior $t$ tokens ($O(t^2)$ total operations).

With **KV Caching**:
- During the first step ("prefill"), Keys and Values for the prompt are computed and stored.
- On subsequent steps ("decode"), only the single new token is passed ($T=1$).
- The new Key and Value are appended to the cache:
  $$K_{\text{all}} = [K_{\text{past}}, K_{\text{new}}], \quad V_{\text{all}} = [V_{\text{past}}, V_{\text{new}}]$$
- Reduces generation complexity from $O(N^2)$ to $O(N)$, delivering **2x–4x+ speedups**.

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/sujanv/nexlm.git
cd nexlm

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and nexlm
pip install -r requirements.txt
pip install -e .
```

### 2. Run the Interactive Demo
Run a quick training, evaluation, and streaming story generation demo:
```bash
python demo.py
```

### 3. Run the KV Cache Benchmark
Compare inference throughput and latency with vs without KV caching:
```bash
python benchmarks/bench_kv_cache.py
```

### 4. Run the Unit Test Suite
```bash
pytest -v
```

---

## 📚 Training on TinyStories

### 1. Download or Generate TinyStories
```bash
# Option A: Download validation split of TinyStories from HuggingFace
python data/download_tinystories.py --output data/tinystories.txt

# Option B: Generate synthetic story dataset locally offline
python data/download_tinystories.py --synthetic --num_stories 5000 --output data/tinystories.txt
```

### 2. Launch Training
```bash
python -m nexlm.training.train \
    --data data/tinystories.txt \
    --preset small \
    --vocab_size 4096 \
    --steps 1000 \
    --batch_size 16 \
    --lr 6e-4 \
    --checkpoint_dir checkpoints/tinystories
```

### 3. Generate from Trained Checkpoint
```bash
python -m nexlm.training.evaluate \
    --checkpoint checkpoints/tinystories/best_model.pt \
    --tokenizer_path data/tokenizer.json \
    --prompt "Once upon a time, Lily found a" \
    --max_tokens 100 \
    --temperature 0.8 \
    --top_k 40 \
    --top_p 0.9
```

---

## 🎛️ Sampling Options

NexLM supports three complementary sampling controls:
- **Temperature ($T$)**:
  - $T \to 0$: Greedy decoding (argmax).
  - $T \approx 0.7 - 0.8$: Creative, coherent generation.
  - $T > 1.2$: Highly diverse / exploratory text.
- **Top-K Filtering**: Limits sampling pool to the $K$ most likely tokens.
- **Top-P (Nucleus) Filtering**: Dynamically cuts off the long tail by selecting the smallest set of tokens whose cumulative probability exceeds $P$ (e.g. $p=0.9$).

---

## 🧪 Model Configurations

| Preset | Layers | Heads | Embedding Dim | Seq Len | Parameters | Target Use |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Tiny** | 4 | 4 | 128 | 256 | ~4.5M | Fast prototyping & testing on CPU/Laptop |
| **Small** | 6 | 8 | 256 | 256 | ~15M | TinyStories training on Mac MPS / Colab |
| **Medium**| 8 | 12 | 384 | 512 | ~35M | Extended training / larger corpora |

---

## 🔬 Benchmark Results (KV Cache)

Results on Apple Silicon (MPS):

| New Tokens | Without KV Cache | With KV Cache | Speedup |
|:---:|:---:|:---:|:---:|
| 32  | 0.38s | 0.20s | **1.9x** |
| 64  | 0.95s | 0.38s | **2.5x** |
| 128 | 2.52s | 0.75s | **3.4x** |
| 200 | 5.21s | 1.18s | **4.4x** |

> *Notice: As output token count increases, the speedup continues growing because KV caching avoids quadratic recomputation.*

---

## 📜 License
MIT License. Created by [Sujan](https://github.com/sujanv).
