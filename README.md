# Mini-Qwen: Autoregressive Language Model from Scratch

A lightweight, modular, and academically rigorous implementation of the Qwen-3 architecture built entirely from scratch using PyTorch. This repository provides an end-to-end pipeline covering modern LLM design elements, custom optimization, Hugging Face dataset tokenization/streaming, and temperature-controlled autoregressive text generation.

---

## 🏗️ Architecture & Component Design

The implementation features state-of-the-art transformer components optimized for training stability and inference throughput:

*   **Grouped-Query Attention (GQA)** ([components/attention.py](file:///E:/qwen_from_scratch/components/attention.py)): Combines multi-query and multi-head attention styles by grouping query heads. This reduces KV cache size and memory access overhead during generation.
*   **Rotary Position Embedding (RoPE)** ([components/rope.py](file:///E:/qwen_from_scratch/components/rope.py)): Applies relative positional information to the key and query projections by rotating them in the 2D complex plane.
*   **SwiGLU Feed-Forward Network** ([components/feedforward.py](file:///E:/qwen_from_scratch/components/feedforward.py)): Employs a Swish-gated Linear Unit activation function ($x \cdot \text{silu}(x \cdot W_g) \cdot W_u$) instead of vanilla ReLU/GELU, improving learning capacity.
*   **Root Mean Square Normalization (RMSNorm)** ([components/block.py](file:///E:/qwen_from_scratch/components/block.py)): Normalizes activations based on their root mean square, which is faster and equally effective compared to standard LayerNorm.
*   **QK Normalization** ([components/attention.py](file:///E:/qwen_from_scratch/components/attention.py)): Normalizes Query and Key tensors before calculating dot-product attention, mitigating entropy collapse and enhancing training stability for deeper networks.
*   **AdamW Optimizer from Scratch** ([optimizer.py](file:///E:/qwen_from_scratch/optimizer.py)): Custom implementation of decoupled weight-decay Adam ($L_2$ regularization decoupled from gradient update steps).

---

## 📂 Repository Structure

```
qwen_from_scratch/
│
├── components/
│   ├── attention.py       # Grouped-Query Attention (GQA) & QK Norm
│   ├── block.py           # TransformerBlock & RMSNorm definitions
│   ├── feedforward.py     # SwiGLU MLP projection layer
│   ├── norm.py            # RMSProp optimizer module (for standalone tests)
│   └── rope.py            # Rotary Position Embeddings (RoPE) cache & application
│
├── data.py                # Hugging Face dataset loading & tokenization pipeline
├── generate.py            # Autoregressive decoding (Top-K / Top-P sampling)
├── model.py               # Main MiniQwen model stack
├── optimizer.py           # Custom AdamW optimizer
├── train.py               # End-to-end training and evaluation script
└── README.md              # Documentation
```

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.8+ installed along with PyTorch, Hugging Face Datasets, and Transformers:

```bash
pip install torch datasets transformers
```

### 2. Training the Model
Run the end-to-end pipeline to load `wikitext`, tokenize text sequences, train the Mini-Qwen model, and evaluate sample generation output:

```bash
python train.py
```

### 3. Autoregressive Generation
You can run standalone generation tests using `generate.py`:

```bash
python generate.py
```

---

## 📊 Pipeline Parameters

The pipeline is preconfigured with the following default hyperparameters:

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `dataset_name` | `"wikitext"` | Hugging Face dataset source |
| `tokenizer_name` | `"gpt2"` | Pretrained tokenizer (50,257 vocab size) |
| `seq_len` | `128` | Max context sequence length |
| `batch_size` | `8` | Training batch size |
| `learning_rate` | `3e-4` | Learning rate for AdamW optimizer |
| `hidden_dim` | `256` | Model embedding hidden dimension |
| `num_layers` | `4` | Number of Transformer block layers |
