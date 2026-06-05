# Mini-Qwen From Scratch

A compact, educational implementation of a Qwen-inspired autoregressive
language model built with PyTorch. The project includes transformer components,
a custom AdamW optimizer, a Hugging Face data pipeline, a training script, and
temperature-controlled text generation.

## Highlights

- Grouped-query attention with QK normalization
- Rotary position embeddings
- SwiGLU feed-forward layers
- RMSNorm transformer blocks
- AdamW optimizer implemented from scratch
- Hugging Face dataset and tokenizer integration
- Top-k and top-p autoregressive sampling

## Repository Structure

```text
qwen_from_scratch/
|-- components/
|   |-- attention.py       # Grouped-query attention and QK normalization
|   |-- block.py           # Transformer block and RMSNorm
|   |-- feedforward.py     # SwiGLU feed-forward network
|   |-- norm.py            # Standalone RMSProp implementation
|   `-- rope.py            # Rotary position embeddings
|-- docs/
|   `-- PROJECT_OVERVIEW.md
|-- data.py                # Dataset loading and tokenization
|-- generate.py            # Autoregressive generation utilities
|-- model.py               # MiniQwen model definition
|-- optimizer.py           # Custom AdamW optimizer
|-- train.py               # Training and evaluation script
|-- requirements.txt
|-- CONTRIBUTING.md
`-- LICENSE
```

## Quick Start

### 1. Create an Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

### 2. Run Smoke Tests

```bash
python model.py
python generate.py
```

### 3. Train

```bash
python train.py
```

The default training script uses:

| Setting | Default |
| --- | --- |
| Dataset | `wikitext` |
| Dataset config | `wikitext-2-raw-v1` |
| Tokenizer | `gpt2` |
| Sequence length | `128` |
| Batch size | `8` |
| Epochs | `3` |
| Learning rate | `3e-4` |
| Hidden dimension | `256` |
| Layers | `4` |
| Attention heads | `8` |
| KV heads | `2` |

## Generation

`generate.py` exposes a reusable `generate` function:

```python
generated = generate(
    model,
    prompt_tensor,
    max_new_tokens=50,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    device="cuda",
)
```

## Documentation

See [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) for a concise map of
the codebase and training flow.

## Notes

- This project is intended for learning and experimentation.
- The default configuration is small and readable, not optimized for benchmark
  performance.
- Training requires network access the first time Hugging Face downloads the
  dataset and tokenizer.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
