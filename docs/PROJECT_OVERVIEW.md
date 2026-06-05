# Project Overview

Mini-Qwen is a compact autoregressive language model implemented from scratch
with PyTorch. The repository is designed for learning and experimentation, with
small, readable modules for the transformer architecture, optimizer, data
pipeline, training loop, and generation utilities.

## Main Modules

| Path | Purpose |
| --- | --- |
| `model.py` | Defines the `MiniQwen` model stack. |
| `train.py` | Runs the full training loop on a Hugging Face text dataset. |
| `generate.py` | Provides autoregressive sampling with temperature, top-k, and top-p filtering. |
| `data.py` | Loads, tokenizes, and chunks Hugging Face datasets for causal language modeling. |
| `optimizer.py` | Implements AdamW from scratch. |
| `components/attention.py` | Implements grouped-query attention and QK normalization. |
| `components/block.py` | Defines RMSNorm and the transformer block. |
| `components/feedforward.py` | Implements the SwiGLU feed-forward network. |
| `components/rope.py` | Builds and applies rotary positional embeddings. |

## Training Flow

1. `train.py` selects hyperparameters and device.
2. `data.py` downloads Wikitext, loads the GPT-2 tokenizer, and chunks tokens.
3. `model.py` builds the Mini-Qwen transformer.
4. `optimizer.py` applies custom AdamW updates.
5. `generate.py` samples text from the trained model.

## Notes

- The default configuration is intentionally small so it can run on modest
  hardware.
- This is an educational project, not a drop-in replacement for production
  Qwen checkpoints.
- Training downloads datasets and tokenizer files from Hugging Face.
