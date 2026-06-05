# Contributing

Thanks for taking an interest in this project.

## Development Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run a quick smoke test:

```bash
python model.py
python generate.py
```

## Contribution Guidelines

- Keep changes focused and easy to review.
- Add short comments only where they clarify non-obvious model logic.
- Avoid committing generated files, checkpoints, virtual environments, or caches.
- Before opening a pull request, make sure the smoke tests above still pass.

## Project Scope

This repository is intended as an educational implementation of a compact,
Qwen-inspired autoregressive transformer. Changes should preserve readability
and make the architecture easier to study.
