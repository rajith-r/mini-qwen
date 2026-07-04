# mini-qwen

A tiny character-level GPT-style transformer built from scratch in PyTorch.

This project is for learning how language models work at a small scale. It includes a basic character tokenizer, token embeddings, masked self-attention, a decoder block, and a simple next-character training loop.

## What Is Inside

- `tokenizer.py` - character-level tokenizer that converts text to token IDs and back.
- `embedding.py` - small embedding experiment for seeing token vectors.
- `model.py` - mini decoder-only language model with:
  - token embeddings
  - masked self-attention
  - residual connections
  - MLP/feed-forward layer
  - language-model head for next-token prediction

## How It Works

The current model trains on the text:

```text
hello cuda
```

It learns to predict the next character:

```text
h -> e
e -> l
l -> l
l -> o
...
```

After training, it starts with `"h"` and generates more characters one at a time.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Install PyTorch:

```bash
pip install torch
```

## Run

Train and generate text:

```bash
python model.py
```

Try the tokenizer:

```bash
python tokenizer.py
```

Try the embedding experiment:

```bash
python embedding.py
```

## Current Limitations

This is intentionally small and beginner-friendly. A few important things are not added yet:

- no positional embeddings yet
- no layer normalization yet
- only one decoder block
- only single-head attention
- trains on a tiny hardcoded string

## Next Steps

Good next upgrades:

1. Add positional embeddings so the model understands token order.
2. Add layer normalization for more stable training.
3. Train on a larger text file instead of one short string.
4. Add multiple decoder blocks.
5. Add multi-head attention.

