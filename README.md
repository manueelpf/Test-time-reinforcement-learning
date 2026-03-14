# Word2Vec from Scratch in Pure NumPy

A clean, interview-friendly implementation of **skip-gram with negative sampling (SGNS)** in pure **NumPy**.

This repository is intentionally structured like a small production-quality ML project rather than a single notebook. The core objective is to satisfy the task:

- pure NumPy training loop
- explicit forward pass, loss, gradients, and parameter updates
- code that is easy to explain in a follow-up interview
- no PyTorch / TensorFlow / JAX / other ML frameworks

## Why this variant?

I chose **skip-gram with negative sampling** because:

1. it is one of the standard Word2Vec variants;
2. it avoids the full softmax over the whole vocabulary at each step;
3. it keeps the gradient derivation compact and explainable;
4. it is practical to train in pure NumPy on a laptop.

## Project layout

```text
jetbrains_word2vec_repo/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  ├─ raw/
│  │  └─ sample_corpus.txt
│  └─ processed/
├─ src/
│  └─ word2vec/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ utils.py
│     ├─ data.py
│     ├─ model.py
│     ├─ trainer.py
│     └─ eval.py
├─ scripts/
│  ├─ train.py
│  └─ inspect.py
└─ tests/
   └─ test_gradients.py
```

## What each part does

- `config.py`: central training configuration.
- `data.py`: tokenization, vocabulary building, subsampling, pair generation, and negative sampling.
- `model.py`: SGNS model, forward pass, loss, gradients, and in-place parameter updates.
- `trainer.py`: epoch loop, batching, logging, and checkpoint saving.
- `eval.py`: cosine similarity and nearest-neighbor inspection.
- `scripts/train.py`: CLI entrypoint for training.
- `scripts/inspect.py`: inspect learned embeddings after training.
- `tests/test_gradients.py`: finite-difference gradient check for the core SGNS step.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate   # Windows PowerShell
pip install -r requirements.txt
```

## Train on the included sample corpus

```bash
python scripts/train.py \
  --text-path data/raw/sample_corpus.txt \
  --output-dir artifacts/run_01
```

## Train on your own corpus

Use any plain-text corpus:

```bash
python scripts/train.py \
  --text-path path/to/corpus.txt \
  --embedding-dim 100 \
  --window-size 4 \
  --negative-samples 5 \
  --epochs 5 \
  --batch-size 256 \
  --output-dir artifacts/run_custom
```

## Inspect learned neighbors

```bash
python scripts/inspect.py \
  --checkpoint artifacts/run_01/model.npz \
  --vocab artifacts/run_01/vocab.json \
  --query king
```

## Run the gradient test

```bash
python -m tests.test_gradients
```

## Implementation notes

### Training objective

For one positive pair `(center, context)` and `K` negative samples `n_1, ..., n_K`, the loss is

```text
L = -log σ(v_c · u_o) - Σ_j log σ(-v_c · u_{n_j})
```

where:

- `v_c` = input embedding of the center word
- `u_o` = output embedding of the true context word
- `u_{n_j}` = output embedding of a sampled negative word

### Gradients used in code

Let:

- `s_pos = v_c · u_o`
- `s_neg_j = v_c · u_{n_j}`

Then:

```text
∂L/∂s_pos   = σ(s_pos) - 1
∂L/∂s_neg_j = σ(s_neg_j)
```

So:

```text
∂L/∂v_c = (σ(s_pos) - 1) u_o + Σ_j σ(s_neg_j) u_{n_j}
∂L/∂u_o = (σ(s_pos) - 1) v_c
∂L/∂u_{n_j} = σ(s_neg_j) v_c
```

The implementation uses `np.add.at(...)` so repeated indices inside a batch are handled correctly.

## Interview talking points

Be ready to explain:

- why row lookup replaces one-hot matrix multiplication;
- why SGNS is much cheaper than full softmax;
- why negative sampling uses a smoothed unigram distribution;
- why we keep separate input and output embedding matrices during training;
- how repeated word indices are accumulated safely in NumPy;
- trade-offs between precomputing all training pairs and streaming them.

## Possible extensions

- CBOW implementation
- hierarchical softmax
- phrase mining
- analogy evaluation set
- learning-rate scheduling
- checkpoint resume
- alias-method negative sampler for even faster sampling
