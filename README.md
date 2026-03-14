# Word2Vec from Scratch in Pure NumPy

A clean, interview-friendly implementation of **Word2Vec (skip-gram with negative sampling)** written entirely in **NumPy**.

The goal of this repository is to demonstrate a full implementation of the **core Word2Vec training loop**, including:

- forward pass
- loss computation
- gradient derivation
- parameter updates

All implemented **without PyTorch, TensorFlow, JAX, or other ML frameworks**.

This project was developed as part of a technical task to demonstrate understanding of **representation learning and optimization in language models**.

---

# Dataset

For the main experiment, the model was trained on the **Project Gutenberg text of *The Importance of Being Earnest* by Oscar Wilde**.

This dataset was chosen because:

- it is **public domain**
- it is **plain text**, making preprocessing simple
- it contains **rich conversational language**
- it is large enough (~24k tokens) to produce meaningful word co-occurrence statistics

The file used in this project is:
```text
data/raw/importance_of_being_earnest.txt
```

Dataset statistics after preprocessing:

| Metric | Value |
|------|------|
| Tokens | ~24,000 |
| Vocabulary size | ~1,575 |
| Skip-gram training pairs | ~144,000 |

The dataset is intentionally modest in size so the full training pipeline can run **quickly on a laptop while still producing meaningful embeddings**.

---

# Why Skip-Gram with Negative Sampling

I chose **skip-gram with negative sampling (SGNS)** because:

1. it is one of the **standard Word2Vec variants**
2. it avoids the expensive **full softmax over the vocabulary**
3. it is **efficient to implement in pure NumPy**
4. the gradient derivation remains **compact and easy to explain**

This makes it ideal for demonstrating understanding of the algorithm in a technical interview.

---

# Project Layout
```text
jetbrains_word2vec_repo/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│ ├─ raw/
│ │ └─ importance_of_being_earnest.txt
│ └─ processed/
├─ src/
│ └─ word2vec/
│ ├─ init.py
│ ├─ config.py
│ ├─ utils.py
│ ├─ data.py
│ ├─ model.py
│ ├─ trainer.py
│ └─ eval.py
├─ scripts/
│ ├─ train.py
│ └─ query_embeddings.py
└─ tests/
└─ test_gradients.py
```


# Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate   # Windows PowerShell

pip install -r requirements.txt
```

# Training the model
To train on the Importance of Being Earnest dataset:
```bash
python scripts/train.py --text-path data/raw/importance_of_being_earnest.txt --embedding-dim 100 --window-size 5 --negative-samples 5 --epochs 10 --batch-size 256 --min-count 2 --dynamic-window --output-dir artifacts/run
```

These are the statistics
```text
Token count:     ~24,000
Vocabulary size: ~1,575
Training pairs:  ~144,000
Final loss:      ~2.39
```
The decreasing loss confirms that the training loop and gradient computations are functioning correctly.

# Inspect Learned Word Embeddings
After training, you can query nearest neighbors:

```bash
python scripts/query_embeddings.py --checkpoint artifacts/run/model.npz --vocab artifacts/run/vocab.json --query marriage
```

Example output:
```text
Nearest neighbors for "marriage"

engagement
proposal
wife
family
```

# Run Gradient Verification

The project includes a finite-difference gradient check to verify the SGNS implementation.

```bash
python -m tests.test_gradients
```

Expected output:
```text
Gradient check passed.
```

This confirms the analytical gradients match the numerical gradients.

# Possible Extensions

Potential improvements to this project:

- CBOW implementation
- hierarchical softmax
- alias-method negative sampler
- phrase detection
- analogy benchmark evaluation
- learning-rate schedulers
- checkpoint resume