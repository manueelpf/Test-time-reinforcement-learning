from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from word2vec.model import SkipGramNegativeSampling
from word2vec.utils import log_sigmoid, sigmoid


def scalar_loss(v_in: np.ndarray, v_out_pos: np.ndarray, v_out_neg: np.ndarray) -> float:
    pos_score = float(v_in @ v_out_pos)
    neg_scores = v_out_neg @ v_in
    return float(-log_sigmoid(np.array([pos_score]))[0] - np.sum(log_sigmoid(-neg_scores)))


def analytical_gradients(v_in: np.ndarray, v_out_pos: np.ndarray, v_out_neg: np.ndarray):
    pos_score = float(v_in @ v_out_pos)
    neg_scores = v_out_neg @ v_in

    grad_pos = float(sigmoid(np.array([pos_score]))[0] - 1.0)
    grad_neg = sigmoid(neg_scores)

    grad_v_in = grad_pos * v_out_pos + np.sum(grad_neg[:, None] * v_out_neg, axis=0)
    grad_v_out_pos = grad_pos * v_in
    grad_v_out_neg = grad_neg[:, None] * v_in[None, :]
    return grad_v_in, grad_v_out_pos, grad_v_out_neg


def numerical_gradient(loss_fn, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]

        x[idx] = original + eps
        loss_plus = loss_fn(x)

        x[idx] = original - eps
        loss_minus = loss_fn(x)

        x[idx] = original
        grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        it.iternext()
    return grad


def main() -> None:
    model = SkipGramNegativeSampling(vocab_size=7, embedding_dim=4, seed=123)

    center_id = 2
    context_id = 5
    negative_ids = np.array([1, 3, 1], dtype=np.int64)

    v_in = model.input_embeddings[center_id].copy()
    v_out_pos = model.output_embeddings[context_id].copy()
    v_out_neg = model.output_embeddings[negative_ids].copy()

    ana_v_in, ana_v_out_pos, ana_v_out_neg = analytical_gradients(v_in, v_out_pos, v_out_neg)

    num_v_in = numerical_gradient(lambda x: scalar_loss(x, v_out_pos, v_out_neg), v_in.copy())
    num_v_out_pos = numerical_gradient(lambda x: scalar_loss(v_in, x, v_out_neg), v_out_pos.copy())
    num_v_out_neg = numerical_gradient(lambda x: scalar_loss(v_in, v_out_pos, x), v_out_neg.copy())

    assert np.allclose(ana_v_in, num_v_in, atol=1e-6), (ana_v_in, num_v_in)
    assert np.allclose(ana_v_out_pos, num_v_out_pos, atol=1e-6), (ana_v_out_pos, num_v_out_pos)
    assert np.allclose(ana_v_out_neg, num_v_out_neg, atol=1e-6), (ana_v_out_neg, num_v_out_neg)

    print("Gradient check passed.")


if __name__ == "__main__":
    main()
