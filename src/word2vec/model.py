from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import log_sigmoid, sigmoid


@dataclass(slots=True)
class StepResult:
    loss: float
    positive_scores: np.ndarray
    negative_scores: np.ndarray


class SkipGramNegativeSampling:
    def __init__(self, vocab_size: int, embedding_dim: int, seed: int = 42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rng = np.random.default_rng(seed)

        scale = 0.5 / max(1, embedding_dim)
        self.input_embeddings = self.rng.uniform(
            low=-scale,
            high=scale,
            size=(vocab_size, embedding_dim),
        ).astype(np.float64)
        self.output_embeddings = self.rng.uniform(
            low=-scale,
            high=scale,
            size=(vocab_size, embedding_dim),
        ).astype(np.float64)

    def get_input_embeddings(self) -> np.ndarray:
        return self.input_embeddings

    def get_output_embeddings(self) -> np.ndarray:
        return self.output_embeddings

    def get_word_vectors(self, combine: str = "mean") -> np.ndarray:
        if combine == "input":
            return self.input_embeddings.copy()
        if combine == "output":
            return self.output_embeddings.copy()
        if combine == "sum":
            return self.input_embeddings + self.output_embeddings
        if combine == "mean":
            return 0.5 * (self.input_embeddings + self.output_embeddings)
        raise ValueError(f"Unknown combine mode: {combine}")

    def train_step(
        self,
        center_ids: np.ndarray,
        context_ids: np.ndarray,
        negative_ids: np.ndarray,
        learning_rate: float,
    ) -> StepResult:
        """
        One minibatch SGD step for SGNS.

        center_ids:   (B,)
        context_ids:  (B,)
        negative_ids: (B, K)
        """
        center_vecs = self.input_embeddings[center_ids]          # (B, D)
        context_vecs = self.output_embeddings[context_ids]       # (B, D)
        negative_vecs = self.output_embeddings[negative_ids]     # (B, K, D)

        positive_scores = np.sum(center_vecs * context_vecs, axis=1)                      # (B,)
        negative_scores = np.einsum("bd,bkd->bk", center_vecs, negative_vecs)            # (B, K)

        loss_vec = -log_sigmoid(positive_scores) - np.sum(log_sigmoid(-negative_scores), axis=1)
        loss = float(np.mean(loss_vec))

        grad_pos = sigmoid(positive_scores) - 1.0                 # (B,)
        grad_neg = sigmoid(negative_scores)                       # (B, K)

        grad_center = (
            grad_pos[:, None] * context_vecs
            + np.sum(grad_neg[:, :, None] * negative_vecs, axis=1)
        )                                                         # (B, D)
        grad_context = grad_pos[:, None] * center_vecs            # (B, D)
        grad_negative = grad_neg[:, :, None] * center_vecs[:, None, :]   # (B, K, D)

        # SGD updates with correct accumulation for repeated indices.
        np.add.at(self.input_embeddings, center_ids, -learning_rate * grad_center)
        np.add.at(self.output_embeddings, context_ids, -learning_rate * grad_context)

        flat_negative_ids = negative_ids.reshape(-1)
        flat_negative_grads = grad_negative.reshape(-1, self.embedding_dim)
        np.add.at(self.output_embeddings, flat_negative_ids, -learning_rate * flat_negative_grads)

        return StepResult(loss=loss, positive_scores=positive_scores, negative_scores=negative_scores)
