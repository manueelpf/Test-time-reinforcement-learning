from __future__ import annotations

import numpy as np

from .utils import cosine_similarity


class EmbeddingInspector:
    def __init__(self, word_vectors: np.ndarray, word_to_id: dict[str, int], id_to_word: list[str]):
        self.word_vectors = word_vectors
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word

    def nearest_neighbors(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if query not in self.word_to_id:
            raise KeyError(f"Word '{query}' is not in the vocabulary.")

        query_id = self.word_to_id[query]
        query_vec = self.word_vectors[query_id]
        sims = cosine_similarity(self.word_vectors, query_vec)
        sims[query_id] = -np.inf

        top_ids = np.argsort(-sims)[:top_k]
        return [(self.id_to_word[idx], float(sims[idx])) for idx in top_ids]

    def analogy(self, a: str, b: str, c: str, top_k: int = 5) -> list[tuple[str, float]]:
        for word in (a, b, c):
            if word not in self.word_to_id:
                raise KeyError(f"Word '{word}' is not in the vocabulary.")

        vec = (
            self.word_vectors[self.word_to_id[b]]
            - self.word_vectors[self.word_to_id[a]]
            + self.word_vectors[self.word_to_id[c]]
        )
        sims = cosine_similarity(self.word_vectors, vec)
        for word in (a, b, c):
            sims[self.word_to_id[word]] = -np.inf

        top_ids = np.argsort(-sims)[:top_k]
        return [(self.id_to_word[idx], float(sims[idx])) for idx in top_ids]
