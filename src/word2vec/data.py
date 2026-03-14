from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")
UNK_TOKEN = "<unk>"


@dataclass(slots=True)
class CorpusData:
    tokens: list[str]
    token_ids: np.ndarray
    word_to_id: dict[str, int]
    id_to_word: list[str]
    counts: np.ndarray

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_word)


class CorpusProcessor:
    def __init__(self, min_count: int = 1, max_vocab_size: int | None = None):
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [tok.lower() for tok in TOKEN_PATTERN.findall(text)]

    def build_vocab(self, tokens: Iterable[str]) -> tuple[dict[str, int], list[str], np.ndarray]:
        counter = Counter(tokens)
        items = [(word, freq) for word, freq in counter.items() if freq >= self.min_count]
        items.sort(key=lambda x: (-x[1], x[0]))

        if self.max_vocab_size is not None:
            items = items[: max(0, self.max_vocab_size - 1)]

        id_to_word = [UNK_TOKEN] + [word for word, _ in items]
        word_to_id = {word: idx for idx, word in enumerate(id_to_word)}

        unk_count = sum(freq for word, freq in counter.items() if word not in word_to_id)
        counts = np.array([unk_count] + [freq for _, freq in items], dtype=np.int64)
        return word_to_id, id_to_word, counts

    def encode_tokens(self, tokens: list[str], word_to_id: dict[str, int]) -> np.ndarray:
        unk_id = word_to_id[UNK_TOKEN]
        return np.array([word_to_id.get(tok, unk_id) for tok in tokens], dtype=np.int64)

    def process_text(self, text: str) -> CorpusData:
        tokens = self.tokenize(text)
        word_to_id, id_to_word, counts = self.build_vocab(tokens)
        token_ids = self.encode_tokens(tokens, word_to_id)
        return CorpusData(tokens, token_ids, word_to_id, id_to_word, counts)


def subsample_token_ids(token_ids: np.ndarray, counts: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray:
    """
    Keep probabilities follow the standard word2vec-style subsampling heuristic.

    On very small corpora, aggressive subsampling can remove too many tokens and produce
    no training pairs. In that case we safely fall back to the original sequence.
    """
    if t <= 0:
        return token_ids

    freqs = counts / counts.sum()
    p_keep = np.minimum(1.0, np.sqrt(t / np.maximum(freqs, 1e-12)) + (t / np.maximum(freqs, 1e-12)))
    keep_mask = rng.random(len(token_ids)) < p_keep[token_ids]
    filtered = token_ids[keep_mask]

    if len(filtered) < max(10, len(token_ids) // 10):
        return token_ids
    return filtered


def build_skipgram_pairs(
    token_ids: np.ndarray,
    window_size: int,
    rng: np.random.Generator,
    dynamic_window: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    centers: list[int] = []
    contexts: list[int] = []

    n = len(token_ids)
    for idx in range(n):
        current_window = int(rng.integers(1, window_size + 1)) if dynamic_window else window_size
        left = max(0, idx - current_window)
        right = min(n, idx + current_window + 1)

        center_id = int(token_ids[idx])
        for j in range(left, right):
            if j == idx:
                continue
            centers.append(center_id)
            contexts.append(int(token_ids[j]))

    return np.array(centers, dtype=np.int64), np.array(contexts, dtype=np.int64)


class NegativeSampler:
    """
    Samples negative word ids from a smoothed unigram distribution p(w)^0.75.
    """

    def __init__(self, counts: np.ndarray, rng: np.random.Generator, power: float = 0.75):
        probs = np.power(counts.astype(np.float64), power)
        probs_sum = probs.sum()
        if probs_sum <= 0:
            raise ValueError("Negative sampling distribution is invalid: sum <= 0.")
        self.probs = probs / probs_sum
        self.cdf = np.cumsum(self.probs)
        self.rng = rng

    def sample(self, batch_size: int, num_negative: int) -> np.ndarray:
        draws = self.rng.random((batch_size, num_negative))
        return np.searchsorted(self.cdf, draws, side="right").astype(np.int64)
