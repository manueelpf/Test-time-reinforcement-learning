from __future__ import annotations

import json
from pathlib import Path

import numpy as np


EPS = 1e-12


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    negative = ~positive
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[negative])
    out[negative] = exp_x / (1.0 + exp_x)
    return out


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable log(sigmoid(x))."""
    return -np.logaddexp(0.0, -x)


def cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    matrix_norm = np.linalg.norm(matrix, axis=1) + EPS
    vector_norm = np.linalg.norm(vector) + EPS
    return (matrix @ vector) / (matrix_norm * vector_norm)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
