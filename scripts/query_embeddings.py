from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from word2vec.eval import EmbeddingInspector
from word2vec.utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect trained word vectors.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--analogy", nargs=3, default=None, metavar=("A", "B", "C"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays = np.load(args.checkpoint)
    vocab = load_json(args.vocab)

    inspector = EmbeddingInspector(
        word_vectors=arrays["word_vectors"],
        word_to_id=vocab["word_to_id"],
        id_to_word=vocab["id_to_word"],
    )

    print(f"Nearest neighbors for '{args.query}':")
    for word, score in inspector.nearest_neighbors(args.query, top_k=args.top_k):
        print(f"  {word:<16} {score:.4f}")

    if args.analogy is not None:
        a, b, c = args.analogy
        print(f"\nAnalogy: {a} : {b} :: {c} : ?")
        for word, score in inspector.analogy(a, b, c, top_k=args.top_k):
            print(f"  {word:<16} {score:.4f}")


if __name__ == "__main__":
    main()
