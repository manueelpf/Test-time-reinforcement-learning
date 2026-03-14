from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from word2vec import CorpusProcessor, SkipGramNegativeSampling, Trainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train skip-gram with negative sampling in pure NumPy.")
    parser.add_argument("--text-path", type=str, required=True, help="Path to a plain-text corpus.")
    parser.add_argument("--output-dir", type=str, default="artifacts/run", help="Where to store model artifacts.")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--negative-samples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--min-learning-rate", type=float, default=0.0001)
    parser.add_argument("--subsampling-t", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dynamic-window", action="store_true", help="Use a random window from 1..window-size.")
    parser.add_argument("--report-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = Path(args.text_path).read_text(encoding="utf-8")

    config = TrainingConfig(
        min_count=args.min_count,
        max_vocab_size=args.max_vocab_size,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        dynamic_window=args.dynamic_window,
        subsampling_t=args.subsampling_t,
        negative_samples=args.negative_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        seed=args.seed,
        report_every=args.report_every,
    )

    processor = CorpusProcessor(min_count=config.min_count, max_vocab_size=config.max_vocab_size)
    corpus = processor.process_text(text)

    model = SkipGramNegativeSampling(
        vocab_size=corpus.vocab_size,
        embedding_dim=config.embedding_dim,
        seed=config.seed,
    )
    trainer = Trainer(model=model, config=config)
    result = trainer.fit(corpus, output_dir=args.output_dir)

    print("\nTraining finished.")
    print(f"Vocabulary size: {result['vocab_size']}")
    print(f"Token count:     {result['num_tokens']}")
    print(f"Training pairs:  {result['num_pairs']}")
    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
