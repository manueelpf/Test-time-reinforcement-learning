from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import TrainingConfig
from .data import CorpusData, NegativeSampler, build_skipgram_pairs, subsample_token_ids
from .model import SkipGramNegativeSampling
from .utils import save_json


class Trainer:
    def __init__(self, model: SkipGramNegativeSampling, config: TrainingConfig):
        self.model = model
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def _learning_rate(self, epoch: int, global_step: int, total_steps: int) -> float:
        _ = epoch
        progress = global_step / max(1, total_steps)
        lr = self.config.learning_rate - progress * (self.config.learning_rate - self.config.min_learning_rate)
        return max(self.config.min_learning_rate, lr)

    def fit(self, corpus: CorpusData, output_dir: str | Path | None = None) -> dict:
        output_dir = Path(output_dir) if output_dir is not None else None
        sampler = NegativeSampler(corpus.counts, rng=self.rng)

        history: list[dict[str, float]] = []

        token_ids = subsample_token_ids(
            corpus.token_ids,
            corpus.counts,
            self.config.subsampling_t,
            rng=self.rng,
        )

        centers, contexts = build_skipgram_pairs(
            token_ids,
            window_size=self.config.window_size,
            rng=self.rng,
            dynamic_window=self.config.dynamic_window,
        )

        num_pairs = len(centers)
        if num_pairs == 0:
            raise ValueError("No training pairs were generated. Check corpus size and preprocessing settings.")

        num_batches = (num_pairs + self.config.batch_size - 1) // self.config.batch_size
        total_steps = self.config.epochs * num_batches
        global_step = 0

        for epoch in range(self.config.epochs):
            permutation = self.rng.permutation(num_pairs)
            centers_epoch = centers[permutation]
            contexts_epoch = contexts[permutation]

            epoch_loss_sum = 0.0
            epoch_examples = 0

            for batch_idx in range(num_batches):
                start = batch_idx * self.config.batch_size
                end = min(num_pairs, start + self.config.batch_size)

                batch_centers = centers_epoch[start:end]
                batch_contexts = contexts_epoch[start:end]
                batch_size = len(batch_centers)
                batch_negatives = sampler.sample(batch_size, self.config.negative_samples)

                lr = self._learning_rate(epoch, global_step, total_steps)
                step = self.model.train_step(
                    center_ids=batch_centers,
                    context_ids=batch_contexts,
                    negative_ids=batch_negatives,
                    learning_rate=lr,
                )

                epoch_loss_sum += step.loss * batch_size
                epoch_examples += batch_size
                global_step += 1

                if (batch_idx + 1) % self.config.report_every == 0 or batch_idx == num_batches - 1:
                    print(
                        f"[epoch {epoch + 1}/{self.config.epochs}] "
                        f"batch {batch_idx + 1}/{num_batches} "
                        f"lr={lr:.6f} loss={step.loss:.6f}"
                    )

            avg_epoch_loss = epoch_loss_sum / max(1, epoch_examples)
            entry = {"epoch": float(epoch + 1), "loss": float(avg_epoch_loss)}
            history.append(entry)
            print(f"==> epoch {epoch + 1} done | avg_loss={avg_epoch_loss:.6f}")

        result = {
            "history": history,
            "num_tokens": int(len(corpus.token_ids)),
            "num_pairs": int(num_pairs),
            "vocab_size": int(corpus.vocab_size),
        }

        if output_dir is not None:
            self.save_artifacts(output_dir, corpus, result)

        return result

    def save_artifacts(self, output_dir: Path, corpus: CorpusData, result: dict) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_dir / "model.npz",
            input_embeddings=self.model.get_input_embeddings(),
            output_embeddings=self.model.get_output_embeddings(),
            word_vectors=self.model.get_word_vectors(combine="mean"),
        )

        save_json(
            output_dir / "vocab.json",
            {
                "id_to_word": corpus.id_to_word,
                "word_to_id": corpus.word_to_id,
                "counts": corpus.counts.tolist(),
            },
        )
        save_json(output_dir / "metrics.json", result)
        save_json(output_dir / "config.json", asdict(self.config))
