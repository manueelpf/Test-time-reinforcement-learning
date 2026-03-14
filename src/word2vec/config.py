from dataclasses import dataclass


@dataclass(slots=True)
class TrainingConfig:
    min_count: int = 1
    max_vocab_size: int | None = None
    embedding_dim: int = 64
    window_size: int = 2
    dynamic_window: bool = True
    subsampling_t: float = 1e-5
    negative_samples: int = 5
    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 0.025
    min_learning_rate: float = 0.0001
    seed: int = 42
    report_every: int = 100
