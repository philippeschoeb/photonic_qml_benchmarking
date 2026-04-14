"""Synthetic 3-class spiral dataset generation for tabular benchmarking."""

from __future__ import annotations

import numpy as np
from merlin.datasets import spiral
from sklearn.model_selection import train_test_split


def generate_spiral_dataset(
    n_samples: int,
    n_features: int,
    n_classes: int = 3,
    test_size: float = 0.2,
    random_state: int | None = None,
):
    """Generate spiral samples and return train/test numpy arrays."""
    if not isinstance(n_features, int) or n_features < 2 or n_features > 100:
        raise ValueError(
            f"Invalid spiral feature dimension d={n_features}. Expected integer in [2, 100]."
        )
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"Invalid n_samples={n_samples}. Expected a positive integer.")
    if not isinstance(n_classes, int) or n_classes < 2:
        raise ValueError(f"Invalid n_classes={n_classes}. Expected integer >= 2.")

    # Keep generation deterministic for a given random_state without leaking RNG state.
    if random_state is None:
        x, y, _ = spiral.get_data(
            num_instances=n_samples,
            num_features=n_features,
            num_classes=n_classes,
        )
    else:
        old_state = np.random.get_state()
        np.random.seed(random_state)
        try:
            x, y, _ = spiral.get_data(
                num_instances=n_samples,
                num_features=n_features,
                num_classes=n_classes,
            )
        finally:
            np.random.set_state(old_state)

    x = np.asarray(x)
    y = np.asarray(y).astype(np.int64)

    return train_test_split(
        x,
        y,
        test_size=test_size,
        train_size=int(n_samples * (1 - test_size)),
        random_state=random_state,
        stratify=y,
    )


__all__ = ["generate_spiral_dataset"]
