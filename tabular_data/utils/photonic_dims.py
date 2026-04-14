"""Photonic mode/photon sizing helpers."""

from __future__ import annotations

import os


def get_photonic_mn(input_size: int, mode: str | None = None) -> tuple[int, int]:
    """
    Return `(m, n)` for photonic models from feature dimension.

    Modes:
    - `feature_plus_one` (default): m = d + 1, n = ceil(d/2)
    - `feature_equal`: m = d, n = ceil(d/2)
    - `feature_double`: m = 2d, n = d
    """
    if not isinstance(input_size, int) or input_size <= 0:
        raise ValueError(f"input_size must be a positive int, got {input_size}")

    resolved_mode = (mode or os.getenv("PHOTONIC_DIM_MODE", "feature_plus_one")).strip()
    if resolved_mode == "feature_plus_one":
        m = input_size + 1
        n = (input_size + 1) // 2
    elif resolved_mode == "feature_equal":
        m = input_size
        n = (input_size + 1) // 2
    elif resolved_mode == "feature_double":
        m = 2 * input_size
        n = input_size
    else:
        raise ValueError(
            f"Unknown PHOTONIC_DIM_MODE='{resolved_mode}'. "
            "Use one of: feature_plus_one, feature_equal, feature_double."
        )

    if n > m:
        raise ValueError(f"Invalid (m, n)=({m}, {n}) for input_size={input_size}")
    return m, n
