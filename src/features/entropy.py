"""Shannon entropy of normalized nonnegative attention rows."""
import numpy as np


def probabilities(row: np.ndarray) -> np.ndarray:
    """Validate and normalize a nonempty attention row."""
    row = np.asarray(row, dtype=np.float64)
    if row.ndim != 1 or not len(row) or not np.isfinite(row).all() or (row < 0).any():
        raise ValueError("Expected a finite, nonnegative, nonempty attention row")
    total = row.sum()
    if total <= 0:
        raise ValueError("An attention row must have positive mass")
    return row / total


def entropy(row: np.ndarray, normalized: bool = False) -> float:
    """Compute natural-log entropy, optionally divided by log(number of keys).

    For a single available key, normalized entropy is defined as zero.
    Zero probabilities remain in the support count.
    """
    p = probabilities(row)
    positive = p[p > 0]
    value = float(-np.sum(positive * np.log(positive)))
    return value / np.log(len(p)) if normalized and len(p) > 1 else (0.0 if normalized else value)
