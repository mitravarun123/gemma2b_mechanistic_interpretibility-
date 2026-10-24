"""Complementary attention concentration metrics."""
import numpy as np
from .entropy import probabilities


def gini(row: np.ndarray) -> float:
    """Finite-support-normalized Gini: zero uniform, one one-hot (n > 1).

    A single available key has no inequality and is assigned zero Gini.
    """
    p = np.sort(probabilities(row))
    n = len(p)
    if n == 1:
        return 0.0
    raw = np.sum((2 * np.arange(1, n + 1) - n - 1) * p) / n
    return float(np.clip(raw * n / (n - 1), 0, 1))


def top1_mass(row: np.ndarray) -> float:
    """Return maximum normalized attention probability."""
    return float(probabilities(row).max())
