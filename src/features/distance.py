"""Attention-weighted distance in original BOS-inclusive token positions."""
import numpy as np
from .entropy import probabilities


def mean_distance(row: np.ndarray, query: int, key_positions: np.ndarray | None = None) -> float:
    """Return E[query-key], preserving distances after removing BOS."""
    p = probabilities(row)
    keys = np.arange(len(p)) if key_positions is None else np.asarray(key_positions)
    if keys.shape != p.shape or np.any(keys > query) or np.any(keys < 0):
        raise ValueError("Key positions must align with the row and be causal")
    return float(np.sum(p * (query - keys)))
