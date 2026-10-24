"""Attention target scores for task-specific query rows."""
import numpy as np
from .entropy import probabilities

TASK_COLUMNS = {"name_tracking": "name_tracking_score", "coreference": "coreference_score",
                "induction": "induction_score", "local_context": "previous_token_score"}


def task_attention_score(row: np.ndarray, target: int, key_positions: np.ndarray | None = None) -> float:
    """Return target mass; structurally masked targets receive zero attention."""
    p = probabilities(row)
    keys = np.arange(len(p)) if key_positions is None else np.asarray(key_positions)
    if keys.shape != p.shape or target < 0:
        raise ValueError("Invalid target or key positions")
    hits = np.flatnonzero(keys == target)
    if len(hits) > 1:
        raise ValueError("Key positions must be unique")
    return float(p[hits[0]]) if len(hits) else 0.0
