"""Vectorized row statistics respecting causal and sliding-window support."""
import numpy as np

CATEGORIES = ("names", "pronouns", "punctuation", "BOS", "function_words", "other")
METRICS = ("entropy", "entropy_normalized", "mean_distance", "gini", "top1_mass",
           "task_score", "wrong_score", "contrast_score", *[f"mass_{c}" for c in CATEGORIES])
BOS_MODES = ("included", "excluded")


def summarize_rows(attention: np.ndarray, queries: np.ndarray, categories: list[str],
                   targets: np.ndarray | None = None, wrong: np.ndarray | None = None,
                   window: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize [heads, queries, keys] twice and return metrics, validity, BOS mass.

    Output metrics have shape [2, heads, queries, metrics]. Undefined rows after
    BOS removal are NaN, never uniform fallbacks. Entropy/Gini use all structurally
    available keys, including keys receiving exactly zero attention.
    """
    a = np.asarray(attention, dtype=np.float64)
    queries = np.asarray(queries, dtype=int)
    if a.ndim != 3 or a.shape[1] != len(queries) or a.shape[2] != len(categories):
        raise ValueError("Attention dimensions do not match query/category positions")
    if not np.isfinite(a).all() or (a < 0).any() or len(queries) == 0:
        raise ValueError("Attention must be finite, nonnegative and contain queries")
    if queries.min() < 0 or queries.max() >= a.shape[2]:
        raise ValueError("Query out of bounds")
    if window is not None and window < 1:
        raise ValueError("Sliding window must be positive")
    if categories[0] != 'BOS' or any(c not in CATEGORIES for c in categories):
        raise ValueError("Unknown token category or missing BOS category")
    keys = np.arange(a.shape[2])
    support = keys[None, :] <= queries[:, None]
    if window is not None:
        support &= keys[None, :] > queries[:, None] - window
    if np.any(a * ~support[None, :, :] > 1e-6):
        raise ValueError("Attention has mass outside its structural causal support")
    sums = a.sum(-1)
    if not np.allclose(sums, 1, atol=0.02, rtol=0.02):
        raise ValueError("Attention rows must sum to one (allowing low-precision rounding)")
    a = a / sums[..., None]
    bos = a[..., 0].copy()
    result = np.full((2, *a.shape[:2], len(METRICS)), np.nan)
    valid = np.zeros((2, *a.shape[:2]), dtype=bool)
    for mode in range(2):
        allowed = support.copy()
        if mode:
            allowed[:, 0] = False
        mass = (a * allowed[None, :, :]).sum(-1)
        ok = mass > 0
        valid[mode] = ok
        p = np.divide(a * allowed[None, :, :], mass[..., None],
            out=np.zeros_like(a), where=mass[..., None] > 0)
        n_keys = allowed.sum(-1)
        logs = np.log(p, out=np.zeros_like(p), where=p > 0)
        ent = -(p * logs).sum(-1)
        denominator = np.log(np.maximum(n_keys, 1))
        norm = np.divide(ent, denominator[None, :], out=np.zeros_like(ent), where=denominator[None, :] > 0)
        distance = (p * (queries[:, None] - keys[None, :])[None, :, :]).sum(-1)
        # Sorting masked zeros first leaves available-key rank weights unchanged.
        ordered = np.sort(p, axis=-1)
        rank = np.arange(1, a.shape[2]+1)[None, :] - (a.shape[2]-n_keys[:, None])
        weights = 2 * rank - n_keys[:, None] - 1
        gini_numerator = (ordered * weights[None, :, :]).sum(-1)
        inequality = np.divide(gini_numerator, (n_keys-1)[None, :], out=np.zeros_like(ent), where=(n_keys-1)[None, :] > 0)
        top1 = p.max(-1)
        def target_mass(positions):
            if positions is None:
                return np.full_like(ent, np.nan)
            positions = np.asarray(positions, dtype=int)
            if positions.shape != queries.shape or np.any(positions >= a.shape[2]) or np.any(positions > queries) or np.any(positions < -1):
                raise ValueError("Target positions must align with queries and be causal")
            values = p[:, np.arange(len(queries)), np.maximum(positions, 0)]
            return np.where(positions[None, :] >= 0, values, np.nan)
        target, wrong_mass = target_mass(targets), target_mass(wrong)
        categories_mass = [(p * (np.asarray(categories) == c)[None, None, :]).sum(-1) for c in CATEGORIES]
        summary = np.stack([ent, norm, distance, np.clip(inequality, 0, 1), top1,
                            target, wrong_mass, target-wrong_mass, *categories_mass], axis=-1)
        result[mode] = np.where(ok[..., None], summary, np.nan)
    return result, valid, bos


def finite_mean(values: np.ndarray, axis: int) -> np.ndarray:
    """Average finite entries, preserving missingness without runtime warnings."""
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    return np.divide(np.where(finite, values, 0).sum(axis=axis), count,
        out=np.full(count.shape, np.nan, dtype=np.float64), where=count > 0)
