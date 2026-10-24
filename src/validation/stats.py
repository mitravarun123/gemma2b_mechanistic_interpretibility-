"""Paired bootstrap uncertainty, paired effect sizes and FDR correction."""
import itertools
import numpy as np


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Adjust finite p values; leave undefined tests undefined."""
    p = np.asarray(p_values,dtype=float)
    valid = np.isfinite(p)
    if np.any((p[valid]<0)|(p[valid]>1)):
        raise ValueError('p values must be in [0,1]')
    result = np.full_like(p,np.nan)
    values = p[valid]; order = np.argsort(values,kind='stable')
    adjusted = values[order]*len(values)/np.arange(1,len(values)+1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted); restored[order] = np.clip(adjusted,0,1)
    result[valid] = restored
    return result


def paired_statistics(differences: np.ndarray, seed: int, repeats: int = 1000,
                      permutation_repeats: int = 1000) -> dict:
    """Summarize finite paired differences with percentile CI and two-sided sign-flip p.

    Exact sign enumeration is used for n <= 12; otherwise Monte Carlo sign flips
    use the plus-one correction. Cohen's dz divides mean by sample SD; nonzero
    constant differences give signed infinity. Fewer than two pairs have no CI/dz.
    """
    if repeats<2 or permutation_repeats<1:
        raise ValueError('Need >=2 bootstrap and >=1 permutation repeats')
    values = np.asarray(differences,dtype=float)
    if values.ndim!=1:
        raise ValueError('Paired differences must be one-dimensional')
    values = values[np.isfinite(values)]; n = len(values)
    result = {'n':n,'mean':np.nan,'ci_low':np.nan,'ci_high':np.nan,'cohens_dz':np.nan,'p_value':np.nan}
    if not n:
        return result
    mean = float(values.mean()); result['mean'] = mean
    rng = np.random.default_rng(seed)
    if n>=2:
        draws = []
        for start in range(0,repeats,256):
            indices = rng.integers(0,n,size=(min(256,repeats-start),n))
            draws.extend(values[indices].mean(axis=1))
        result['ci_low'],result['ci_high'] = np.quantile(draws,[0.025,0.975])
        sd = values.std(ddof=1)
        result['cohens_dz'] = mean/sd if sd>0 else (0.0 if mean==0 else float(np.copysign(np.inf,mean)))
    tolerance = 8*np.finfo(float).eps*float(np.mean(np.abs(values)))
    cutoff = max(0,abs(mean)-tolerance)
    if n<=12:
        count = sum(abs(np.dot(signs,values)/n)>=cutoff for signs in itertools.product((-1,1),repeat=n))
        result['p_value'] = count/(2**n)
    else:
        count = 0
        for start in range(0,permutation_repeats,256):
            size = min(256,permutation_repeats-start)
            signs = rng.choice([-1,1],size=(size,n))
            count += int((np.abs((signs*values).mean(axis=1))>=cutoff).sum())
        result['p_value'] = (count+1)/(permutation_repeats+1)
    return result


def recovery_score(patched: np.ndarray, clean: np.ndarray, corrupted: np.ndarray,
                   epsilon: float = 1e-6) -> np.ndarray:
    """Return signed normalized recovery; tiny clean-corrupt gaps are undefined."""
    if not np.isfinite(epsilon) or epsilon<=0:
        raise ValueError('Recovery epsilon must be positive')
    patched,clean,corrupted = np.broadcast_arrays(patched,clean,corrupted)
    gap = clean-corrupted
    return np.divide(patched-corrupted,gap,out=np.full(gap.shape,np.nan,dtype=float),
        where=np.isfinite(gap)&(np.abs(gap)>epsilon))
