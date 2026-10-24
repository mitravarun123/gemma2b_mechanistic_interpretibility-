"""Known paired-statistic results and degenerate measurement handling."""
import numpy as np
import pytest
from src.validation.stats import paired_statistics,benjamini_hochberg,recovery_score
from src.validation.consistency import safe_spearman


def test_bootstrap_paired_effect_and_exact_test():
    values = np.array([1.,2.,3.,4.])
    stats = paired_statistics(values,42,200,100)
    assert stats['mean']==2.5 and stats['n']==4
    assert stats['cohens_dz']==pytest.approx(values.mean()/values.std(ddof=1))
    assert stats['ci_low']<=2.5<=stats['ci_high']
    assert stats['p_value']==pytest.approx(2/16)
    assert stats==paired_statistics(values,42,200,100)
    zero = paired_statistics(np.zeros(4),42,10,10)
    assert zero['p_value']==1 and zero['cohens_dz']==0
    empty = paired_statistics(np.array([np.nan]),42,10,10)
    assert empty['n']==0 and np.isnan(empty['mean'])


def test_bh_and_recovery_endpoints():
    actual = benjamini_hochberg(np.array([0.01,0.04,0.03,np.nan]))
    np.testing.assert_allclose(actual[:3],[0.03,0.04,0.04])
    assert np.isnan(actual[3])
    clean = np.array([3.,2.,1.]); bad = np.array([1.,4.,1.])
    np.testing.assert_allclose(recovery_score(clean,clean,bad)[:2],1)
    np.testing.assert_allclose(recovery_score(bad,clean,bad)[:2],0)
    assert np.isnan(recovery_score(clean,clean,bad)[2])
    assert np.isnan(recovery_score(np.array([1.]),np.array([1.+1e-8]),np.array([1.]))).all()
    rho,n,status = safe_spearman([1,2,3],[3,2,1])
    assert rho==-1 and n==3 and status=='ok'
    assert safe_spearman([1,1,1],[1,2,3])[2]=='constant_vector'
    assert safe_spearman([],[])[2]=='insufficient_pairs'


def test_noising_and_denoising_use_same_baseline_gap():
    from src.validation.patching import patch_effects
    clean = np.array([[1.,4.,0.8],[0.,-1.,0.1]])
    bad = np.array([[0.,1.,0.2],[1.,2.,0.4]])
    np.testing.assert_allclose(patch_effects(clean,clean,bad,0,1e-6)[:,3:5],1)
    np.testing.assert_allclose(patch_effects(bad,clean,bad,0,1e-6)[:,3:5],0)
    np.testing.assert_allclose(patch_effects(bad,clean,bad,1,1e-6)[:,3:5],1)
    assert np.isnan(patch_effects(clean,clean,bad,0,1e-6)[1,5:]).all()


def test_sign_flip_p_values_are_scale_invariant():
    original = paired_statistics(np.array([1.,2.,3.,4.]),42,20,20)
    scaled = paired_statistics(np.array([1.,2.,3.,4.])*1e-14,42,20,20)
    assert original['p_value']==scaled['p_value']
