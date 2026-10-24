"""Analytical checks of attention feature definitions and BOS edge cases."""
import numpy as np
import pytest
from src.features.entropy import entropy
from src.features.distance import mean_distance
from src.features.sparsity import gini, top1_mass
from src.features.task_score import task_attention_score
from src.features.rows import METRICS, summarize_rows


def test_entropy_uniform_and_one_hot():
    assert entropy(np.ones(5)) == pytest.approx(np.log(5))
    assert entropy(np.ones(5), normalized=True) == pytest.approx(1)
    assert entropy(np.array([0, 0, 1, 0])) == 0
    assert entropy(np.array([0, 0, 1, 0]), normalized=True) == 0
    assert entropy(np.array([1]), normalized=True) == 0
    with pytest.raises(ValueError):
        entropy(np.zeros(4))


def test_sparsity_uniform_one_hot_and_distance():
    assert gini(np.array([0, 1, 0, 0])) == pytest.approx(1)
    assert top1_mass(np.array([0, 1, 0, 0])) == 1
    assert gini(np.ones(4)) == pytest.approx(0)
    assert top1_mass(np.ones(4)) == pytest.approx(0.25)
    assert mean_distance(np.array([0, 0, 1, 0]), query=3) == 1
    assert mean_distance(np.array([0, 1, 0]), query=3, key_positions=np.array([1, 2, 3])) == 1
    assert task_attention_score(np.array([0.2, 0.8]), 0, np.array([2, 3])) == 0


def test_vectorized_matches_scalar_and_bos_removal():
    queries = np.array([1, 3])
    attention = np.array([[[0.8, 0.2, 0, 0], [0.5, 0, 0.5, 0]],
                          [[1, 0, 0, 0], [0.2, 0.1, 0.3, 0.4]]])
    summary, valid, sink = summarize_rows(attention, queries, ['BOS', 'names', 'other', 'pronouns'],
        np.array([1, 2]), np.array([-1, 1]))
    np.testing.assert_allclose(sink, [[0.8, 0.5], [1, 0.2]])
    for mode in range(2):
        for head in range(2):
            for index, query in enumerate(queries):
                keys = np.arange(mode, query+1)
                row = attention[head, index, keys]
                if row.sum() == 0:
                    assert not valid[mode, head, index]
                    assert np.isnan(summary[mode, head, index]).all()
                    continue
                values = summary[mode, head, index]
                assert values[METRICS.index('entropy')] == pytest.approx(entropy(row))
                assert values[METRICS.index('entropy_normalized')] == pytest.approx(entropy(row, True))
                assert values[METRICS.index('gini')] == pytest.approx(gini(row))
                assert values[METRICS.index('top1_mass')] == pytest.approx(top1_mass(row))
                assert values[METRICS.index('mean_distance')] == pytest.approx(mean_distance(row, query, keys))
                assert values[METRICS.index('task_score')] == pytest.approx(task_attention_score(row, [1, 2][index], keys))
                assert sum(values[METRICS.index('mass_'+c)] for c in ['names', 'pronouns', 'punctuation', 'BOS', 'function_words', 'other']) == pytest.approx(1)
    assert summary[1, 0, 1, METRICS.index('mean_distance')] == 1
    assert summary[1, 0, 1, METRICS.index('mass_BOS')] == 0


def test_sliding_support_and_undefined_bos_query():
    summary, valid, _ = summarize_rows(np.array([[[0, 0, 0.5, 0.5]]]), np.array([3]),
        ['BOS', 'other', 'other', 'other'], np.array([1]), window=2)
    assert summary[0, 0, 0, METRICS.index('entropy_normalized')] == pytest.approx(1)
    assert summary[0, 0, 0, METRICS.index('task_score')] == 0
    assert summary[0, 0, 0, METRICS.index('gini')] == 0
    summary, valid, _ = summarize_rows(np.ones((1, 1, 1)), np.array([0]), ['BOS'])
    assert valid[0, 0, 0] and not valid[1, 0, 0]
    assert np.isnan(summary[1]).all()
    with pytest.raises(ValueError, match='structural'):
        summarize_rows(np.array([[[0.5, 0, 0.5, 0]]]), np.array([3]), ['BOS','other','other','other'], window=2)
    with pytest.raises(ValueError, match='sum to one'):
        summarize_rows(np.array([[[0.1, 0.1]]]), np.array([1]), ['BOS','other'])
