"""Tiny-model integration of saved attention, feature tables and resumability."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import GemmaConfig, GemmaForCausalLM, Gemma2Config, Gemma2ForCausalLM
from test_dataset import tokenizer
from src.dataset.names import build_name_pools
from src.dataset.generate import generate_split, figure_example
from src.extraction.attention import extract_attention, extraction_batches
from src.features.compute import aggregate_features
from src.features.rows import METRICS
from src.utils.io import read_config


@pytest.fixture(params=['gemma', 'gemma2'])
def extracted(tmp_path, request):
    tok = tokenizer()
    pool = build_name_pools(tok, json.loads(Path('data/raw/names.json').read_text()), 42)
    records = []
    for split in ('discovery', 'validation'):
        subset, _ = generate_split(tok, pool[split], split, 4, 42)
        records.extend(subset)
    records.append(figure_example(tok))
    config = read_config('configs/debug.yaml')
    config['extraction']['example_prompts'] = 2
    config['batch_size'] = 4
    types = (GemmaConfig, GemmaForCausalLM) if request.param == 'gemma' else (Gemma2Config, Gemma2ForCausalLM)
    torch.manual_seed(42)
    model = types[1](types[0](vocab_size=len(tok), hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=256, sliding_window=4, attn_implementation='eager')).eval()
    folder = tmp_path / 'attention'
    extract_attention(model, records, folder, config)
    return model, records, folder, config, request.param


def test_extraction_and_saved_feature_pipeline(extracted, tmp_path, monkeypatch):
    model, records, folder, config, variant = extracted
    index = json.loads((folder/'index.json').read_text())
    assert index['complete'] and len(index['records']) == len(records)
    assert sum(e['has_full_attention'] for e in index['records'].values()) == 2
    assert index['records']['figure1-coreference']['has_full_attention']
    assert index['sliding_windows'] == ([4, None] if variant == 'gemma2' else [None, None])
    for record in records:
        entry = index['records'][record['id']]
        with np.load(folder/entry['file']) as data:
            assert data['rows'].shape[:3] == (2, 2, 4)
            assert data['selected_summary'].shape == (2, 2, 4, len(METRICS))
            np.testing.assert_allclose(data['rows'][0].sum(-1), 1, atol=1e-6)
            assert np.all(data['all_valid_query_counts'][1] == len(record['token_ids'])-1)
            assert data['all_summary'][0, 0, 0, METRICS.index('top1_mass')] > 0
            if entry['has_full_attention']:
                assert np.isnan(data['full_attention'][1, :, :, 0]).all()
    # Compare retained rows against a separate real forward pass.
    record = records[0]
    with torch.no_grad():
        attentions = model(torch.tensor([record['token_ids']]), output_attentions=True, use_cache=False).attentions
    with np.load(folder/index['records'][record['id']]['file']) as data:
        queries = data['query_positions']
        np.testing.assert_allclose(data['rows'][0, 0], attentions[0][0].numpy()[:, queries], atol=1e-6)
    features = tmp_path/'features'
    outputs = aggregate_features(folder, features, config)
    heads = pd.read_csv(outputs['head_features.csv'])
    assert len(heads) == 2*2*2*4
    assert set(heads['split']) == {'discovery','validation'}
    assert set(heads['bos_mode']) == {'included','excluded'}
    assert (heads['valid_tasks'] == 4).all()
    assert np.isfinite(heads['specialization']).all()
    np.testing.assert_allclose(heads['specialization'], heads[[f'z_{name}' for name in ['name_tracking_score','coreference_score','induction_score','previous_token_score']]].max(axis=1))
    excluded = heads[heads['bos_mode']=='excluded']
    np.testing.assert_allclose(excluded['mass_BOS'], 0)
    for (_, mode), group in heads.groupby(['split','bos_mode']):
        for column in ['name_tracking_score','coreference_score','induction_score','previous_token_score']:
            np.testing.assert_allclose(group['z_'+column].mean(), 0, atol=1e-6)
    tasks = pd.read_csv(outputs['task_features.csv'])
    group = tasks[(tasks['split']=='discovery') & (tasks['bos_mode']=='included') & (tasks['layer']==0) & (tasks['head']==0)]
    head = heads[(heads['split']=='discovery') & (heads['bos_mode']=='included') & (heads['layer']==0) & (heads['head']==0)].iloc[0]
    assert head['entropy_normalized'] == pytest.approx(group['entropy_normalized'].mean())
    first_ids = [r['id'] for r in records if r['split']=='discovery' and r['task']=='coreference']
    expected = []
    for ident in first_ids:
        with np.load(folder/index['records'][ident]['file']) as shard:
            expected.append(shard['selected_summary'][0, 0, 0, METRICS.index('task_score')])
    assert head['coreference_score'] == pytest.approx(np.mean(expected))
    if variant == 'gemma2':
        assert head['coreference_score'] == 0  # Antecedents lie outside layer zero's window.
    before = (features/'head_features.csv').read_bytes()
    # Verified extraction resume must not call model.forward.
    monkeypatch.setattr(model, 'forward', lambda *a, **k: (_ for _ in ()).throw(AssertionError('cache missed')))
    extract_attention(model, records, folder, config)
    aggregate_features(folder, features, config)
    assert (features/'head_features.csv').read_bytes() == before
    layer = pd.read_csv(outputs['layer_summaries.csv'])
    assert (layer['ci_method']=='bootstrap_heads').all()
    assert (layer['specialization_ci_low'] <= layer['specialization_ci_high']).all()


def test_memory_budget_and_missing_results(tmp_path):
    records = [{'id': 'x', 'token_ids': [1]*100} for _ in range(4)]
    batches = extraction_batches(records, 4, 2, 4, 0.4)
    assert [len(b) for b in batches] == [1,1,1,1]
    with pytest.raises(MemoryError):
        extraction_batches(records, 4, 2, 4, 0.1)
    with pytest.raises(FileNotFoundError, match='step 02'):
        aggregate_features(tmp_path/'missing', tmp_path/'features', read_config('configs/debug.yaml'))


def test_shard_repair_and_interrupted_index_resume(extracted, tmp_path, monkeypatch):
    model, records, folder, config, _ = extracted
    index = json.loads((folder/'index.json').read_text())
    ident = records[0]['id']
    path = folder/index['records'][ident]['file']
    path.write_bytes(b'corrupted shard')
    with pytest.raises(ValueError, match='checksum mismatch'):
        aggregate_features(folder, tmp_path/'features', config)
    original = model.forward
    calls = []
    def count_forward(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(model, 'forward', count_forward)
    extract_attention(model, records, folder, config)
    assert len(calls) == 1
    index = json.loads((folder/'index.json').read_text())
    index['records'].pop(ident)
    index['complete'] = False
    (folder/'index.json').write_text(json.dumps(index))
    monkeypatch.setattr(model, 'forward', lambda *a, **k: (_ for _ in ()).throw(AssertionError('sidecar resume missed')))
    extract_attention(model, records, folder, config)
    assert json.loads((folder/'index.json').read_text())['complete']
