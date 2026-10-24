"""Discovery-only fit, prompt bootstrap, shortlist and saved-transform contracts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score
from test_extraction import extracted
from src.clustering.cluster_heads import (cluster_heads, choose_k, standardized_space,
    COLUMNS, candidate_shortlist, prompt_arrays, resampled_frame)
from src.features.compute import aggregate_features
from src.features.task_score import TASK_COLUMNS
from src.utils.cache import file_hash
from src.utils.io import write_json


def test_partition_choice_and_group_weighting():
    rng = np.random.default_rng(42)
    space = np.concatenate([rng.normal(-4,0.05,(12,5)),rng.normal(4,0.05,(12,5))])
    for method in ('kmeans','gmm'):
        labels, _, scores = choose_k(space,method,{'k_range':[2,3,24], 'n_init':3,'gmm_n_init':2},42)
        assert next(row['k'] for row in scores if row['selected']) == 2
        assert adjusted_rand_score(labels,np.repeat([0,1],12)) == 1
        assert next(row['status'] for row in scores if row['k']==24)=='invalid_size'
    frame = pd.DataFrame(space,columns=COLUMNS)
    transformed, scaler, factors = standardized_space(frame,{})
    np.testing.assert_allclose(factors,[1,1,1/np.sqrt(2),1/np.sqrt(2),1])
    np.testing.assert_allclose(transformed,scaler.transform(frame.to_numpy())*factors)
    with pytest.raises(ValueError,match='No valid'):
        choose_k(np.zeros((4,5)),'kmeans',{'k_range':[2,3]},42)


def test_shortlist_rule_and_empty_candidates():
    rows = []
    for head in range(4):
        row = {'layer':0,'head':head,'cluster':0 if head<2 else 1,
            'specialization':2.0 if head==0 else 1.0,'best_task':'name_tracking' if head<2 else 'induction'}
        for column in [*COLUMNS,*TASK_COLUMNS.values()]:
            row[column] = 0.1
        for task,label in TASK_COLUMNS.items():
            row['z_'+label] = 1.5 if (task=='name_tracking' and head<2) else 0.0
        rows.append(row)
    heads = pd.DataFrame(rows)
    profiles, shortlist = candidate_shortlist(heads,{'shortlist_per_cluster':1})
    assert len(shortlist)==1 and shortlist.iloc[0]['head']==0
    assert (shortlist['status']=='discovery_candidate').all()
    assert not shortlist['causally_validated'].any()
    assert profiles['task_aligned'].tolist()==[True,False]
    _, empty = candidate_shortlist(heads,{'selection':{'min_cluster_task_z':100}})
    assert empty.empty and 'head' in empty.columns


def test_clustering_pipeline_and_no_validation_leakage(extracted,tmp_path,monkeypatch):
    _, _, attention, config, _ = extracted
    features = tmp_path/'features'
    aggregate_features(attention,features,config)
    folder = tmp_path/'clusters'
    outputs = cluster_heads(features,attention,folder,config)
    assignments = pd.read_csv(outputs['assignments.csv'])
    assert len(assignments)==8 and set(assignments['split'])=={'discovery'}
    assert assignments[['layer','head']].duplicated().sum()==0
    k_selection = pd.read_csv(outputs['k_selection.csv'])
    assert k_selection.groupby('method')['selected'].sum().to_dict()=={'gmm':1,'kmeans':1}
    boot = pd.read_csv(outputs['bootstrap_stability.csv'])
    assert len(boot)==2*config['clustering']['bootstrap_repeats']
    assert boot['adjusted_rand_index'].between(-1,1).all()
    distribution = pd.read_csv(outputs['layer_distribution.csv'])
    np.testing.assert_allclose(distribution.groupby('layer')['fraction_in_layer'].sum(),1)
    coords = pd.read_csv(outputs['pca_coordinates.csv'])
    with np.load(outputs['fit_parameters.npz']) as fit:
        space = (assignments[COLUMNS].to_numpy()-fit['scaler_mean'])/fit['scaler_scale']*fit['group_factors']
        expected = (space-fit['pca_mean'])@fit['pca_components'].T
        np.testing.assert_allclose(coords[['pc1','pc2']],expected,atol=1e-10)
    index = json.loads((attention/'index.json').read_text())
    arrays,strata = prompt_arrays(index,attention,tmp_path/'independent-bootstrap', 'excluded',assignments)
    assert all(len(a)==4 for a in arrays.values())
    rebuilt = resampled_frame(arrays,strata,assignments,None)
    np.testing.assert_allclose(rebuilt[COLUMNS],assignments[COLUMNS],rtol=1e-5,atol=1e-7)
    sampled = resampled_frame(arrays,strata,assignments,np.random.default_rng(7))
    same = resampled_frame(arrays,strata,assignments,np.random.default_rng(7))
    pd.testing.assert_frame_equal(sampled,same)
    # Completed verified outputs skip all fitting.
    before = outputs['assignments.csv'].stat().st_mtime_ns
    import src.clustering.cluster_heads as module
    original = module.fit_partition
    monkeypatch.setattr(module,'fit_partition',lambda *a,**k: (_ for _ in ()).throw(AssertionError('cache missed')))
    cluster_heads(features,attention,folder,config)
    assert outputs['assignments.csv'].stat().st_mtime_ns==before
    monkeypatch.setattr(module,'fit_partition',original)
    # Alter held-out feature values and corrupt a held-out shard: discovery remains identical.
    table = pd.read_csv(features/'head_features.csv')
    mask = table['split']=='validation'
    for column in [*COLUMNS,*TASK_COLUMNS.values()]:
        table.loc[mask,column] = np.nan
    table.to_csv(features/'head_features.csv',index=False)
    completion = json.loads((features/'completion.json').read_text())
    completion['hashes']['head_features.csv'] = file_hash(features/'head_features.csv')
    write_json(features/'completion.json',completion)
    validation_entry = next(e for e in index['records'].values() if e['split']=='validation')
    (attention/validation_entry['file']).write_bytes(b'held-out shard deliberately unavailable')
    outputs = cluster_heads(features,attention,folder,config)
    pd.testing.assert_frame_equal(assignments,pd.read_csv(outputs['assignments.csv']))
    pd.testing.assert_frame_equal(coords,pd.read_csv(outputs['pca_coordinates.csv']))
    # Repair final output while reusing per-repeat checkpoints.
    outputs['assignments.csv'].write_text('corrupt output')
    fit_calls = []
    def counted(*args,**kwargs):
        fit_calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'fit_partition',counted)
    cluster_heads(features,attention,folder,config)
    assert len(fit_calls)==int((k_selection['status']=='ok').sum())
    pd.testing.assert_frame_equal(assignments,pd.read_csv(outputs['assignments.csv']))


def test_missing_clustering_inputs(tmp_path):
    with pytest.raises(FileNotFoundError,match='steps 02 and 03'):
        cluster_heads(tmp_path/'features',tmp_path/'attention',tmp_path/'clusters',{'seed':42,'clustering':{}})
