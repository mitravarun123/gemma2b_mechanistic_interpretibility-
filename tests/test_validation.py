"""Real tiny-model intervention, held-out measurement and resume contracts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from test_extraction import extracted
from src.dataset.generate import generate_split
from src.dataset.names import build_name_pools
from test_dataset import tokenizer
from src.features.compute import aggregate_features
from src.model.loader import architecture
from src.model.hooks import capture_heads,intervene
from src.validation.ablation import run_ablation,reference_mean,intervention_groups
from src.validation.patching import run_patching,aligned_twins
from src.validation.consistency import run_consistency
from src.validation.assessment import assess_candidates
from src.validation.common import forward_inputs,behavior_metrics
from src.validation.stats import recovery_score


def test_validation_interventions_and_checkpoints(extracted,tmp_path,monkeypatch):
    model,records,attention,config,_ = extracted
    config['validation']['reference_size']=4
    config['validation']['bootstrap_repeats']=20
    config['validation']['random_control_repeats']=3
    features = tmp_path/'features'; aggregate_features(attention,features,config)
    discovery = [r for r in records if r['split']=='discovery']
    validation = [r for r in records if r['split']=='validation']
    tok = tokenizer()
    pools = build_name_pools(tok,json.loads(Path('data/raw/names.json').read_text()),42)
    _,twins = generate_split(tok,pools['validation'],'validation',4,42)
    shortlist = pd.DataFrame([{'layer':1,'head':0,'specialization':2.0,'task':'name_tracking'}])
    con = run_consistency(attention,tmp_path/'consistency',config)
    consistency = pd.read_csv(con)
    assert len(consistency)==2*2*4*4
    assert consistency['template_spearman'].isna().all()
    assert set(consistency['template_status'])=={'disjoint_templates_no_mapping'}
    assert consistency['above_threshold_fraction'].dropna().between(0,1).all()
    groups,status = intervention_groups(architecture(model.config),shortlist,config)
    targets = {g['id']:g for g in groups if g['kind']!='random'}
    for control in (g for g in groups if g['kind']=='random'):
        target = targets[control['target_id']]
        for layer in range(2):
            assert sum(l==layer for l,_ in control['heads'])==sum(l==layer for l,_ in target['heads'])
        assert not set(control['heads'])&set(target['heads'])
    output = run_ablation(model,validation,discovery,shortlist,tmp_path/'ablation',config)
    summary = pd.read_csv(output)
    assert set(summary['kind'])=={'single','group'}
    assert len(summary[summary['kind']=='single'])==2*4*3*3
    assert summary['q_value'].dropna().between(0,1).all()
    reference_set = json.loads((tmp_path/'ablation/reference_set.json').read_text())
    assert all(ident.startswith('discovery-') for ident in reference_set['prompt_ids'])
    # Explicitly recompute the reference mean from its saved discovery IDs.
    ref_records = [r for r in discovery if r['id'] in reference_set['prompt_ids']]
    pooled = np.zeros((2,4,8)); n_tokens=0
    for record in ref_records:
        with torch.no_grad(),capture_heads(model,to_cpu=True) as states:
            model(**forward_inputs(model,[record]))
        for layer,values in states.items():
            pooled[layer]+=values.double().sum(dim=(0,1)).numpy()
        n_tokens+=len(record['token_ids'])
    with np.load(tmp_path/'ablation/reference_checkpoint.npz') as checkpoint:
        mean = checkpoint['values'].sum(axis=0).reshape(2,4,8)/n_tokens
    np.testing.assert_allclose(mean,pooled/n_tokens,atol=1e-8)
    patch = run_patching(model,validation,twins,tmp_path/'patching',config,features=features/'head_features.csv')
    table = pd.read_csv(patch)
    assert set(table['direction'])=={'denoising','noising'}
    assert len(table)==2*4*2*3*7
    assert (tmp_path/'patching/feature_correlations.csv').exists()
    prompt = pd.read_csv(tmp_path/'patching/per_prompt.csv')
    baseline = pd.read_csv(tmp_path/'patching/baseline.csv')
    expected = prompt.merge(baseline,on=['id','task'])
    good = expected['clean_logit_gap'].abs()>config['validation']['recovery_epsilon']
    np.testing.assert_allclose(expected.loc[good,'recovery_logit'],
        expected.loc[good,'raw_logit_difference']/expected.loc[good,'clean_logit_gap'],atol=1e-10)
    # A direct target-position denoising patch must agree with the saved evaluation.
    clean = next(r for r in validation if r['correct_token_id'] is not None)
    bad = aligned_twins([clean],twins)[clean['id']]
    with torch.no_grad(),capture_heads(model,to_cpu=True) as states:
        model(**forward_inputs(model,[clean]))
    with torch.no_grad(),intervene(model,{(0,0):states[0][:,-1:,0,:]},positions=[-1]):
        measured = behavior_metrics(model(**forward_inputs(model,[bad])).logits,[clean])
    base = baseline[baseline['id']==clean['id']].iloc[0]
    observed = prompt[(prompt['id']==clean['id'])&(prompt['layer']==0)&(prompt['head']==0)&(prompt['direction']=='denoising')].iloc[0]
    assert observed['raw_logit_difference']==pytest.approx(measured[0,1]-base['corrupted_logit_difference'],abs=1e-6)
    assessed = assess_candidates(shortlist,con,output,patch,tmp_path/'patching',config)
    assert set(pd.read_csv(assessed)['status'])<= {'validated','did_not_pass_causal_and_consistency_thresholds'}
    # Full completed caches must not execute any model forwards.
    monkeypatch.setattr(model,'forward',lambda *a,**k: (_ for _ in ()).throw(AssertionError('unexpected model evaluation')))
    run_ablation(model,validation,discovery,shortlist,tmp_path/'ablation',config)
    run_patching(model,validation,twins,tmp_path/'patching',config,features=features/'head_features.csv')
    # Damaged final summaries are rebuilt from verified intervention checkpoints.
    output.write_text('damaged final table')
    run_ablation(model,validation,discovery,shortlist,tmp_path/'ablation',config)
    patch.write_text('damaged final table')
    run_patching(model,validation,twins,tmp_path/'patching',config,features=features/'head_features.csv')


def test_twin_alignment_guard():
    clean = {'id':'clean','corrupted_id':'bad','token_ids':[1,2],'correct_token_id':2,'distractor_token_id':3,
        'split':'validation','task':'name_tracking'}
    twin = {'id':'bad','clean_id':'clean','token_ids':[1,3,4],'correct_token_id':3,'distractor_token_id':2,
        'split':'validation','task':'name_tracking'}
    with pytest.raises(ValueError,match='align/invert'):
        aligned_twins([clean],[twin])


def test_zero_ablation_and_all_position_patch(extracted,tmp_path):
    model,records,_,config,_ = extracted
    config['validation']['ablation']='zero'
    config['validation']['patch_positions']='all'
    config['validation']['bootstrap_repeats']=5
    config['validation']['random_control_repeats']=2
    validation = [r for r in records if r['split']=='validation' and r['correct_token_id'] is not None][:4]
    discovery = [r for r in records if r['split']=='discovery']
    tok = tokenizer(); pool = build_name_pools(tok,json.loads(Path('data/raw/names.json').read_text()),42)
    _,twins = generate_split(tok,pool['validation'],'validation',4,42)
    empty = pd.DataFrame(columns=['layer','head','specialization','task'])
    ablation = run_ablation(model,validation,discovery,empty,tmp_path/'zero',config)
    assert set(pd.read_csv(ablation)['method'])=={'zero'}
    assert not (tmp_path/'zero/reference_set.json').exists()
    status = pd.read_csv(tmp_path/'zero/group_status.csv')
    assert set(status['status'])=={'insufficient_candidates'}
    patch = run_patching(model,validation,twins,tmp_path/'all',config)
    assert set(pd.read_csv(patch)['positions'])=={'all'}
    # Both noising and denoising for all positions agree with direct model interventions.
    clean = validation[0]; bad = aligned_twins([clean],twins)[clean['id']]
    with torch.no_grad(),capture_heads(model,to_cpu=True) as clean_states:
        clean_logits = model(**forward_inputs(model,[clean])).logits
    with torch.no_grad(),capture_heads(model,to_cpu=True) as bad_states:
        bad_logits = model(**forward_inputs(model,[bad])).logits
    clean_metric,bad_metric = behavior_metrics(clean_logits,[clean]),behavior_metrics(bad_logits,[clean])
    prompts = pd.read_csv(tmp_path/'all/per_prompt.csv')
    for direction,donor,recipient in [('denoising',clean_states,bad),('noising',bad_states,clean)]:
        with torch.no_grad(),intervene(model,{(0,0):donor[0][:,:,0,:]}):
            measured = behavior_metrics(model(**forward_inputs(model,[recipient])).logits,[clean])
        row = prompts[(prompts['id']==clean['id'])&(prompts['layer']==0)&(prompts['head']==0)&(prompts['direction']==direction)].iloc[0]
        expected = measured[0,2]-bad_metric[0,2] if direction=='denoising' else clean_metric[0,2]-measured[0,2]
        assert row['raw_correct_probability']==pytest.approx(expected,abs=1e-7)
    with torch.no_grad(),intervene(model,{(0,0):clean_states[0][:,:,0,:]}):
        self_logits = model(**forward_inputs(model,[clean])).logits
    torch.testing.assert_close(self_logits,clean_logits,rtol=0,atol=0)


def test_control_overlap_is_explicit_when_disjoint_match_impossible():
    from src.model.loader import Architecture
    info = Architecture(1,4,1,8,None,None,None)
    shortlist = pd.DataFrame([{'layer':0,'head':h,'specialization':4-h,'task':'name_tracking'} for h in range(3)])
    config = {'seed':42,'validation':{'top_k':[3],'random_control_repeats':3}}
    groups,_ = intervention_groups(info,shortlist,config)
    controls = [g for g in groups if g.get('target_id')=='top_3']
    assert all(len(g['heads'])==3 and g['overlap_count']>=2 for g in controls)
