"""Held-out head ablation with discovery reference means and matched controls."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from src.model.loader import architecture,Architecture
from src.model.hooks import capture_heads,intervene
from src.utils.cache import write_csv,write_csv_rows
from src.utils.io import write_json
from src.validation.common import (METRICS,stage_signature,cached_stage,finish_stage,load_checkpoint,
    save_checkpoint,behavioral_records,forward_inputs,behavior_metrics,validation_batches,
    baseline_metrics,summarize_differences,adjust_tests)


@torch.inference_mode()
def reference_mean(model: torch.nn.Module,records: list[dict],folder: Path,signature: str,config: dict) -> np.ndarray:
    """Compute token-pooled head means on seeded discovery prompts, including BOS."""
    info = architecture(model.config)
    pool = behavioral_records(records,'discovery')
    count = config['validation']['reference_size']
    if not isinstance(count,int) or not 1<=count<=len(pool):
        raise ValueError('reference_size must be between 1 and discovery behavioral count')
    rng = np.random.default_rng(config['seed'])
    indices = sorted(rng.choice(len(pool),size=count,replace=False).tolist())
    selected = [pool[i] for i in indices]
    path = folder/'reference_checkpoint.npz'
    width = info.n_layers*info.n_heads*info.head_dim
    values,done = load_checkpoint(path,signature,(count,width))
    slots = {record['id']:i for i,record in enumerate(selected)}
    tokens = sum(len(record['token_ids']) for record in selected)
    write_json(folder/'reference_set.json',{'split':'discovery','prompt_ids':[r['id'] for r in selected],
        'n_tokens':tokens,'definition':'Token-pooled head output mean including BOS; broadcast at every position'})
    for batch in tqdm(validation_batches([r for i,r in enumerate(selected) if not done[i]],model,config),desc='Discovery reference mean'):
        indices = [slots[r['id']] for r in batch]
        pooled = np.empty((len(batch),info.n_layers,info.n_heads,info.head_dim))
        with capture_heads(model,to_cpu=True) as states:
            model(**forward_inputs(model,batch))
        for layer,state in states.items():
            pooled[:,layer] = state.double().sum(dim=1).numpy()
        values[indices] = pooled.reshape(len(batch),width)
        done[indices] = True
        save_checkpoint(path,signature,values,done)
    return (values.sum(axis=0)/tokens).reshape(info.n_layers,info.n_heads,info.head_dim)


def intervention_groups(info: Architecture,shortlist: pd.DataFrame,config: dict) -> tuple[list[dict],pd.DataFrame]:
    """Define every single head, feasible shortlisted top-k groups and matched controls."""
    heads = [(layer,head) for layer in range(info.n_layers) for head in range(info.n_heads)]
    targets = [{'id':f'L{l}H{h}','kind':'single','heads':[(l,h)],'layer':l,'head':h} for l,h in heads]
    status = []
    if len(shortlist):
        if shortlist[['layer','head']].duplicated().any():
            raise ValueError('Shortlist contains duplicate heads')
        if not set(map(tuple,shortlist[['layer','head']].values))<=set(heads):
            raise ValueError('Shortlist contains an invalid head')
    ranked = shortlist.sort_values(['specialization','layer','head'],ascending=[False,True,True]) if len(shortlist) else shortlist
    for k in config['validation']['top_k']:
        if not isinstance(k,int) or k<1:
            raise ValueError('top_k entries must be positive integers')
        enough = k<=len(ranked)
        status.append({'requested_k':k,'available_candidates':len(ranked),'status':'evaluated' if enough else 'insufficient_candidates'})
        if enough:
            selected = [(int(r.layer),int(r.head)) for r in ranked.head(k).itertuples()]
            targets.append({'id':f'top_{k}','kind':'group','heads':selected,'layer':-1,'head':-1})
    rng = np.random.default_rng(config['seed'])
    groups = list(targets)
    repeats = config['validation']['random_control_repeats']
    if repeats<1:
        raise ValueError('random_control_repeats must be positive')
    for target in targets:
        for repeat in range(repeats):
            chosen = []
            for layer in sorted({l for l,_ in target['heads']}):
                n = sum(l==layer for l,_ in target['heads'])
                choices = [h for h in range(info.n_heads) if (layer,h) not in target['heads']]
                if len(choices)<n:
                    choices = list(range(info.n_heads))
                chosen.extend((layer,int(h)) for h in rng.choice(choices,size=n,replace=False))
            groups.append({'id':f'{target["id"]}_random_{repeat}','kind':'random','heads':chosen,
                'target_id':target['id'],'repeat':repeat,'layer':-1,'head':-1,
                'overlap_count':len(set(chosen)&set(target['heads']))})
    return groups,pd.DataFrame(status,columns=['requested_k','available_candidates','status'])


@torch.inference_mode()
def run_ablation(model: torch.nn.Module,records: list[dict],discovery: list[dict],shortlist: pd.DataFrame,
                 folder: Path,config: dict,input_paths: list[Path] | None = None) -> Path:
    """Ablate all heads/groups at all positions on held-out behavioral prompts."""
    folder = Path(folder); evaluation = behavioral_records(records,'validation')
    method = config['validation']['ablation']
    if method not in ('mean','zero'):
        raise ValueError('validation.ablation must be mean or zero')
    signature = stage_signature('ablation',config,input_paths or [],{
        'model_config':model.config.to_dict(),'evaluation':evaluation,'discovery':discovery,
        'shortlist':shortlist.to_dict('records')})
    if cached_stage(folder,signature):
        return folder/'summary.csv'
    folder.mkdir(parents=True,exist_ok=True)
    info = architecture(model.config)
    groups,status = intervention_groups(info,shortlist,config)
    baseline = baseline_metrics(model,evaluation,folder,signature,config)
    reference = reference_mean(model,discovery,folder,signature,config) if method=='mean' else None
    slots = {r['id']:i for i,r in enumerate(evaluation)}
    cache,rows,controls = {},[],[]
    for group in tqdm(groups,desc=f'{method.title()} ablation and controls'):
        key = tuple(sorted(group['heads']))
        if key not in cache:
            name = '__'.join(f'L{l}H{h}' for l,h in key)
            path = folder/'interventions'/f'{name}.npz'
            values,done = load_checkpoint(path,signature,(len(evaluation),len(METRICS)))
            for batch in validation_batches([r for i,r in enumerate(evaluation) if not done[i]],model,config):
                indices = [slots[r['id']] for r in batch]
                replacements = {(l,h):None if reference is None else torch.from_numpy(reference[l,h]) for l,h in key}
                with intervene(model,replacements):
                    values[indices] = behavior_metrics(model(**forward_inputs(model,batch)).logits,batch)
                done[indices] = True
                save_checkpoint(path,signature,values,done)
            cache[key] = values
        values = cache[key]; drops = baseline-values
        ident = {'intervention':group['id'],'kind':group['kind'],'layer':group['layer'],'head':group['head'],'method':method}
        if group['kind']=='random':
            for task in ['all',*sorted({r['task'] for r in evaluation})]:
                indices = [i for i,r in enumerate(evaluation) if task=='all' or r['task']==task]
                for metric_i,metric in enumerate(METRICS):
                    controls.append({'target_id':group['target_id'],'repeat':group['repeat'],'task':task,'metric':metric,
                        'mean_drop':float(drops[indices,metric_i].mean()),'heads':json.dumps(group['heads']),
                        'overlap_count':group['overlap_count']})
        else:
            rows.extend(summarize_differences(drops,evaluation,ident,config))
    summary = pd.DataFrame(rows)
    # Per-head FDR families include all task strata; group tests form separate families.
    adjust_tests(summary,['metric','kind'])
    random = pd.DataFrame(controls)
    bands = []
    for (target,task,metric),group in random.groupby(['target_id','task','metric'],sort=True):
        lo,hi = np.quantile(group['mean_drop'],[0.025,0.975])
        bands.append({'target_id':target,'task':task,'metric':metric,'mean_drop':group['mean_drop'].mean(),
            'p025':lo,'p975':hi,'n_repeats':len(group),'definition':'Empirical range over matched control groups'})
    write_csv(folder/'summary.csv',summary)
    def per_prompt_rows():
        for group in groups:
            if group['kind']=='random':
                continue
            values = cache[tuple(sorted(group['heads']))]
            for i,record in enumerate(evaluation):
                row = {'intervention':group['id'],'kind':group['kind'],'layer':group['layer'],'head':group['head'],
                    'method':method,'id':record['id'],'task':record['task']}
                for column,metric in enumerate(METRICS):
                    row[f'baseline_{metric}'],row[f'ablated_{metric}'],row[f'drop_{metric}'] = baseline[i,column],values[i,column],baseline[i,column]-values[i,column]
                yield row
    write_csv_rows(folder/'per_prompt.csv',per_prompt_rows())
    write_csv(folder/'random_controls.csv',random)
    write_csv(folder/'random_control_bands.csv',pd.DataFrame(bands))
    write_csv(folder/'group_status.csv',status)
    baseline_table = pd.DataFrame(baseline,columns=METRICS)
    baseline_table['id'],baseline_table['task'] = [r['id'] for r in evaluation],[r['task'] for r in evaluation]
    write_csv(folder/'baseline.csv',baseline_table)
    baseline_summaries = []
    for task in ['all',*sorted({r['task'] for r in evaluation})]:
        indices = [i for i,r in enumerate(evaluation) if task=='all' or r['task']==task]
        for column,metric in enumerate(METRICS):
            baseline_summaries.append({'task':task,'metric':metric,'n':len(indices),'mean':float(baseline[indices,column].mean())})
    write_csv(folder/'baseline_summary.csv',pd.DataFrame(baseline_summaries))
    write_json(folder/'groups.json',groups)
    write_json(folder/'definitions.json',{'evaluation_split':'validation','effect_sign':'baseline minus ablated; positive means harmful ablation',
        'reference_split':'discovery' if method=='mean' else None,'fdr':'BH within metric and kind across heads/groups and task strata',
        'random_controls':'Matched in count per layer; exclude target when feasible, otherwise sample the full layer and report overlap_count; repeated selections may recur',
        'accuracy':'Forced choice correct vs distractor; ties score 0.5','positions':'all'})
    outputs = ['summary.csv','per_prompt.csv','random_controls.csv','random_control_bands.csv','group_status.csv','baseline.csv','baseline_summary.csv','groups.json','definitions.json']
    if method=='mean':
        outputs.append('reference_set.json')
    finish_stage(folder,signature,config,model.config.to_dict(),outputs)
    return folder/'summary.csv'
