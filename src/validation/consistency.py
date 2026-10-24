"""Held-out task-score consistency without inventing template correspondences."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm
from src.features.rows import BOS_MODES,METRICS
from src.features.task_score import TASK_COLUMNS
from src.utils.cache import file_hash,write_csv
from src.utils.io import write_json
from src.validation.common import stage_signature,cached_stage,finish_stage


def safe_spearman(x: np.ndarray,y: np.ndarray,minimum: int = 3) -> tuple[float,int,str]:
    """Return an undefined statistic for sparse or constant matched vectors."""
    x,y = np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if x.shape!=y.shape:
        raise ValueError('Correlation vectors must align')
    valid = np.isfinite(x)&np.isfinite(y); n = int(valid.sum())
    if n<minimum:
        return np.nan,n,'insufficient_pairs'
    if np.ptp(x[valid])==0 or np.ptp(y[valid])==0:
        return np.nan,n,'constant_vector'
    return float(spearmanr(x[valid],y[valid]).statistic),n,'ok'


def mean_vector(values: np.ndarray) -> np.ndarray:
    """Average available prompt observations, keeping entirely missing heads undefined."""
    valid = np.isfinite(values); counts = valid.sum(axis=0)
    return np.divide(np.where(valid,values,0).sum(axis=0),counts,
        out=np.full(values.shape[1:],np.nan),where=counts>0)


def run_consistency(attention: Path,folder: Path,config: dict) -> Path:
    """Compute fractions, genuine template pairs, condition transfer and stratified reliability."""
    attention,folder = Path(attention),Path(folder)
    index_path,metadata_path = attention/'index.json',attention/'metadata.json'
    for path in (index_path,metadata_path):
        if not path.exists():
            raise FileNotFoundError(f'Missing consistency input: {path}; run step 02')
    index = json.loads(index_path.read_text())
    if not index['complete'] or set(index['records'])!=set(index['expected_ids']):
        raise ValueError('Consistency requires completed discovery and validation extraction')
    entries = [entry for entry in index['records'].values() if entry['split'] in ('discovery','validation')]
    for entry in tqdm(entries,desc='Verify consistency inputs'):
        path = attention/entry['file']
        if not path.exists() or file_hash(path)!=entry['sha256']:
            raise ValueError(f'Missing/damaged consistency shard: {path}; resume step 02')
    signature = stage_signature('consistency',config,[index_path,metadata_path],{})
    if cached_stage(folder,signature):
        return folder/'consistency.csv'
    settings = config['validation']; threshold = settings['consistency_threshold']
    minimum = settings.get('minimum_correlation_pairs',3)
    repeats = settings.get('split_half_repeats',20)
    if not 0<=threshold<=1 or minimum<3 or repeats<1:
        raise ValueError('Invalid consistency threshold/correlation/split-half settings')
    pairs = settings.get('template_pairs',[])
    n_layers,n_heads = index['architecture']['n_layers'],index['architecture']['n_heads']
    grouped = {}
    for entry in tqdm(entries,desc='Read held-out task scores'):
        key = entry['split'],entry['task']
        with np.load(attention/entry['file'],allow_pickle=False) as shard:
            score = shard['selected_summary'][...,METRICS.index('task_score')]
            q_count = len(shard['query_positions'])
        grouped.setdefault(key,[]).append((entry,score,q_count))
    rows = []
    for task in TASK_COLUMNS:
        if ('discovery',task) not in grouped or ('validation',task) not in grouped:
            raise ValueError(f'Missing consistency task/split: {task}')
        discovery = grouped['discovery',task]; heldout = grouped['validation',task]
        def grouped_means(records,key_function):
            group = {}
            for entry,score,count in records:
                group.setdefault(key_function(entry,count),[]).append(score)
            return {key:mean_vector(np.stack(values)) for key,values in group.items()}
        d_templates = grouped_means(discovery,lambda e,q:e['template_id'])
        v_templates = grouped_means(heldout,lambda e,q:e['template_id'])
        explicit = [(p['discovery'],p['validation']) for p in pairs if p['task']==task]
        if len({d for d,_ in explicit})!=len(explicit) or len({v for _,v in explicit})!=len(explicit):
            raise ValueError('Template pairs must define a one-to-one mapping')
        for d,v in explicit:
            if d not in d_templates or v not in v_templates:
                raise ValueError(f'Unknown template mapping: {d} -> {v}')
        matched = explicit or [(key,key) for key in sorted(set(d_templates)&set(v_templates))]
        d_conditions = grouped_means(discovery,lambda e,q:e['num_distractors'])
        v_conditions = grouped_means(heldout,lambda e,q:e['num_distractors'])
        conditions = sorted(set(d_conditions)&set(v_conditions))
        heldout_scores = np.stack([score for _,score,_ in heldout])
        strata = {}
        for entry,score,q in heldout:
            strata.setdefault((entry['template_id'],entry['num_distractors'],q),[]).append(score)
        strata = [np.stack(values) for values in strata.values() if len(values)>=2]
        rng = np.random.default_rng(np.random.SeedSequence([config['seed'],list(TASK_COLUMNS).index(task)]))
        halves = []
        for repeat in range(repeats):
            a,b = [],[]
            for values in strata:
                order = rng.permutation(len(values)); middle = len(order)//2
                a.append(mean_vector(values[order[:middle]])); b.append(mean_vector(values[order[middle:]]))
            halves.append((np.stack(a),np.stack(b)) if a else (None,None))
        for mode,bos_mode in enumerate(BOS_MODES):
            for layer in range(n_layers):
                for head in range(n_heads):
                    scores = heldout_scores[:,mode,layer,head]; valid = np.isfinite(scores)
                    fraction = float((scores[valid]>threshold).mean()) if valid.any() else np.nan
                    d = np.array([d_templates[key][mode,layer,head] for key,_ in matched])
                    v = np.array([v_templates[key][mode,layer,head] for _,key in matched])
                    rho,n,status = safe_spearman(d,v,minimum)
                    if not matched:
                        status = 'disjoint_templates_no_mapping'
                    dc = [d_conditions[key][mode,layer,head] for key in conditions]
                    vc = [v_conditions[key][mode,layer,head] for key in conditions]
                    condition_rho,condition_n,condition_status = safe_spearman(dc,vc,minimum)
                    reliability = []
                    for a,b in halves:
                        if a is not None:
                            r,_,_ = safe_spearman(a[:,mode,layer,head],b[:,mode,layer,head],minimum)
                            if np.isfinite(r):
                                reliability.append(r)
                    half_rho = float(np.mean(reliability)) if reliability else np.nan
                    corrected = 2*half_rho/(1+half_rho) if np.isfinite(half_rho) and half_rho>-1 else np.nan
                    rows.append({'layer':layer,'head':head,'task':task,'bos_mode':bos_mode,
                        'split':'validation','threshold':threshold,'n_prompts':len(scores),'n_valid_prompts':int(valid.sum()),
                        'above_threshold_fraction':fraction,'template_spearman':rho,'template_pairs':n,'template_status':status,
                        'condition_spearman':condition_rho,'condition_pairs':condition_n,'condition_status':condition_status,
                        'split_half_rho':half_rho,'spearman_brown_reliability':corrected,
                        'reliability_valid_repeats':len(reliability),'reliability_strata':len(strata),
                        'reliability_status':'ok' if reliability else 'insufficient_or_constant_strata'})
    folder.mkdir(parents=True,exist_ok=True)
    write_csv(folder/'consistency.csv',pd.DataFrame(rows))
    write_json(folder/'definitions.json',{'fraction':'Held-out prompt mean task score strictly above threshold',
        'template_correlation':'Only explicit user-configured discovery/validation template pairs or identical IDs; disjoint IDs remain undefined',
        'template_pairs':pairs,'condition_correlation':'Separate Spearman statistic over shared distractor-count conditions',
        'split_half':'Random split within held-out template/distractor/query-count strata; correlate per-head means across strata',
        'reliability':'Mean valid split-half Spearman then Spearman-Brown correction; negative values preserved',
        'minimum_pairs':minimum,'repeats':repeats,'missingness':'Undefined statistics remain NaN with status/count columns'})
    metadata = json.loads(metadata_path.read_text())
    finish_stage(folder,signature,config,metadata['model_config'],['consistency.csv','definitions.json'])
    return folder/'consistency.csv'
