"""All-head denoising/noising activation patches on held-out corrupted twins."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm
from src.model.loader import architecture
from src.model.hooks import capture_heads,intervene
from src.utils.cache import write_csv,write_csv_rows
from src.utils.io import write_json
from src.validation.stats import benjamini_hochberg
from src.validation.common import (METRICS,stage_signature,cached_stage,finish_stage,load_checkpoint,
    save_checkpoint,behavioral_records,forward_inputs,behavior_metrics,validation_batches,
    summarize_differences,adjust_tests)


def aligned_twins(records: list[dict],twins: list[dict]) -> dict[str,dict]:
    """Require one length-aligned answer-inverting twin for every evaluated clean prompt."""
    by_id = {r['id']:r for r in twins}
    if len(by_id)!=len(twins):
        raise ValueError('Duplicate corrupted prompt IDs')
    result = {}
    for clean in records:
        if clean['corrupted_id'] not in by_id:
            raise ValueError(f'Missing corrupted twin for {clean["id"]}')
        twin = by_id[clean['corrupted_id']]
        if len(twin['token_ids'])!=len(clean['token_ids']) or twin['correct_token_id']!=clean['distractor_token_id'] or twin['distractor_token_id']!=clean['correct_token_id']:
            raise ValueError(f'Corrupted twin does not align/invert answers: {clean["id"]}')
        if twin['clean_id']!=clean['id'] or twin['split']!=clean['split'] or twin['task']!=clean['task']:
            raise ValueError(f'Corrupted twin metadata mismatch: {clean["id"]}')
        result[clean['id']] = twin
    return result


def feature_correlations(features: Path,summary: pd.DataFrame,mode: str) -> pd.DataFrame:
    """Correlate discovery features with held-out denoising effects across all heads."""
    frame = pd.read_csv(features)
    frame = frame[(frame['split']=='discovery') & (frame['bos_mode']==mode)]
    columns = [column for column in frame.select_dtypes(include='number').columns
        if column not in ('layer','head','n_prompts','valid_tasks') and not column.endswith('_valid_prompts')]
    rows = []
    selected = summary[(summary['direction']=='denoising') & summary['metric'].isin(['raw_logit_difference','recovery_logit_positive_gap'])]
    for (task,metric),group in selected.groupby(['task','metric'],sort=True):
        joined = frame.merge(group[['layer','head','mean']],on=['layer','head'],validate='one_to_one')
        if len(joined)!=len(group):
            raise ValueError('Discovery features must cover all causally evaluated heads')
        for column in columns:
            x,y = joined[column].to_numpy(dtype=float),joined['mean'].to_numpy(dtype=float)
            valid = np.isfinite(x)&np.isfinite(y)
            rho,p = np.nan,np.nan
            if valid.sum()>=3 and np.ptp(x[valid])>0 and np.ptp(y[valid])>0:
                statistic = spearmanr(x[valid],y[valid]); rho,p = float(statistic.statistic),float(statistic.pvalue)
            rows.append({'task':task,'causal_metric':metric,'feature':column,'n_heads':int(valid.sum()),
                'spearman_rho':rho,'p_value':p,'status':'ok' if np.isfinite(rho) else 'insufficient_or_constant'})
    table = pd.DataFrame(rows)
    table['q_value'] = benjamini_hochberg(table['p_value'].to_numpy())
    return table


EFFECT_METRICS = ('raw_accuracy','raw_logit_difference','raw_correct_probability','recovery_logit','recovery_probability',
                 'recovery_logit_positive_gap','recovery_probability_positive_gap')


def patch_effects(patched: np.ndarray,clean: np.ndarray,bad: np.ndarray,direction: int,epsilon: float) -> np.ndarray:
    """Compute raw effects and signed/positive-gap normalized scores without clipping."""
    if not np.isfinite(epsilon) or epsilon<=0 or direction not in (0,1):
        raise ValueError('Invalid recovery epsilon/direction')
    gap = clean-bad
    raw = patched-bad if direction==0 else clean-patched
    logit = np.divide(raw[:,1],gap[:,1],out=np.full(len(clean),np.nan),where=np.abs(gap[:,1])>epsilon)
    probability = np.divide(raw[:,2],gap[:,2],out=np.full(len(clean),np.nan),where=np.abs(gap[:,2])>epsilon)
    return np.column_stack([raw,logit,probability,np.where(gap[:,1]>epsilon,logit,np.nan),
        np.where(gap[:,2]>epsilon,probability,np.nan)])


@torch.inference_mode()
def run_patching(model: torch.nn.Module,records: list[dict],twins: list[dict],folder: Path,config: dict,
                 input_paths: list[Path] | None = None,features: Path | None = None) -> Path:
    """Patch every query head in both directions, at target or all aligned positions."""
    folder = Path(folder); evaluation = behavioral_records(records,'validation')
    corrupted = aligned_twins(evaluation,twins)
    policy = config['validation']['patch_positions']
    if policy not in ('target','all'):
        raise ValueError('patch_positions must be target or all')
    epsilon = config['validation'].get('recovery_epsilon',1e-6)
    paths = list(input_paths or [])
    if features is not None and Path(features) not in paths:
        paths.append(Path(features))
    signature = stage_signature('patching',config,paths,{
        'model_config':model.config.to_dict(),'evaluation':evaluation,'twins':corrupted})
    if cached_stage(folder,signature):
        return folder/'summary.csv'
    folder.mkdir(parents=True,exist_ok=True)
    info = architecture(model.config)
    baseline_path = folder/'baseline_checkpoint.npz'
    baseline,base_done = load_checkpoint(baseline_path,signature,(2,len(evaluation),len(METRICS)))
    slots = {r['id']:i for i,r in enumerate(evaluation)}
    heads = [(layer,head) for layer in range(info.n_layers) for head in range(info.n_heads)]
    # Head checkpoints are small (two directions x prompts x three metrics).
    values,done = {},{}
    for key in heads:
        path = folder/'heads'/f'L{key[0]}H{key[1]}.npz'
        values[key],done[key] = load_checkpoint(path,signature,(2,len(evaluation),len(METRICS)))
    for batch in tqdm(validation_batches(evaluation,model,config),desc='All-head activation patching'):
        indices = [slots[r['id']] for r in batch]
        pending = [key for key in heads if not done[key][:,indices].all()]
        if not pending and base_done[:,indices].all():
            continue
        bad_batch = [corrupted[r['id']] for r in batch]
        with capture_heads(model,to_cpu=True) as clean_states:
            clean_output = model(**forward_inputs(model,batch)).logits
        baseline[0,indices] = behavior_metrics(clean_output,batch)
        del clean_output
        with capture_heads(model,to_cpu=True) as bad_states:
            bad_output = model(**forward_inputs(model,bad_batch)).logits
        # Both baselines and patches use CLEAN answers, even on corrupted inputs.
        baseline[1,indices] = behavior_metrics(bad_output,batch)
        del bad_output
        base_done[:,indices] = True
        save_checkpoint(baseline_path,signature,baseline,base_done)
        positions = None if policy=='all' else [-1]
        for layer,head in pending:
            key = layer,head
            for direction in range(2):
                if done[key][direction,indices].all():
                    continue
                donor = clean_states if direction==0 else bad_states
                inputs = bad_batch if direction==0 else batch
                replacement = donor[layer][:,:,head,:]
                if policy=='target':
                    replacement = replacement[:,-1:,:]
                with intervene(model,{key:replacement},positions=positions):
                    output = model(**forward_inputs(model,inputs)).logits
                values[key][direction,indices] = behavior_metrics(output,batch)
                del output
                done[key][direction,indices] = True
            save_checkpoint(folder/'heads'/f'L{layer}H{head}.npz',signature,values[key],done[key])
        del clean_states,bad_states
    rows = []
    clean,bad = baseline[0],baseline[1]
    gap = clean-bad
    for layer,head in tqdm(heads,desc='Patching uncertainty'):
        patched = values[layer,head]
        for direction,label in enumerate(('denoising','noising')):
            data = patch_effects(patched[direction],clean,bad,direction,epsilon)
            rows.extend(summarize_differences(data,evaluation,{'layer':layer,'head':head,'direction':label,'positions':policy},config,EFFECT_METRICS))
    summary = pd.DataFrame(rows); adjust_tests(summary,['metric','direction'])
    write_csv(folder/'summary.csv',summary)
    def per_prompt_rows():
        for layer,head in heads:
            for direction,label in enumerate(('denoising','noising')):
                data = patch_effects(values[layer,head][direction],clean,bad,direction,epsilon)
                for i,record in enumerate(evaluation):
                    row = {'id':record['id'],'task':record['task'],'layer':layer,'head':head,'direction':label,
                        'positions':policy,'clean_logit_gap':gap[i,1],'clean_probability_gap':gap[i,2],
                        'gap_status':'clean_better' if gap[i,1]>epsilon else ('inverted' if gap[i,1]<-epsilon else 'tiny_gap')}
                    row.update(zip(EFFECT_METRICS,data[i].tolist()))
                    for column,metric in enumerate(METRICS):
                        row['patched_'+metric] = values[layer,head][direction,i,column]
                    yield row
    write_csv_rows(folder/'per_prompt.csv',per_prompt_rows())
    baseline_rows = []
    for i,record in enumerate(evaluation):
        row = {'id':record['id'],'corrupted_id':record['corrupted_id'],'task':record['task']}
        for column,metric in enumerate(METRICS):
            row['clean_'+metric],row['corrupted_'+metric] = clean[i,column],bad[i,column]
        baseline_rows.append(row)
    write_csv(folder/'baseline.csv',pd.DataFrame(baseline_rows))
    outputs = ['summary.csv','per_prompt.csv','baseline.csv','definitions.json']
    if features is not None:
        write_csv(folder/'feature_correlations.csv',feature_correlations(features,summary,config['clustering'].get('bos_mode','excluded')))
        outputs.append('feature_correlations.csv')
    write_json(folder/'definitions.json',{'evaluation_split':'validation','answers':'Clean correct/distractor tokens for both twins',
        'denoising':'Corrupted input, clean donor; positive raw effect is patched minus corrupted',
        'noising':'Clean input, corrupted donor; positive raw effect is clean minus patched',
        'recovery':'Raw effect divided by clean-corrupted baseline gap; no clipping',
        'epsilon':epsilon,'inverted_gaps':'Signed ratios retained; positive-gap summaries reported separately',
        'all_heads':True,'positions':policy,'fdr':'BH by metric/direction across all heads and task strata',
        'correlations':'Discovery feature vs held-out causal effect; BH across feature/metric/task comparisons'})
    finish_stage(folder,signature,config,model.config.to_dict(),outputs)
    return folder/'summary.csv'
