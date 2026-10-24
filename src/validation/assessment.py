"""Join discovery candidates with task-matched held-out causal validation."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.cache import write_csv
from src.utils.io import write_json


def assess_candidates(shortlist: pd.DataFrame,consistency: Path,ablation: Path,patching: Path,
                      folder: Path,config: dict) -> Path:
    """Label candidates validated only by matching behavioral task and positive FDR effects."""
    con,abl,patch = pd.read_csv(consistency),pd.read_csv(ablation),pd.read_csv(patching)
    mode = config['clustering'].get('bos_mode','excluded')
    alpha = config['validation'].get('fdr_alpha',0.05)
    minimum = config['validation'].get('minimum_consistency_fraction',0.5)
    if not 0<alpha<1 or not 0<=minimum<=1:
        raise ValueError('Invalid FDR or consistency validation threshold')
    rows = []
    for candidate in shortlist.to_dict('records'):
        layer,head,task = int(candidate['layer']),int(candidate['head']),candidate['task']
        row = {'layer':layer,'head':head,'task':task,'discovery_specialization':candidate['specialization'],
            'causally_validated':False,'status':'no_task_matched_behavioral_target',
            'consistency_fraction':np.nan,'ablation_probability_drop':np.nan,'ablation_q':np.nan,
            'denoising_logit_effect':np.nan,'denoising_q':np.nan}
        if task in ('name_tracking','coreference'):
            c = con[(con['layer']==layer)&(con['head']==head)&(con['task']==task)&(con['bos_mode']==mode)]
            a = abl[(abl['layer']==layer)&(abl['head']==head)&(abl['task']==task)&(abl['kind']=='single')&(abl['metric']=='correct_probability')]
            p = patch[(patch['layer']==layer)&(patch['head']==head)&(patch['task']==task)&(patch['direction']=='denoising')&(patch['metric']=='raw_logit_difference')]
            if len(c)!=1 or len(a)!=1 or len(p)!=1:
                raise ValueError(f'Missing/duplicate candidate validation measurements for L{layer}H{head}')
            c,a,p = c.iloc[0],a.iloc[0],p.iloc[0]
            row.update(consistency_fraction=c['above_threshold_fraction'],ablation_probability_drop=a['mean'],
                ablation_q=a['q_value'],denoising_logit_effect=p['mean'],denoising_q=p['q_value'])
            passed = (c['above_threshold_fraction']>=minimum and a['mean']>0 and a['ci_low']>0 and a['q_value']<=alpha
                and p['mean']>0 and p['ci_low']>0 and p['q_value']<=alpha)
            row['causally_validated'] = bool(passed)
            row['status'] = 'validated' if passed else 'did_not_pass_causal_and_consistency_thresholds'
        rows.append(row)
    columns = ['layer','head','task','discovery_specialization','causally_validated','status',
        'consistency_fraction','ablation_probability_drop','ablation_q','denoising_logit_effect','denoising_q']
    path = folder/'validated_heads.csv'
    write_csv(path,pd.DataFrame(rows,columns=columns))
    write_json(folder/'assessment_rules.json',{'fdr_alpha':alpha,'minimum_consistency_fraction':minimum,
        'requirements':'Discovery shortlist + task-matched held-out consistency + positive ablation probability drop and denoising logit effect, both CI low > 0 and BH q <= alpha',
        'unsupported_tasks':'Local-context and induction lack matched behavioral targets in this dataset; remain unvalidated',
        'empty_result':'No passing candidates is a valid outcome'})
    return path
