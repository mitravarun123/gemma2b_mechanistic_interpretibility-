"""Shared behavioral evaluation and numeric checkpoint infrastructure."""
import hashlib
import json
import logging
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from src.dataset.generate import length_batches
from src.model.loader import architecture
from src.utils.cache import fingerprint,file_hash,write_npz,write_csv
from src.utils.io import write_json,read_jsonl
from src.utils.logging import save_metadata
from src.validation.stats import paired_statistics,benjamini_hochberg

METRICS = ('accuracy','logit_difference','correct_probability')


def stage_signature(stage: str,config: dict,paths: list[Path],extra: dict) -> str:
    """Bind caches to inputs, validation/hook code, config and model snapshot."""
    src = Path(__file__).resolve().parents[1]
    return fingerprint(config,[*paths,*sorted((src/'validation').glob('*.py')),src/'model/hooks.py',src/'utils/cache.py'],
        extra={'stage':stage,**extra})


def cached_stage(folder: Path,signature: str) -> bool:
    """Verify final stage artifact checksums before skipping work."""
    path = folder/'completion.json'
    if not path.exists():
        return False
    completion = json.loads(path.read_text())
    return completion['signature']==signature and bool(completion['hashes']) and all(
        (folder/name).exists() and file_hash(folder/name)==checksum for name,checksum in completion['hashes'].items())


def finish_stage(folder: Path,signature: str,config: dict,model_config: dict,outputs: list[str]) -> None:
    """Record provenance and publish completion only after every artifact is saved."""
    save_metadata(folder,config,model_config)
    names = [*outputs,'metadata.json']
    write_json(folder/'completion.json',{'signature':signature,'hashes':{name:file_hash(folder/name) for name in names}})


def load_checkpoint(path: Path,signature: str,shape: tuple) -> tuple[np.ndarray,np.ndarray]:
    """Load a checksummed partial numeric evaluation, or restart a damaged cache."""
    meta = path.with_suffix('.json')
    if path.exists() and meta.exists():
        record = json.loads(meta.read_text())
        if record['signature']==signature and file_hash(path)==record['sha256']:
            with np.load(path,allow_pickle=False) as data:
                values,done = data['values'],data['done']
            if values.shape!=shape or done.shape!=shape[:-1] or done.dtype!=np.dtype(bool):
                raise ValueError(f'Invalid evaluation checkpoint dimensions: {path}')
            return values,done
        logging.warning('Restarting mismatched/damaged evaluation checkpoint: %s',path)
    return np.full(shape,np.nan,dtype=float),np.zeros(shape[:-1],dtype=bool)


def save_checkpoint(path: Path,signature: str,values: np.ndarray,done: np.ndarray) -> None:
    """Atomically persist evaluation values and completion mask with a checksum."""
    write_npz(path,values=values,done=done)
    write_json(path.with_suffix('.json'),{'signature':signature,'sha256':file_hash(path)})


def behavioral_records(records: list[dict],split: str) -> list[dict]:
    """Select unique behavioral records only from the requested split."""
    selected = sorted([r for r in records if r['split']==split and r['task'] in ('name_tracking','coreference')
        and r['correct_token_id'] is not None],key=lambda r:r['id'])
    if not selected or len({r['id'] for r in selected})!=len(selected):
        raise ValueError(f'Need unique behavioral prompts in {split}')
    if any(r['correct_token_id']==r['distractor_token_id'] for r in selected):
        raise ValueError('Correct and distractor tokens must differ')
    return selected


def forward_inputs(model: torch.nn.Module,records: list[dict]) -> dict:
    """Construct identical-length token batches with an explicit attention mask."""
    if len({len(r['token_ids']) for r in records})!=1:
        raise ValueError('Behavioral batches must have identical lengths')
    tokens = torch.tensor([r['token_ids'] for r in records],device=next(model.parameters()).device)
    inputs = {'input_ids':tokens,'attention_mask':torch.ones_like(tokens),'use_cache':False}
    parameters = inspect.signature(model.forward).parameters
    if 'logits_to_keep' in parameters:
        inputs['logits_to_keep'] = 1
    elif 'num_logits_to_keep' in parameters:
        inputs['num_logits_to_keep'] = 1
    return inputs


def behavior_metrics(logits: torch.Tensor,records: list[dict]) -> np.ndarray:
    """Measure logits using each CLEAN prompt's correct/distractor answer convention."""
    if logits.ndim!=3 or logits.shape[0]!=len(records):
        raise ValueError('Logits must align with behavioral prompt batch')
    logits = logits[:,-1].float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Non-finite next-token logits; inspect model dtype/device')
    indices = torch.arange(len(records),device=logits.device)
    correct = torch.tensor([r['correct_token_id'] for r in records],device=logits.device)
    wrong = torch.tensor([r['distractor_token_id'] for r in records],device=logits.device)
    delta = logits[indices,correct]-logits[indices,wrong]
    accuracy = torch.where(delta>0,1.0,torch.where(delta<0,0.0,0.5))
    probability = logits.softmax(-1)[indices,correct]
    return torch.stack([accuracy,delta,probability],dim=-1).cpu().numpy().astype(float)


def validation_batches(records: list[dict],model: torch.nn.Module,config: dict) -> list[list[dict]]:
    """Bound cached clean/corrupted head activations and eager attention workspace."""
    info = architecture(model.config)
    limit = config['validation'].get('max_activation_mb',256)*1024**2
    attention_limit = config['extraction'].get('max_attention_mb',256)*1024**2
    forward_limit = config['validation'].get('max_forward_mb',512)*1024**2
    parameters = inspect.signature(model.forward).parameters
    last_only = 'logits_to_keep' in parameters or 'num_logits_to_keep' in parameters
    vocab = getattr(getattr(model.config,'text_config',model.config),'vocab_size')
    if limit<=0 or attention_limit<=0 or forward_limit<=0:
        raise ValueError('Validation memory budgets must be positive')
    result = []
    for batch in length_batches(records,config['batch_size']):
        length = len(batch[0]['token_ids'])
        bytes_per = 2*info.n_layers*info.n_heads*info.head_dim*length*4
        attention_bytes = info.n_heads*length*length*4
        logits_bytes = vocab*(1 if last_only else length)*4
        maximum = min(int(limit//bytes_per),int(attention_limit//attention_bytes),
            int(forward_limit//(logits_bytes+attention_bytes+bytes_per)))
        if maximum<1:
            raise MemoryError(f'Validation prompt {batch[0]["id"]} exceeds memory budget')
        for start in range(0,len(batch),maximum):
            result.append(batch[start:start+maximum])
    return result


@torch.inference_mode()
def baseline_metrics(model: torch.nn.Module,records: list[dict],folder: Path,signature: str,config: dict) -> np.ndarray:
    """Resume baseline clean-answer measurements batch by batch."""
    path = folder/'baseline_checkpoint.npz'
    values,done = load_checkpoint(path,signature,(len(records),len(METRICS)))
    slots = {r['id']:i for i,r in enumerate(records)}
    for batch in tqdm(validation_batches([r for i,r in enumerate(records) if not done[i]],model,config),desc='Behavioral baseline'):
        indices = [slots[r['id']] for r in batch]
        values[indices] = behavior_metrics(model(**forward_inputs(model,batch)).logits,batch)
        done[indices] = True
        save_checkpoint(path,signature,values,done)
    return values


def summarize_differences(values: np.ndarray,records: list[dict],identifiers: dict,config: dict,metric_names: tuple = METRICS) -> list[dict]:
    """Compute paired statistics by task and across all held-out behavioral prompts."""
    if values.shape!=(len(records),len(metric_names)):
        raise ValueError('Difference columns must align with records and metric names')
    rows = []
    settings = config['validation']
    groups = [('all',np.arange(len(records)))] + [(task,np.array([i for i,r in enumerate(records) if r['task']==task]))
        for task in sorted({r['task'] for r in records})]
    for task,indices in groups:
        for column,metric in enumerate(metric_names):
            # Stable seed per comparison makes reruns and repairs reproduce intervals.
            key = json.dumps({**identifiers,'task':task,'metric':metric},sort_keys=True)
            seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'little') ^ config['seed']
            stats = paired_statistics(values[indices,column],seed,settings['bootstrap_repeats'],settings.get('permutation_repeats',1000))
            rows.append({**identifiers,'task':task,'metric':metric,**stats})
    return rows


def adjust_tests(table: pd.DataFrame,keys: list[str]) -> None:
    """Apply BH separately for each metric/direction family across heads and tasks."""
    table['q_value'] = np.nan
    for _,indices in table.groupby(keys,sort=True).groups.items():
        table.loc[indices,'q_value'] = benjamini_hochberg(table.loc[indices,'p_value'].to_numpy())


def read_validation_inputs(root: Path,config: dict) -> tuple[list[dict],list[dict],list[dict],pd.DataFrame,list[Path]]:
    """Read hash-verified saved prompts, twins and discovery shortlist for scripts 05–06."""
    processed = root/config.get('data_dir','data')/'processed'
    results = root/config['results_dir']
    paths = [processed/'discovery.jsonl',processed/'validation.jsonl',processed/'corrupted/validation.jsonl',
        processed/'manifest.json',processed/'metadata.json',results/'features/head_features.csv',
        results/'features/completion.json',results/'clusters/shortlist.csv',results/'clusters/completion.json',
        results/'attention/metadata.json',results/'attention/index.json']
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f'Missing validation input: {path}; run stages 01–04')
    manifest = json.loads((processed/'manifest.json').read_text())
    for path in paths[:3]:
        if file_hash(path)!=manifest['hashes'].get(str(path.relative_to(processed))):
            raise ValueError(f'Dataset checksum mismatch: {path}')
    for folder,name in ((results/'features','head_features.csv'),(results/'clusters','shortlist.csv')):
        completion = json.loads((folder/'completion.json').read_text())
        if file_hash(folder/name)!=completion['hashes'].get(name):
            raise ValueError(f'Stage artifact checksum mismatch: {folder/name}')
    provenance = json.loads((results/'attention/metadata.json').read_text())['config']
    for key in ('name','revision','dtype'):
        if provenance['model'].get(key)!=config['model'].get(key):
            raise ValueError(f'Validation model.{key} differs from discovery extraction; rerun earlier stages')
    return read_jsonl(paths[0]),read_jsonl(paths[1]),read_jsonl(paths[2]),pd.read_csv(paths[7]),paths


def validate_model_inputs(model: torch.nn.Module,tokenizer,records: list[dict],attention_metadata: Path) -> None:
    """Reject tokenization, architecture or resolved-checkpoint drift before interventions."""
    prior = json.loads(attention_metadata.read_text())['model_config']
    prior = prior.get('text_config',prior)
    current = getattr(model.config,'text_config',model.config)
    for key in ('num_hidden_layers','num_attention_heads','head_dim','num_key_value_heads'):
        if prior.get(key)!=getattr(current,key,None):
            raise ValueError(f'Model dimension {key} differs from discovery extraction')
    if prior.get('_commit_hash') and prior['_commit_hash']!=getattr(model.config,'_commit_hash',None):
        raise ValueError('Resolved model checkpoint changed since discovery extraction; pin revision')
    for record in records:
        if [tokenizer.bos_token_id,*tokenizer.encode(record['text'],add_special_tokens=False)]!=record['token_ids']:
            raise ValueError(f'Tokenization mismatch for {record["id"]}')
