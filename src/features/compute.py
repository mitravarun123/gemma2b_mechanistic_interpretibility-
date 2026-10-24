"""Streaming head-feature aggregation from saved extraction summaries only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.features.rows import BOS_MODES, METRICS, finite_mean
from src.features.task_score import TASK_COLUMNS
from src.utils.cache import file_hash, fingerprint, write_csv
from src.utils.io import write_json
from src.utils.logging import save_metadata

OUTPUTS = ('task_features.csv', 'template_features.csv', 'head_features.csv',
           'layer_summaries.csv', 'zscore_parameters.csv')


def aggregate_features(extraction_folder: Path, folder: Path, config: dict) -> dict[str, Path]:
    """Aggregate prompt-weighted task features and task-balanced head features.

    Reads shards one at a time; does not load a model or full attention matrices.
    Split-specific z scores are descriptive. Discovery parameters are also saved
    so later validation can apply a discovery-fitted transform without leakage.
    """
    extraction_folder, folder = Path(extraction_folder), Path(folder)
    index_path = extraction_folder / 'index.json'
    if not index_path.exists():
        raise FileNotFoundError(f'Missing extraction index: {index_path}; run step 02')
    index = json.loads(index_path.read_text())
    if not index['complete'] or set(index['records']) != set(index['expected_ids']):
        raise ValueError('Extraction is incomplete; resume step 02 before computing features')
    if index['metrics'] != list(METRICS) or index['bos_modes'] != list(BOS_MODES):
        raise ValueError('Unsupported extraction feature schema')
    metadata_path = extraction_folder / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f'Missing extraction provenance: {metadata_path}')
    for ident, entry in tqdm(index['records'].items(), desc='Verify extraction shards'):
        path = extraction_folder / entry['file']
        if not path.exists():
            raise FileNotFoundError(f'Missing attention shard for {ident}: {path}')
        if file_hash(path) != entry['sha256']:
            raise ValueError(f'Attention shard checksum mismatch: {path}; resume step 02')
    code = sorted(Path(__file__).resolve().parent.glob('*.py'))
    signature = fingerprint(config, [index_path, metadata_path, *code, Path(__file__).resolve().parents[1] / 'utils/cache.py'], extra={'stage': 'features'})
    completion = folder / 'completion.json'
    if completion.exists():
        previous = json.loads(completion.read_text())
        if previous['signature'] == signature and set(previous['hashes']) == set(OUTPUTS) and all(
            (folder / name).exists() and file_hash(folder / name) == checksum for name, checksum in previous['hashes'].items()):
            return {name: folder / name for name in OUTPUTS}
    n_layers, n_heads = index['architecture']['n_layers'], index['architecture']['n_heads']
    shape = (2, n_layers, n_heads, len(METRICS))
    groups = {}
    # Collect metrics + all-token references + BOS sinks + valid query counts.
    def update(key, payload):
        if key not in groups:
            groups[key] = [np.zeros_like(payload, dtype=np.float64), np.zeros_like(payload, dtype=np.int64), 0]
        sums, counts, total = groups[key]
        valid = np.isfinite(payload)
        sums += np.where(valid, payload, 0)
        counts += valid
        groups[key][2] = total + 1
    for entry in tqdm(index['records'].values(), desc='Aggregate prompt features'):
        if entry['split'] == 'example':
            continue
        if entry['split'] not in ('discovery', 'validation') or entry['task'] not in TASK_COLUMNS:
            raise ValueError(f'Unknown split or task in extraction index: {entry["id"]}')
        with np.load(extraction_folder / entry['file'], allow_pickle=False) as shard:
            selected, all_summary = shard['selected_summary'], shard['all_summary']
            if selected.shape != shape or all_summary.shape != shape:
                raise ValueError(f'Feature dimensions mismatch for {entry["id"]}')
            sink = np.broadcast_to(shard['bos_sink'][None, :, :, None], (2, n_layers, n_heads, 1))
            all_sink = np.broadcast_to(shard['all_bos_sink'][None, :, :, None], sink.shape)
            payload = np.concatenate([selected, all_summary, sink, all_sink,
                shard['valid_query_counts'][..., None], shard['all_valid_query_counts'][..., None]], axis=-1)
        update((entry['split'], entry['task'], '__task__'), payload)
        update((entry['split'], entry['task'], entry['template_id']), payload)
    expected = {(split, task, '__task__') for split in ('discovery', 'validation') for task in TASK_COLUMNS}
    if not expected <= set(groups):
        raise ValueError(f'Missing required split/task summaries: {sorted(expected-set(groups))}')
    labels = [*METRICS, *[f'{m}_all' for m in METRICS], 'bos_sink', 'bos_sink_all',
              'valid_queries', 'all_valid_queries']
    task_rows, template_rows = [], []
    for (split, task, template), (sums, counts, total) in sorted(groups.items()):
        means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        for mode, bos_mode in enumerate(BOS_MODES):
            for layer in range(n_layers):
                for head in range(n_heads):
                    row = {'layer': layer, 'head': head, 'split': split, 'bos_mode': bos_mode,
                           'task': task, 'template_id': template, 'n_prompts': total,
                           'valid_task_prompts': int(counts[mode, layer, head, METRICS.index('task_score')])}
                    row.update(zip(labels, means[mode, layer, head].tolist()))
                    (task_rows if template == '__task__' else template_rows).append(row)
    tasks, templates = pd.DataFrame(task_rows), pd.DataFrame(template_rows)
    head_rows = []
    # These summaries average tasks equally rather than letting prompt/query count dominate.
    general = [m for m in METRICS if m not in ('task_score', 'wrong_score', 'contrast_score')]
    general += [f'{m}_all' for m in general] + ['bos_sink', 'bos_sink_all']
    for (split, mode, layer, head), group in tasks.groupby(['split', 'bos_mode', 'layer', 'head'], sort=True):
        row = {'layer': layer, 'head': head, 'split': split, 'bos_mode': mode,
               'n_prompts': int(group['n_prompts'].sum())}
        for label in general:
            values = group[label].to_numpy()
            row[label] = float(finite_mean(values, axis=0))
        for task, label in TASK_COLUMNS.items():
            subset = group[group['task'] == task].iloc[0]
            row[label] = subset['task_score']
            row[f'{task}_valid_prompts'] = int(subset['valid_task_prompts'])
            if task in ('name_tracking', 'coreference'):
                row[f'{task}_wrong_score'] = subset['wrong_score']
                row[f'{task}_contrast_score'] = subset['contrast_score']
        head_rows.append(row)
    heads = pd.DataFrame(head_rows)
    parameters = []
    task_labels = list(TASK_COLUMNS.values())
    for (split, mode), indices in heads.groupby(['split', 'bos_mode'], sort=True).groups.items():
        scores = heads.loc[indices, task_labels].to_numpy(dtype=float)
        z = np.full_like(scores, np.nan)
        for column, label in enumerate(task_labels):
            finite = np.isfinite(scores[:, column])
            mean = float(scores[finite, column].mean()) if finite.any() else np.nan
            std = float(scores[finite, column].std(ddof=0)) if finite.any() else np.nan
            if finite.any():
                if std <= np.finfo(float).eps * max(1.0, abs(mean)):
                    std = 0.0
                z[finite, column] = (scores[finite, column]-mean) / std if std > 0 else 0.0
            parameters.append({'split': split, 'bos_mode': mode, 'feature': label, 'mean': mean, 'std': std})
            heads.loc[indices, 'z_'+label] = z[:, column]
        available = np.isfinite(z)
        values = np.where(available, z, -np.inf)
        choice = values.argmax(axis=1)
        specialization = values.max(axis=1)
        specialization[~available.any(axis=1)] = np.nan
        heads.loc[indices, 'specialization'] = specialization
        heads.loc[indices, 'best_task_score'] = specialization
        heads.loc[indices, 'best_task'] = [list(TASK_COLUMNS)[i] if available[j].any() else 'unclear' for j, i in enumerate(choice)]
        heads.loc[indices, 'valid_tasks'] = available.sum(axis=1)
    rng = np.random.default_rng(config['seed'])
    repeats = config.get('features', {}).get('layer_bootstrap_repeats', 1000)
    if repeats < 2:
        raise ValueError('features.layer_bootstrap_repeats must be at least 2')
    layer_rows = []
    numeric = [*general, *task_labels, 'specialization']
    for (split, mode, layer), group in heads.groupby(['split', 'bos_mode', 'layer'], sort=True):
        row = {'layer': layer, 'split': split, 'bos_mode': mode, 'n_heads': len(group),
               'ci_method': 'bootstrap_heads', 'bootstrap_repeats': repeats}
        for label in numeric:
            row['mean_'+label] = float(finite_mean(group[label].to_numpy(), axis=0))
        values = group['specialization'].to_numpy()
        values = values[np.isfinite(values)]
        if len(values):
            samples = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
            row['specialization_ci_low'], row['specialization_ci_high'] = np.quantile(samples, [0.025, 0.975])
        else:
            row['specialization_ci_low'] = row['specialization_ci_high'] = np.nan
        layer_rows.append(row)
    folder.mkdir(parents=True, exist_ok=True)
    tables = {'task_features.csv': tasks, 'template_features.csv': templates,
              'head_features.csv': heads, 'layer_summaries.csv': pd.DataFrame(layer_rows),
              'zscore_parameters.csv': pd.DataFrame(parameters)}
    for name, table in tables.items():
        write_csv(folder / name, table)
    metadata = json.loads(metadata_path.read_text())
    save_metadata(folder, config, metadata['model_config'])
    write_json(folder / 'feature_definitions.json', {
        'aggregation': 'Mean query statistics within each prompt; equal prompt weight within task; equal task weight for general head summaries',
        'gini': 'Finite-support normalized Gini; 0 uniform, 1 one-hot when support size > 1; 0 for singleton support',
        'entropy_normalization': 'Natural log of structurally available causal keys, respecting sliding windows and BOS removal',
        'bos_sink': 'Mean original normalized BOS mass, saved identically in both modes',
        'missingness': 'NaN when no non-BOS mass remains; valid prompt and query counts are saved',
        'all_tokens': 'Included mode includes BOS query; excluded mode omits undefined zero-support queries',
        'specialization': 'Maximum of per-task z scores across all heads within the same split/BOS mode (population std)',
        'layer_ci': 'Percentile bootstrap over finite head specialization values within a layer; describes across-head variability, not prompt sampling uncertainty',
        'discovery_transform': 'zscore_parameters.csv stores discovery mean/std for future held-out transforms; no validation data used in discovery features'})
    write_json(completion, {'signature': signature, 'hashes': {name: file_hash(folder / name) for name in OUTPUTS},
                            'extraction_signature': index['signature']})
    return {name: folder / name for name in OUTPUTS}
