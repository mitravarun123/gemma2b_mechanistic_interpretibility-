"""Memory-bounded attention extraction into resumable per-prompt numeric shards."""
import json
import hashlib
import logging
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from src.dataset.generate import length_batches
from src.features.rows import BOS_MODES, METRICS, finite_mean, summarize_rows
from src.model.loader import architecture
from src.utils.cache import file_hash, fingerprint, write_npz
from src.utils.io import write_json
from src.utils.logging import save_metadata


def sliding_windows(model) -> list[int | None]:
    """Read effective per-layer sliding-window policies from HF attention modules."""
    info = architecture(model.config)
    modules = [module for name, module in model.named_modules() if name.endswith('.self_attn')]
    if len(modules) != info.n_layers:
        raise ValueError('Cannot identify every decoder self-attention module')
    config = getattr(model.config, 'text_config', model.config)
    types = getattr(config, 'layer_types', None)
    windows = []
    for index, module in enumerate(modules):
        window = getattr(module, 'sliding_window', None)
        if getattr(module, 'is_sliding', True) is False:
            window = None
        if types is not None:
            if len(types) != info.n_layers:
                raise ValueError('layer_types count does not match decoder layers')
            window = getattr(config, 'sliding_window', window) if types[index] == 'sliding_attention' else None
        windows.append(int(window) if window is not None else None)
    return windows


def extraction_batches(records: list[dict], batch_size: int, layers: int, heads: int,
                       max_attention_mb: float) -> list[list[dict]]:
    """Cap materialized attention tensor bytes in addition to configured batch size.

    This conservative float32 estimate covers attention outputs, not model weights
    or temporary eager-attention workspace; lower the limit if device memory is tight.
    """
    if max_attention_mb <= 0:
        raise ValueError('max_attention_mb must be positive')
    batches = []
    for batch in length_batches(records, batch_size):
        length = len(batch[0]['token_ids'])
        bytes_per_prompt = layers * heads * length * length * 4
        maximum = int(max_attention_mb * 1024**2 // bytes_per_prompt)
        if maximum < 1:
            raise MemoryError(f'Prompt {batch[0]["id"]} exceeds attention budget; raise extraction.max_attention_mb')
        for start in range(0, len(batch), maximum):
            batches.append(batch[start:start + maximum])
    return batches


def validate_record(record: dict, vocab_size: int) -> None:
    """Reject malformed or misaligned saved records before model execution."""
    ids = record['token_ids']
    if not ids or any(not isinstance(i, int) or not 0 <= i < vocab_size for i in ids):
        raise ValueError(f'Invalid token IDs in {record["id"]}')
    if len(record['token_categories']) != len(ids):
        raise ValueError(f'Category length mismatch in {record["id"]}')
    spans = record['key_spans']
    if not spans:
        raise ValueError(f'No task query spans in {record["id"]}')
    queries = [span['query'] for span in spans]
    if len(set(queries)) != len(queries):
        raise ValueError('Each task query must occur once in a prompt')
    for span in spans:
        if not 0 <= span['key'] < span['query'] < len(ids):
            raise ValueError(f'Noncausal/out-of-range task span in {record["id"]}')
        if 'wrong_key' in span and not 0 <= span['wrong_key'] < span['query']:
            raise ValueError(f'Invalid wrong-antecedent span in {record["id"]}')


def extraction_signature(config: dict, input_paths: list[Path], records: list[dict],
                         model_config: dict, windows: list[int | None]) -> str:
    """Fingerprint extraction inputs, implementation and recorded model snapshot."""
    source_root = Path(__file__).resolve().parents[1]
    code = [Path(__file__), *sorted((source_root / 'features').glob('*.py')),
            source_root / 'utils/cache.py', source_root / 'model/loader.py',
            source_root.parent / 'scripts/02_extract_attention.py']
    return fingerprint(config, [*input_paths, *code], extra={
        'model_config': model_config, 'windows': windows, 'records': records})


def cached_extraction(folder: Path, config: dict, records: list[dict], input_paths: list[Path]) -> bool:
    """Verify a completed extraction using its recorded snapshot before loading weights.

    Resume uses the previously resolved model revision. Pin model.revision for an
    immutable remote checkpoint; removing the attention folder explicitly requests a fresh run.
    """
    index_path, metadata_path = folder / 'index.json', folder / 'metadata.json'
    if not index_path.exists() or not metadata_path.exists():
        return False
    index, metadata = json.loads(index_path.read_text()), json.loads(metadata_path.read_text())
    if not index['complete'] or set(index['records']) != {r['id'] for r in records}:
        return False
    signature = extraction_signature(config, input_paths, records, metadata['model_config'], index['sliding_windows'])
    if signature != index['signature']:
        return False
    return all((folder / entry['file']).exists() and file_hash(folder / entry['file']) == entry['sha256']
               for entry in index['records'].values())


@torch.inference_mode()
def extract_attention(model, records: list[dict], folder: Path, config: dict,
                      input_paths: list[Path] | None = None) -> Path:
    """Save task-query rows and per-prompt head summaries, resuming verified shards.

    Full matrices are retained only for a configurable number of examples. All-token
    reference summaries include BOS query zero in the included mode; excluded mode
    drops undefined rows with no non-BOS support and records valid query counts.
    """
    folder = Path(folder)
    info = architecture(model.config)
    windows = sliding_windows(model)
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Prompt IDs must be unique across splits')
    for record in records:
        validate_record(record, getattr(getattr(model.config, 'text_config', model.config), 'vocab_size'))
    signature = extraction_signature(config, input_paths or [], records, model.config.to_dict(), windows)
    example_count = config['extraction'].get('example_prompts', 4)
    if example_count < 0:
        raise ValueError('example_prompts cannot be negative')
    # Prefer the explicit Figure 1 diagnostic, then diversify retained task examples.
    ordered = sorted(records, key=lambda r: (r['split'] != 'example', r['id']))
    examples = []
    seen = set()
    for record in ordered:
        group = (record['split'], record['task'])
        if group not in seen and len(examples) < example_count:
            examples.append(record['id'])
            seen.add(group)
    for record in ordered:
        if len(examples) < example_count and record['id'] not in examples:
            examples.append(record['id'])
    index_path = folder / 'index.json'
    previous = json.loads(index_path.read_text()) if index_path.exists() else {}
    cached = previous.get('records', {}) if previous.get('signature') == signature else {}
    verified = {}
    for record in records:
        entry = cached.get(record['id'])
        sidecar = folder / 'prompts' / (hashlib.sha256(record['id'].encode()).hexdigest() + '.json')
        if entry is None and sidecar.exists():
            checkpoint = json.loads(sidecar.read_text())
            if checkpoint['signature'] == signature:
                entry = checkpoint['entry']
        if entry and (folder / entry['file']).exists() and file_hash(folder / entry['file']) == entry['sha256']:
            verified[record['id']] = entry
    index = {'schema_version': 1, 'signature': signature, 'metrics': list(METRICS),
             'bos_modes': list(BOS_MODES), 'architecture': vars(info),
             'sliding_windows': windows, 'example_ids': examples, 'records': verified,
             'expected_ids': [r['id'] for r in records], 'complete': False}
    folder.mkdir(parents=True, exist_ok=True)
    save_metadata(folder, config, model.config.to_dict())
    write_json(index_path, index)
    pending = [r for r in records if r['id'] not in verified]
    batches = extraction_batches(pending, config['batch_size'], info.n_layers, info.n_heads,
        config['extraction'].get('max_attention_mb', 256))
    logging.info('Extracting %d pending prompts; %d cached; sliding windows=%s', len(pending), len(verified), windows)
    device = next(model.parameters()).device
    model.eval()
    for batch in tqdm(batches, desc='Attention extraction'):
        tokens = torch.tensor([r['token_ids'] for r in batch], device=device)
        output = model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
            output_attentions=True, use_cache=False, return_dict=True)
        if output.attentions is None or len(output.attentions) != info.n_layers:
            raise RuntimeError('Model did not return all attention tensors; load attn_implementation="eager"')
        for b, record in enumerate(batch):
            length = len(record['token_ids'])
            queries = np.array([s['query'] for s in record['key_spans']], dtype=np.int64)
            targets = np.array([s['key'] for s in record['key_spans']], dtype=np.int64)
            wrong = np.array([s.get('wrong_key', -1) for s in record['key_spans']], dtype=np.int64)
            rows = np.empty((2, info.n_layers, info.n_heads, len(queries), length), dtype=np.float32)
            selected = np.empty((2, info.n_layers, info.n_heads, len(METRICS)), dtype=np.float32)
            all_tokens = np.empty_like(selected)
            counts = np.empty((2, info.n_layers, info.n_heads), dtype=np.int64)
            all_counts = np.empty_like(counts)
            sink = np.empty((info.n_layers, info.n_heads), dtype=np.float32)
            all_sink = np.empty_like(sink)
            full = np.empty((2, info.n_layers, info.n_heads, length, length), dtype=np.float32) if record['id'] in examples else None
            for layer, tensor in enumerate(output.attentions):
                if tensor is None or tuple(tensor.shape) != (len(batch), info.n_heads, length, length):
                    raise RuntimeError(f'Unexpected attention dimensions at layer {layer}')
                a = tensor[b].float().cpu().numpy()
                metrics, valid, bos = summarize_rows(a[:, queries, :], queries, record['token_categories'], targets, wrong, windows[layer])
                selected[:, layer] = finite_mean(metrics, axis=2)
                counts[:, layer] = valid.sum(axis=-1)
                sink[layer] = bos.mean(axis=-1)
                reference, reference_valid, reference_bos = summarize_rows(a, np.arange(length), record['token_categories'], window=windows[layer])
                all_tokens[:, layer] = finite_mean(reference, axis=2)
                all_counts[:, layer] = reference_valid.sum(axis=-1)
                all_sink[layer] = reference_bos.mean(axis=-1)
                a = a / a.sum(-1, keepdims=True)
                without_bos = a.copy()
                without_bos[..., 0] = 0
                denominator = without_bos.sum(-1, keepdims=True)
                without_bos = np.divide(without_bos, denominator, out=np.full_like(without_bos, np.nan), where=denominator > 0)
                rows[0, layer], rows[1, layer] = a[:, queries], without_bos[:, queries]
                if full is not None:
                    full[0, layer], full[1, layer] = a, without_bos
            arrays = {'query_positions': queries, 'target_positions': targets, 'wrong_positions': wrong,
                      'rows': rows, 'selected_summary': selected, 'all_summary': all_tokens,
                      'valid_query_counts': counts, 'all_valid_query_counts': all_counts,
                      'bos_sink': sink, 'all_bos_sink': all_sink,
                      'token_ids': np.array(record['token_ids'], dtype=np.int64),
                      'token_categories': np.array(record['token_categories'])}
            if full is not None:
                arrays['full_attention'] = full
            # IDs are hashed into filenames to prevent path traversal and collisions.
            filename = 'prompts/' + hashlib.sha256(record['id'].encode()).hexdigest() + '.npz'
            path = folder / filename
            write_npz(path, **arrays)
            index['records'][record['id']] = {'file': filename, 'sha256': file_hash(path),
                'id': record['id'], 'task': record['task'], 'split': record['split'],
                'template_id': record['template_id'], 'num_distractors': record['num_distractors'],
                'text': record['text'], 'has_full_attention': full is not None}
            write_json(path.with_suffix('.json'), {'signature': signature, 'entry': index['records'][record['id']]})
        del output
    index['records'] = dict(sorted(index['records'].items()))
    index['complete'] = set(index['records']) == set(index['expected_ids'])
    write_json(index_path, index)
    return index_path
