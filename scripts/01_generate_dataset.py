"""Generate tokenizer-validated datasets and run pretrained baseline calibration."""
import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tqdm import tqdm
from transformers import AutoTokenizer
from src.dataset.names import build_name_pools
from src.dataset.generate import generate_split, figure_example
from src.dataset.templates import TEMPLATES
from src.dataset.calibration import calibrate
from src.model.loader import load_model
from src.utils.io import read_config, read_jsonl, write_json, write_jsonl
from src.utils.seed import seed_all
from src.utils.logging import configure_logging, save_metadata


def main() -> None:
    """Resume only artifacts with a matching config, tokenizer and code fingerprint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--dataset-only', action='store_true', help='Explicitly defer pretrained calibration')
    args = parser.parse_args()
    configure_logging()
    config = read_config(args.config)
    seed_all(config['seed'])
    root = Path(__file__).resolve().parents[1]
    data = root / config.get('data_dir', 'data')
    raw, processed = data / 'raw', data / 'processed'
    settings = config['model']
    tokenizer = AutoTokenizer.from_pretrained(settings['name'], revision=settings.get('revision', 'main'),
        token=os.environ.get('HF_TOKEN'), use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError('A fast tokenizer is required for reliable span offsets')
    candidates = json.loads((raw / 'names.json').read_text())
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    fingerprint.update(tokenizer.backend_tokenizer.to_str().encode())
    for path in sorted((root / 'src/dataset').glob('*.py')) + [raw / 'names.json', Path(__file__)]:
        fingerprint.update(path.read_bytes())
    signature = fingerprint.hexdigest()
    manifest = processed / 'manifest.json'
    outputs = [processed / f'{split}.jsonl' for split in ('discovery', 'validation')]
    outputs += [processed / 'corrupted' / f'{split}.jsonl' for split in ('discovery', 'validation')]
    outputs += [processed / 'name_pools.json', processed / 'templates.json', processed / 'examples.jsonl']
    valid_cache = False
    if manifest.exists() and all(path.exists() for path in outputs):
        previous = json.loads(manifest.read_text())
        valid_cache = previous['signature'] == signature and all(
            previous['hashes'].get(str(path.relative_to(processed))) == hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs)
    if valid_cache:
        logging.info('Dataset cache verified; skipping generation')
    else:
        pools = build_name_pools(tokenizer, candidates, config['seed'])
        write_json(processed / 'name_pools.json', pools)
        write_json(processed / 'templates.json', TEMPLATES)
        write_jsonl(processed / 'examples.jsonl', [figure_example(tokenizer)])
        for split in tqdm(('discovery', 'validation'), desc='Generate splits'):
            records, twins = generate_split(tokenizer, pools[split], split,
                config['dataset'][f'{split}_per_task'], config['seed'] + (split == 'validation'))
            write_jsonl(processed / f'{split}.jsonl', records)
            write_jsonl(processed / 'corrupted' / f'{split}.jsonl', twins)
        write_json(manifest, {'signature': signature, 'hashes': {
            str(path.relative_to(processed)): hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs}})
    save_metadata(processed, config, {'tokenizer': settings['name'], 'bos_token_id': tokenizer.bos_token_id})
    if args.dataset_only:
        logging.info('Dataset-only run requested; calibration deferred')
        return
    folder = root / config['results_dir'] / 'calibration'
    completion = folder / 'completion.json'
    if completion.exists():
        previous = json.loads(completion.read_text())
        if previous['signature'] == signature and all((folder / name).exists() and hashlib.sha256((folder / name).read_bytes()).hexdigest() == checksum for name, checksum in previous['hashes'].items()):
            logging.info('Calibration cache verified; skipping model load')
            return
    model, _, info = load_model(config)
    records = [record for split in ('discovery', 'validation') for record in read_jsonl(processed / f'{split}.jsonl')]
    checkpoint_path = folder / 'baseline.jsonl'
    run_path = folder / 'run.json'
    existing = []
    if run_path.exists() and checkpoint_path.exists() and json.loads(run_path.read_text())['signature'] == signature:
        existing = read_jsonl(checkpoint_path)
    write_json(run_path, {'signature': signature})
    rows, report = calibrate(model, records, config['batch_size'], existing=existing,
        checkpoint=lambda rows: write_jsonl(checkpoint_path, rows))
    write_jsonl(folder / 'baseline.jsonl', rows)
    write_json(folder / 'report.json', report)
    save_metadata(folder, config, model.config.to_dict())
    write_json(completion, {'signature': signature, 'hashes': {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in ('baseline.jsonl', 'report.json')}})
    for row in report:
        logging.info('Calibration: %s', row)


if __name__ == '__main__':
    main()
