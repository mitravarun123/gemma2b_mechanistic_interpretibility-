"""Extract attention from verified saved datasets using the configured model."""
import argparse
import json
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.extraction.attention import extract_attention, cached_extraction
from src.model.loader import load_model
from src.utils.cache import file_hash
from src.utils.io import read_config, read_jsonl
from src.utils.logging import configure_logging
from src.utils.seed import seed_all


def main() -> None:
    """Read step 01 artifacts, verify tokenizer provenance and run step 02."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    config = read_config(args.config)
    configure_logging()
    seed_all(config['seed'])
    root = Path(__file__).resolve().parents[1]
    processed = root / config.get('data_dir', 'data') / 'processed'
    required = [processed / 'discovery.jsonl', processed / 'validation.jsonl',
                processed / 'examples.jsonl', processed / 'manifest.json', processed / 'metadata.json']
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f'Missing step 01 input: {path}; run 01_generate_dataset.py with the same config')
    manifest = json.loads((processed / 'manifest.json').read_text())
    for path in required[:3]:
        if manifest['hashes'].get(path.name) != file_hash(path):
            raise ValueError(f'Dataset checksum mismatch: {path}; rerun step 01')
    metadata = json.loads((processed / 'metadata.json').read_text())
    prior = metadata['config']
    for key in ('name', 'revision'):
        if prior['model'].get(key) != config['model'].get(key):
            raise ValueError(f'Dataset tokenizer model.{key} differs from extraction config; rerun step 01')
    if prior['seed'] != config['seed'] or prior['dataset'] != config['dataset']:
        raise ValueError('Dataset seed/sizes differ from config; rerun step 01')
    records = [r for path in required[:3] for r in read_jsonl(path)]
    folder = root / config['results_dir'] / 'attention'
    if cached_extraction(folder, config, records, required):
        logging.info('Completed attention cache verified; skipping model load')
        return
    model, tokenizer, _ = load_model(config)
    if metadata['model_config']['bos_token_id'] != tokenizer.bos_token_id:
        raise ValueError('Dataset BOS differs from loaded tokenizer')
    # Check tokenization against the loaded revision, including an unpinned main update.
    for record in records:
        ids = [tokenizer.bos_token_id, *tokenizer.encode(record['text'], add_special_tokens=False)]
        if ids != record['token_ids']:
            raise ValueError(f'Tokenizer mismatch for {record["id"]}; rerun step 01')
    extract_attention(model, records, folder, config, required)


if __name__ == '__main__':
    main()
