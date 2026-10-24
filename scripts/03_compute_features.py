"""Compute head features solely from step 02 numeric artifacts."""
import argparse
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.compute import aggregate_features
from src.utils.io import read_config
from src.utils.logging import configure_logging
from src.utils.seed import seed_all


def main() -> None:
    """Aggregate attention summaries and save CSV feature tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    config = read_config(args.config)
    configure_logging()
    seed_all(config['seed'])
    root = Path(__file__).resolve().parents[1]
    results = root / config['results_dir']
    outputs = aggregate_features(results / 'attention', results / 'features', config)
    logging.info('Feature outputs: %s', outputs)


if __name__ == '__main__':
    main()
