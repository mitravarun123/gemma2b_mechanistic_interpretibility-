"""Cluster discovery heads and save projections, stability and candidate shortlist."""
import argparse
import logging
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.clustering.cluster_heads import cluster_heads
from src.utils.io import read_config
from src.utils.logging import configure_logging
from src.utils.seed import seed_all


def main() -> None:
    """Read saved features and attention summaries; run seeded discovery clustering."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/default.yaml')
    args = parser.parse_args()
    config = read_config(args.config)
    configure_logging()
    seed_all(config['seed'])
    root = Path(__file__).resolve().parents[1]/config['results_dir']
    outputs = cluster_heads(root/'features',root/'attention',root/'clusters',config)
    logging.info('Clustering outputs: %s',outputs)


if __name__=='__main__':
    main()
