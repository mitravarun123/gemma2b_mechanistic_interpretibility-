"""Render the eleven publication figures solely from saved pipeline artifacts."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.figures.generate import generate_figures
from src.utils.io import read_config
from src.utils.logging import configure_logging


def main() -> None:
    """Generate all figures or an explicitly requested subset, without loading a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--figures', nargs='+', type=int, choices=range(1, 12))
    args = parser.parse_args()
    configure_logging()
    generate_figures(Path(__file__).resolve().parents[1], read_config(args.config), args.figures)


if __name__ == '__main__':
    main()
