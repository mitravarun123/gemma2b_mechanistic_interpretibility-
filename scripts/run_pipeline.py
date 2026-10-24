"""Run the debug pipeline by default, or a configured full/restarted run."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import run_pipeline
from src.utils.logging import configure_logging


def main() -> int:
    """Parse the requested stages and preserve a failing stage's exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/debug.yaml')
    parser.add_argument('--from-stage', type=int, choices=range(1, 8), default=1)
    parser.add_argument('--through-stage', type=int, choices=range(1, 8), default=7)
    parser.add_argument('--figures', nargs='+', type=int, choices=range(1, 12), help='Explicit figure subset; stage 7 must be included')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without running stages or writing artifacts')
    args = parser.parse_args()
    if args.from_stage > args.through_stage:
        parser.error('--from-stage must not exceed --through-stage')
    if args.figures is not None and args.through_stage != 7:
        parser.error('--figures requires stage 7')
    configure_logging()
    return run_pipeline(ROOT, args.config, args.from_stage, args.through_stage, args.figures, args.dry_run)


if __name__ == '__main__':
    raise SystemExit(main())
