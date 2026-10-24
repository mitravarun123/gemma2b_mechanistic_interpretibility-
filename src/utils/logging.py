"""Run provenance and logging."""
import importlib.metadata
import logging
import subprocess
from pathlib import Path
from .io import write_json


def configure_logging() -> None:
    """Enable timestamped progress messages."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def save_metadata(folder: str | Path, config: dict, model_config: dict) -> None:
    """Record package versions, configuration and Git revision without secrets."""
    packages = ["torch", "transformers", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn", "tqdm", "pyyaml", "pytest"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    write_json(Path(folder) / "metadata.json", {"packages": versions, "config": config, "model_config": model_config, "git_hash": git.stdout.strip() if git.returncode == 0 else None})
