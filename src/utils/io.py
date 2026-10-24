"""Configuration and atomic result writing."""
import json
import os
import tempfile
from pathlib import Path
import yaml


def read_config(path: str | Path) -> dict:
    """Read a YAML mapping; reject malformed configuration."""
    with open(path) as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    return config


def write_json(path: str | Path, value: object) -> None:
    """Atomically replace a JSON artifact in its destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_jsonl(path: str | Path) -> list[dict]:
    """Read structured prompt or result records from disk."""
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    """Write a complete JSONL artifact atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as stream:
            for record in records:
                stream.write(json.dumps(record) + '\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
