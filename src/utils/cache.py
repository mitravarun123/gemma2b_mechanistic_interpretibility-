"""Content fingerprints and atomic numeric artifacts for resumable stages."""
import hashlib
from collections.abc import Iterable
import json
import os
import tempfile
from pathlib import Path
import numpy as np


def fingerprint(config: dict, paths: list[Path], extra: dict | None = None) -> str:
    """Hash configuration and input/code bytes without modification timestamps."""
    digest = hashlib.sha256(json.dumps({"config": config, "extra": extra}, sort_keys=True, default=str).encode())
    for path in sorted(paths):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    """Compute a streaming SHA256 checksum."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Atomically write a compressed NumPy archive with no Python objects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.npz')
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv(path: Path, table) -> None:
    """Atomically write a pandas table as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.csv')
    try:
        with os.fdopen(fd, 'w') as stream:
            table.to_csv(stream, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv_rows(path: Path,rows: Iterable[dict]) -> None:
    """Atomically stream nonempty dictionaries to CSV without retaining the full table."""
    import csv
    iterator = iter(rows)
    first = next(iterator,None)
    if first is None:
        raise ValueError('Streaming CSV needs at least one row to define its schema')
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary = tempfile.mkstemp(dir=path.parent,suffix='.csv')
    try:
        with os.fdopen(fd,'w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(first))
            writer.writeheader(); writer.writerow(first); writer.writerows(iterator)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
