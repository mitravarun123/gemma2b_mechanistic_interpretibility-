"""Configuration, provenance and deterministic seeding checks."""
import random
import numpy as np
import torch
from src.utils.seed import seed_all
from src.utils.io import read_config, write_json
from src.utils.logging import save_metadata
import json


def test_seed():
    seed_all(7)
    first = (random.random(), np.random.rand(), torch.rand(2))
    seed_all(7)
    second = (random.random(), np.random.rand(), torch.rand(2))
    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2])


def test_configs_and_provenance(tmp_path):
    config = read_config('configs/debug.yaml')
    assert config['dataset']['discovery_per_task'] == 20
    assert config['model']['name'] == 'google/gemma-2b'
    save_metadata(tmp_path, config, {'num_hidden_layers': 2})
    metadata = json.loads((tmp_path / 'metadata.json').read_text())
    assert metadata['packages']['torch']
    assert 'git_hash' in metadata
    write_json(tmp_path / 'nested' / 'result.json', {'ok': True})
    assert json.loads((tmp_path / 'nested' / 'result.json').read_text())['ok']
