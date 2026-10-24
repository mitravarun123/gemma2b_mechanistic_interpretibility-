"""Strict saved-result requirements for figure generation."""
from pathlib import Path
import json
import pytest
from src.figures.common import FigureContext, grid_values
from src.figures.generate import generate_figures
from src.utils.io import read_config
import pandas as pd


def test_missing_results_do_not_draw_placeholders(tmp_path):
    config = read_config('configs/debug.yaml')
    config['results_dir'] = str(tmp_path/'results')
    with pytest.raises(ValueError,match='Missing figure input'):
        generate_figures(Path.cwd(),config,[2,3])
    folder = tmp_path/'results/figures'
    manifest = json.loads((folder/'manifest.json').read_text())
    assert not manifest['complete']
    assert len(manifest['figures'])==2
    assert all(entry['status']=='blocked' for entry in manifest['figures'].values())
    assert not list(folder.glob('*.png')) and not list(folder.glob('*.pdf'))


def test_physical_head_grid_rejects_missing_or_duplicate_measurements():
    frame = pd.DataFrame({'layer':[0,0],'head':[0,0],'mean':[0.1,0.2]})
    with pytest.raises(ValueError,match='exactly one measured row'):
        grid_values(frame,'mean',1,2)


def test_figure_options_are_validated_before_rendering(tmp_path):
    config = read_config('configs/debug.yaml')
    for numbers in ([],[0],[12]):
        with pytest.raises(ValueError,match='figure numbers'):
            generate_figures(tmp_path,config,numbers)
    config['figures']['bos_mode']='typo'
    with pytest.raises(ValueError,match='BOS mode'):
        generate_figures(tmp_path,config,[3])
