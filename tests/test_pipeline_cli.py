"""Exercise the actual debug-sized scripts without gated weights or research outputs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd
import torch
import yaml
from transformers import GemmaConfig, GemmaForCausalLM
from test_dataset import tokenizer
from src.utils.io import read_config


def test_debug_sized_cli_handoffs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    model_dir = tmp_path / 'random-gemma-test-model'
    tok = tokenizer()
    torch.manual_seed(42)
    model = GemmaForCausalLM(GemmaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=256, attn_implementation='eager')).eval()
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    config = read_config('configs/debug.yaml')
    config['model']['name'] = str(model_dir)
    config['model']['device'] = 'cpu'
    config['data_dir'] = str(tmp_path / 'data')
    config['results_dir'] = str(tmp_path / 'results')
    config['figures']['run_label'] = 'Random tiny-model test · not pretrained results'
    raw = tmp_path / 'data/raw'
    raw.mkdir(parents=True)
    (raw/'names.json').write_bytes((root/'data/raw/names.json').read_bytes())
    path = tmp_path / 'debug-local.yaml'
    path.write_text(yaml.safe_dump(config))
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1')
    result = subprocess.run([sys.executable, str(root/'scripts/run_pipeline.py'), '--config', str(path),
        '--figures', *[str(i) for i in range(2,12)]], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    pipeline_report = json.loads((tmp_path/'results/pipeline_run.json').read_text())
    assert pipeline_report['status']=='complete'
    assert pipeline_report['requested_stages']==list(range(1,8))
    assert len(pipeline_report['stages'])==7
    assert all(stage['status']=='complete' for stage in pipeline_report['stages'])
    index = json.loads((tmp_path/'results/attention/index.json').read_text())
    assert index['complete'] and len(index['records']) == 161
    report = json.loads((tmp_path/'results/calibration/report.json').read_text())
    assert {r['split'] for r in report} == {'discovery','validation'}
    assert sum(r['n'] for r in report if r['template_id']=='__task__') == 80
    heads = pd.read_csv(tmp_path/'results/features/head_features.csv')
    assert len(heads) == 16
    # Repeated script 02 must avoid loading the model after verifying cache.
    result = subprocess.run([sys.executable, str(root/'scripts/02_extract_attention.py'), '--config', str(path)],
        cwd=root, env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert 'skipping model load' in result.stderr
    before = (tmp_path/'results/features/head_features.csv').stat().st_mtime_ns
    result = subprocess.run([sys.executable, str(root/'scripts/03_compute_features.py'), '--config', str(path)],
        cwd=root, env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert before == (tmp_path/'results/features/head_features.csv').stat().st_mtime_ns

    clusters = pd.read_csv(tmp_path/'results/clusters/assignments.csv')
    assert len(clusters)==4 and set(clusters['split'])=={'discovery'}

    assert (tmp_path/'results/consistency/consistency.csv').exists()
    assert (tmp_path/'results/ablation/random_control_bands.csv').exists()
    patches = pd.read_csv(tmp_path/'results/patching/summary.csv')
    assert set(patches['head'])=={0,1,2,3}
    assert set(patches['direction'])=={'denoising','noising'}
    for script in ['05_run_ablation.py','06_run_patching.py']:
        result = subprocess.run([sys.executable,str(root/'scripts'/script),'--config',str(path)],cwd=root,env=environment,capture_output=True,text=True,timeout=120)
        assert result.returncode==0,result.stderr
        assert 'skipping model load' in result.stderr

    # Figures consume the actual saved random-model measurements, without model access.
    from src.figures.generate import generate_figures
    from src.figures.common import FigureContext, InsufficientResultsError
    from src.figures import fig01_attention, fig07_ablation, fig10_atlas
    from src.figures.style import save_figure
    import matplotlib.pyplot as plt
    import numpy as np
    import pytest
    from PIL import Image
    command = [sys.executable,str(root/'scripts/run_pipeline.py'),'--from-stage','7','--config',str(path),
               '--figures',*[str(i) for i in range(2,12)]]
    result = subprocess.run(command,cwd=root,env=environment,capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    folder = tmp_path/'results/figures'
    manifest = json.loads((folder/'manifest.json').read_text())
    assert manifest['complete'] and len(manifest['figures'])==10
    assert len(list(folder.glob('*.png')))==len(list(folder.glob('*.pdf')))==10
    for entry in manifest['figures'].values():
        assert entry['inputs'] and len(entry['outputs'])==2
        for name in entry['outputs']:
            if name.endswith('.png'):
                with Image.open(name) as picture:
                    assert all(abs(dpi-300)<0.1 for dpi in picture.info['dpi'])
                    assert min(picture.size)>1000
            else:
                assert Path(name).read_bytes().startswith(b'%PDF-')
    before = {p:p.stat().st_mtime_ns for p in folder.glob('*.png')}
    result = subprocess.run(command,cwd=root,env=environment,capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stderr
    assert result.stderr.count('skipping render')==10
    assert before=={p:p.stat().st_mtime_ns for p in folder.glob('*.png')}

    context = FigureContext(root,config)
    figure = fig07_ablation.build(context)
    saved = pd.read_csv(tmp_path/'results/ablation/summary.csv')
    expected = saved[(saved['kind']=='single')&(saved['task']=='all')&(saved['metric']=='accuracy')]
    np.testing.assert_allclose(figure.axes[0].images[0].get_array(),expected.sort_values(['layer','head'])['mean'].to_numpy().reshape(1,4))
    plt.close(figure)
    validation = pd.read_csv(tmp_path/'results/patching/validated_heads.csv')
    assert not validation['causally_validated'].any()
    with pytest.raises(InsufficientResultsError,match='none passed'):
        generate_figures(root,config,[1])
    assert not (folder/'fig01_attention_heatmap.png').exists()
    blocked = json.loads((folder/'manifest.json').read_text())
    assert not blocked['complete'] and blocked['figures']['1']['status']=='blocked'
    figure = fig10_atlas.build(FigureContext(root,config))
    assert np.isin(figure.axes[0].images[0].get_array(),[0,1]).all()
    plt.close(figure)

    # Exercise Figure 1's rendering using actual attention; explicitly unvalidated.
    example = json.loads((tmp_path/'data/processed/examples.jsonl').read_text().splitlines()[0])
    entry = index['records'][example['id']]
    with np.load(tmp_path/'results/attention'/entry['file']) as shard:
        figure = fig01_attention.render_panels(shard['full_attention'][1],example['token_labels'],
            shard['query_positions'],shard['target_positions'],[(0,0),(0,1)],
            ['Random test head L0H0 (unvalidated)','Random test head L0H1 (unvalidated)'])
        np.testing.assert_allclose(figure.axes[0].images[0].get_array(),shard['full_attention'][1,0,0])
    save_figure(figure,tmp_path/'renderer-smoke','unvalidated_attention_test')
    plt.close(figure)

    # Modified saved results must invalidate caches and be rejected by checksums.
    summary = tmp_path/'results/ablation/summary.csv'
    original = summary.read_bytes()
    summary.write_bytes(original+b'\n')
    with pytest.raises(InsufficientResultsError,match='checksum mismatch'):
        generate_figures(root,config,[7])
    assert not (folder/'fig07_ablation_accuracy.png').exists()

    summary.write_bytes(original)
