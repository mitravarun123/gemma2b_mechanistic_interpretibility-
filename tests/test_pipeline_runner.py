"""Runner failure propagation and side-effect-free command planning."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.pipeline import run_pipeline


def config_file(tmp_path):
    """Write only the configuration fields the orchestration layer needs."""
    path = tmp_path/'run.yaml'
    path.write_text('results_dir: results\n')
    return path


def test_failure_stops_downstream_and_preserves_exit_code(tmp_path, monkeypatch):
    calls = []
    def fail(command, **kwargs):
        calls.append((command,kwargs))
        return SimpleNamespace(returncode=23)
    monkeypatch.setattr('src.pipeline.subprocess.run',fail)
    config = config_file(tmp_path)
    assert run_pipeline(tmp_path,config)==23
    assert len(calls)==1 and Path(calls[0][0][1]).name=='01_generate_dataset.py'
    report = json.loads((tmp_path/'results/pipeline_run.json').read_text())
    assert report['status']=='failed' and report['failed_stage']==1
    assert report['stages'][0]['returncode']==23


def test_dry_run_has_no_subprocess_or_results(tmp_path, monkeypatch, capsys):
    def unexpected(*args,**kwargs):
        pytest.fail('Dry run launched a subprocess')
    monkeypatch.setattr('src.pipeline.subprocess.run',unexpected)
    assert run_pipeline(tmp_path,config_file(tmp_path),first=6,figures=[2,9],dry_run=True)==0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines)==2 and '06_run_patching.py' in lines[0]
    assert '07_make_figures.py' in lines[1] and '--figures 2 9' in lines[1]
    assert not (tmp_path/'results').exists()


def test_invalid_ranges_do_not_write_results(tmp_path):
    config = config_file(tmp_path)
    for first,last in [(0,7),(7,1),(1,8)]:
        with pytest.raises(ValueError,match='from-stage'):
            run_pipeline(tmp_path,config,first,last)
    with pytest.raises(ValueError,match='figure subset'):
        run_pipeline(tmp_path,config,last=6,figures=[2])
    assert not (tmp_path/'results').exists()
