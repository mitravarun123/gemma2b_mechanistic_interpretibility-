"""Sequential CLI orchestration; stage-owned caches remain the source of truth."""
from datetime import datetime, timezone
import logging
from pathlib import Path
import shlex
import subprocess
import sys
from src.utils.cache import file_hash
from src.utils.io import read_config, write_json

STAGES = ('01_generate_dataset.py', '02_extract_attention.py', '03_compute_features.py',
          '04_cluster_heads.py', '05_run_ablation.py', '06_run_patching.py', '07_make_figures.py')


def run_pipeline(root: Path, config_path: Path, first: int = 1, last: int = 7,
                 figures: list[int] | None = None, dry_run: bool = False) -> int:
    """Run requested stages in order, recording failures without continuing downstream."""
    if not 1 <= first <= last <= len(STAGES):
        raise ValueError('Require 1 <= from-stage <= through-stage <= 7')
    if figures is not None and (last != 7 or not figures or any(n not in range(1, 12) for n in figures)):
        raise ValueError('A figure subset requires stage 7 and numbers from 1 through 11')
    root, config_path = root.resolve(), config_path.resolve()
    config = read_config(config_path)
    commands = []
    for number in range(first, last+1):
        command = [sys.executable, str(root/'scripts'/STAGES[number-1]), '--config', str(config_path)]
        if number == 7 and figures is not None:
            command.extend(['--figures', *map(str, figures)])
        commands.append((number, command))
    if dry_run:
        for _, command in commands:
            print(shlex.join(command))
        return 0
    report_path = root/config['results_dir']/'pipeline_run.json'
    report = {'config_path': str(config_path), 'config_sha256': file_hash(config_path),
              'started_at': datetime.now(timezone.utc).isoformat(),
              'requested_stages': list(range(first, last+1)),
              'requested_figures': (figures if figures is not None else list(range(1, 12))) if last == 7 else [],
              'status': 'running', 'stages': []}
    write_json(report_path, report)
    for number, command in commands:
        stage = {'stage': number, 'script': STAGES[number-1], 'status': 'running'}
        report['stages'].append(stage)
        write_json(report_path, report)
        logging.info('Stage %02d/07: %s', number, shlex.join(command))
        try:
            code = subprocess.run(command, cwd=root, check=False).returncode
            code = code if code >= 0 else 128-code
        except KeyboardInterrupt:
            code = 130
        except OSError as error:
            stage['error'] = str(error)
            code = 1
        stage.update(returncode=code, status='complete' if code == 0 else 'failed')
        if code:
            report.update(status='failed', failed_stage=number, finished_at=datetime.now(timezone.utc).isoformat())
            write_json(report_path, report)
            logging.error('Stopped at stage %02d (exit %s). Resolve the error above and rerun; stage caches verify saved work. Report: %s', number, code, report_path)
            return code
        write_json(report_path, report)
    report.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat())
    write_json(report_path, report)
    logging.info('Requested stages completed. Report: %s', report_path)
    return 0
