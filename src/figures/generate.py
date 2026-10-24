"""Render saved scientific results with per-figure provenance and verified caching."""
import importlib
import json
import logging
from pathlib import Path
from src.figures.style import apply_style, save_figure
import matplotlib.pyplot as plt
from src.figures.common import FigureContext, InsufficientResultsError
from src.utils.cache import fingerprint, file_hash
from src.utils.io import write_json
from src.utils.logging import save_metadata

MODULES = ('fig01_attention', 'fig02_pipeline', 'fig03_entropy', 'fig04_distance',
           'fig05_clusters', 'fig06_tasks', 'fig07_ablation', 'fig08_probability',
           'fig09_patching', 'fig10_atlas', 'fig11_specialization')


def generate_figures(root: Path, config: dict, numbers: list[int] | None = None) -> dict:
    """Export requested figures; record unsupported figures and fail without placeholders."""
    numbers = list(range(1, 12)) if numbers is None else sorted(set(numbers))
    if not numbers or any(n not in range(1, 12) for n in numbers):
        raise ValueError('Choose figure numbers from 1 through 11')
    options = config.get('figures', {})
    if options.get('bos_mode', config['clustering'].get('bos_mode', 'excluded')) not in ('included', 'excluded'):
        raise ValueError('Figure BOS mode must be included or excluded')
    if options.get('top_heads', 8) < 1:
        raise ValueError('figures.top_heads must be positive')
    folder = root/config['results_dir']/'figures'
    manifest_path = folder/'manifest.json'
    previous = json.loads(manifest_path.read_text()).get('figures', {}) if manifest_path.exists() else {}
    source = Path(__file__).resolve().parent
    code = [*source.glob('*.py'), source.parent/'utils/cache.py', source.parent/'utils/io.py',
            source.parent/'utils/logging.py', source.parents[1]/'scripts/07_generate_figures.py']
    signature = fingerprint(config, code)
    manifest = {'requested': numbers, 'complete': False, 'figures': {}}
    errors = []
    apply_style()
    logging.getLogger('fontTools.subset').setLevel(logging.WARNING)
    for number in numbers:
        module = importlib.import_module('src.figures.'+MODULES[number-1])
        context = FigureContext(root, config)
        old = previous.get(str(number), {})
        if (old.get('status') == 'complete' and old.get('signature') == signature
                and old.get('inputs') and len(old.get('outputs', {})) == 2
                and all(Path(path).is_file() and file_hash(Path(path)) == digest
                        for path, digest in {**old['inputs'], **old['outputs']}.items())):
            manifest['figures'][str(number)] = old
            logging.info('Figure %s cache verified; skipping render', number)
            continue
        try:
            context.json('attention/metadata.json')
            figure = module.build(context)
            if options.get('run_label'):
                # A figure-level legend keeps the label outside the scientific axes.
                figure.legend([], [], title=options['run_label'], loc='outside lower center', frameon=False)
            outputs = save_figure(figure, folder, module.STEM)
            manifest['figures'][str(number)] = {
                'status': 'complete', 'signature': signature, 'selection': context.selection,
                'inputs': {str(p.resolve()): file_hash(p) for p in sorted(context.inputs)},
                'outputs': {str(p.resolve()): file_hash(p) for p in outputs}}
            logging.info('Saved figure %s: %s', number, module.STEM)
        except (FileNotFoundError, InsufficientResultsError, ValueError) as error:
            errors.append(f'Figure {number}: {error}')
            manifest['figures'][str(number)] = {'status': 'blocked', 'reason': str(error)}
            # A failed rebuild must not leave stale pictures appearing current.
            for suffix in ('png', 'pdf'):
                (folder/f'{module.STEM}.{suffix}').unlink(missing_ok=True)
            logging.error('%s', errors[-1])
        finally:
            plt.close('all')
            write_json(manifest_path, manifest)
    metadata_path = root/config['results_dir']/'attention/metadata.json'
    if metadata_path.exists():
        save_metadata(folder, config, json.loads(metadata_path.read_text())['model_config'])
    manifest['complete'] = not errors
    write_json(manifest_path, manifest)
    if errors:
        raise InsufficientResultsError('Requested figure export incomplete:\n'+'\n'.join(errors))
    return manifest
