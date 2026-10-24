"""Specification-compatible alias for the saved-result figure exporter."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('07_generate_figures.py')), run_name='__main__')
