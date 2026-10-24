"""Shared print-ready, colorblind-safe figure appearance and atomic exports."""
import os
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'gemma-head-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PALETTE = ('#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9','#F0E442','#000000')
TASK_LABELS = {'name_tracking':'Name tracking','coreference':'Coreference','induction':'Induction','local_context':'Local context'}


def apply_style() -> None:
    """Use readable sans-serif type, restrained spines and embedded PDF fonts."""
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
        'pdf.fonttype':42,'ps.fonttype':42,'figure.constrained_layout.use':True})


def save_figure(figure,folder: Path,stem: str) -> tuple[Path,Path]:
    """Atomically export the same figure as 300 dpi PNG and vector-capable PDF."""
    folder.mkdir(parents=True,exist_ok=True)
    paths = []
    for extension in ('png','pdf'):
        target = folder/f'{stem}.{extension}'
        fd,temporary = tempfile.mkstemp(dir=folder,suffix='.'+extension); os.close(fd)
        try:
            figure.savefig(temporary,format=extension,dpi=300,bbox_inches='tight')
            os.replace(temporary,target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        paths.append(target)
    return tuple(paths)
