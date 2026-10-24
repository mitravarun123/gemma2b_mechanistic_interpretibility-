"""Strict saved-result access and shared rendering helpers; no model/statistic fitting."""
from dataclasses import dataclass,field
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.utils.cache import file_hash
from src.figures.style import PALETTE


class InsufficientResultsError(ValueError):
    """A required scientific claim is unsupported by the saved measurements."""


@dataclass
class FigureContext:
    """Paths/config and a trace of every saved artifact used by one figure."""
    root: Path
    config: dict
    inputs: set[Path] = field(default_factory=set)
    selection: dict = field(default_factory=dict)

    @property
    def results(self) -> Path:
        """Resolve the configured results directory."""
        return self.root/self.config['results_dir']

    @property
    def processed(self) -> Path:
        """Resolve saved tokenizer-validated datasets."""
        return self.root/self.config.get('data_dir','data')/'processed'

    @property
    def mode(self) -> str:
        """Use the configured plotting BOS mode, defaulting to the clustering mode."""
        return self.config.get('figures',{}).get('bos_mode',self.config['clustering'].get('bos_mode','excluded'))

    @property
    def split(self) -> str:
        """Resolve the plotting split for descriptive head features."""
        return self.config.get('figures',{}).get('split','discovery')

    @property
    def task(self) -> str:
        """Resolve the held-out behavioral task stratum used in causal plots."""
        return self.config.get('figures',{}).get('behavioral_task','all')

    def require(self,path: Path) -> Path:
        """Reject missing or altered artifacts and track provenance dependencies."""
        if not path.exists():
            raise FileNotFoundError(f'Missing figure input: {path}; run its generating stage')
        self.inputs.add(path)
        completion = path.parent/'completion.json'
        if completion.exists() and path!=completion:
            record = json.loads(completion.read_text())
            self.inputs.add(completion)
            if path.name in record['hashes'] and file_hash(path)!=record['hashes'][path.name]:
                raise ValueError(f'Figure input checksum mismatch: {path}')
        return path

    def table(self,relative: str,columns: list[str] | None = None) -> pd.DataFrame:
        """Load a saved CSV and enforce its expected schema."""
        path = self.require(self.results/relative)
        frame = pd.read_csv(path)
        if columns and not set(columns)<=set(frame.columns):
            raise ValueError(f'Invalid figure input schema: {path}; rerun its generating stage')
        return frame

    def json(self,relative: str) -> dict:
        """Load structured saved results without deriving measurements."""
        return json.loads(self.require(self.results/relative).read_text())

    def records(self,name: str) -> list[dict]:
        """Read saved dataset records and verify the generation manifest when present."""
        path = self.require(self.processed/name)
        manifest = self.processed/'manifest.json'
        if manifest.exists():
            self.inputs.add(manifest)
            hashes = json.loads(manifest.read_text())['hashes']
            if hashes.get(name)!=file_hash(path):
                raise ValueError(f'Dataset checksum mismatch: {path}')
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def heads(self) -> pd.DataFrame:
        """Select measured head features for one split/BOS mode in physical head order."""
        frame = self.table('features/head_features.csv',['layer','head','split','bos_mode','specialization'])
        selected = frame[(frame['split']==self.split)&(frame['bos_mode']==self.mode)].sort_values(['layer','head'])
        if selected.empty or selected[['layer','head']].duplicated().any():
            raise ValueError('No unique saved head features for plotting split/BOS mode')
        return selected.reset_index(drop=True)

    def architecture(self) -> tuple[int,int]:
        """Read physical grid dimensions from extraction results."""
        info = self.json('attention/index.json')['architecture']
        return info['n_layers'],info['n_heads']


def finite_table(frame: pd.DataFrame,columns: list[str],message: str) -> pd.DataFrame:
    """Select finite saved observations; fail if none support the requested plot."""
    mask = np.isfinite(frame[columns].to_numpy(dtype=float)).all(axis=1)
    selected = frame[mask]
    if selected.empty:
        raise InsufficientResultsError(message)
    return selected


def head_labels(frame: pd.DataFrame) -> list[str]:
    """Format measured physical head identifiers."""
    return [f'L{int(r.layer)}H{int(r.head)}' for r in frame.itertuples()]


def grid_values(frame: pd.DataFrame,value: str,layers: int,heads: int) -> np.ndarray:
    """Reshape a complete saved per-head table without averaging or filling missing heads."""
    expected = {(l,h) for l in range(layers) for h in range(heads)}
    if len(frame)!=len(expected) or set(zip(frame['layer'],frame['head']))!=expected:
        raise ValueError('Figure requires exactly one measured row for every model head')
    return frame.sort_values(['layer','head'])[value].to_numpy(dtype=float).reshape(layers,heads)


def draw_grid(ax,values: np.ndarray,title: str,cmap: str = 'viridis',signed: bool = False):
    """Draw finite saved grid cells; undefined cells remain visibly masked."""
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise InsufficientResultsError(f'No finite values for {title}')
    colors = __import__('matplotlib').colormaps[cmap].copy(); colors.set_bad('#d9d9d9')
    limit = float(np.max(np.abs(finite))) if signed else None
    image = ax.imshow(np.ma.masked_invalid(values),aspect='auto',interpolation='nearest',cmap=colors,
        vmin=-limit if signed and limit>0 else (-1 if signed else None),
        vmax=limit if signed and limit>0 else (1 if signed else None))
    ax.set(title=title,xlabel='Head',ylabel='Layer')
    ax.set_xticks(range(values.shape[1])); ax.set_yticks(range(values.shape[0]))
    return image


def interval_points(ax,frame: pd.DataFrame,color: str = PALETTE[0]) -> None:
    """Plot saved means and CI endpoints without changing an asymmetric interval."""
    x = np.arange(len(frame))
    ax.vlines(x,frame['ci_low'],frame['ci_high'],color=color,lw=1.4)
    ax.scatter(x,frame['mean'],color=color,s=25,zorder=3)
    ax.axhline(0,color='#777777',lw=0.8,ls='--')
