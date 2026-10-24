"""Display saved per-layer head distributions without refitting features."""
import numpy as np
import matplotlib.pyplot as plt
from src.figures.common import FigureContext,finite_table
from src.figures.style import PALETTE


def layer_distribution(context: FigureContext,column: str,title: str,ylabel: str):
    """Plot finite measured heads with violin density where support permits it."""
    heads = finite_table(context.heads(),[column],f'No finite saved {column} values')
    figure,ax = plt.subplots(figsize=(max(7.5,min(15,heads['layer'].nunique()*0.5)),5))
    rng = np.random.default_rng(context.config['seed'])
    layers = sorted(heads['layer'].unique())
    for position,layer in enumerate(layers):
        values = heads[heads['layer']==layer][column].to_numpy()
        if len(values)>=3 and np.ptp(values)>0:
            violin = ax.violinplot([values],positions=[position],widths=0.75,showextrema=False)
            violin['bodies'][0].set(facecolor=PALETTE[0],edgecolor='none',alpha=0.22)
        ax.scatter(position+rng.uniform(-0.16,0.16,len(values)),values,s=18,color=PALETTE[0],alpha=0.7,zorder=3)
    ax.set_xticks(range(len(layers)),[str(int(layer)) for layer in layers])
    ax.set(xlabel='Layer',ylabel=ylabel,title=f'{title}\n{context.split.capitalize()} · BOS {context.mode}')
    ax.grid(axis='y',alpha=0.15)
    return figure
