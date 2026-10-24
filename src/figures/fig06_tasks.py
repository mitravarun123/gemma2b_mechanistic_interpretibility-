"""Task attention score heatmap with data-selected specialized heads."""
import numpy as np
import matplotlib.pyplot as plt
from src.figures.common import FigureContext,head_labels,InsufficientResultsError
from src.figures.style import TASK_LABELS
from src.features.task_score import TASK_COLUMNS
STEM = 'fig06_task_attention_heatmap'


def build(context: FigureContext):
    """Display actual four-task attention scores; top marks denote specialization only."""
    heads = context.heads(); columns = list(TASK_COLUMNS.values())
    values = heads[columns].to_numpy(dtype=float)
    if not np.isfinite(values).any():
        raise InsufficientResultsError('Task heatmap has no finite scores')
    figure,ax = plt.subplots(figsize=(8,max(5,min(28,len(heads)*0.14))))
    cmap = plt.colormaps['viridis'].copy(); cmap.set_bad('#dddddd')
    image = ax.imshow(np.ma.masked_invalid(values),aspect='auto',interpolation='nearest',vmin=0,vmax=1,cmap=cmap)
    ax.set_xticks(range(4),[TASK_LABELS[task] for task in TASK_COLUMNS],rotation=20,ha='right')
    labels = head_labels(heads)
    if len(heads)<=60:
        ax.set_yticks(range(len(heads)),labels,fontsize=8)
    else:
        ticks = [int(group.index[0]) for _,group in heads.groupby('layer',sort=True)]
        ax.set_yticks(ticks,[f'Layer {int(heads.iloc[index]["layer"])}' for index in ticks])
    top_count = context.config.get('figures',{}).get('top_heads',8)
    selected = heads.sort_values(['specialization','layer','head'],ascending=[False,True,True]).head(top_count)
    for index,row in selected.iterrows():
        ax.text(3.62,index,f'{labels[index]} ★',fontsize=8,va='center',clip_on=False)
    ax.set(xlabel='Task',ylabel='Heads in layer/head order',title=f'Task attention scores · {context.split}\nBOS {context.mode}; ★ highest specialization, not causal validation')
    ax.set_xlim(-0.5,4.65)
    figure.colorbar(image,ax=ax,shrink=0.6,pad=0.13,label='Mean target attention probability')
    context.selection['annotated_heads'] = head_labels(selected)
    return figure
