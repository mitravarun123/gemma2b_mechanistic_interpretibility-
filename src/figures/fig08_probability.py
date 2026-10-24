"""Actual correct-name probability loss with saved paired-bootstrap intervals."""
import numpy as np
import matplotlib.pyplot as plt
from src.figures.common import FigureContext,finite_table,head_labels,interval_points
STEM = 'fig08_causal_probability_effect'


def build(context: FigureContext):
    """Plot every finite individual-head probability effect and its existing CI."""
    frame = context.table('ablation/summary.csv',['layer','head','kind','task','metric','mean','ci_low','ci_high'])
    frame = frame[(frame['kind']=='single')&(frame['task']==context.task)&(frame['metric']=='correct_probability')].sort_values(['layer','head']).reset_index(drop=True)
    frame = finite_table(frame,['mean','ci_low','ci_high'],'No saved finite probability effects with bootstrap CIs').reset_index(drop=True)
    figure,ax = plt.subplots(figsize=(max(8,min(20,len(frame)*0.08)),5.8))
    interval_points(ax,frame)
    if len(frame)<=32:
        ax.set_xticks(range(len(frame)),head_labels(frame),rotation=50,ha='right')
    else:
        groups = [(int(group.index[len(group)//2]),int(layer)) for layer,group in frame.groupby('layer',sort=True)]
        ax.set_xticks([x for x,_ in groups],[f'L{layer}' for _,layer in groups])
    ax.set(xlabel='Individual heads in layer/head order',ylabel='Correct-name probability decrease',
        title=f'Causal effect on probability · {context.task.replace("_"," ")}\nSaved 95% paired-bootstrap intervals')
    ax.grid(axis='y',alpha=0.15)
    return figure
