"""Held-out per-head accuracy loss and layer-matched random-control ranges."""
import numpy as np
import matplotlib.pyplot as plt
from src.figures.common import FigureContext,grid_values,draw_grid,head_labels,finite_table
from src.figures.style import PALETTE
STEM = 'fig07_ablation_accuracy'


def build(context: FigureContext):
    """Plot saved accuracy drops and saved baseline/control statistics."""
    table = context.table('ablation/summary.csv',['layer','head','kind','task','metric','mean'])
    table = table[(table['kind']=='single')&(table['task']==context.task)&(table['metric']=='accuracy')]
    layers,heads = context.architecture(); values = grid_values(table,'mean',layers,heads)
    baseline = context.table('ablation/baseline_summary.csv',['task','metric','mean'])
    baseline = baseline[(baseline['task']==context.task)&(baseline['metric']=='accuracy')]
    if len(baseline)!=1:
        raise ValueError('Missing unique saved baseline accuracy summary; rerun stage 05')
    controls = context.table('ablation/random_control_bands.csv',['target_id','task','metric','p025','p975'])
    controls = controls[(controls['task']==context.task)&(controls['metric']=='accuracy')]
    top = finite_table(table,['mean'],'No finite accuracy drops').sort_values(['mean','layer','head'],ascending=[False,True,True]).head(context.config.get('figures',{}).get('top_heads',8))
    labels = head_labels(top)
    selected = top.copy(); selected['target_id'] = labels
    selected = selected.merge(controls[['target_id','p025','p975']],on='target_id',validate='one_to_one')
    if len(selected)!=len(top):
        raise ValueError('Missing random control band for a plotted ablation head')
    figure,axes = plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1.2,1]})
    image = draw_grid(axes[0],values,'Every head · accuracy decrease',cmap='RdBu_r',signed=True)
    figure.colorbar(image,ax=axes[0],shrink=0.8,label='Baseline − ablated accuracy')
    x = np.arange(len(selected))
    axes[1].fill_between(x,selected['p025'],selected['p975'],color=PALETTE[1],alpha=0.3,label='Matched random controls: 2.5–97.5% range')
    axes[1].bar(x,selected['mean'],color=PALETTE[0],width=0.55,label='Observed head ablation')
    axes[1].axhline(0,color='#777777',lw=0.8,ls='--')
    axes[1].set_xticks(x,labels,rotation=45,ha='right')
    axes[1].set(ylabel='Accuracy decrease',title='Largest observed losses')
    axes[1].legend(frameon=False,fontsize=8,loc='best')
    figure.suptitle(f'Held-out ablation · {context.task.replace("_"," ")}\nBaseline forced-choice accuracy = {baseline.iloc[0]["mean"]:.3f}')
    context.selection['ranked_heads'] = labels
    return figure
