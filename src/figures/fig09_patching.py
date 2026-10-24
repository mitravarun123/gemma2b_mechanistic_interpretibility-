"""Causal patch schematic, all-head recovery grid and top-head uncertainty."""
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from src.figures.common import FigureContext,grid_values,draw_grid,finite_table,head_labels,interval_points
from src.figures.style import PALETTE
STEM = 'fig09_activation_patching'


def example_measurement(context: FigureContext,layer: int,head: int) -> dict:
    """Select a genuine saved patch outcome in chunks; prefer observed wrong-to-correct cases."""
    path = context.require(context.results/'patching/per_prompt.csv')
    selected = []
    required = ['id','task','layer','head','direction','raw_logit_difference','patched_accuracy','patched_correct_probability']
    for chunk in pd.read_csv(path,chunksize=50000,usecols=required):
        mask = (chunk['layer']==layer)&(chunk['head']==head)&(chunk['direction']=='denoising')
        if context.task!='all':
            mask &= chunk['task']==context.task
        selected.append(chunk[mask])
    frame = pd.concat(selected,ignore_index=True)
    frame = finite_table(frame,['raw_logit_difference','patched_correct_probability'],'No observed denoising example for the selected head')
    baseline = context.table('patching/baseline.csv',['id','corrupted_accuracy','corrupted_correct_probability'])
    frame = frame.merge(baseline,on=['id','task'],validate='many_to_one')
    frame['recovered_correct'] = (frame['corrupted_accuracy']==0)&(frame['patched_accuracy']==1)
    return frame.sort_values(['recovered_correct','raw_logit_difference','id'],ascending=[False,False,True]).iloc[0].to_dict()


def build(context: FigureContext):
    """Render saved recovery and CI data, using observed outcomes rather than assumed success."""
    metric = context.config.get('figures',{}).get('recovery_metric','recovery_logit_positive_gap')
    if metric not in ('recovery_logit_positive_gap','recovery_logit'):
        raise ValueError('Figure 9 recovery_metric must be a saved logit recovery metric')
    frame = context.table('patching/summary.csv',['layer','head','direction','task','metric','mean','ci_low','ci_high'])
    frame = frame[(frame['direction']=='denoising')&(frame['task']==context.task)&(frame['metric']==metric)]
    layers,heads = context.architecture(); values = grid_values(frame,'mean',layers,heads)
    top = finite_table(frame,['mean','ci_low','ci_high'],'Figure 9 has no recovery scores with finite saved CIs').sort_values(['mean','layer','head'],ascending=[False,True,True]).head(context.config.get('figures',{}).get('top_heads',8)).reset_index(drop=True)
    layer,head = int(top.iloc[0]['layer']),int(top.iloc[0]['head'])
    observed = example_measurement(context,layer,head)
    records = {r['id']:r for r in context.records('validation.jsonl')}; twins = {r['id']:r for r in context.records('corrupted/validation.jsonl')}
    clean = records[observed['id']]; bad = twins[clean['corrupted_id']]
    figure = plt.figure(figsize=(13,8.4)); layout = figure.add_gridspec(2,2,height_ratios=[1.35,3])
    schematic = figure.add_subplot(layout[0,:]); left = figure.add_subplot(layout[1,0]); right = figure.add_subplot(layout[1,1])
    schematic.axis('off'); schematic.set_xlim(0,1); schematic.set_ylim(0,1)
    def box(x,width,title,detail,color):
        schematic.add_patch(FancyBboxPatch((x,0.16),width,0.60,boxstyle='round,pad=0.015',facecolor=color,alpha=0.12,edgecolor=color,lw=1.3))
        schematic.text(x+width/2,0.65,title,ha='center',va='center',fontweight='bold',fontsize=10)
        schematic.text(x+width/2,0.39,textwrap.fill(detail,30),ha='center',va='center',fontsize=8.5)
    short = textwrap.shorten(bad['text'],width=95,placeholder='…')
    before = 'Wrong forced choice' if observed['corrupted_accuracy']==0 else ('Correct forced choice' if observed['corrupted_accuracy']==1 else 'Tied forced choice')
    after = 'Correct forced choice' if observed['patched_accuracy']==1 else ('Wrong forced choice' if observed['patched_accuracy']==0 else 'Tied forced choice')
    box(0.02,0.28,'A  Corrupted input',short,PALETTE[4])
    box(0.36,0.24,'Measured baseline',f'{before}\np(correct name) = {observed["corrupted_correct_probability"]:.4f}',PALETTE[1])
    box(0.68,0.29,f'Clean head patch · L{layer}H{head}',f'{after}\np(correct name) = {observed["patched_correct_probability"]:.4f}',PALETTE[2])
    for start,end in [((0.30,0.46),(0.36,0.46)),((0.60,0.46),(0.68,0.46))]:
        schematic.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':1.5,'color':'#555555'})
    schematic.text(0.02,0.94,f'Observed held-out example: {clean["id"]}',fontsize=9,va='top')
    label = 'Positive clean–corrupted gaps' if metric.endswith('positive_gap') else 'Signed nonzero clean–corrupted gaps'
    image = draw_grid(left,values,'B  Denoising recovery · every head',cmap='RdBu_r',signed=True)
    figure.colorbar(image,ax=left,shrink=0.8,label='Normalized logit recovery (unclipped)')
    right.bar(np.arange(len(top)),top['mean'],color=PALETTE[2],alpha=0.65)
    interval_points(right,top,color='#333333'); right.set_xticks(range(len(top)),head_labels(top),rotation=45,ha='right')
    right.set(ylabel='Normalized logit recovery',title='C  Highest recovery · 95% bootstrap CIs')
    figure.suptitle(f'Activation patching · {context.task.replace("_"," ")}\n{label}',fontsize=14)
    context.selection.update(schematic_head=[layer,head],example_id=clean['id'],observed_wrong_to_correct=bool(observed['recovered_correct']),ranked_heads=head_labels(top),recovery_metric=metric)
    return figure
