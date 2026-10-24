"""Research pipeline with genuine saved clean/corrupted examples."""
import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from src.figures.common import FigureContext,InsufficientResultsError
from src.figures.style import PALETTE

STEM = 'fig02_pipeline'


def build(context: FigureContext):
    """Show the experimental flow and stored text; make no correctness claims."""
    records = [r for r in context.records('validation.jsonl') if r.get('corrupted_id')]
    twins = {r['id']:r for r in context.records('corrupted/validation.jsonl')}
    if not records:
        raise InsufficientResultsError('Pipeline examples require saved behavioral prompts')
    clean = min(records,key=lambda r:(len(r['token_ids']),r['id']))
    if clean['corrupted_id'] not in twins:
        raise ValueError('Missing pipeline corrupted twin')
    bad = twins[clean['corrupted_id']]
    figure,ax = plt.subplots(figsize=(12,7)); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
    stages = [('Dataset generation','Disjoint names and templates'),('Attention extraction','BOS included / removed'),
        ('Head features','Task-query and all-token summaries'),('Clustering','Discovery only + prompt bootstrap'),
        ('Ablation','Held-out effects + matched controls'),('Activation patching','All heads, two causal directions')]
    positions = [(0.04,0.68),(0.37,0.68),(0.70,0.68),(0.70,0.40),(0.37,0.40),(0.04,0.40)]
    for i,((title,detail),(x,y)) in enumerate(zip(stages,positions)):
        box = FancyBboxPatch((x,y),0.26,0.19,boxstyle='round,pad=0.012',facecolor=PALETTE[i],alpha=0.13,edgecolor=PALETTE[i],lw=1.4)
        ax.add_patch(box); ax.text(x+0.13,y+0.13,title,ha='center',va='center',fontweight='bold',fontsize=10)
        ax.text(x+0.13,y+0.06,textwrap.fill(detail,27),ha='center',va='center',fontsize=9)
    edges = [((0.30,0.775),(0.37,0.775)),((0.63,0.775),(0.70,0.775)),((0.83,0.68),(0.83,0.59)),
             ((0.70,0.495),(0.63,0.495)),((0.37,0.495),(0.30,0.495))]
    for start,end in edges:
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':1.8,'color':'#555555'})
    for y,label,record in [(0.25,'Clean',clean),(0.11,'Corrupted',bad)]:
        ax.text(0.04,y,label,fontsize=10,fontweight='bold',va='top',color=PALETTE[0 if label=='Clean' else 4])
        ax.text(0.16,y,textwrap.fill(record['text'],110),fontsize=9,va='top')
    figure.suptitle('Attention head specialization study',fontsize=15)
    context.selection.update(example_id=clean['id'],corrupted_id=bad['id'])
    return figure
