"""Functional head atlas labeling only discovery-selected, causally validated heads."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
from src.figures.common import FigureContext
from src.figures.style import PALETTE,TASK_LABELS
STEM = 'fig10_functional_head_atlas'


def build(context: FigureContext):
    """Keep unsupported/unvalidated cells neutral; never infer validation from attention."""
    layers,heads = context.architecture()
    shortlist = context.table('clusters/shortlist.csv',['layer','head','task'])
    validation = context.table('patching/validated_heads.csv',['layer','head','task','causally_validated'])
    if not validation['causally_validated'].isin([True,False]).all():
        raise ValueError('Invalid atlas validation flags')
    candidates = {(int(r.layer),int(r.head)):r.task for r in shortlist.itertuples()}
    functions = ['unclear','unvalidated','local_context','name_tracking','coreference','induction']
    colors = ['#eeeeee','#aaaaaa',PALETTE[0],PALETTE[1],PALETTE[2],PALETTE[3]]
    values = np.zeros((layers,heads),dtype=int); labels = []
    for (layer,head),task in candidates.items():
        if not 0<=layer<layers or not 0<=head<heads:
            raise ValueError('Atlas candidate is outside physical model dimensions')
        values[layer,head] = 1
    for row in validation.itertuples():
        if row.causally_validated:
            key = int(row.layer),int(row.head)
            if key not in candidates or candidates[key]!=row.task or row.task not in functions[2:]:
                raise ValueError('Causal atlas label lacks a matching discovery candidate')
            values[key] = functions.index(row.task); labels.append(key)
    figure,ax = plt.subplots(figsize=(max(8,heads*0.8),max(5,layers*0.35)))
    ax.imshow(values,cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-0.5,len(functions)+0.5),len(functions)),aspect='auto',interpolation='nearest')
    ax.set_xticks(range(heads)); ax.set_yticks(range(layers)); ax.set(xlabel='Head',ylabel='Layer',title='Functional head atlas\nLabels require discovery selection and task-matched causal validation')
    for layer,head in labels:
        ax.text(head,layer,f'L{layer}H{head}',ha='center',va='center',fontsize=7,color='#111111',fontweight='bold')
    ax.legend([Patch(facecolor=color) for color in colors],[TASK_LABELS.get(function,function.capitalize()) for function in functions],
        loc='upper center',bbox_to_anchor=(0.5,-0.12),ncol=3,frameon=False)
    context.selection['validated_heads'] = [list(key) for key in labels]
    return figure
