"""Saved layer-wise specialization means with across-head bootstrap bands."""
import matplotlib.pyplot as plt
from src.figures.common import FigureContext,finite_table
from src.figures.style import PALETTE
STEM = 'fig11_layer_specialization'


def build(context: FigureContext):
    """Show discovery/validation curves without recomputing specialization or intervals."""
    frame = context.table('features/layer_summaries.csv',['layer','split','bos_mode','mean_specialization','specialization_ci_low','specialization_ci_high','ci_method'])
    frame = frame[frame['bos_mode']==context.mode]
    if not frame['ci_method'].eq('bootstrap_heads').all():
        raise ValueError('Figure 11 requires saved across-head bootstrap intervals')
    figure,ax = plt.subplots(figsize=(8,5.4))
    for i,split in enumerate(('discovery','validation')):
        selected = finite_table(frame[frame['split']==split],['mean_specialization','specialization_ci_low','specialization_ci_high'],f'No specialization curve for {split}').sort_values('layer')
        if selected['layer'].duplicated().any():
            raise ValueError('Layer specialization summaries are not unique')
        ax.plot(selected['layer'],selected['mean_specialization'],marker='o',ms=4,color=PALETTE[i],label=split.capitalize())
        ax.fill_between(selected['layer'],selected['specialization_ci_low'],selected['specialization_ci_high'],color=PALETTE[i],alpha=0.18)
    layers = sorted(frame['layer'].unique()); ax.set_xticks(layers)
    if len(layers)==1:
        ax.set_xlim(layers[0]-0.5,layers[0]+0.5)
    ax.set(xlabel='Layer',ylabel='Mean specialization (maximum task z)',title=f'Layer-wise specialization · BOS {context.mode}\n95% bootstrap across heads; not prompt-sampling uncertainty')
    ax.legend(frameon=False); ax.grid(axis='y',alpha=0.15)
    return figure
