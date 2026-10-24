"""Specified coreference heatmaps for a validated head and measured low-specialization control."""
import numpy as np
import matplotlib.pyplot as plt
from src.figures.common import FigureContext,InsufficientResultsError
from src.utils.cache import file_hash

STEM = 'fig01_attention_heatmap'


def render_panels(attention: np.ndarray,token_labels: list[str],queries: np.ndarray,targets: np.ndarray,
                  heads: list[tuple[int,int]],titles: list[str]):
    """Render actual saved matrices and mark the supplied pronoun/antecedent cells."""
    figure,axes = plt.subplots(1,2,figsize=(13,6.5))
    cmap = plt.colormaps['viridis'].copy(); cmap.set_bad('#d9d9d9')
    for ax,(layer,head),title in zip(axes,heads,titles):
        matrix = attention[layer,head]
        if matrix.shape!=(len(token_labels),len(token_labels)):
            raise ValueError('Saved example token labels and full attention dimensions disagree')
        image = ax.imshow(np.ma.masked_invalid(matrix),vmin=0,vmax=1,cmap=cmap,interpolation='nearest')
        # Annotate saved attention probabilities; choose black/white from cell luminance.
        font_size = max(5,min(8,180/len(token_labels)))
        for query in range(matrix.shape[0]):
            for key in range(matrix.shape[1]):
                value = matrix[query,key]
                if not np.isfinite(value):
                    continue
                red,green,blue,_ = cmap(float(np.clip(value,0,1)))
                linear = [c/12.92 if c<=0.04045 else ((c+0.055)/1.055)**2.4 for c in (red,green,blue)]
                luminance = 0.2126*linear[0]+0.7152*linear[1]+0.0722*linear[2]
                color = '#111111' if luminance>0.42 else 'white'
                ax.text(key,query,f'{value:.2f}',ha='center',va='center',fontsize=font_size,
                    color=color,zorder=3,clip_on=True)
        labels = [token.replace('▁',' ').replace('Ġ',' ').strip() for token in token_labels]
        ax.set_xticks(range(len(labels)),labels,rotation=65,ha='right',fontsize=8)
        ax.set_yticks(range(len(labels)),labels,fontsize=8)
        ax.set(title=title,xlabel='Key token',ylabel='Query token')
        for i,(query,key) in enumerate(zip(queries,targets)):
            ax.scatter([key],[query],s=110,facecolors='none',edgecolors='#56B4E9',linewidths=1.7,zorder=4)
            ax.annotate(f'{labels[query]} → {labels[key]}',xy=(key,query),xytext=(8,20 if i==0 else -25),
                textcoords='offset points',fontsize=8,color='#0072B2',fontweight='bold',
                arrowprops={'arrowstyle':'->','color':'#56B4E9'},bbox={'facecolor':'white','alpha':0.9,'edgecolor':'none'})
    figure.colorbar(image,ax=axes,shrink=0.75,label='Attention probability')
    return figure


def build(context: FigureContext):
    """Require actual validated candidates; never promote an unvalidated example head."""
    validated = context.table('patching/validated_heads.csv',['layer','head','task','causally_validated','denoising_logit_effect'])
    if not validated['causally_validated'].isin([True,False]).all():
        raise ValueError('Invalid causal validation flags')
    passed = validated[validated['causally_validated']==True].copy()
    if passed.empty:
        raise InsufficientResultsError('Figure 1 requires a causally validated discovery head; none passed. No illustrative validated head will be substituted.')
    passed['coreference_first'] = passed['task']=='coreference'
    selected = passed.sort_values(['coreference_first','denoising_logit_effect','layer','head'],ascending=[False,False,True,True]).iloc[0]
    heads = context.heads()
    excluded = set(zip(passed['layer'],passed['head']))
    threshold = context.config.get('figures',{}).get('control_specialization_max',0.5)
    controls = heads[(heads['specialization']<=threshold)&[pair not in excluded for pair in zip(heads['layer'],heads['head'])]]
    if controls.empty:
        raise InsufficientResultsError('Figure 1 has no measured nonvalidated head below control_specialization_max')
    control = controls.sort_values(['specialization','layer','head']).iloc[0]
    record = next((r for r in context.records('examples.jsonl') if r['id']=='figure1-coreference'),None)
    if record is None or 'token_labels' not in record:
        raise ValueError('Missing saved Figure 1 diagnostic/token labels; rerun stage 01')
    index = context.json('attention/index.json')
    entry = index['records'].get(record['id'])
    if not entry or not entry['has_full_attention']:
        raise InsufficientResultsError('Figure 1 diagnostic full matrix was not saved; increase extraction.example_prompts and rerun stage 02')
    path = context.require(context.results/'attention'/entry['file'])
    if file_hash(path)!=entry['sha256']:
        raise ValueError('Figure 1 attention shard checksum mismatch')
    with np.load(path,allow_pickle=False) as shard:
        full = shard['full_attention'][0 if context.mode=='included' else 1]
        queries,targets = shard['query_positions'],shard['target_positions']
    chosen = [(int(selected['layer']),int(selected['head'])),(int(control['layer']),int(control['head']))]
    titles = [f'L{chosen[0][0]}H{chosen[0][1]} · validated for {selected["task"].replace("_"," ")}',
              f'L{chosen[1][0]}H{chosen[1][1]} · low specialization, unvalidated']
    figure = render_panels(full,record['token_labels'],queries,targets,chosen,titles)
    figure.suptitle('John gave Mary a book. She thanked him.\n'+f'BOS {context.mode}',fontsize=13)
    context.selection.update(validated_head=list(chosen[0]),control_head=list(chosen[1]),example_id=record['id'])
    return figure
