"""Discovery PCA biplot from saved coordinates and loadings."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.figures.common import FigureContext,finite_table
from src.figures.style import PALETTE
STEM = 'fig05_head_clustering_pca'


def build(context: FigureContext):
    """Color primary clusters and mark layer thirds without refitting PCA."""
    points = context.table('clusters/pca_coordinates.csv',['layer','head','cluster','pc1','pc2'])
    points = finite_table(points,['pc1','pc2'],'No finite saved PCA coordinates')
    loadings = context.table('clusters/pca_loadings.csv',['feature','pc1_loading','pc2_loading'])
    layers,_ = context.architecture()
    figure,ax = plt.subplots(figsize=(9,6.8))
    clusters = sorted(points['cluster'].unique()); markers = ('o','s','^'); group_labels = ('Early layers','Middle layers','Late layers')
    groups = np.minimum(2,((points['layer'].to_numpy()+0.5)*3/layers).astype(int))
    for i,cluster in enumerate(clusters):
        for group,marker in enumerate(markers):
            mask = (points['cluster']==cluster)&(groups==group)
            ax.scatter(points.loc[mask,'pc1'],points.loc[mask,'pc2'],c=PALETTE[i%len(PALETTE)],marker=marker,s=48,edgecolors='white',linewidths=0.5)
    extent = max(float(np.abs(points[['pc1','pc2']].to_numpy()).max()),1)
    maximum = max(float(np.abs(loadings[['pc1_loading','pc2_loading']].to_numpy()).max()),1e-12)
    scale = 0.6*extent/maximum  # Graphic length only; stored loading values are unchanged.
    labels = {'entropy_normalized':'Entropy','mean_distance':'Distance','gini':'Gini','top1_mass':'Top-1 mass','best_task_score':'Best task z'}
    annotations = []
    for row in loadings.itertuples():
        end = row.pc1_loading*scale,row.pc2_loading*scale
        ax.annotate('',xy=end,xytext=(0,0),arrowprops={'arrowstyle':'->','color':'#555555','lw':1.1})
        annotations.append((end,labels.get(row.feature,row.feature)))
    ax.update_datalim(loadings[['pc1_loading','pc2_loading']].to_numpy()*scale)
    ax.autoscale_view()
    ax.margins(x=0.22,y=0.22)
    cluster_legend = ax.legend([Line2D([],[],marker='o',ls='',color=PALETTE[i%len(PALETTE)]) for i in range(len(clusters))],
        [f'Cluster {int(cluster)}' for cluster in clusters],loc='upper left',frameon=False,ncol=2)
    ax.add_artist(cluster_legend)
    ax.legend([Line2D([],[],marker=marker,ls='',color='#555555') for marker in markers],group_labels,loc='lower right',frameon=False)
    ax.set(xlabel='Principal component 1',ylabel='Principal component 2',title='Discovery head clustering · PCA\nFeature loading arrows scaled for display')
    ax.axhline(0,color='#dddddd',lw=0.6); ax.axvline(0,color='#dddddd',lw=0.6)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = []
    for end,label in annotations:
        annotation = ax.annotate(label,xy=end,xytext=(8,8),textcoords='offset points',fontsize=9,color='#444444')
        for offset in (8,-16,28,-36,48,-56,68,-76,88):
            annotation.set_position((8,offset))
            box = annotation.get_window_extent(renderer).expanded(1.08,1.2)
            if not any(box.overlaps(previous) for previous in boxes):
                break
        boxes.append(box)
    return figure
