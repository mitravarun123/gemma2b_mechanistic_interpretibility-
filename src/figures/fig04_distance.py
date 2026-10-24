"""Per-layer measured mean attention distance distribution."""
from src.figures.distributions import layer_distribution
from src.figures.common import FigureContext
STEM = 'fig04_attention_distance'


def build(context: FigureContext):
    """Render saved task-query mean distance in original token positions."""
    return layer_distribution(context,'mean_distance','Attention distance across heads','Mean distance (tokens)')
