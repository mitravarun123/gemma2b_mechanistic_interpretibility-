"""Per-layer measured entropy distribution."""
from src.figures.distributions import layer_distribution
from src.figures.common import FigureContext
STEM = 'fig03_entropy_distribution'


def build(context: FigureContext):
    """Render saved normalized task-query entropy across physical heads."""
    return layer_distribution(context,'entropy_normalized','Attention entropy across heads','Normalized entropy')
