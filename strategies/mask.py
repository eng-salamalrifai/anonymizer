from core.registry import register_strategy
from strategies.base import Strategy


@register_strategy("mask")
class Mask(Strategy):
    """Fixed-size mask. Not length-preserving on purpose: a mask as long as
    the name would leak how long the name is."""
    def replace(self, entity, mapper):
        return "*****"
