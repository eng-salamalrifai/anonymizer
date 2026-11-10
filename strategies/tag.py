from core.registry import register_strategy
from strategies.base import Strategy


@register_strategy("tag")
class Tag(Strategy):
    """PER_1, PER_2, ORG_1 ... The same entity always gets the same tag."""
    def replace(self, entity, mapper):
        return mapper.get_or_create(
            entity.label, entity.text,
            lambda: f"{entity.label}_{mapper.next_index(entity.label)}",
        )
