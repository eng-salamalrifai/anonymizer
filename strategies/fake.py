from core.registry import register_strategy
from strategies.base import Strategy

# canonical label -> Faker provider. A company must not be replaced by a person's name.
_PROVIDERS = {"PER": "name", "ORG": "company", "LOC": "city"}


@register_strategy("fake")
class Fake(Strategy):
    def replace(self, entity, mapper):
        provider = _PROVIDERS.get(entity.label)
        if provider is None:                      # label Faker has no provider for
            return mapper.get_or_create(
                entity.label, entity.text,
                lambda: f"{entity.label}_{mapper.next_index(entity.label)}")

        def create():
            used = mapper.used_values()
            value = getattr(mapper.faker, provider)()
            for _ in range(20):                   # avoid two people sharing one fake name
                if value not in used:
                    break
                value = getattr(mapper.faker, provider)()
            return value

        return mapper.get_or_create(entity.label, entity.text, create)
