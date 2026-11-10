"""Consistent entity -> replacement mapping (one Mapper per document).

"John Smith", "john smith" and "JOHN  SMITH" must all become the SAME
replacement, otherwise the anonymized text contradicts itself.
"""
import re
from collections import defaultdict


class Mapper:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._map: dict = {}                       # (label, norm text) -> replacement
        self._counters = defaultdict(int)          # label -> last tag number
        self._faker = None                         # created lazily (see below)

    @staticmethod
    def key(label: str, text: str) -> tuple:
        # lower-case + collapse whitespace so trivial variants collide.
        # NOTE: we deliberately do NOT merge "Smith" into "John Smith"
        # (two different Smiths would be wrongly joined). Reported as a limitation.
        return (label, re.sub(r"\s+", " ", text).strip().lower())

    def get_or_create(self, label: str, text: str, create):
        """Return the stored replacement, or call create() once and store it."""
        k = self.key(label, text)
        if k not in self._map:
            self._map[k] = create()
        return self._map[k]

    def next_index(self, label: str) -> int:
        self._counters[label] += 1
        return self._counters[label]

    def used_values(self) -> set:
        return set(self._map.values())

    @property
    def faker(self):
        """Seeded Faker, imported only when a fake-data strategy needs it,
        so mask/tag work even if Faker is not installed."""
        if self._faker is None:
            from faker import Faker
            self._faker = Faker()
            self._faker.seed_instance(self.seed)   # reproducible output
        return self._faker

    def table(self) -> dict:
        """{'PER:john smith': 'PER_1', ...} - handy for the UI and the report."""
        return {f"{l}:{t}": r for (l, t), r in self._map.items()}
