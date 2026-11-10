"""A tiny stand-in NER model for tests: any run of Capitalised Words = PER.
Registered once under the name "fake"; both test files import it."""
import re
from core.registry import register_model
from core.types import Entity
from models.base import NERModel


@register_model("fake")
class FakeModel(NERModel):
    name = "fake"
    supported_labels = {"PER", "ORG", "LOC"}
    max_tokens = 40            # -> window 32 words, overlap 4: forces chunking in tests

    def predict(self, text, labels, threshold):
        # Worst case on purpose: a single (possibly cut-off) word scores HIGHER
        # than the full multi-word name, like a real model sometimes does.
        out = []
        for m in re.finditer(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*", text):
            score = 0.99 if " " not in m.group() else 0.90
            if "PER" in labels:
                out.append(Entity(m.start(), m.end(), "PER", m.group(), score))
        return out
