import re

from core.registry import register_model
from core.types import Entity
from models.base import NERModel
from models.local import resolve

# Canonical label -> the natural-language wording the zero-shot model is prompted with.
# Wording matters a lot for zero-shot models ("organization" vs "company" can change
# results), so it lives in ONE visible place that can be experimented with.
LABEL_WORDING = {
    "PER": "person",
    "ORG": "organization",
    "LOC": "location",
    "DATE": "date",
    "EMAIL": "email address",
    "PHONE": "phone number",
}

# GLiNER splits text into words with this pattern; chunking by the same units keeps
# our window sizes consistent with what the model counts.
_WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")


def _wording(label: str) -> str:
    """Canonical label -> prompt text. Unknown/custom labels ("JOB_TITLE") become "job title"."""
    return LABEL_WORDING.get(label, label.lower().replace("_", " "))


@register_model("nuner")
class NuNERZero(NERModel):
    name = "numind/NuNER_Zero"
    supported_labels = set(LABEL_WORDING)      # suggestions for the UI; any custom label also works
    fixed_labels = False                       # zero-shot: the USER chooses what to detect
    max_tokens = 384                           # words; lowered at load time if the model says less

    def __init__(self):
        from gliner import GLiNER              # lazy import (heavy, and optional)
        self._model = GLiNER.from_pretrained(resolve(self.name))
        cfg_len = getattr(getattr(self._model, "config", None), "max_len", None)
        if isinstance(cfg_len, int) and cfg_len > 0:
            self.max_tokens = min(self.max_tokens, cfg_len)   # never exceed the model's own limit

    def token_spans(self, text):
        return [m.span() for m in _WORD.finditer(text)]

    def predict(self, text, labels, threshold):
        wording_to_label = {_wording(l): l for l in labels}
        raw = self._model.predict_entities(text, list(wording_to_label), threshold=threshold)
        entities = []
        for r in raw:
            label = wording_to_label.get(r["label"]) or wording_to_label.get(str(r["label"]).lower())
            if label is None:
                continue
            start, end = r["start"], r["end"]
            entities.append(Entity(start, end, label, text[start:end], float(r["score"])))
        return entities
