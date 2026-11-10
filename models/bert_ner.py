from core.registry import register_model
from core.types import Entity
from models.base import NERModel
from models.local import resolve

# checkpoint label -> canonical label. Anything not listed (MISC) is ignored.
_LABEL_MAP = {"PER": "PER", "ORG": "ORG", "LOC": "LOC"}


@register_model("bert")
class BertNER(NERModel):
    name = "dslim/bert-base-NER"
    supported_labels = {"PER", "ORG", "LOC"}
    fixed_labels = True
    max_tokens = 512

    def __init__(self):
        # Heavy imports live here, NOT at the top of the file, so the registry
        # (and the tests) work even where transformers is not installed.
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(resolve(self.name))
        model = AutoModelForTokenClassification.from_pretrained(resolve(self.name))
        self._pipe = pipeline(
            "token-classification", model=model, tokenizer=self.tokenizer,
            aggregation_strategy="simple",   # merges B-PER/I-PER pieces, returns char offsets
        )

    def token_spans(self, text):
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return [tuple(span) for span in enc["offset_mapping"]]

    def predict(self, text, labels, threshold):
        entities = []
        for r in self._pipe(text):
            label = _LABEL_MAP.get(r["entity_group"])
            if label is None or label not in labels or r["score"] < threshold:
                continue
            start, end = r["start"], r["end"]
            while start < end and text[start].isspace():      # trim stray spaces
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start == end:
                continue
            # use text[start:end], not r["word"]: "word" can contain '##' pieces
            entities.append(Entity(start, end, label, text[start:end], float(r["score"])))
        return entities
