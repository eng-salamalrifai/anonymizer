import re
from abc import ABC, abstractmethod


class NERModel(ABC):
    """Contract every NER model must follow."""
    name: str                      # human-readable / checkpoint name
    supported_labels: set          # canonical labels the model can output
    fixed_labels: bool = True      # BERT: True. Zero-shot (NuNER): False -> UI shows entity picker
    max_tokens: int = 512          # model input limit; the PIPELINE chunks, not the model

    def token_spans(self, text: str) -> list:
        """Character span (start, end) of every token, used for chunking.
        Default = whitespace words. Models with a tokenizer override this so
        chunk sizes are measured in the model's real tokens."""
        return [m.span() for m in re.finditer(r"\S+", text)]

    @abstractmethod
    def predict(self, text: str, labels, threshold: float) -> list:
        """Return Entity objects (canonical labels, offsets relative to `text`).
        Must only return labels in `labels` and scores >= threshold."""
