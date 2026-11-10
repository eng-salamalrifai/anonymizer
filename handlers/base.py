from abc import ABC, abstractmethod
from core.types import Document


class DocumentHandler(ABC):
    """Turns bytes into segments, and (anonymized) segments back into bytes."""

    @abstractmethod
    def read(self, data: bytes) -> Document: ...

    @abstractmethod
    def write(self, doc: Document, new_texts: dict, edits: dict = None) -> bytes:
        """new_texts: segment id -> anonymized text.
        edits: segment id -> [(start, end, replacement), ...] in ORIGINAL offsets.
        Handlers that only need the final text (txt) ignore `edits`; handlers that
        must preserve formatting (docx) use them."""
