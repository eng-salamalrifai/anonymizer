import codecs
from core.registry import register_handler
from core.types import Document, Segment
from handlers.base import DocumentHandler


@register_handler(".txt")
class TxtHandler(DocumentHandler):
    """The whole file is ONE segment. We never split into lines, so blank
    lines, CRLF vs LF, and a missing final newline all survive untouched
    (str.decode/encode do not translate newlines)."""

    def read(self, data: bytes) -> Document:
        bom = data.startswith(codecs.BOM_UTF8)
        body = data[len(codecs.BOM_UTF8):] if bom else data
        try:
            text, enc = body.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            text, enc = body.decode("latin-1"), "latin-1"      # never fails
        return Document([Segment("0", text)], raw={"encoding": enc, "bom": bom})

    def write(self, doc: Document, new_texts: dict, edits: dict = None) -> bytes:
        out = new_texts["0"].encode(doc.raw["encoding"])
        return codecs.BOM_UTF8 + out if doc.raw["bom"] else out
