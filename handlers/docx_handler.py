"""Word (.docx) handler - formatting-preserving.

Why this is harder than .txt: Word stores a paragraph as several RUNS, each with
its own formatting. A name can be split over runs:

    run1 "my name is "   run2 "John" (bold)   run3 " Smith" (italic)   run4 " and ..."

NER needs the whole paragraph text ("my name is John Smith and ..."), but the
replacement must be written back into the runs without destroying the formatting
of the text around the name.

Supported : body paragraphs, tables (incl. nested + merged cells), headers, footers,
            text inside hyperlinks.
Not supported (documented limitation): text boxes, content controls (w:sdt),
            footnotes/endnotes, comments, text inside images.
"""
import io

from docx import Document as open_docx
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.text.run import Run

from core.registry import register_handler
from core.types import Document, Segment
from handlers.base import DocumentHandler


def _paragraphs_in(container_elm, parent):
    """Yield paragraphs in document order, descending into tables.
    We walk the XML <w:tc> cells directly: python-docx's row.cells repeats a merged
    cell once per grid column, which would anonymize the same paragraph twice."""
    for child in container_elm.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            for tr in child.iterchildren(qn("w:tr")):
                for tc in tr.iterchildren(qn("w:tc")):
                    yield from _paragraphs_in(tc, parent)      # also handles nested tables


def _all_paragraphs(docx) -> list:
    paragraphs = list(_paragraphs_in(docx.element.body, docx))
    seen = []
    for section in docx.sections:
        for part in (section.header, section.first_page_header, section.even_page_header,
                     section.footer, section.first_page_footer, section.even_page_footer):
            if part.is_linked_to_previous:        # inherits another section's header: skip,
                continue                          # otherwise it would be processed twice
            elm = part._element
            if any(elm is s for s in seen):
                continue
            seen.append(elm)
            paragraphs.extend(_paragraphs_in(elm, part))
    return paragraphs


def _runs(paragraph: Paragraph) -> list:
    """Runs of THIS paragraph, including runs inside hyperlinks, but not runs that
    belong to a paragraph nested in a text box inside this one."""
    out = []
    for r in paragraph._p.xpath(".//w:r"):
        owner = r.xpath("ancestor::w:p[1]")
        if owner and owner[0] is paragraph._p:
            out.append(Run(r, paragraph))
    return out


def _paragraph_text(paragraph: Paragraph) -> str:
    return "".join(r.text for r in _runs(paragraph))


def _apply_edits(paragraph: Paragraph, edits: list) -> None:
    """Run-aware replacement. `edits` = [(start, end, replacement)] in offsets of
    the ORIGINAL paragraph text.

    For each entity: the replacement goes into the FIRST run the entity touches
    (so it takes that run's formatting); the entity's characters are removed from
    any other run it touches; runs it does not touch are never modified.
    Edits are applied from the END, so earlier offsets stay valid."""
    runs = _runs(paragraph)
    texts = [r.text for r in runs]
    bounds, pos = [], 0
    for t in texts:
        bounds.append((pos, pos + len(t)))
        pos += len(t)
    changed = set()

    for start, end, replacement in sorted(edits, reverse=True):
        first = True
        for i, (a, b) in enumerate(bounds):
            if b <= start or a >= end:                    # run not touched by this entity
                continue
            local_start, local_end = max(start, a) - a, min(end, b) - a
            texts[i] = texts[i][:local_start] + (replacement if first else "") + texts[i][local_end:]
            first = False
            changed.add(i)

    for i in changed:
        runs[i].text = texts[i]


def _fallback_set(paragraph: Paragraph, new_text: str) -> None:
    """Last resort: whole paragraph text into the first run, other runs emptied."""
    runs = _runs(paragraph)
    if not runs:
        return
    runs[0].text = new_text
    for r in runs[1:]:
        r.text = ""


@register_handler(".docx")
class DocxHandler(DocumentHandler):
    def __init__(self):
        self.warnings = []

    def read(self, data: bytes) -> Document:
        docx = open_docx(io.BytesIO(data))
        paragraphs = _all_paragraphs(docx)
        # segment id = position in the paragraph list (empty paragraphs just have no segment)
        segments = [Segment(str(i), t) for i, p in enumerate(paragraphs)
                    if (t := _paragraph_text(p)).strip()]
        return Document(segments, raw={"docx": docx, "paragraphs": paragraphs})

    def write(self, doc: Document, new_texts: dict, edits: dict = None) -> bytes:
        paragraphs = doc.raw["paragraphs"]
        for seg_id, new_text in new_texts.items():
            p = paragraphs[int(seg_id)]
            if edits is not None and not edits.get(seg_id):
                continue                                   # nothing replaced here: leave untouched
            if edits is None:
                if _paragraph_text(p) != new_text:
                    _fallback_set(p, new_text)
                continue
            _apply_edits(p, edits[seg_id])
            # Safety net: the paragraph must now read exactly as the pipeline expects.
            if _paragraph_text(p) != new_text:
                self.warnings.append(f"paragraph {seg_id}: run-aware edit failed, formatting simplified")
                _fallback_set(p, new_text)
        buf = io.BytesIO()
        doc.raw["docx"].save(buf)
        return buf.getvalue()
