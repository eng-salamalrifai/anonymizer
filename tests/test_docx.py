import io

import tests.fake_model  # noqa: F401
from docx import Document as new_docx
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from core.pipeline import run


def build(mutator) -> bytes:
    d = new_docx()
    mutator(d)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def anonymize(data: bytes, strategy="tag"):
    res = run(data, "t.docx", "fake", strategy_name=strategy)
    return new_docx(io.BytesIO(res.output)), res


def split_name_paragraph(d):
    p = d.add_paragraph()
    p.add_run("my name is ")
    p.add_run("John").bold = True
    p.add_run(" Smith").italic = True
    p.add_run(" and i work at Acme.")


def test_name_split_across_runs_keeps_surrounding_formatting():
    out, _ = anonymize(build(split_name_paragraph))
    runs = out.paragraphs[0].runs
    assert out.paragraphs[0].text == "my name is PER_1 and i work at PER_2."
    assert runs[0].text == "my name is " and not runs[0].bold      # untouched run
    assert runs[1].text == "PER_1" and runs[1].bold                 # replacement takes first run's format
    assert runs[2].text == "" and runs[2].italic                    # leftover of the name removed
    assert runs[3].text == " and i work at PER_2." and not runs[3].bold


def test_entity_spanning_three_runs():
    def mut(d):
        p = d.add_paragraph()
        for t in ("see Anna ", "Maria ", "Lee today"):
            p.add_run(t)
    out, _ = anonymize(build(mut))
    assert out.paragraphs[0].text == "see PER_1 today"


def test_tables_nested_tables_and_merged_cells():
    def mut(d):
        t = d.add_table(rows=2, cols=2)
        t.cell(0, 0).text = "Anna Lee"
        t.cell(0, 0).merge(t.cell(0, 1))                    # merged cell must be processed ONCE
        t.cell(1, 0).text = "plain text"
        inner = t.cell(1, 1).add_table(rows=1, cols=1)      # nested table
        inner.cell(0, 0).text = "Bob Ray"
    out, res = anonymize(build(mut))
    assert out.tables[0].cell(0, 0).text == "PER_1"
    nested = out.tables[0].cell(1, 1).tables[0]
    assert nested.cell(0, 0).text == "PER_2"
    assert out.tables[0].cell(1, 0).text == "plain text"
    assert sum(len(v) for v in res.entities.values()) == 2


def test_header_and_footer_and_linked_sections():
    def mut(d):
        d.add_paragraph("body only")
        s1 = d.sections[0]
        s1.header.paragraphs[0].text = "prepared by Bob Ray"
        s1.footer.paragraphs[0].text = "contact Anna Lee"
        d.add_section()                                     # section 2 inherits (linked) header/footer
    out, res = anonymize(build(mut))
    assert out.sections[0].header.paragraphs[0].text == "prepared by PER_1"
    assert out.sections[0].footer.paragraphs[0].text == "contact PER_2"
    assert len(res.rows) == 2                               # linked header/footer not processed twice


def test_hyperlink_text_is_anonymized():
    def mut(d):
        p = d.add_paragraph("write to ")
        link = parse_xml(f'<w:hyperlink {nsdecls("w")} w:anchor="x"><w:r><w:t>Anna Lee</w:t></w:r></w:hyperlink>')
        p._p.append(link)
    out, _ = anonymize(build(mut))
    assert out.paragraphs[0].text == "write to PER_1"


def test_same_name_gets_same_replacement_across_parts():
    def mut(d):
        d.add_paragraph("by Anna Lee")
        d.sections[0].header.paragraphs[0].text = "ref Anna Lee"
        t = d.add_table(rows=1, cols=1)
        t.cell(0, 0).text = "to ANNA  LEE"
    out, res = anonymize(build(mut))
    assert {r["replaced_with"] for r in res.rows} == {"PER_1"}      # one mapper per document


def test_no_entities_leaves_document_xml_identical():
    def mut(d):
        split_name_paragraph(d)
        d.paragraphs[0].runs[1].text = "john"             # lowercase -> fake model finds nothing
        d.paragraphs[0].runs[2].text = " smith"
        d.paragraphs[0].runs[3].text = " and i work at acme."
    data = build(mut)
    before = new_docx(io.BytesIO(data)).element.xml
    out, res = anonymize(data)
    assert res.rows == [] and out.element.xml == before


def test_no_leftover_original_names_anywhere_and_no_warnings():
    out_doc, res = anonymize(build(split_name_paragraph), strategy="mask")
    assert "John" not in out_doc.paragraphs[0].text and "Smith" not in out_doc.paragraphs[0].text
    assert res.warnings == []
