# challenges.md  (raw material for the report's most important section)

Status: **verified** = a test reproduced it. **designed** = handled in the code, not yet seen on real BERT output.

| # | Problem | Cause | Fix | Status |
|---|---|---|---|---|
| 1 | Replacements land in the wrong place | Each replacement changes the text length, so later offsets shift | Decide replacements in reading order, then splice from END to START (`apply_replacements`) | verified |
| 2 | Part of a name leaks at a chunk border (e.g. "Smith" survives) | A chunk cut through "John Smith" yields the partial entity "John"; the model may score the fragment higher than the full name, so "highest score wins" keeps the fragment | Drop entities that touch an inner chunk border; the 51-token overlap guarantees the neighbouring chunk sees the full name | verified (test failed at n=27 with the fix disabled) |
| 3 | Offsets from a chunk point to the wrong place in the document | Model offsets are relative to the chunk | `global = chunk_start + local` for start and end; `Entity.text` is re-sliced from the original text | verified |
| 4 | Same entity reported twice | Overlap zones are processed by two chunks | De-duplicate on `(start, end, label)`, keep the higher score | verified |
| 5 | Blank lines / CRLF / missing final newline change after anonymizing | Splitting a file into lines and re-joining is lossy | .txt is ONE segment; byte-exact round-trip test (also BOM and latin-1) | verified |
| 6 | "New York" and "York" both returned | Overlapping predictions | Higher score wins, then the longer span; touching entities are both kept | verified |
| 7 | Same person gets different replacements ("John Smith" / "JOHN  SMITH") | Case and whitespace differences | Mapper key = `(label, lower-cased, whitespace-collapsed text)` | verified |
| 8 | PER_1 numbers look random | Replacements were decided in model-output order | Decide in reading order before splicing | verified |
| 9 | A company could get a person's name from Faker | One Faker call for all labels | Provider per label: PER=name, ORG=company, LOC=city | designed |
| 10 | My first boundary test passed even with the bug present | The stand-in model gave equal scores, so the tie rule ("longer wins") masked the bug | Made the stand-in score a fragment higher than the full name, then re-ran with the fix disabled | verified (testing lesson) |
| 11 | A name split over Word runs ("John" bold + " Smith" italic) cannot be replaced run by run | Word stores formatting in runs; NER needs the whole paragraph text | Build paragraph text from runs, record each run's character range, map entity offsets back to runs; replacement goes in the first touched run, the rest of the name is removed from the others | verified |
| 12 | Putting the whole new paragraph in the first run destroys formatting of the text around the name | Naive approach rewrites every run | Run-aware edit; only runs an entity touches are modified. Test fails if the naive approach is used | verified (mutation test) |
| 13 | Merged table cells could be anonymized twice | python-docx `row.cells` repeats a merged cell once per grid column | Walk the XML `<w:tc>` elements directly (also gives nested tables for free) | designed + tested (not mutation-checked) |
| 14 | Reading a *linked* header/footer silently changed the file | python-docx creates an empty header definition and unlinks it when you touch a linked one | Skip `is_linked_to_previous` headers; they inherit text that is processed once in the section that owns it | verified (document XML changed when the skip was removed) |
| 15 | Names inside hyperlinks were invisible | `paragraph.runs` does not include runs inside `<w:hyperlink>` | Collect runs with XPath `.//w:r`, keep only runs whose nearest paragraph is this one (excludes text boxes) | verified |
| 16 | docx writer could not be run-aware | `write()` only received the final text, not which characters changed | `write()` now also receives the exact edits `(start, end, replacement)` per segment; txt ignores them | design change |
| 17 | After editing, can we be sure the paragraph reads correctly? | Run text round-trips (tabs, breaks) could differ | Safety net: re-read the paragraph; if it differs from the expected text, fall back to first-run replacement and report a warning | designed |
| 18 | Zero-shot model needs natural-language labels | NuNER is prompted with words, not fixed classes | One visible `LABEL_WORDING` map (PER -> "person", ...); custom labels become lower-case words; results mapped back to canonical labels | designed - **measure wording sensitivity on the real model** |
| 19 | BERT counts sub-word tokens, NuNER counts words | Different tokenisers | Each model supplies its own `token_spans()`; chunk window is 80% of that model's limit | designed |

## Known limitations (report: Limitations section)
- "Smith" is not linked to "John Smith" (two different Smiths could be wrongly merged).
- An entity longer than the chunk overlap (about 51 tokens for BERT) is not protected at chunk borders; never expected for names.
- Word: text boxes, content controls, footnotes/endnotes and comments are not processed. A replacement spanning runs takes the formatting of the first run it touches.
- Evaluation gold set is small (3 documents, ~200 words, 38 entities) and annotated by one person (me). Results are indicative, not statistically strong; extend it with your own documents.
- Zero-shot scores and BERT scores are not calibrated alike, so one shared threshold is not a perfectly fair comparison.
- Not yet measured on real BERT output: subword offsets, accuracy per label, speed.
