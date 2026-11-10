# Text Obfuscation (anonymizer)

Detects PER / ORG / LOC (and, with the zero-shot model, any label you choose) in a
.txt or .docx file and replaces them (mask, tag, or fake data).

## Setup
    pip install -r requirements.txt
    # first use of each model downloads it: dslim/bert-base-NER (~430 MB), numind/NuNER_Zero

## Use
    streamlit run app.py                                   # web UI
    python run.py input.txt                                # mask: *****
    python run.py report.docx --strategy tag               # PER_1, ORG_1 ... (formatting kept)
    python run.py input.txt --model nuner --labels PER ORG "job title" --strategy fake
    python run.py input.txt --threshold 0.8 -o out.txt
    python run.py input.txt --chunk-window 30              # force small chunks to see chunking on a short file

## Segments and chunks view
In the app, the result page has a **Segments and chunks** section. Set *Advanced > Chunk window* (e.g. 30) to see a short text split into overlapping chunks. The setting only changes how the text is cut, never the final output.

## Evaluate (BERT vs NuNER, same pipeline, same data)
    python -m evaluation.evaluate --models bert nuner      # writes evaluation/results.md
Gold data: `evaluation/data/*.txt`, annotated inline as `[[text|LABEL]]`.

## Tests (no model download needed)
    python -m pytest

## Architecture
    bytes -> handler.read -> segments -> NER (chunked) -> resolve overlaps
          -> strategy.replace (via Mapper) -> handler.write -> bytes

`core/pipeline.py` is the only file that wires the steps. It only talks to the three registries.
`app.py` and `run.py` both just call `pipeline.run()`.

## Adding things later (one file + one import line)
| To add | Create | Then add to |
|---|---|---|
| model | `models/x.py` with `@register_model("x")`, subclass `NERModel` | `models/__init__.py` |
| file format | `handlers/x.py` with `@register_handler(".x")`, subclass `DocumentHandler` | `handlers/__init__.py` |
| strategy | `strategies/x.py` with `@register_strategy("x")`, subclass `Strategy` | `strategies/__init__.py` |
The UI dropdowns fill themselves from the registries.

## Word support
Supported: paragraphs, tables (nested, merged), headers, footers, hyperlinks; formatting of untouched text is kept.
Not supported: text boxes, content controls, footnotes/endnotes, comments.

## Local models (offline)
    python download_models.py        # once, with internet -> ./models_local/
The app then loads from that folder. Or run once online and set HF_HUB_OFFLINE=1 to use the Hugging Face cache.
