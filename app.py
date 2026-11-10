"""Streamlit UI.   Run with:  streamlit run app.py

This file contains NO processing logic. It reads the registries to fill the
dropdowns, calls core.pipeline.run(), and displays the Result - exactly what
run.py does on the command line.
"""
import os

import streamlit as st

import core.pipeline  # noqa: F401  (importing it fills the registries)
from core.pipeline import run
from core.registry import HANDLERS, MODELS, STRATEGIES

STRATEGY_HELP = {"mask": "*****", "tag": "PER_1, ORG_1 ...", "fake": "realistic fake data (Faker)"}
MIME = {".txt": "text/plain",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}

st.set_page_config(page_title="Document Anonymizer", layout="wide")
st.title("Document Anonymizer")
st.caption("Detects people, organisations and locations and replaces them.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")

    model_name = st.selectbox(
        "NER model", sorted(MODELS),
        format_func=lambda n: f"{n}  ({MODELS[n].name})",
    )
    model_cls = MODELS[model_name]              # class attributes: no model is loaded here

    if model_cls.fixed_labels:                  # pre-trained model: fixed entity types
        labels = set(model_cls.supported_labels)
        st.markdown("**Entities:** " + ", ".join(sorted(labels)) + "  \n*(fixed for this model)*")
    else:                                       # zero-shot model: the user chooses
        chosen = st.multiselect("Entity types", sorted(model_cls.supported_labels),
                                default=[l for l in ("PER", "ORG", "LOC") if l in model_cls.supported_labels])
        custom = st.text_input("Extra entity types (comma-separated)",
                               help="Any name works, e.g. 'job title, medical condition'")
        labels = set(chosen) | {c.strip().upper().replace(" ", "_") for c in custom.split(",") if c.strip()}

    strategy_name = st.selectbox(
        "Anonymization", sorted(STRATEGIES),
        format_func=lambda n: f"{n}  ->  {STRATEGY_HELP.get(n, '')}",
    )
    threshold = st.slider("Confidence threshold", 0.10, 0.99, 0.50, 0.05,
                          help="Higher = fewer detections, fewer false alarms. "
                               "Scores of different models are not directly comparable.")
    with st.expander("Advanced"):
        chunk_window = int(st.number_input(
            "Chunk window (tokens, 0 = automatic)", min_value=0, value=0, step=10,
            help="Long texts are cut into overlapping windows. Set e.g. 30 to see chunking "
                 "on a short file. Smaller than 12 is raised to 12."))
    seed = 42
    if strategy_name == "fake":
        seed = int(st.number_input("Faker seed", value=42, step=1,
                                   help="Same seed + same document = same fake names"))

# ------------------------------------------------------------------ input
extensions = [e.lstrip(".") for e in sorted(HANDLERS)]
uploaded = st.file_uploader("Upload a document", type=extensions)

if uploaded is not None and st.button("Anonymize", type="primary"):
    try:
        with st.spinner("Running NER... (the first run loads the model, which can take a while)"):
            result = run(uploaded.getvalue(), uploaded.name, model_name,
                         labels, strategy_name, threshold, seed,
                         chunk_window or None)
        st.session_state["last"] = {"result": result, "name": uploaded.name}
    except ImportError as err:
        st.error(f"A required package is missing for this model: {err}. "
                 "Install it with pip (see README) and restart the app.")
    except Exception as err:                    # show the problem instead of a stack trace
        st.error(f"Could not process this file: {err}")

# ------------------------------------------------------------------ output
last = st.session_state.get("last")      # kept in session_state: the download button reruns the script
if last:
    result, name = last["result"], last["name"]
    base, ext = os.path.splitext(name)

    for w in result.warnings:
        st.warning(w)

    n_found = len(result.rows)
    counts = {}
    for row in result.rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    cols = st.columns(3 + len(counts))
    cols[0].metric("Entities replaced", n_found)
    cols[1].metric("Model load (s)", f"{result.timings['model_load_s']:.1f}")
    cols[2].metric("Inference (s)", f"{result.timings['inference_s']:.2f}")
    for col, (label, n) in zip(cols[3:], sorted(counts.items())):
        col.metric(label, n)

    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.text_area("original", "\n".join(result.originals.values()), height=350,
                     label_visibility="collapsed")
    with right:
        st.subheader("Anonymized")
        st.text_area("anonymized", "\n".join(result.texts.values()), height=350,
                     label_visibility="collapsed")

    with st.expander("Segments and chunks (how the document was split)", expanded=False):
        st.caption("Segment = the unit sent to the NER model (txt: the whole file; Word: one paragraph / "
                   "table cell / header). Long segments are cut into overlapping chunks; an entity "
                   "touching an inner chunk border is ignored because the overlap lets the next chunk "
                   "see it whole.")
        info = result.segment_info
        st.dataframe(
            [{"segment": i["id"], "characters": i["chars"], "words": i["words"],
              "chunks": len(i["chunks"]), "entities": i["entities"]} for i in info],
            use_container_width=True)
        MAX_SHOWN = 20
        multi = [i for i in info if len(i["chunks"]) > 1]
        if not multi:
            st.info("Every segment fits in a single chunk. Use Advanced > Chunk window to force smaller chunks.")
        for i in multi[:MAX_SHOWN]:
            st.markdown(f"**Segment {i['id']}** - {len(i['chunks'])} chunks")
            text = result.originals[i["id"]]
            st.dataframe(
                [{"chunk": n + 1, "tokens": c["tokens"], "chars": f"{c['start']}-{c['end']}",
                  "starts with": text[c["start"]:c["start"] + 50].replace("\n", " ")}
                 for n, c in enumerate(i["chunks"])],
                use_container_width=True)
        if len(multi) > MAX_SHOWN:
            st.caption(f"... and {len(multi) - MAX_SHOWN} more segments with several chunks (not shown).")

    st.download_button(
        f"Download {base}.anonymized{ext}", data=result.output,
        file_name=f"{base}.anonymized{ext}", mime=MIME.get(ext, "application/octet-stream"),
    )
    if ext == ".docx":
        st.caption("Word output keeps tables, headers, footers and formatting. "
                   "Not supported: text boxes, content controls, footnotes.")

    with st.expander("What was replaced"):
        if result.rows:
            st.dataframe(result.rows, use_container_width=True)
        else:
            st.write("No entities found. Try lowering the confidence threshold.")
    with st.expander("Replacement table (same entity -> same replacement)"):
        st.json(result.mapping)
