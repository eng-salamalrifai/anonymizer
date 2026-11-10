"""The ONLY place where the steps are wired together.

    bytes -> handler.read -> segments -> NER (chunked) -> resolve overlaps
          -> strategy replace -> handler.write -> bytes

It never imports a concrete model/handler/strategy; it asks the registries.
It also never imports Streamlit, so the CLI, the UI and the tests all call
exactly the same code.
"""
import os
import time
from dataclasses import dataclass, field

# Importing these packages runs their @register_* decorators.
import handlers, models, strategies  # noqa: F401
from core.chunking import chunk_info, predict_chunked
from core.mapper import Mapper
from core.registry import HANDLERS, MODELS, STRATEGIES

_MODEL_CACHE: dict = {}


def get_model(name: str):
    """Load a model once per process (loading BERT takes seconds)."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")
    if name not in _MODEL_CACHE:
        t0 = time.perf_counter()
        model = MODELS[name]()
        model.load_seconds = time.perf_counter() - t0
        _MODEL_CACHE[name] = model
    return _MODEL_CACHE[name]


def resolve_overlaps(entities: list) -> list:
    """Keep non-overlapping entities. Winner rule: higher score, then longer span.
    (So "New York" beats "York".) Adjacent-but-touching entities are both kept."""
    ranked = sorted(entities, key=lambda e: (-e.score, -(e.end - e.start), e.start))
    kept = []
    for e in ranked:
        if all(e.end <= k.start or e.start >= k.end for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e.start)


def plan_replacements(entities: list, strategy, mapper) -> list:
    """[(entity, replacement)] in READING ORDER. Doing this forward means PER_1 is
    the first name read, and the mapper sees entities in document order."""
    return [(e, strategy.replace(e, mapper)) for e in sorted(entities, key=lambda e: e.start)]


def apply_replacements(text: str, entities: list, strategy, mapper) -> str:
    """Decide forward (plan_replacements), then splice BACKWARD (end -> start) so
    replacing a later entity can never shift the offsets of an earlier one."""
    for e, replacement in reversed(plan_replacements(entities, strategy, mapper)):
        text = text[:e.start] + replacement + text[e.end:]
    return text


def anonymize_text(text, model, labels, strategy, mapper, threshold=0.5, chunks=None):
    """One segment in -> (new_text, entities, plan) out."""
    entities = resolve_overlaps(predict_chunked(model, text, labels, threshold, chunks=chunks))
    plan = plan_replacements(entities, strategy, mapper)
    new_text = text
    for e, replacement in reversed(plan):
        new_text = new_text[:e.start] + replacement + new_text[e.end:]
    return new_text, entities, plan


@dataclass
class Result:
    output: bytes                                   # file content to save / download
    originals: dict = field(default_factory=dict)   # segment id -> original text
    texts: dict = field(default_factory=dict)       # segment id -> anonymized text
    entities: dict = field(default_factory=dict)    # segment id -> [Entity]
    edits: dict = field(default_factory=dict)       # segment id -> [(start, end, replacement)]
    rows: list = field(default_factory=list)        # flat table for the UI / report
    segment_info: list = field(default_factory=list)  # per segment: size + its chunks (for display)
    mapping: dict = field(default_factory=dict)     # original -> replacement table
    timings: dict = field(default_factory=dict)     # model_load_s / inference_s
    warnings: list = field(default_factory=list)    # handler warnings (e.g. docx fallback)


def run(data: bytes, filename: str, model_name="bert", labels=None,
        strategy_name="mask", threshold=0.5, seed=42, chunk_window=None) -> Result:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in HANDLERS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(HANDLERS)}")
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {sorted(STRATEGIES)}")

    model = get_model(model_name)
    labels = set(labels) if labels else set(model.supported_labels)
    if model.fixed_labels:                       # pre-trained model: labels are not user-selectable
        labels &= model.supported_labels
    if not labels:
        raise ValueError(f"No valid labels. {model_name} supports {sorted(model.supported_labels)}")

    handler, strategy, mapper = HANDLERS[ext](), STRATEGIES[strategy_name](), Mapper(seed)

    doc = handler.read(data)
    result = Result(output=b"")
    t0 = time.perf_counter()
    for seg in doc.segments:
        chunks = chunk_info(model, seg.text, chunk_window)       # computed once: used for NER AND for display
        new_text, ents, plan = anonymize_text(seg.text, model, labels, strategy, mapper, threshold, chunks)
        result.segment_info.append({"id": seg.id, "chars": len(seg.text), "words": len(seg.text.split()),
                                    "entities": len(ents), "chunks": chunks})
        result.originals[seg.id], result.texts[seg.id] = seg.text, new_text
        result.entities[seg.id] = ents
        result.edits[seg.id] = [(e.start, e.end, r) for e, r in plan]
        result.rows += [{"segment": seg.id, "text": e.text, "label": e.label,
                         "score": round(e.score, 3), "replaced_with": r} for e, r in plan]
    result.timings = {"model_load_s": getattr(model, "load_seconds", 0.0),
                      "inference_s": time.perf_counter() - t0}
    result.mapping = mapper.table()
    result.output = handler.write(doc, result.texts, result.edits)
    result.warnings = list(getattr(handler, "warnings", []))
    return result
