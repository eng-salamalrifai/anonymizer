"""Token-based sliding window so long texts fit the model's input limit.

Three things can go wrong and each is handled here:
  1. offsets are relative to the chunk  -> add chunk_start (global = start + local)
  2. an entity is cut by a chunk border -> drop entities touching an inner border;
                                           the overlap guarantees another chunk
                                           contains them whole
  3. overlap zones are seen twice       -> de-duplicate identical entities
"""
from core.types import Entity


MIN_WINDOW = 12   # smallest allowed manual window: keeps the overlap larger than a name


def window_settings(max_tokens: int, window_override: int = None) -> tuple:
    """Stay well under the limit: BERT adds [CLS]/[SEP], and re-tokenising a
    chunk that starts mid-word can produce a few extra word-pieces.

    window_override lets the UI force SMALLER windows (never bigger than the safe
    size) so a short demo text is split into several chunks."""
    window = int(max_tokens * 0.8)       # 512 -> 409 tokens
    if window_override:
        window = max(MIN_WINDOW, min(window_override, window))
    overlap = max(4, window // 8)        # 409 -> 51 tokens (must exceed the longest entity)
    return window, overlap


def make_chunks(spans: list, text_len: int, window: int, overlap: int) -> list:
    """spans = [(char_start, char_end), ...] one per token.
    Returns [(char_start, char_end), ...] one per chunk."""
    if overlap >= window:
        raise ValueError("overlap must be smaller than window")
    if len(spans) <= window:
        return [(0, text_len)]                      # fits in one go
    chunks, i, step = [], 0, window - overlap
    while True:
        j = min(i + window, len(spans))
        chunks.append((spans[i][0], spans[j - 1][1]))
        if j == len(spans):
            return chunks
        i += step


def chunk_info(model, text: str, window_override: int = None) -> list:
    """[{'start', 'end', 'tokens'}, ...] - the chunks of one segment, for the
    pipeline AND for display (the UI shows exactly what the model will receive)."""
    window, overlap = window_settings(model.max_tokens, window_override)
    spans = model.token_spans(text)
    out = []
    for cs, ce in make_chunks(spans, len(text), window, overlap):
        out.append({"start": cs, "end": ce,
                    "tokens": sum(1 for a, b in spans if a >= cs and b <= ce)})
    return out


def predict_chunked(model, text: str, labels, threshold: float,
                    window_override: int = None, chunks: list = None) -> list:
    if not text.strip():
        return []
    if chunks is None:
        chunks = chunk_info(model, text, window_override)
    chunks = [(c["start"], c["end"]) for c in chunks]

    found = {}
    for idx, (cs, ce) in enumerate(chunks):
        sub = text[cs:ce]
        is_first, is_last = idx == 0, idx == len(chunks) - 1
        for e in model.predict(sub, labels, threshold):
            # touches an INNER border -> may be cut in half, skip it
            if (not is_first and e.start == 0) or (not is_last and e.end == len(sub)):
                continue
            g_start, g_end = e.start + cs, e.end + cs          # global offsets
            k = (g_start, g_end, e.label)
            if k not in found or e.score > found[k].score:     # de-duplicate
                found[k] = Entity(g_start, g_end, e.label, text[g_start:g_end], e.score)
    return list(found.values())
