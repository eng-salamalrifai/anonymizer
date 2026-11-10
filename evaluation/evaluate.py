"""Evaluate NER models through the SAME code path the application uses
(predict_chunked -> resolve_overlaps), so the comparison between models is fair.

    python -m evaluation.evaluate --models bert nuner

Metric: ENTITY-LEVEL EXACT MATCH. A prediction is correct only if start, end AND
label all equal the gold annotation. "John" for gold "John Smith" is both a false
positive and a false negative. Model load time is reported separately from
inference time, and a warm-up call is made before timing starts.
"""
import argparse
import glob
import os
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.chunking import predict_chunked                 # noqa: E402
from core.pipeline import get_model, resolve_overlaps     # noqa: E402

LABELS = {"PER", "ORG", "LOC"}
_MARKUP = re.compile(r"\[\[(.+?)\|([A-Z_]+)\]\]")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def parse_gold(markup: str):
    """'[[John|PER]] met Anna' -> ('John met Anna', [(0, 4, 'PER')])"""
    text, spans, last = [], [], 0
    pos = 0
    for m in _MARKUP.finditer(markup):
        before = markup[last:m.start()]
        text.append(before)
        pos += len(before)
        spans.append((pos, pos + len(m.group(1)), m.group(2)))
        text.append(m.group(1))
        pos += len(m.group(1))
        last = m.end()
    text.append(markup[last:])
    return "".join(text), spans


def load_docs(data_dir=DATA_DIR) -> list:
    docs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(path, encoding="utf-8", newline="") as f:
            text, gold = parse_gold(f.read())
        docs.append({"name": os.path.basename(path), "text": text, "gold": gold})
    return docs


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn, "support": tp + fn}


def score(gold: list, pred: list) -> dict:
    """gold/pred = [(start, end, label)] -> per-label tp/fp/fn counts + error lists."""
    g, p = set(gold), set(pred)
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for s in g & p:
        counts[s[2]]["tp"] += 1
    for s in p - g:
        counts[s[2]]["fp"] += 1
    for s in g - p:
        counts[s[2]]["fn"] += 1
    return {"counts": dict(counts), "fp": sorted(p - g), "fn": sorted(g - p)}


def evaluate_model(model, docs: list, labels=LABELS, threshold=0.5) -> dict:
    total = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    errors, words, elapsed = [], 0, 0.0
    predict_chunked(model, "Warm up with John Smith in Paris.", labels, threshold)   # warm-up, not timed
    for d in docs:
        t0 = time.perf_counter()
        ents = resolve_overlaps(predict_chunked(model, d["text"], labels, threshold))
        elapsed += time.perf_counter() - t0
        words += len(d["text"].split())
        pred = [(e.start, e.end, e.label) for e in ents]
        s = score(d["gold"], pred)
        for label, c in s["counts"].items():
            for k in c:
                total[label][k] += c[k]
        for kind in ("fp", "fn"):
            for st, en, lb in s[kind]:
                errors.append((kind.upper(), d["name"], d["text"][st:en], lb))
    per_label = {l: prf(**total[l]) for l in sorted(labels)}
    micro = prf(sum(c["tp"] for c in total.values()),
                sum(c["fp"] for c in total.values()),
                sum(c["fn"] for c in total.values()))
    return {"per_label": per_label, "micro": micro, "errors": errors,
            "words": words, "seconds": elapsed}


def benchmark(model, docs: list, labels=LABELS, threshold=0.5, min_words=3000) -> dict:
    """Time ONE long text (gold documents repeated) -> seconds per 1,000 words.
    A long text also exercises chunking on the real model."""
    parts, n = [], 0
    while n < min_words:
        for d in docs:
            parts.append(d["text"])
            n += len(d["text"].split())
    text = "\n\n".join(parts)
    predict_chunked(model, "Warm up with John Smith in Paris.", labels, threshold)
    t0 = time.perf_counter()
    predict_chunked(model, text, labels, threshold)
    sec = time.perf_counter() - t0
    words = len(text.split())
    return {"words": words, "seconds": sec, "sec_per_1000_words": sec / words * 1000}


def to_markdown(results: dict) -> str:
    out = ["## Entity-level exact match (start, end and label must all match)\n",
           "| Model | Label | Precision | Recall | F1 | Support |", "|---|---|---|---|---|---|"]
    for name, r in results.items():
        for label, m in r["eval"]["per_label"].items():
            out.append(f"| {name} | {label} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['support']} |")
        m = r["eval"]["micro"]
        out.append(f"| **{name}** | **micro** | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['support']} |")
    out += ["\n## Speed (model load reported separately)\n",
            "| Model | Load time (s) | Benchmark words | Inference (s) | Seconds per 1,000 words |", "|---|---|---|---|---|"]
    for name, r in results.items():
        b = r["bench"]
        out.append(f"| {name} | {r['load_s']:.1f} | {b['words']} | {b['seconds']:.2f} | {b['sec_per_1000_words']:.2f} |")
    out.append("\n## Errors (for the report's error analysis)\n")
    for name, r in results.items():
        out.append(f"**{name}**\n")
        out += [f"- {k} `{t}` ({lb}) in {doc}" for k, doc, t, lb in r["eval"]["errors"]] or ["- none"]
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["bert"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(DATA_DIR), "results.md"))
    args = ap.parse_args()

    docs = load_docs()
    print(f"{len(docs)} documents, {sum(len(d['gold']) for d in docs)} gold entities", file=sys.stderr)
    results = {}
    for name in args.models:
        try:
            model = get_model(name)
        except ImportError as err:
            print(f"skipping '{name}': {err}", file=sys.stderr)
            continue
        print(f"evaluating {name} ...", file=sys.stderr)
        results[name] = {"eval": evaluate_model(model, docs, LABELS, args.threshold),
                         "bench": benchmark(model, docs, LABELS, args.threshold),
                         "load_s": getattr(model, "load_seconds", 0.0)}
    md = to_markdown(results)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
