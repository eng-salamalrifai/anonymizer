"""Command-line entry point:  python run.py input.txt --strategy tag"""
import argparse
import os
import sys

from core.pipeline import run
from core.registry import HANDLERS, MODELS, STRATEGIES


def main():
    p = argparse.ArgumentParser(description="Anonymize a document.")
    p.add_argument("input", nargs="?", help="path to a .txt or .docx file")
    p.add_argument("-o", "--output", help="output path (default: <name>.anonymized<ext>)")
    p.add_argument("--model", default="bert", choices=sorted(MODELS))
    p.add_argument("--strategy", default="mask", choices=sorted(STRATEGIES))
    p.add_argument("--labels", nargs="+", help="e.g. PER ORG LOC (only used by zero-shot models)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--chunk-window", type=int, default=None,
                   help="force smaller chunks (tokens) to see chunking on short text")
    p.add_argument("--seed", type=int, default=42, help="seed for Faker (reproducible output)")
    args = p.parse_args()

    if not args.input:
        p.error("input file required")
    with open(args.input, "rb") as f:
        data = f.read()

    result = run(data, args.input, args.model, args.labels, args.strategy, args.threshold, args.seed,
                 args.chunk_window)

    base, ext = os.path.splitext(args.input)
    out_path = args.output or f"{base}.anonymized{ext}"
    with open(out_path, "wb") as f:
        f.write(result.output)

    n = sum(len(v) for v in result.entities.values())
    print(f"{n} entities replaced -> {out_path}", file=sys.stderr)
    n_chunks = sum(len(i["chunks"]) for i in result.segment_info)
    print(f"{len(result.segment_info)} segment(s), {n_chunks} chunk(s)", file=sys.stderr)
    print(f"model load {result.timings['model_load_s']:.2f}s | "
          f"inference {result.timings['inference_s']:.2f}s", file=sys.stderr)
    for original, repl in result.mapping.items():
        print(f"  {original}  ->  {repl}", file=sys.stderr)


if __name__ == "__main__":
    main()
