"""Download models into ./models_local  (needs internet).

Usage:
    python download_models.py            # asks which model(s) to download
    python download_models.py bert       # only BERT
    python download_models.py nuner      # only NuNER Zero
    python download_models.py all        # both
"""
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(os.environ.get("ANONYMIZER_MODELS_DIR",
                           Path(__file__).resolve().parent / "models_local"))

MODELS = {
    "bert": {
        "repo": "dslim/bert-base-NER",
        # Only what local inference needs (skips TensorFlow/Flax copies).
        # A name that does not exist in the repo is skipped silently.
        "files": [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "added_tokens.json",
        ],
        "must_have": ["config.json", "vocab.txt", "tokenizer_config.json", "model.safetensors"],
    },
    "nuner": {
        "repo": "numind/NuNER_Zero",
        "files": None,          # file list not verified, so download the whole repo
        "must_have": [],
    },
}


def folder_for(repo: str) -> Path:
    return ROOT / repo.replace("/", "__")          # must match models/local.py


def download(key: str) -> None:
    info = MODELS[key]
    target = folder_for(info["repo"])
    target.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {info['repo']}\n  -> {target}")
    snapshot_download(repo_id=info["repo"], local_dir=str(target),
                      allow_patterns=info["files"])

    missing = [f for f in info["must_have"] if not (target / f).exists()]
    if missing:
        raise FileNotFoundError(f"{info['repo']}: missing {missing}")
    files = sorted(p.name for p in target.iterdir() if p.is_file())
    print(f"Done. {len(files)} files: {files}")


def ask() -> list:
    print("Which model do you want to download?")
    print("  1) bert   - dslim/bert-base-NER (fixed PER/ORG/LOC)")
    print("  2) nuner  - numind/NuNER_Zero   (zero-shot, custom labels)")
    print("  3) all")
    choice = input("Enter 1, 2 or 3: ").strip().lower()
    return {"1": ["bert"], "bert": ["bert"],
            "2": ["nuner"], "nuner": ["nuner"],
            "3": list(MODELS), "all": list(MODELS)}.get(choice, [])


def main() -> None:
    args = [a.lower() for a in sys.argv[1:]]
    if not args:
        keys = ask()
    elif args == ["all"]:
        keys = list(MODELS)
    else:
        keys = args

    unknown = [k for k in keys if k not in MODELS]
    if unknown or not keys:
        sys.exit(f"Unknown or no choice {unknown or ''}. Options: {', '.join(MODELS)}, all")

    for key in keys:
        download(key)
    print("\nAll requested downloads finished.")


if __name__ == "__main__":
    main()