"""Prefer a model stored in ./models_local/ over downloading it.

Layout:  models_local/dslim__bert-base-NER/   models_local/numind__NuNER_Zero/
Override the folder with the ANONYMIZER_MODELS_DIR environment variable.
"""
import os

ROOT = os.environ.get(
    "ANONYMIZER_MODELS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models_local"),
)


def resolve(repo_id: str) -> str:
    """Local folder if it exists, otherwise the Hugging Face repo id (cache/download)."""
    path = os.path.join(ROOT, repo_id.replace("/", "__"))
    return path if os.path.isdir(path) else repo_id
