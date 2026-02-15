#!/usr/bin/env python3
"""One-time script to upload model files to HuggingFace Hub.

Usage:
    pip install huggingface_hub
    huggingface-cli login
    python scripts/upload_models_to_hf.py

This creates the repo naijaml/naijaml-models and uploads all model JSON files.
After uploading, the local files in src/naijaml/nlp/ can be removed from
git tracking (they are now downloaded on demand from HF).
"""
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "naijaml/naijaml-models"
MODEL_DIR = Path(__file__).parent.parent / "src" / "naijaml" / "nlp"

MODELS = [
    "word_diacritic_model.json",
    "diacritic_model.json",
    "dot_below_model.json",
    "igbo_diacritic_model.json",
    "lang_model.json",
    "sentiment_model.json",
]


def main():
    api = HfApi()

    # Create repo (no-op if exists)
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
    print(f"Repo: https://huggingface.co/{REPO_ID}")

    for filename in MODELS:
        local_path = MODEL_DIR / filename
        if not local_path.exists():
            print(f"  SKIP {filename} (not found at {local_path})")
            continue

        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  Uploading {filename} ({size_mb:.1f} MB) ...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=REPO_ID,
        )
        print(f"  Done: {filename}")

    print(f"\nAll models uploaded to https://huggingface.co/{REPO_ID}")
    print("You can now remove the local model files from git tracking.")


if __name__ == "__main__":
    main()
