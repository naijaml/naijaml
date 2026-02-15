"""Model download and caching utility for NaijaML.

Downloads pre-trained model files from HuggingFace Hub and caches them
locally at ~/.cache/naijaml/models/ for offline use.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

HF_REPO = "naijaml/naijaml-models"
HF_BASE_URL = "https://huggingface.co/{repo}/resolve/main/{filename}"

_CACHE_DIR_ENV = "NAIJAML_CACHE_DIR"
_DEFAULT_CACHE_DIR = os.path.join("~", ".cache", "naijaml")


def get_models_cache_dir() -> Path:
    """Get the models cache directory, creating it if needed.

    Uses $NAIJAML_CACHE_DIR/models/ if env var is set,
    otherwise ~/.cache/naijaml/models/.

    Returns:
        Path to the models cache directory.
    """
    base = Path(os.environ.get(_CACHE_DIR_ENV, _DEFAULT_CACHE_DIR)).expanduser()
    models_dir = base / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_model_path(filename: str, repo: Optional[str] = None) -> Path:
    """Get path to a model file, downloading from HuggingFace if needed.

    1. Checks ~/.cache/naijaml/models/{filename}
    2. If missing, downloads from HuggingFace Hub
    3. Returns local path

    Args:
        filename: Name of the model file (e.g. "lang_model.json").
        repo: HuggingFace repo ID. Defaults to "naijaml/naijaml-models".

    Returns:
        Path to the local model file.

    Raises:
        RuntimeError: If download fails and no cached version exists.
    """
    cache_dir = get_models_cache_dir()
    local_path = cache_dir / filename

    if local_path.exists():
        return local_path

    # Download from HuggingFace
    if repo is None:
        repo = HF_REPO

    url = HF_BASE_URL.format(repo=repo, filename=filename)
    logger.info("Downloading %s from %s ...", filename, url)

    import requests

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    response = requests.get(url, stream=True, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(
            "Failed to download model '%s' (HTTP %d). "
            "Check your internet connection or install models manually.\n"
            "URL: %s" % (filename, response.status_code, url)
        )

    total_size = int(response.headers.get("content-length", 0))

    # Atomic write: download to temp file, then rename
    tmp_fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    try:
        if has_tqdm and total_size > 0:
            progress = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=filename,
            )
        else:
            progress = None

        with os.fdopen(tmp_fd, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                if progress is not None:
                    progress.update(len(chunk))

        if progress is not None:
            progress.close()

        # Atomic rename
        Path(tmp_path).rename(local_path)
        logger.info("Downloaded %s to %s", filename, local_path)

    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return local_path
