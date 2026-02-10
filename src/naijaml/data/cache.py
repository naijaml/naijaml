"""Offline-first caching system for NaijaML datasets.

Caches downloaded datasets to ~/.cache/naijaml/ so they work without
internet after first download.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CACHE_DIR_ENV = "NAIJAML_CACHE_DIR"
_DEFAULT_CACHE_DIR = os.path.join("~", ".cache", "naijaml")


def get_cache_dir() -> Path:
    """Get the cache directory path, creating it if needed.

    Uses $NAIJAML_CACHE_DIR if set, otherwise ~/.cache/naijaml/.

    Returns:
        Path to the cache directory.
    """
    cache_dir = Path(os.environ.get(_CACHE_DIR_ENV, _DEFAULT_CACHE_DIR)).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _dataset_cache_path(dataset_name: str, lang: Optional[str], split: str) -> Path:
    """Build the cache file path for a specific dataset/lang/split combo."""
    parts = [dataset_name]
    if lang:
        parts.append(lang)
    parts.append(split)
    filename = "_".join(parts) + ".json"
    return get_cache_dir() / dataset_name / filename


def is_cached(dataset_name: str, lang: Optional[str] = None, split: str = "train") -> bool:
    """Check if a dataset is already cached locally.

    Args:
        dataset_name: Name of the dataset (e.g. 'naijasenti').
        lang: Language code filter (e.g. 'yor').
        split: Dataset split ('train', 'test', 'validation').

    Returns:
        True if the cached file exists.
    """
    return _dataset_cache_path(dataset_name, lang, split).exists()


def save_to_cache(
    data: List[Dict[str, Any]],
    dataset_name: str,
    lang: Optional[str] = None,
    split: str = "train",
) -> Path:
    """Save dataset to local cache as JSON.

    Args:
        data: List of dicts to cache.
        dataset_name: Name of the dataset.
        lang: Language code filter.
        split: Dataset split.

    Returns:
        Path to the cached file.
    """
    path = _dataset_cache_path(dataset_name, lang, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    logger.info("Cached %d records to %s", len(data), path)
    return path


def load_from_cache(
    dataset_name: str,
    lang: Optional[str] = None,
    split: str = "train",
) -> List[Dict[str, Any]]:
    """Load dataset from local cache.

    Args:
        dataset_name: Name of the dataset.
        lang: Language code filter.
        split: Dataset split.

    Returns:
        List of dicts loaded from cache.

    Raises:
        FileNotFoundError: If the dataset is not cached.
    """
    path = _dataset_cache_path(dataset_name, lang, split)
    if not path.exists():
        raise FileNotFoundError(
            "Dataset '%s' (lang=%s, split=%s) is not cached. "
            "Load it once with internet to cache it." % (dataset_name, lang, split)
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d records from cache %s", len(data), path)
    return data


def clear_cache(dataset_name: Optional[str] = None) -> None:
    """Clear cached datasets.

    Args:
        dataset_name: If provided, only clear cache for this dataset.
            If None, clear the entire cache.
    """
    import shutil

    cache_dir = get_cache_dir()
    if dataset_name:
        target = cache_dir / dataset_name
        if target.exists():
            shutil.rmtree(target)
            logger.info("Cleared cache for %s", dataset_name)
    else:
        for item in cache_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        logger.info("Cleared entire NaijaML cache")
