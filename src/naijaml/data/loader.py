"""Core dataset loading logic for NaijaML.

Provides load_dataset() which dispatches to dataset-specific loaders.
Each loader handles its own caching, downloading, and parsing.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from naijaml.data.registry import validate_lang, validate_split

logger = logging.getLogger(__name__)

# Map dataset name -> loader function (lazy imports to avoid circular deps)
_LOADERS = {}  # type: Dict[str, Callable]


def _get_loader(name: str) -> Callable:
    """Get the loader function for a dataset, importing lazily."""
    if name not in _LOADERS:
        if name == "naijasenti":
            from naijaml.data.datasets.naijasenti import load_naijasenti
            _LOADERS[name] = load_naijasenti
        elif name == "masakhaner":
            from naijaml.data.datasets.masakhaner import load_masakhaner
            _LOADERS[name] = load_masakhaner
        elif name == "masakhanews":
            from naijaml.data.datasets.masakhanews import load_masakhanews
            _LOADERS[name] = load_masakhanews
        else:
            raise ValueError(
                "Dataset '%s' does not have a loader yet. "
                "Available loaders: naijasenti, masakhaner, masakhanews" % name
            )
    return _LOADERS[name]


def load_dataset(
    name: str,
    lang: Optional[str] = None,
    split: str = "train",
) -> List[Dict[str, Any]]:
    """Load a Nigerian NLP dataset.

    Checks local cache first, then downloads if needed.
    After first download, works fully offline.

    Args:
        name: Dataset name (e.g. 'naijasenti', 'masakhaner').
            Use list_datasets() to see all available names.
        lang: Language code to filter by (e.g. 'yor', 'hau', 'ibo', 'pcm').
            If None, loads all languages (where applicable).
        split: Dataset split — 'train', 'validation', or 'test'.

    Returns:
        List of dicts. Structure depends on the dataset:
        - Sentiment: [{"text": "...", "label": "positive"}, ...]
        - NER: [{"tokens": [...], "ner_tags": [...]}, ...]
        - Classification: [{"text": "...", "label": "...", "headline": "..."}, ...]

    Raises:
        ValueError: If dataset name, language, or split is invalid.

    Example:
        >>> from naijaml.data import load_dataset
        >>> data = load_dataset("naijasenti", lang="yor", split="train")
        >>> print(data[0])
        {"text": "...", "label": "positive"}
    """
    validate_split(name, split)
    validate_lang(name, lang)

    loader = _get_loader(name)
    return loader(lang=lang, split=split)
