"""MasakhaNER v2 dataset loader.

Loads the MasakhaNER 2.0 named entity recognition dataset for African languages.
Source: https://github.com/masakhane-io/masakhane-ner
HuggingFace: masakhane/masakhaner2
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from naijaml.data.cache import is_cached, load_from_cache, save_to_cache

logger = logging.getLogger(__name__)

_BASE_URL = (
    "https://github.com/masakhane-io/masakhane-ner/raw/main/"
    "MasakhaNER2.0/data/{lang}/{split}.txt"
)

_SPLIT_MAP = {
    "train": "train",
    "validation": "dev",
    "test": "test",
}

_LANGUAGES = ["yor", "hau", "ibo"]


def load_masakhaner(
    lang: Optional[str] = None,
    split: str = "train",
) -> List[Dict[str, Any]]:
    """Load the MasakhaNER v2 NER dataset.

    Args:
        lang: Language code ('yor', 'hau', 'ibo').
            If None, loads all languages.
        split: Dataset split ('train', 'validation', 'test').

    Returns:
        List of dicts with keys 'tokens' (list of str) and
        'ner_tags' (list of str like 'B-PER', 'I-LOC', 'O').
    """
    if is_cached("masakhaner", lang, split):
        return load_from_cache("masakhaner", lang, split)

    langs = [lang] if lang else _LANGUAGES
    records = []
    for lg in langs:
        records.extend(_download_lang(lg, split))

    save_to_cache(records, "masakhaner", lang, split)
    return records


def _download_lang(lang: str, split: str) -> List[Dict[str, Any]]:
    """Download and parse MasakhaNER CoNLL-format file for one language/split."""
    import requests

    file_split = _SPLIT_MAP.get(split, split)
    url = _BASE_URL.format(lang=lang, split=file_split)
    logger.info("Downloading MasakhaNER %s/%s from %s", lang, split, url)

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    records = []
    tokens = []
    ner_tags = []

    for line in resp.text.splitlines():
        line = line.strip()
        if not line:
            # Blank line = sentence boundary
            if tokens:
                records.append({"tokens": tokens, "ner_tags": ner_tags})
                tokens = []
                ner_tags = []
        else:
            parts = line.split(" ")
            if len(parts) >= 2:
                tokens.append(parts[0])
                ner_tags.append(parts[1])

    # Last sentence if file doesn't end with blank line
    if tokens:
        records.append({"tokens": tokens, "ner_tags": ner_tags})

    logger.info("Loaded %d sentences for MasakhaNER %s/%s", len(records), lang, split)
    return records
