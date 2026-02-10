"""NaijaSenti dataset loader.

Loads the NaijaSenti Twitter sentiment corpus for Nigerian languages.
Source: https://github.com/hausanlp/NaijaSenti
HuggingFace: HausaNLP/NaijaSenti-Twitter
"""
from __future__ import annotations

import csv
import io
import logging
from typing import Any, Dict, List, Optional

from naijaml.data.cache import is_cached, load_from_cache, save_to_cache

logger = logging.getLogger(__name__)

_BASE_URL = (
    "https://raw.githubusercontent.com/hausanlp/NaijaSenti/main/"
    "data/annotated_tweets/{lang}/{split}.tsv"
)

_SPLIT_MAP = {
    "train": "train",
    "validation": "dev",
    "test": "test",
}

_LANGUAGES = ["yor", "hau", "ibo", "pcm"]


def load_naijasenti(
    lang: Optional[str] = None,
    split: str = "train",
) -> List[Dict[str, Any]]:
    """Load the NaijaSenti sentiment dataset.

    Args:
        lang: Language code ('yor', 'hau', 'ibo', 'pcm').
            If None, loads all languages.
        split: Dataset split ('train', 'validation', 'test').

    Returns:
        List of dicts with keys 'text' and 'label'.
        Label is one of: 'positive', 'neutral', 'negative'.
    """
    if is_cached("naijasenti", lang, split):
        return load_from_cache("naijasenti", lang, split)

    langs = [lang] if lang else _LANGUAGES
    records = []
    for lg in langs:
        records.extend(_download_lang(lg, split))

    save_to_cache(records, "naijasenti", lang, split)
    return records


def _download_lang(lang: str, split: str) -> List[Dict[str, Any]]:
    """Download and parse NaijaSenti TSV for one language/split."""
    import requests

    file_split = _SPLIT_MAP.get(split, split)
    url = _BASE_URL.format(lang=lang, split=file_split)
    logger.info("Downloading NaijaSenti %s/%s from %s", lang, split, url)

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    records = []
    reader = csv.DictReader(io.StringIO(resp.text), delimiter="\t")
    for row in reader:
        text = row.get("tweet", "").strip()
        label = row.get("label", "").strip()
        if text and label:
            records.append({"text": text, "label": label})

    logger.info("Loaded %d records for NaijaSenti %s/%s", len(records), lang, split)
    return records
