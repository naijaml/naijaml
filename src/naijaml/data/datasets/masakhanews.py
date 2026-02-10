"""MasakhaNEWS dataset loader.

Loads the MasakhaNEWS news topic classification dataset for African languages.
Source: https://huggingface.co/datasets/masakhane/masakhanews
"""
from __future__ import annotations

import csv
import io
import logging
from typing import Any, Dict, List, Optional

from naijaml.data.cache import is_cached, load_from_cache, save_to_cache

logger = logging.getLogger(__name__)

_BASE_URL = (
    "https://huggingface.co/datasets/masakhane/masakhanews/resolve/main/"
    "data/{lang}/{split}.tsv"
)

_SPLIT_MAP = {
    "train": "train",
    "validation": "dev",
    "test": "test",
}

_LANGUAGES = ["yor", "hau", "ibo", "pcm"]


def load_masakhanews(
    lang: Optional[str] = None,
    split: str = "train",
) -> List[Dict[str, Any]]:
    """Load the MasakhaNEWS topic classification dataset.

    Args:
        lang: Language code ('yor', 'hau', 'ibo', 'pcm').
            If None, loads all languages.
        split: Dataset split ('train', 'validation', 'test').

    Returns:
        List of dicts with keys 'text', 'label' (topic category),
        'headline', and 'url'.
    """
    if is_cached("masakhanews", lang, split):
        return load_from_cache("masakhanews", lang, split)

    langs = [lang] if lang else _LANGUAGES
    records = []
    for lg in langs:
        records.extend(_download_lang(lg, split))

    save_to_cache(records, "masakhanews", lang, split)
    return records


def _download_lang(lang: str, split: str) -> List[Dict[str, Any]]:
    """Download and parse MasakhaNEWS TSV for one language/split."""
    import requests

    file_split = _SPLIT_MAP.get(split, split)
    url = _BASE_URL.format(lang=lang, split=file_split)
    logger.info("Downloading MasakhaNEWS %s/%s from %s", lang, split, url)

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    records = []
    reader = csv.DictReader(io.StringIO(resp.text), delimiter="\t")
    for row in reader:
        text = row.get("text", "").strip()
        label = row.get("category", "").strip()
        headline = row.get("headline", "").strip()
        article_url = row.get("url", "").strip()
        if text and label:
            records.append({
                "text": text,
                "label": label,
                "headline": headline,
                "url": article_url,
            })

    logger.info("Loaded %d records for MasakhaNEWS %s/%s", len(records), lang, split)
    return records
