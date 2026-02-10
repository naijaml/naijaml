"""Dataset registry with metadata for all supported NaijaML datasets."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Each entry: name -> metadata dict
_REGISTRY = {
    "naijasenti": {
        "name": "naijasenti",
        "description": "NaijaSenti: Twitter sentiment dataset for four Nigerian languages.",
        "languages": ["yor", "hau", "ibo", "pcm"],
        "task": "sentiment",
        "splits": ["train", "validation", "test"],
        "hf_id": "HausaNLP/NaijaSenti-Twitter",
        "citation": (
            "Muhammad, S. H., et al. (2022). NaijaSenti: A Nigerian Twitter "
            "Sentiment Corpus for Multilingual Sentiment Analysis."
        ),
    },
    "masakhaner": {
        "name": "masakhaner",
        "description": "MasakhaNER v2: Named entity recognition for African languages.",
        "languages": ["yor", "hau", "ibo"],
        "task": "ner",
        "splits": ["train", "validation", "test"],
        "hf_id": "masakhane/masakhaner2",
        "citation": (
            "Adelani, D. I., et al. (2022). MasakhaNER 2.0: Africa-centric "
            "Transfer Learning for Named Entity Recognition."
        ),
    },
    "masakhapos": {
        "name": "masakhapos",
        "description": "MasakhaPOS: Part-of-speech tagging for African languages.",
        "languages": ["yor", "hau", "ibo"],
        "task": "pos",
        "splits": ["train", "validation", "test"],
        "hf_id": "masakhane/masakhapos",
        "citation": (
            "Dione, C. M. B., et al. (2023). MasakhaPOS: Part-of-Speech "
            "Tagging for Typologically Diverse African Languages."
        ),
    },
    "masakhanews": {
        "name": "masakhanews",
        "description": "MasakhaNEWS: News topic classification for African languages.",
        "languages": ["yor", "hau", "ibo", "pcm"],
        "task": "classification",
        "splits": ["train", "validation", "test"],
        "hf_id": "masakhane/masakhanews",
        "citation": (
            "Adelani, D. I., et al. (2023). MasakhaNEWS: News Topic "
            "Classification for African Languages."
        ),
    },
    "nollysenti": {
        "name": "nollysenti",
        "description": "NollySenti: Sentiment dataset from Nollywood movie reviews.",
        "languages": ["yor", "hau", "ibo", "pcm", "eng"],
        "task": "sentiment",
        "splits": ["train", "validation", "test"],
        "hf_id": "HausaNLP/NollySenti",
        "citation": (
            "Muhammad, S. H., et al. (2023). NollySenti: Leveraging Transfer "
            "Learning and Machine Translation for Nigerian Movie Sentiment."
        ),
    },
    "menyo20k": {
        "name": "menyo20k",
        "description": "MENYO-20k: Yorùbá-English parallel translation corpus.",
        "languages": ["yor", "eng"],
        "task": "translation",
        "splits": ["train", "validation", "test"],
        "hf_id": "masakhane/menyo20k_mt",
        "citation": (
            "Adelani, D. I., et al. (2021). The Effect of Domain and "
            "Diacritics in Yoruba-English Neural Machine Translation."
        ),
    },
    "afriqa": {
        "name": "afriqa",
        "description": "AfriQA: Cross-lingual open-retrieval question answering for African languages.",
        "languages": ["yor", "hau"],
        "task": "qa",
        "splits": ["train", "validation", "test"],
        "hf_id": "masakhane/afriqa",
        "citation": (
            "Ogundepo, O., et al. (2023). AfriQA: Cross-lingual "
            "Open-Retrieval Question Answering for African Languages."
        ),
    },
}


def list_datasets() -> List[str]:
    """List all available dataset names.

    Returns:
        Sorted list of dataset name strings.
    """
    return sorted(_REGISTRY.keys())


def dataset_info(name: str) -> Dict[str, Any]:
    """Get metadata for a dataset.

    Args:
        name: Dataset name (e.g. 'naijasenti').

    Returns:
        Dict with keys: name, description, languages, task, splits, hf_id, citation.

    Raises:
        ValueError: If the dataset name is not in the registry.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            "Unknown dataset '%s'. Available datasets: %s" % (name, available)
        )
    return dict(_REGISTRY[name])


def get_hf_id(name: str) -> str:
    """Get the HuggingFace dataset ID for a dataset.

    Args:
        name: Dataset name.

    Returns:
        HuggingFace dataset identifier string.

    Raises:
        ValueError: If the dataset name is not in the registry.
    """
    info = dataset_info(name)
    return info["hf_id"]


def validate_lang(name: str, lang: Optional[str]) -> None:
    """Validate that a language is supported by a dataset.

    Args:
        name: Dataset name.
        lang: Language code to validate, or None (skip validation).

    Raises:
        ValueError: If the language is not supported by the dataset.
    """
    if lang is None:
        return
    info = dataset_info(name)
    supported = info["languages"]
    if lang not in supported:
        raise ValueError(
            "Language '%s' is not supported by '%s'. "
            "Supported languages: %s" % (lang, name, ", ".join(supported))
        )


def validate_split(name: str, split: str) -> None:
    """Validate that a split exists for a dataset.

    Args:
        name: Dataset name.
        split: Split name to validate.

    Raises:
        ValueError: If the split does not exist for the dataset.
    """
    info = dataset_info(name)
    valid_splits = info["splits"]
    if split not in valid_splits:
        raise ValueError(
            "Split '%s' is not valid for '%s'. "
            "Valid splits: %s" % (split, name, ", ".join(valid_splits))
        )
