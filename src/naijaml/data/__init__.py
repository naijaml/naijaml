"""NaijaML data loading utilities.

Provides access to Nigerian NLP datasets with offline-first caching.

Example:
    >>> from naijaml.data import load_dataset, list_datasets
    >>> list_datasets()
    ['afriqa', 'masakhaner', 'masakhanews', 'masakhapos', 'menyo20k', 'naijasenti', 'nollysenti']
    >>> data = load_dataset("naijasenti", lang="yor", split="train")
"""
from naijaml.data.cache import clear_cache
from naijaml.data.loader import load_dataset
from naijaml.data.registry import dataset_info, list_datasets

__all__ = ["load_dataset", "list_datasets", "dataset_info", "clear_cache"]
