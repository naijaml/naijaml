"""NaijaML - Open-source ML tools for the Nigerian ecosystem."""
__version__ = "0.2.0"

from naijaml.data import clear_cache, dataset_info, list_datasets, load_dataset

__all__ = ["load_dataset", "list_datasets", "dataset_info", "clear_cache"]
