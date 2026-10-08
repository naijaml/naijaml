"""Training and evaluation helpers for the Yorùbá diacritizer models."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from naijaml.nlp._yoruba_models import DotBelowDiacritizer, WordLevelDiacritizer, YorubaDiacritizer
from naijaml.nlp._yoruba_text import normalize_yoruba, strip_diacritics

logger = logging.getLogger(__name__)


# =============================================================================
# Training data collection
# =============================================================================

def _get_fallback_training_data() -> List[str]:
    """Get built-in training sentences when datasets unavailable."""
    return [
        # Common greetings and phrases
        "Ẹ kú àárọ̀, báwo ni?",
        "Ọjọ́ dára púpọ̀ o",
        "Ẹ ṣé púpọ̀, mo dúpẹ́",
        "Ọlọ́run a bukun fún ẹ",
        "Mo fẹ́ràn rẹ púpọ̀",
        "Kí ni orúkọ rẹ?",
        "Orúkọ mi ni Adé",
        "Báwo ni ọjà ṣe wà?",
        "Ó ti lọ sí ilé ìwé",
        "Àwọn ọmọdé ń ṣeré ní ọ̀nà",
        "Ẹ jọ̀wọ́ ẹ fún mi ní omi",
        "Mo ń lọ sí ọjà láti ra oúnjẹ",
        # Words with ọ/ẹ distinction
        "Ọmọ mi dára",
        "Ẹ̀ṣẹ̀ púpọ̀ fún oúnjẹ náà",
        "Ọkọ rẹ ti dé",
        "Ẹyẹ ń fò lókè",
        "Ọ̀gá mi ń bọ̀",
        "Ẹrú ń bà mí",
        # Tonal minimal pairs
        "Igbá kan wà níbẹ̀",  # igbá = calabash
        "Ó gbà owó náà",  # gbà = receive
        "Ọjọ́ ọ̀sẹ̀ yìí dára",
        "Òjò ń rọ̀",  # òjò = rain
        "Ojó ti dé",  # ojó = a name
        # More sentences
        "Ìwọ ni mo fẹ́ràn jù lọ",
        "Ọrẹ mi dára púpọ̀ láti bá mi sọ̀rọ̀",
        "Ẹ má bínú mo ti pẹ́ díẹ̀",
        "Nígbà tí mo dé ilé ó ti lọ",
        "Àwọn ará ìlú náà ń ṣiṣẹ́ dáadáa",
        "Oúnjẹ yìí dùn gan ẹ ṣe é dáadáa",
        "Mo ń kọ́ èdè Yorùbá ní ilé ẹ̀kọ́",
        "Àwọn akẹ́kọ̀ọ́ ń kàwé ní ilé ìkàwé",
        "Ìyá mi ti ṣe oúnjẹ àárọ̀",
        "Bàbá mi ń ṣiṣẹ́ ní ilé iṣẹ́",
        "Ẹ̀gbọ́n mi ń gbé ní ìlú Lagos",
        "Àbúrò mi ti lọ sí ilé ẹ̀kọ́ gíga",
        "A ó rí ara wa lọ́la ẹ máa rìn dáadáa",
        "Ọ̀rọ̀ yìí dára púpọ̀ mo gbọ́ ọ dáadáa",
        "Àwa Yorùbá ń gbé ní gúúsù ìwọ̀ oòrùn Nàìjíríà",
        "Ìṣẹ̀lẹ̀ náà ṣẹlẹ̀ ní ọjọ́ ọ̀sẹ̀ tó kọjá",
        "Ó dára kí a máa bá ara wa sọ̀rọ̀",
        "Ẹ̀rọ yìí ṣiṣẹ́ dáadáa gan ni",
        "Mo ń wá iṣẹ́ ní ilú náà",
        "Àwọn oníṣòwò ń ta ọjà ní ọjà",
        "Ọmọdé kò gbọdọ̀ máa ṣe bẹ́ẹ̀",
        "Olùkọ́ ń kọ́ àwọn akẹ́kọ̀ọ́ ní yàrá",
        "Ìròyìn dùn mo gbọ́ ẹ ṣeun",
        "Ó ṣe pàtàkì láti kọ́ èdè míì",
        # Common words in context
        "Ilé ni ilé",
        "Omi tútù dára fún ara",
        "Owó kò tó",
        "Iṣẹ́ ń pa mí",
        "Àlàáfíà ni",
        "Odindi ọjọ́ náà dára",
        "Mo rí i pé ó dára",
        "Ṣé o ti jẹun?",
        "Rárá, mi ò tí ì jẹun",
        "Ó dára, má ṣe wàhálà",
    ]


def _collect_training_data() -> List[str]:
    """Collect training data from available sources.

    Uses bumie-e/Yoruba-diacritics-vs-non-diacritics (676k sentences)
    which has proper tonal marks AND dot-below characters.

    Returns:
        List of diacritized Yorùbá sentences.
    """
    texts = _get_fallback_training_data()
    logger.info("Starting with %d fallback sentences", len(texts))

    # Try to load from HuggingFace dataset (best quality)
    try:
        from datasets import load_dataset

        logger.info("Loading Yorùbá diacritics dataset from HuggingFace...")
        ds = load_dataset("bumie-e/Yoruba-diacritics-vs-non-diacritics", split="train")

        # Sample to avoid huge model (take ~50k sentences)
        max_samples = 50000
        if len(ds) > max_samples:
            import random
            random.seed(42)
            indices = random.sample(range(len(ds)), max_samples)
        else:
            indices = range(len(ds))

        for i in indices:
            diacritized = ds[i].get("diacritcs", "")  # Note: typo in dataset column name
            if diacritized and any(c in diacritized for c in "ọẹṣáàéèíìóòúù"):
                texts.append(diacritized)

        logger.info("Loaded %d sentences from Yorùbá diacritics dataset", len(texts))
        return texts

    except ImportError:
        logger.info("datasets library not available, trying MENYO-20k...")
    except Exception as e:
        logger.warning("Failed to load HuggingFace dataset: %s", e)

    # Fallback to MENYO-20k from Zenodo
    try:
        import requests

        url = "https://zenodo.org/api/records/4297448/files/train.tsv/content"
        logger.info("Downloading MENYO-20k from Zenodo...")

        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            lines = response.text.strip().split("\n")
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) >= 2:
                    yo_text = parts[1].strip()
                    if yo_text:
                        texts.append(yo_text)

            logger.info("Loaded %d sentences from MENYO-20k", len(texts))
        else:
            logger.warning("Failed to download MENYO-20k: HTTP %d", response.status_code)

    except ImportError:
        logger.info("requests not available, using fallback data")
    except Exception as e:
        logger.warning("Failed to load MENYO-20k: %s", e)

    return texts


def train_and_save_model(
    path: Optional[Path] = None,
    train_word_level: bool = True,
    train_syllable: bool = True,
) -> Dict[str, Path]:
    """Train new diacritizer models and save them.

    Trains both word-level (primary) and syllable-level (fallback) models.
    Requires the 'datasets' package for best results.

    Args:
        path: Base path for models. If None, saves to cache directory.
              Word model saved to path or cache/word_diacritic_model.json,
              syllable model saved to path with '_syllable' suffix or cache/diacritic_model.json.
        train_word_level: Whether to train word-level model (default True).
        train_syllable: Whether to train syllable model (default True).

    Returns:
        Dict with paths where models were saved:
        {'word_level': Path, 'syllable': Path}
    """
    from naijaml.nlp import diacritizer as api

    saved_paths = {}

    # Collect training data
    diacritized_texts = []
    undiacritized_texts = []

    try:
        from datasets import load_dataset as hf_load_dataset

        logger.info("Loading Yorùbá diacritics dataset from HuggingFace...")
        ds = hf_load_dataset("bumie-e/Yoruba-diacritics-vs-non-diacritics", split="train")

        for item in ds:
            diac = item.get("diacritcs", "")
            undiac = item.get("no_diacritcs", "")
            if diac and undiac:
                diacritized_texts.append(diac)
                undiacritized_texts.append(undiac)

        logger.info("Loaded %d sentence pairs from HuggingFace", len(diacritized_texts))

    except ImportError:
        logger.warning("datasets library not available, using fallback training data")
        diacritized_texts = _collect_training_data()
        undiacritized_texts = None
    except Exception as e:
        logger.warning("Failed to load HuggingFace dataset: %s, using fallback", e)
        diacritized_texts = _collect_training_data()
        undiacritized_texts = None

    # Train word-level model
    if train_word_level:
        from naijaml.utils.download import get_models_cache_dir
        word_path = path if path else get_models_cache_dir() / "word_diacritic_model.json"
        model = WordLevelDiacritizer(min_word_freq=5, min_bigram_freq=3)
        model.train(diacritized_texts, undiacritized_texts)
        model.save(word_path)
        api._WORD_MODEL = model
        saved_paths["word_level"] = word_path
        logger.info("Saved word-level model to %s", word_path)

    # Train syllable model
    if train_syllable:
        from naijaml.utils.download import get_models_cache_dir
        syllable_path = get_models_cache_dir() / "diacritic_model.json"
        if path:
            syllable_path = path.parent / (path.stem + "_syllable" + path.suffix)

        model = YorubaDiacritizer()
        model.train(diacritized_texts)
        model.save(syllable_path)
        api._MODEL = model
        saved_paths["syllable"] = syllable_path
        logger.info("Saved syllable model to %s", syllable_path)

    return saved_paths


def train_and_save_dot_below_model(path: Optional[Path] = None) -> Path:
    """Train a new dot-below-only diacritizer model and save it.

    Args:
        path: Path to save the model. Defaults to bundled location.

    Returns:
        Path where the model was saved.
    """
    from naijaml.nlp import diacritizer as api

    if path is None:
        from naijaml.utils.download import get_models_cache_dir
        path = get_models_cache_dir() / "dot_below_model.json"

    texts = _collect_training_data()

    model = DotBelowDiacritizer()
    model.train(texts)
    model.save(path)

    api._DOT_BELOW_MODEL = model

    return path


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_dot_below_accuracy(
    test_texts: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate dot-below-only diacritization accuracy.

    Measures how accurately the model restores ọ, ẹ, ṣ (no tones).

    Args:
        test_texts: List of diacritized test sentences. If None, uses fallback.
        verbose: Whether to print detailed results.

    Returns:
        Dict with accuracy metrics:
        - char_accuracy: Character-level accuracy
        - word_accuracy: Word-level accuracy (whole word correct)
        - dot_below_precision: Precision for dot-below prediction
        - dot_below_recall: Recall for dot-below prediction
        - dot_below_f1: F1 score for dot-below
    """
    if test_texts is None:
        test_texts = _get_fallback_training_data()

    from naijaml.nlp.diacritizer import _get_dot_below_model

    model = _get_dot_below_model()

    total_chars = 0
    correct_chars = 0
    total_words = 0
    correct_words = 0

    # For dot-below specific metrics
    true_positives = 0  # Correctly predicted dot-below
    false_positives = 0  # Predicted dot-below when shouldn't
    false_negatives = 0  # Missed dot-below

    dot_below_chars = set("ọẹṣỌẸṢ")

    for text in test_texts:
        # Expected: text with only dot-below (no tones)
        expected = strip_diacritics(text, tones_only=True)
        # Input: fully undiacritized
        input_text = strip_diacritics(text)
        # Predicted
        predicted = model.diacritize(input_text)

        # Character-level accuracy
        for exp_char, pred_char in zip(expected, predicted):
            total_chars += 1
            if exp_char == pred_char:
                correct_chars += 1

            # Dot-below specific
            exp_has_dot = exp_char in dot_below_chars
            pred_has_dot = pred_char in dot_below_chars

            if exp_has_dot and pred_has_dot and exp_char.lower() == pred_char.lower():
                true_positives += 1
            elif pred_has_dot and not exp_has_dot:
                false_positives += 1
            elif exp_has_dot and not pred_has_dot:
                false_negatives += 1

        # Word-level accuracy
        exp_words = expected.split()
        pred_words = predicted.split()
        for exp_word, pred_word in zip(exp_words, pred_words):
            total_words += 1
            if exp_word == pred_word:
                correct_words += 1

    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
    word_accuracy = correct_words / total_words if total_words > 0 else 0.0

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results = {
        "char_accuracy": char_accuracy,
        "word_accuracy": word_accuracy,
        "dot_below_precision": precision,
        "dot_below_recall": recall,
        "dot_below_f1": f1,
        "total_chars": total_chars,
        "total_words": total_words,
    }

    if verbose:
        print("=" * 60)
        print("DOT-BELOW-ONLY DIACRITIZATION EVALUATION")
        print("=" * 60)
        print(f"Test samples: {len(test_texts)}")
        print(f"Total characters: {total_chars}")
        print(f"Total words: {total_words}")
        print("-" * 60)
        print(f"Character-level accuracy: {char_accuracy:.2%}")
        print(f"Word-level accuracy: {word_accuracy:.2%}")
        print("-" * 60)
        print("Dot-below specific metrics (ọ, ẹ, ṣ):")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall: {recall:.2%}")
        print(f"  F1 Score: {f1:.2%}")
        print("=" * 60)

    return results


def evaluate_full_diacritization_accuracy(
    test_texts: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate full diacritization accuracy (tones + dot-below).

    Args:
        test_texts: List of diacritized test sentences. If None, uses fallback.
        verbose: Whether to print detailed results.

    Returns:
        Dict with accuracy metrics.
    """
    if test_texts is None:
        test_texts = _get_fallback_training_data()

    from naijaml.nlp.diacritizer import _get_model

    model = _get_model()

    total_chars = 0
    correct_chars = 0
    total_words = 0
    correct_words = 0

    for text in test_texts:
        expected = normalize_yoruba(text)
        input_text = strip_diacritics(text)
        predicted = model.diacritize(input_text)

        # Character-level
        for exp_char, pred_char in zip(expected, predicted):
            total_chars += 1
            if exp_char == pred_char:
                correct_chars += 1

        # Word-level
        exp_words = expected.split()
        pred_words = predicted.split()
        for exp_word, pred_word in zip(exp_words, pred_words):
            total_words += 1
            if exp_word == pred_word:
                correct_words += 1

    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
    word_accuracy = correct_words / total_words if total_words > 0 else 0.0

    results = {
        "char_accuracy": char_accuracy,
        "word_accuracy": word_accuracy,
        "total_chars": total_chars,
        "total_words": total_words,
    }

    if verbose:
        print("=" * 60)
        print("FULL DIACRITIZATION EVALUATION (tones + dot-below)")
        print("=" * 60)
        print(f"Test samples: {len(test_texts)}")
        print(f"Total characters: {total_chars}")
        print(f"Total words: {total_words}")
        print("-" * 60)
        print(f"Character-level accuracy: {char_accuracy:.2%}")
        print(f"Word-level accuracy: {word_accuracy:.2%}")
        print("=" * 60)

    return results


def compare_diacritization_methods(
    test_texts: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare dot-below-only vs full diacritization accuracy.

    Args:
        test_texts: List of diacritized test sentences. If None, uses fallback.

    Returns:
        Dict with results for both methods.
    """
    print("\n" + "=" * 60)
    print("COMPARING DIACRITIZATION METHODS")
    print("=" * 60 + "\n")

    if test_texts is None:
        test_texts = _get_fallback_training_data()

    print("Method 1: Dot-Below Only (ọ, ẹ, ṣ - no tones)")
    print("-" * 60)
    dot_below_results = evaluate_dot_below_accuracy(test_texts, verbose=True)

    print("\nMethod 2: Full Diacritization (tones + dot-below)")
    print("-" * 60)
    full_results = evaluate_full_diacritization_accuracy(test_texts, verbose=True)

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Dot-below only char accuracy: {dot_below_results['char_accuracy']:.2%}")
    print(f"Full diacritization char accuracy: {full_results['char_accuracy']:.2%}")
    diff = dot_below_results['char_accuracy'] - full_results['char_accuracy']
    print(f"Difference: {diff:+.2%} ({'dot-below better' if diff > 0 else 'full better'})")
    print("=" * 60)

    return {
        "dot_below_only": dot_below_results,
        "full_diacritization": full_results,
    }
