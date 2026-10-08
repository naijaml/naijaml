"""Yorùbá diacritizer using word-level lookup with syllable fallback.

Restores diacritics (tonal marks and dot-below) to undiacritized Yorùbá text
using a two-stage approach:
1. Word-level lookup for known words (~83% accuracy, 99% coverage)
2. Syllable-based fallback for unknown words

This is a lightweight, CPU-only implementation. It gets 80.3% of words right on
the MENYO-20k test set (reproduce with ``python scripts/eval_heldout.py``).

This module holds the public functions and the model cache. The character
tables and syllabifier live in ``_yoruba_text``, the model classes in
``_yoruba_models`` and the training and evaluation helpers in
``_yoruba_training``; their public names are re-exported here.
"""
from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

from naijaml.nlp._yoruba_models import (
    DotBelowDiacritizer,
    WordLevelDiacritizer,
    YorubaDiacritizer,
)
from naijaml.nlp._yoruba_text import (  # noqa: F401  (re-exported)
    COMBINING_MARKS,
    DOTTED_CONSONANTS,
    DOTTED_VOWELS,
    TONE_ACUTE,
    TONE_GRAVE,
    TONE_MACRON,
    UNDOTTED_CONS_TO_DOTTED,
    UNDOTTED_TO_DOTTED,
    YORUBA_CHARS,
    YORUBA_CONSONANTS,
    YORUBA_VOWELS,
    normalize_yoruba,
    strip_diacritics,
    syllabify,
    syllabify_text,
)
from naijaml.nlp._yoruba_training import (  # noqa: F401  (re-exported)
    _collect_training_data,
    _get_fallback_training_data,
    compare_diacritization_methods,
    evaluate_dot_below_accuracy,
    evaluate_full_diacritization_accuracy,
    train_and_save_dot_below_model,
    train_and_save_model,
)
from naijaml.utils.download import get_model_path

# Cached models
_MODEL: Optional["YorubaDiacritizer"] = None
_WORD_MODEL: Optional["WordLevelDiacritizer"] = None
_WORD_MODEL_LOAD_FAILED = False


def _load_pretrained_word_model() -> Optional[WordLevelDiacritizer]:
    """Load the pre-trained word-level model (cached, or downloaded from HF).

    Returns None if it cannot be loaded. Never trains a model, and only
    attempts the download once per process so offline callers are not
    slowed down by repeated network timeouts.
    """
    global _WORD_MODEL, _WORD_MODEL_LOAD_FAILED

    if _WORD_MODEL is not None:
        return _WORD_MODEL
    if _WORD_MODEL_LOAD_FAILED:
        return None

    try:
        model_path = get_model_path("word_diacritic_model.json")
        _WORD_MODEL = WordLevelDiacritizer.load(model_path)
        logger.debug("Loaded pre-trained word-level diacritizer from %s", model_path)
    except Exception as e:
        _WORD_MODEL_LOAD_FAILED = True
        logger.warning("Failed to load word-level model: %s", e)

    return _WORD_MODEL


def _get_word_model() -> WordLevelDiacritizer:
    """Get the word-level diacritizer model (loading from file or training)."""
    global _WORD_MODEL

    # Try to load pre-trained model (downloads from HF if needed)
    model = _load_pretrained_word_model()
    if model is not None:
        return model

    # Train a new model from HuggingFace dataset
    logger.info("Training new word-level diacritizer model...")

    diacritized_texts = []
    undiacritized_texts = []

    try:
        from datasets import load_dataset as hf_load_dataset

        logger.info("Loading Yorùbá diacritics dataset from HuggingFace...")
        ds = hf_load_dataset("bumie-e/Yoruba-diacritics-vs-non-diacritics", split="train")

        for item in ds:
            diac = item.get("diacritcs", "")  # Note: typo in dataset column name
            undiac = item.get("no_diacritcs", "")
            if diac and undiac:
                diacritized_texts.append(diac)
                undiacritized_texts.append(undiac)

        logger.info("Loaded %d sentence pairs", len(diacritized_texts))

    except ImportError:
        logger.warning("datasets library not available, using fallback training data")
        diacritized_texts = _get_fallback_training_data()
        undiacritized_texts = None
    except Exception as e:
        logger.warning("Failed to load HuggingFace dataset: %s, using fallback", e)
        diacritized_texts = _get_fallback_training_data()
        undiacritized_texts = None

    _WORD_MODEL = WordLevelDiacritizer(min_word_freq=5, min_bigram_freq=3)
    _WORD_MODEL.train(diacritized_texts, undiacritized_texts)

    return _WORD_MODEL


# =============================================================================
# Dot-Below-Only Diacritizer (Stage 1)
# =============================================================================

# Bundled dot-below model (ships with pip install)
_BUNDLED_DOT_BELOW_PATH = Path(__file__).parent / "dot_below_model.json"

# Cached dot-below model
_DOT_BELOW_MODEL: Optional["DotBelowDiacritizer"] = None


def _get_dot_below_model() -> DotBelowDiacritizer:
    """Get the dot-below diacritizer model (loading from file or training)."""
    global _DOT_BELOW_MODEL

    if _DOT_BELOW_MODEL is not None:
        return _DOT_BELOW_MODEL

    # Try bundled model first, then HF download
    try:
        if _BUNDLED_DOT_BELOW_PATH.exists():
            model_path = _BUNDLED_DOT_BELOW_PATH
        else:
            model_path = get_model_path("dot_below_model.json")
        _DOT_BELOW_MODEL = DotBelowDiacritizer.load(model_path)
        logger.debug("Loaded pre-trained dot-below model from %s", model_path)
        return _DOT_BELOW_MODEL
    except Exception as e:
        logger.warning("Failed to load dot-below model: %s", e)

    # Train a new model
    logger.info("Training new dot-below diacritizer model...")
    texts = _collect_training_data()

    _DOT_BELOW_MODEL = DotBelowDiacritizer()
    _DOT_BELOW_MODEL.train(texts)

    return _DOT_BELOW_MODEL


# =============================================================================
# Model loading
# =============================================================================

def _get_model() -> YorubaDiacritizer:
    """Get the diacritizer model (loading from file or training)."""
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    # Try to load pre-trained model (downloads from HF if needed)
    try:
        model_path = get_model_path("diacritic_model.json")
        _MODEL = YorubaDiacritizer.load(model_path)
        logger.debug("Loaded pre-trained diacritizer from %s", model_path)
        return _MODEL
    except Exception as e:
        logger.warning("Failed to load diacritizer: %s", e)

    # Train a new model
    logger.info("Training new diacritizer model...")
    texts = _collect_training_data()

    _MODEL = YorubaDiacritizer()
    _MODEL.train(texts)

    return _MODEL


# =============================================================================
# Public API
# =============================================================================

def diacritize(text: str, use_word_level: bool = True) -> str:
    """Restore diacritics to undiacritized Yorùbá text.

    Uses a two-stage approach for best accuracy:
    1. Word-level lookup for known words (~83% accuracy, 99% coverage)
    2. Syllable-based fallback for unknown words

    Args:
        text: Undiacritized Yorùbá text.
        use_word_level: If True (default), use word-level lookup with syllable
                       fallback. If False, use syllable-only approach.

    Returns:
        Text with diacritics restored.

    Example:
        >>> diacritize("Ojo dara pupo")
        'Ọjọ́ dára púpọ̀'
        >>> diacritize("E ku ise")
        'Ẹ kú iṣẹ́'
        >>> diacritize("Ọjọ dara")
        'Ọjọ́ dára'

    Marks already in the input are kept: a word that carries a tone mark is
    returned as given, and a word with dot-below only gains tones.

    Note:
        Gets 80.3% of words right on the MENYO-20k test set (reproduce with
        ``python scripts/eval_heldout.py``).
        The model was trained on the bumie-e/Yoruba-diacritics-vs-non-diacritics
        dataset containing 676k sentence pairs.
    """
    if not text or not text.strip():
        return text

    if use_word_level:
        model = _get_word_model()
    else:
        model = _get_model()

    # The models expect undiacritized input, so predict on stripped text and
    # put the marks the caller already supplied back afterwards.
    plain = strip_diacritics(text)
    predicted = model.diacritize(plain)
    if plain == normalize_yoruba(text):
        return predicted
    return _merge_existing_marks(text, predicted)


def _mark_clusters(text: str) -> List[Tuple[str, str]]:
    """Split text into (base character, combining marks) pairs."""
    clusters = []  # type: List[Tuple[str, str]]
    for char in unicodedata.normalize("NFD", text):
        if unicodedata.category(char) == "Mn" and clusters:
            base, marks = clusters[-1]
            clusters[-1] = (base, marks + char)
        else:
            clusters.append((char, ""))
    return clusters


def _word_runs(clusters: List[Tuple[str, str]]) -> List[List[Tuple[str, str]]]:
    """Group clusters into alternating runs of letters and non-letters."""
    runs = []  # type: List[List[Tuple[str, str]]]
    for cluster in clusters:
        if runs and runs[-1][0][0].isalpha() == cluster[0].isalpha():
            runs[-1].append(cluster)
        else:
            runs.append([cluster])
    return runs


def _merge_existing_marks(text: str, predicted: str) -> str:
    """Combine predicted diacritics with the marks already present in text.

    Marks in the input are never removed or changed. A word that already
    carries a tone mark is taken to be fully marked and is returned as it
    was given. A word with dot-below only keeps its dots and gains the
    predicted tones and any further predicted dots.
    """
    dot = "̣"
    given_runs = _word_runs(_mark_clusters(text))
    predicted_runs = _word_runs(_mark_clusters(predicted))
    if len(given_runs) != len(predicted_runs):
        return predicted

    result = []
    for given, pred in zip(given_runs, predicted_runs):
        given_marks = "".join(marks for _, marks in given)
        if not given_marks:
            result.extend(base + marks for base, marks in pred)
            continue

        has_tone = any(mark != dot for mark in given_marks)
        same_letters = [b.lower() for b, _ in given] == [b.lower() for b, _ in pred]
        if has_tone or not same_letters:
            result.extend(base + marks for base, marks in given)
            continue

        for (base, marks), (_, pred_marks) in zip(given, pred):
            result.append(base + marks + "".join(m for m in pred_marks if m not in marks))

    return normalize_yoruba("".join(result))


# =============================================================================
# Dot-Below-Only Public API
# =============================================================================

def diacritize_dot_below(text: str, use_word_level: bool = True) -> str:
    """Restore ONLY dot-below characters to undiacritized Yorùbá text.

    This is a simpler task than full diacritization, achieving higher accuracy.
    Restores ọ, ẹ, ṣ without adding tonal marks (á, à, é, è, etc.).

    Use this when:
    - You need reliable diacritization for display/reading
    - Tonal accuracy is not critical for your use case
    - You prefer higher accuracy over completeness

    Args:
        text: Undiacritized Yorùbá text.
        use_word_level: If True (default), run the word-level restorer and
                       drop its tone marks. The word-level model is downloaded
                       once and cached; if it is unavailable (e.g. offline on
                       first use) the bundled syllable model is used instead.
                       If False, always use the bundled syllable model, which
                       needs no download.

    Returns:
        Text with dot-below characters (ọ, ẹ, ṣ) restored, no tones.

    Example:
        >>> diacritize_dot_below("Ojo dara pupo")
        'Ọjọ dara pupọ'
        >>> diacritize_dot_below("E ku ise")
        'Ẹ ku iṣẹ'

    Note:
        On the MENYO-20k test set (6,633 sentences) the word-level route
        gets 93.3% of words right, against 85.7% for the bundled syllable
        model. Reproduce with ``python scripts/eval_heldout.py``.
    """
    if not text or not text.strip():
        return text

    if use_word_level:
        word_model = _load_pretrained_word_model()
        if word_model is not None:
            return _dot_below_via_word_model(text, word_model)

    model = _get_dot_below_model()
    return model.diacritize(text)


def _dot_below_via_word_model(text: str, word_model: WordLevelDiacritizer) -> str:
    """Restore dot-below with the word-level model, keeping the input's dots and letter case.

    The word-level model expects undiacritized input, so existing marks are
    stripped before lookup and the tones it restores are dropped afterwards.
    """
    given = strip_diacritics(text, tones_only=True)
    predicted = strip_diacritics(word_model.diacritize(strip_diacritics(text)), tones_only=True)
    if len(predicted) != len(given):
        return predicted

    dotted = set(DOTTED_VOWELS) | set(DOTTED_CONSONANTS)
    result = []
    for g, p in zip(given, predicted):
        if g in dotted:
            result.append(g)
        elif g.isupper():
            result.append(p.upper())
        elif g.islower():
            result.append(p.lower())
        else:
            result.append(p)
    return "".join(result)
