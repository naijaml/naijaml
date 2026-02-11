"""Evaluation metrics for NaijaML NLP models."""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

logger = logging.getLogger(__name__)


@dataclass
class DiacritizerMetrics:
    """Evaluation metrics for a diacritizer model."""

    # Basic info
    language: str
    model_version: str
    test_size: int
    date: str = field(default_factory=lambda: datetime.now().isoformat()[:10])

    # Accuracy metrics
    word_accuracy: float = 0.0          # % of words exactly correct
    sentence_accuracy: float = 0.0      # % of sentences exactly correct
    character_accuracy: float = 0.0     # % of characters correct

    # Diacritic-specific metrics
    diacritic_precision: float = 0.0    # when we add a diacritic, is it correct?
    diacritic_recall: float = 0.0       # did we find all diacritics that should exist?
    diacritic_f1: float = 0.0           # harmonic mean of precision and recall

    # Error analysis
    total_words: int = 0
    correct_words: int = 0
    total_sentences: int = 0
    correct_sentences: int = 0

    # Sample errors (for debugging)
    sample_errors: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Return human-readable summary."""
        return f"""
=== {self.language.upper()} Diacritizer Evaluation ===
Model: {self.model_version}
Date: {self.date}
Test size: {self.test_size} sentences

ACCURACY:
  Word accuracy:      {self.word_accuracy:.1%} ({self.correct_words}/{self.total_words})
  Sentence accuracy:  {self.sentence_accuracy:.1%} ({self.correct_sentences}/{self.total_sentences})
  Character accuracy: {self.character_accuracy:.1%}

DIACRITIC METRICS:
  Precision: {self.diacritic_precision:.1%}
  Recall:    {self.diacritic_recall:.1%}
  F1 Score:  {self.diacritic_f1:.1%}

SAMPLE ERRORS ({len(self.sample_errors)} shown):
""" + "\n".join(
            f"  '{e['input']}' → '{e['predicted']}' (expected: '{e['expected']}')"
            for e in self.sample_errors[:5]
        )


def _get_diacritics(text: str, language: str) -> set:
    """Get positions of diacritized characters in text."""
    if language == "yoruba":
        diacritic_chars = set("ọẹṣáàéèíìóòúùọ́ọ̀ẹ́ẹ̀ỌẸṢÁÀÉÈÍÌÓÒÚÙ")
    elif language == "igbo":
        diacritic_chars = set("ịọụáàóòúùịọụỊỌỤÁÀÓÒÚÙ")
    else:
        diacritic_chars = set()

    return {(i, c) for i, c in enumerate(text) if c in diacritic_chars}


def _calculate_diacritic_metrics(
    predicted: str,
    expected: str,
    language: str
) -> Tuple[int, int, int]:
    """Calculate true positives, false positives, false negatives for diacritics.

    Returns:
        (true_positives, false_positives, false_negatives)
    """
    pred_diacritics = _get_diacritics(predicted, language)
    exp_diacritics = _get_diacritics(expected, language)

    # True positives: diacritics in both predicted and expected
    tp = len(pred_diacritics & exp_diacritics)
    # False positives: diacritics in predicted but not expected
    fp = len(pred_diacritics - exp_diacritics)
    # False negatives: diacritics in expected but not predicted
    fn = len(exp_diacritics - pred_diacritics)

    return tp, fp, fn


def evaluate_diacritizer(
    language: str,
    test_data: Optional[List[str]] = None,
    test_size: int = 500,
    strip_fn: Optional[Callable[[str], str]] = None,
    diacritize_fn: Optional[Callable[[str], str]] = None,
    model_version: str = "v1",
    seed: int = 42,
) -> DiacritizerMetrics:
    """Evaluate a diacritizer model on held-out test data.

    Args:
        language: "yoruba" or "igbo"
        test_data: List of properly diacritized sentences for testing.
                   If None, will fetch from appropriate dataset.
        test_size: Number of sentences to test on.
        strip_fn: Function to strip diacritics. If None, uses default.
        diacritize_fn: Function to restore diacritics. If None, uses default.
        model_version: Version string for tracking.
        seed: Random seed for reproducibility.

    Returns:
        DiacritizerMetrics with all evaluation results.
    """
    random.seed(seed)

    # Get functions based on language
    if language == "yoruba":
        if strip_fn is None:
            from naijaml.nlp.diacritizer import strip_diacritics
            strip_fn = strip_diacritics
        if diacritize_fn is None:
            from naijaml.nlp.diacritizer import diacritize
            diacritize_fn = diacritize
        if test_data is None:
            test_data = _get_yoruba_test_data()
    elif language == "igbo":
        if strip_fn is None:
            from naijaml.nlp.igbo_diacritizer import strip_diacritics
            strip_fn = strip_diacritics
        if diacritize_fn is None:
            from naijaml.nlp.igbo_diacritizer import diacritize_igbo
            diacritize_fn = diacritize_igbo
        if test_data is None:
            test_data = _get_igbo_test_data()
    else:
        raise ValueError(f"Unknown language: {language}")

    # Sample test data
    if len(test_data) > test_size:
        test_data = random.sample(test_data, test_size)

    # Initialize counters
    total_words = 0
    correct_words = 0
    total_sentences = len(test_data)
    correct_sentences = 0
    total_chars = 0
    correct_chars = 0

    # Diacritic metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Collect errors
    errors = []

    for expected in test_data:
        # Strip diacritics and re-diacritize
        stripped = strip_fn(expected)
        predicted = diacritize_fn(stripped)

        # Sentence-level accuracy
        if predicted.lower() == expected.lower():
            correct_sentences += 1
        else:
            if len(errors) < 20:  # Keep up to 20 error examples
                errors.append({
                    "input": stripped,
                    "predicted": predicted,
                    "expected": expected,
                })

        # Word-level accuracy
        pred_words = predicted.lower().split()
        exp_words = expected.lower().split()

        for pred_word, exp_word in zip(pred_words, exp_words):
            total_words += 1
            if pred_word == exp_word:
                correct_words += 1

        # Character-level accuracy
        for pred_char, exp_char in zip(predicted.lower(), expected.lower()):
            total_chars += 1
            if pred_char == exp_char:
                correct_chars += 1

        # Diacritic precision/recall
        tp, fp, fn = _calculate_diacritic_metrics(predicted, expected, language)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Calculate final metrics
    word_accuracy = correct_words / total_words if total_words > 0 else 0
    sentence_accuracy = correct_sentences / total_sentences if total_sentences > 0 else 0
    character_accuracy = correct_chars / total_chars if total_chars > 0 else 0

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return DiacritizerMetrics(
        language=language,
        model_version=model_version,
        test_size=len(test_data),
        word_accuracy=word_accuracy,
        sentence_accuracy=sentence_accuracy,
        character_accuracy=character_accuracy,
        diacritic_precision=precision,
        diacritic_recall=recall,
        diacritic_f1=f1,
        total_words=total_words,
        correct_words=correct_words,
        total_sentences=total_sentences,
        correct_sentences=correct_sentences,
        sample_errors=errors,
    )


def _get_yoruba_test_data() -> List[str]:
    """Get Yorùbá test sentences from MENYO-20k (held-out from training)."""
    try:
        import requests
        import random

        logger.info("Fetching Yorùbá test data from MENYO-20k...")
        # Use the API endpoint to get correct file URL
        url = "https://zenodo.org/api/records/4297448/files/train.tsv/content"
        response = requests.get(url, timeout=60)

        if response.status_code == 200:
            texts = []
            lines = response.text.strip().split("\n")
            for line in lines[1:]:  # Skip header
                parts = line.split("\t")
                if len(parts) >= 2:
                    yo_text = parts[1].strip()
                    # Only include sentences with diacritics
                    if yo_text and any(c in yo_text for c in "ọẹṣáàéèíìóòúù"):
                        texts.append(yo_text)

            # Use last 20% as test data (not used in training)
            random.seed(42)
            random.shuffle(texts)
            test_size = len(texts) // 5
            test_texts = texts[:test_size]

            logger.info("Loaded %d Yorùbá test sentences (20%% held out)", len(test_texts))
            return test_texts
    except Exception as e:
        logger.warning("Failed to load MENYO-20k test data: %s", e)

    # Fallback
    return [
        "Ọjọ́ dára púpọ̀",
        "Ẹ kú àárọ̀",
        "Mo fẹ́ràn rẹ",
        "Ọmọ mi dára",
        "Ẹ ṣé púpọ̀",
    ]


def _get_igbo_test_data() -> List[str]:
    """Get Igbo test sentences from JW300."""
    try:
        from datasets import load_dataset

        logger.info("Fetching Igbo test data from JW300...")
        ds = load_dataset("Tommy0201/JW300_Igbo_To_Eng", split="test")

        texts = []
        for item in ds:
            text = item.get("igbo", "")
            # Only include well-diacritized sentences
            if text and len(text) > 20:
                # Calculate diacritization ratio
                text_lower = text.lower()
                dotted = sum(1 for c in text_lower if c in "ịọụ")
                undotted = sum(1 for c in text_lower if c in "iou")
                total = dotted + undotted
                if total > 0 and (dotted / total) >= 0.4:
                    texts.append(text)

        logger.info("Loaded %d high-quality Igbo test sentences", len(texts))
        return texts

    except Exception as e:
        logger.warning("Failed to load JW300 test data: %s", e)

    # Fallback
    return [
        "Ọ bụ ezie na anyị na-anatakwa ihe ọbụla",
        "Kedụ ka ị mere?",
        "Chineke nọnyeere anyị",
        "Ụmụaka na-amụ ihe",
    ]


def save_evaluation_results(
    metrics: DiacritizerMetrics,
    path: Optional[Path] = None
) -> Path:
    """Save evaluation results to JSON file.

    Args:
        metrics: Evaluation metrics to save.
        path: Path to save to. If None, uses default location.

    Returns:
        Path where results were saved.
    """
    if path is None:
        path = Path(__file__).parent / "evaluation_results.json"

    # Load existing results or create new
    if path.exists():
        with open(path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {"evaluations": []}

    # Add new result
    all_results["evaluations"].append(metrics.to_dict())
    all_results["last_updated"] = datetime.now().isoformat()

    # Save
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info("Saved evaluation results to %s", path)
    return path


def run_all_evaluations(test_size: int = 500) -> Dict[str, DiacritizerMetrics]:
    """Run evaluation on all diacritizer models.

    Args:
        test_size: Number of test sentences per model.

    Returns:
        Dictionary mapping language to metrics.
    """
    results = {}

    print("=" * 60)
    print("NAIJAML DIACRITIZER EVALUATION")
    print("=" * 60)

    for language in ["yoruba", "igbo"]:
        print(f"\nEvaluating {language.upper()} diacritizer...")
        metrics = evaluate_diacritizer(language, test_size=test_size)
        results[language] = metrics
        print(metrics.summary())

    # Save results
    save_evaluation_results(results["yoruba"])
    save_evaluation_results(results["igbo"])

    return results
