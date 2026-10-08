"""Evaluate Yoruba and Igbo diacritizers on unseen data.

Measures word/sentence/character accuracy, diacritic precision/recall/F1,
homograph resolution accuracy. Results saved to evaluation_reports/.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Windows pipes and older consoles default to a legacy code page; Yorùbá/Igbo text and "→" need UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "eval"
REPORTS_DIR = PROJECT_ROOT / "evaluation_reports"
REPORTS_DIR.mkdir(exist_ok=True)


def strip_yoruba_diacritics(text: str) -> str:
    """Strip Yoruba diacritics from text."""
    from naijaml.nlp.diacritizer import strip_diacritics
    return strip_diacritics(text)


def strip_igbo_diacritics(text: str) -> str:
    """Strip Igbo diacritics from text."""
    from naijaml.nlp.igbo_diacritizer import strip_diacritics
    return strip_diacritics(text)


def evaluate_word_accuracy(expected_sentences: List[str], strip_fn, diacritize_fn) -> Dict:
    """Evaluate word-level, sentence-level, and character-level accuracy."""
    total_words = 0
    correct_words = 0
    total_sentences = len(expected_sentences)
    correct_sentences = 0
    total_chars = 0
    correct_chars = 0
    errors = []

    for expected in expected_sentences:
        stripped = strip_fn(expected)
        predicted = diacritize_fn(stripped)

        # Sentence accuracy
        if predicted.lower() == expected.lower():
            correct_sentences += 1
        else:
            if len(errors) < 30:
                errors.append({
                    "input": stripped,
                    "predicted": predicted,
                    "expected": expected,
                })

        # Word accuracy
        pred_words = predicted.lower().split()
        exp_words = expected.lower().split()
        for pw, ew in zip(pred_words, exp_words):
            total_words += 1
            if pw == ew:
                correct_words += 1
        # Count unmatched words from longer sentence
        total_words += abs(len(pred_words) - len(exp_words))

        # Character accuracy
        for pc, ec in zip(predicted.lower(), expected.lower()):
            total_chars += 1
            if pc == ec:
                correct_chars += 1
        total_chars += abs(len(predicted) - len(expected))

    return {
        "word_accuracy": round(correct_words / total_words, 4) if total_words > 0 else 0,
        "sentence_accuracy": round(correct_sentences / total_sentences, 4) if total_sentences > 0 else 0,
        "character_accuracy": round(correct_chars / total_chars, 4) if total_chars > 0 else 0,
        "total_words": total_words,
        "correct_words": correct_words,
        "total_sentences": total_sentences,
        "correct_sentences": correct_sentences,
        "sample_errors": errors,
    }


def evaluate_homographs(test_cases: List[Dict], diacritize_fn) -> Dict:
    """Evaluate homograph resolution accuracy."""
    total = 0
    correct = 0
    errors = []

    for case in test_cases:
        undiacritized = case["undiacritized"]
        expected = case["expected"]
        predicted = diacritize_fn(undiacritized)

        # Compare word by word
        pred_words = predicted.lower().split()
        exp_words = expected.lower().split()

        for pw, ew in zip(pred_words, exp_words):
            total += 1
            if pw == ew:
                correct += 1

        if predicted.lower() != expected.lower():
            if len(errors) < 20:
                errors.append({
                    "input": undiacritized,
                    "predicted": predicted,
                    "expected": expected,
                    "note": case.get("note", ""),
                })

    return {
        "homograph_word_accuracy": round(correct / total, 4) if total > 0 else 0,
        "total_homograph_words": total,
        "correct_homograph_words": correct,
        "total_cases": len(test_cases),
        "sample_errors": errors,
    }


def evaluate_yoruba() -> Dict:
    """Evaluate Yoruba diacritizer."""
    from naijaml.nlp.diacritizer import diacritize

    logger.info("Evaluating Yoruba diacritizer...")

    # Load unseen sentences
    yor_file = FIXTURES_DIR / "unseen_yoruba_sentences.json"
    if not yor_file.exists():
        logger.error("Missing fixture: %s", yor_file)
        return {}

    with open(yor_file, encoding="utf-8") as f:
        yor_data = json.load(f)
    sentences = yor_data["sentences"]

    # Word accuracy on unseen data
    accuracy = evaluate_word_accuracy(sentences, strip_yoruba_diacritics, diacritize)

    # Homograph evaluation
    homograph_file = FIXTURES_DIR / "homograph_test_cases.json"
    homograph_result = {}
    if homograph_file.exists():
        with open(homograph_file, encoding="utf-8") as f:
            homograph_data = json.load(f)
        homograph_result = evaluate_homographs(homograph_data["test_cases"], diacritize)

    return {
        "language": "yoruba",
        "unseen_accuracy": accuracy,
        "homograph_evaluation": homograph_result,
    }


def evaluate_igbo() -> Dict:
    """Evaluate Igbo diacritizer."""
    from naijaml.nlp.igbo_diacritizer import diacritize_igbo

    logger.info("Evaluating Igbo diacritizer...")

    ibo_file = FIXTURES_DIR / "unseen_igbo_sentences.json"
    if not ibo_file.exists():
        logger.error("Missing fixture: %s", ibo_file)
        return {}

    with open(ibo_file, encoding="utf-8") as f:
        ibo_data = json.load(f)
    sentences = ibo_data["sentences"]

    accuracy = evaluate_word_accuracy(sentences, strip_igbo_diacritics, diacritize_igbo)

    return {
        "language": "igbo",
        "unseen_accuracy": accuracy,
    }


def print_report(result: Dict) -> None:
    """Print human-readable report."""
    print("\n" + "=" * 60)
    print("DIACRITIZER EVALUATION")
    print("=" * 60)

    for lang_key in ["yoruba", "igbo"]:
        lang_result = result.get(lang_key)
        if not lang_result:
            continue

        acc = lang_result.get("unseen_accuracy", {})
        print(f"\n--- {lang_key.upper()} ---")
        print(f"Test sentences: {acc.get('total_sentences', 0)}")
        print(f"Word accuracy:      {acc.get('word_accuracy', 0):.1%} "
              f"({acc.get('correct_words', 0)}/{acc.get('total_words', 0)})")
        print(f"Sentence accuracy:  {acc.get('sentence_accuracy', 0):.1%} "
              f"({acc.get('correct_sentences', 0)}/{acc.get('total_sentences', 0)})")
        print(f"Character accuracy: {acc.get('character_accuracy', 0):.1%}")

        # Homograph results (Yoruba only)
        homo = lang_result.get("homograph_evaluation", {})
        if homo:
            print(f"\nHomograph resolution:")
            print(f"  Word accuracy: {homo.get('homograph_word_accuracy', 0):.1%} "
                  f"({homo.get('correct_homograph_words', 0)}/{homo.get('total_homograph_words', 0)})")

        # Sample errors
        errors = acc.get("sample_errors", [])
        if errors:
            print(f"\nSample errors ({len(errors)} shown):")
            for err in errors[:5]:
                print(f"  Input:    {err['input'][:70]}")
                print(f"  Expected: {err['expected'][:70]}")
                print(f"  Got:      {err['predicted'][:70]}")
                print()


def main():
    result = {
        "module": "diacritizer",
        "date": datetime.now().isoformat(),
    }

    result["yoruba"] = evaluate_yoruba()
    result["igbo"] = evaluate_igbo()

    print_report(result)

    # Save report
    report_path = REPORTS_DIR / f"{datetime.now().strftime('%Y%m%d')}_diacritizer.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Report saved to %s", report_path)

    return result


if __name__ == "__main__":
    main()
