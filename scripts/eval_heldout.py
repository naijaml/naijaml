"""Evaluate the Yoruba diacritizers on the MENYO-20k test set.

MENYO-20k (Adelani et al., 2021) is a multi-domain Yoruba-English corpus with
carefully diacritized Yoruba. Its test set is not part of the diacritizer's
training data, apart from 109 sentences that also occur in it; those are
excluded by default (see tests/fixtures/eval/menyo_test_train_overlap.json,
which stores their SHA-1 hashes).

The test set is downloaded on first run and cached; it is not redistributed
with NaijaML. Results are saved to evaluation_reports/.

Usage:
    python scripts/eval_heldout.py
    python scripts/eval_heldout.py --keep-overlap --limit 500
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

# Windows pipes and older consoles default to a legacy code page; Yorùbá text needs UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "eval"
REPORTS_DIR = PROJECT_ROOT / "evaluation_reports"

MENYO_TEST_URL = "https://raw.githubusercontent.com/uds-lsv/menyo-20k_MT/master/data/test.tsv"
OVERLAP_FILE = FIXTURES_DIR / "menyo_test_train_overlap.json"


def normalize(text: str) -> str:
    """NFC-normalize, lowercase and collapse whitespace."""
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def sentence_hash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()


def load_menyo_test(keep_overlap: bool = False) -> List[str]:
    """Download (once) and return the diacritized Yoruba side of the MENYO-20k test set."""
    from naijaml.utils.download import get_models_cache_dir

    cache = get_models_cache_dir().parent / "eval" / "menyo20k_test.tsv"
    if not cache.exists():
        import requests

        logger.info("Downloading MENYO-20k test set from %s ...", MENYO_TEST_URL)
        response = requests.get(MENYO_TEST_URL, timeout=120)
        response.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(response.content)

    rows = csv.reader(io.StringIO(cache.read_text(encoding="utf-8")), delimiter="\t", quoting=csv.QUOTE_NONE)
    sentences = [row[1].strip() for row in list(rows)[1:] if len(row) > 1 and row[1].strip()]

    if not keep_overlap:
        with open(OVERLAP_FILE, encoding="utf-8") as f:
            overlap = set(json.load(f)["sha1"])
        sentences = [s for s in sentences if sentence_hash(s) not in overlap]

    return sentences


def score(gold: List[str], predicted: List[str], strip_fn: Callable[[str], str]) -> Dict:
    """Word and sentence accuracy of predicted against gold, compared after normalization."""
    total_words = correct_words = correct_sentences = 0
    marked_words = correct_marked = 0
    errors = []

    for expected, got in zip(gold, predicted):
        exp_words, got_words = normalize(expected).split(), normalize(got).split()
        if exp_words == got_words:
            correct_sentences += 1
        elif len(errors) < 20:
            errors.append({"expected": expected, "predicted": got})

        for ew, gw in zip(exp_words, got_words):
            total_words += 1
            correct_words += ew == gw
            if ew != normalize(strip_fn(ew)):  # the gold word carries at least one mark
                marked_words += 1
                correct_marked += ew == gw
        total_words += abs(len(exp_words) - len(got_words))

    return {
        "word_accuracy": correct_words / total_words,
        "sentence_accuracy": correct_sentences / len(gold),
        "accuracy_on_marked_words": correct_marked / marked_words if marked_words else None,
        "share_of_words_marked": marked_words / total_words,
        "total_words": total_words,
        "total_sentences": len(gold),
        "sample_errors": errors,
    }


def run_evaluation(keep_overlap: bool = False, limit: int = 0) -> Dict:
    from naijaml.nlp.diacritizer import diacritize, diacritize_dot_below, strip_diacritics

    gold = load_menyo_test(keep_overlap=keep_overlap)
    if limit:
        gold = gold[:limit]
    inputs = [strip_diacritics(s) for s in gold]
    gold_dot_below = [strip_diacritics(s, tones_only=True) for s in gold]

    def strip_dots(word: str) -> str:
        return strip_diacritics(word)

    systems = {
        "full: no restoration (baseline)": (gold, inputs),
        "full: diacritize": (gold, [diacritize(s) for s in inputs]),
        "dot-below: no restoration (baseline)": (gold_dot_below, inputs),
        "dot-below: diacritize_dot_below": (gold_dot_below, [diacritize_dot_below(s) for s in inputs]),
        "dot-below: bundled syllable model": (
            gold_dot_below, [diacritize_dot_below(s, use_word_level=False) for s in inputs]),
    }
    return {
        "dataset": "MENYO-20k test",
        "train_overlap_excluded": not keep_overlap,
        "sentences": len(gold),
        "results": {name: score(g, p, strip_dots) for name, (g, p) in systems.items()},
    }


def print_report(result: Dict) -> None:
    print("\n" + "=" * 72)
    print("YORUBA DIACRITIZER - HELD-OUT EVALUATION ({})".format(result["dataset"]))
    print("=" * 72)
    print("Sentences: {} (training overlap {})".format(
        result["sentences"], "excluded" if result["train_overlap_excluded"] else "kept"))
    print("\n{:<40} {:>10} {:>10} {:>9}".format("System", "Word acc", "Sent acc", "Words"))
    print("-" * 72)
    for name, r in result["results"].items():
        print("{:<40} {:>9.1%} {:>9.1%} {:>9,}".format(
            name, r["word_accuracy"], r["sentence_accuracy"], r["total_words"]))


def main() -> Dict:
    parser = argparse.ArgumentParser(description="Evaluate Yoruba diacritizers on MENYO-20k test")
    parser.add_argument("--keep-overlap", action="store_true",
                        help="Keep the 109 test sentences that also occur in the training data")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N sentences")
    args = parser.parse_args()

    result = run_evaluation(keep_overlap=args.keep_overlap, limit=args.limit)
    result["date"] = datetime.now().isoformat()
    print_report(result)

    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / "{}_heldout.json".format(datetime.now().strftime("%Y%m%d"))
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\nReport saved to: {}".format(report_path))
    return result


if __name__ == "__main__":
    main()
