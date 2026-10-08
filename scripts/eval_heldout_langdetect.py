"""Evaluate language detection on the NaijaSenti and MasakhaNEWS test splits.

The detector is trained on the train and validation splits of NaijaSenti,
MasakhaNEWS, NollySenti and MasakhaNER, so the test splits are held out.

Three test sets are reported separately because they differ a lot in difficulty:

- NaijaSenti test: tweets in Yoruba, Hausa, Igbo and Pidgin. Tweets are short
  and often code-mixed, and the label is the language the tweet was collected
  for, so a Pidgin-labelled tweet written in plain English counts as an error.
- MasakhaNEWS test, headlines: one short line per article, five languages.
- MasakhaNEWS test, articles: the full article text, five languages.

The test sets are downloaded on first run and cached; they are not
redistributed with NaijaML. Results are saved to evaluation_reports/.

Usage:
    python scripts/eval_heldout_langdetect.py
    python scripts/eval_heldout_langdetect.py --limit 200
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Windows pipes and older consoles default to a legacy code page; Yorùbá/Igbo text needs UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR = PROJECT_ROOT / "evaluation_reports"

NIGERIAN_LANGS = ["yor", "hau", "ibo", "pcm"]
MASAKHANEWS_ENG_TEST_URL = (
    "https://huggingface.co/datasets/masakhane/masakhanews/resolve/main/data/eng/test.tsv"
)

Sample = Tuple[str, str]  # (text, language code)


def load_masakhanews_eng_test() -> List[Dict[str, str]]:
    """Download (once) the English MasakhaNEWS test split, which the package loader does not cover."""
    from naijaml.utils.download import get_models_cache_dir

    cache = get_models_cache_dir().parent / "eval" / "masakhanews_eng_test.tsv"
    if not cache.exists():
        import requests

        logger.info("Downloading MasakhaNEWS English test set from %s ...", MASAKHANEWS_ENG_TEST_URL)
        response = requests.get(MASAKHANEWS_ENG_TEST_URL, timeout=120)
        response.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(response.content)

    csv.field_size_limit(10 ** 9)
    reader = csv.DictReader(io.StringIO(cache.read_text(encoding="utf-8")), delimiter="\t")
    return [row for row in reader if (row.get("text") or "").strip()]


def load_test_sets(limit: int = 0) -> Dict[str, List[Sample]]:
    from naijaml.data import load_dataset

    tweets = []  # type: List[Sample]
    headlines = []  # type: List[Sample]
    articles = []  # type: List[Sample]

    def take(rows: list) -> list:
        return rows[:limit] if limit else rows

    for lang in NIGERIAN_LANGS:
        for item in take(load_dataset("naijasenti", lang=lang, split="test")):
            tweets.append((item["text"], lang))
        for item in take(load_dataset("masakhanews", lang=lang, split="test")):
            articles.append((item["text"], lang))
            if item.get("headline"):
                headlines.append((item["headline"], lang))

    for row in take(load_masakhanews_eng_test()):
        articles.append((row["text"].strip(), "eng"))
        if (row.get("headline") or "").strip():
            headlines.append((row["headline"].strip(), "eng"))

    return {
        "NaijaSenti test (tweets)": tweets,
        "MasakhaNEWS test (headlines)": headlines,
        "MasakhaNEWS test (articles)": articles,
    }


def score(samples: List[Sample]) -> Dict:
    """Accuracy, per-language recall and precision, and the confusion counts."""
    from naijaml.nlp.langdetect import detect_language

    confusion = defaultdict(Counter)  # type: Dict[str, Counter]
    for text, lang in samples:
        confusion[lang][detect_language(text) or "none"] += 1

    predicted_totals = Counter()  # type: Counter
    for row in confusion.values():
        predicted_totals.update(row)

    per_language = {}
    for lang, row in confusion.items():
        total = sum(row.values())
        per_language[lang] = {
            "samples": total,
            "recall": row[lang] / total,
            "precision": row[lang] / predicted_totals[lang] if predicted_totals[lang] else None,
        }

    correct = sum(row[lang] for lang, row in confusion.items())
    return {
        "samples": len(samples),
        "accuracy": correct / len(samples),
        "macro_recall": sum(v["recall"] for v in per_language.values()) / len(per_language),
        "per_language": per_language,
        "confusion": {lang: dict(row) for lang, row in confusion.items()},
    }


def print_report(result: Dict) -> None:
    print("\n" + "=" * 72)
    print("LANGUAGE DETECTION - HELD-OUT EVALUATION")
    print("=" * 72)
    for name, r in result["results"].items():
        print("\n{}: {:.1%} accuracy, {:.1%} macro recall, {:,} samples".format(
            name, r["accuracy"], r["macro_recall"], r["samples"]))
        print("  {:<6} {:>8} {:>8} {:>10}   {}".format("Lang", "Samples", "Recall", "Precision", "Predicted as"))
        for lang, v in r["per_language"].items():
            precision = "{:.1%}".format(v["precision"]) if v["precision"] is not None else "-"
            predicted = ", ".join(
                "{} {}".format(k, n) for k, n in sorted(r["confusion"][lang].items(), key=lambda kv: -kv[1]))
            print("  {:<6} {:>8,} {:>7.1%} {:>10}   {}".format(lang, v["samples"], v["recall"], precision, predicted))


def main() -> Dict:
    parser = argparse.ArgumentParser(description="Evaluate language detection on held-out test splits")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N samples per language")
    args = parser.parse_args()

    result = {
        "date": datetime.now().isoformat(),
        "results": {name: score(samples) for name, samples in load_test_sets(args.limit).items()},
    }
    print_report(result)

    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / "{}_heldout_langdetect.json".format(datetime.now().strftime("%Y%m%d"))
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\nReport saved to: {}".format(report_path))
    return result


if __name__ == "__main__":
    main()
