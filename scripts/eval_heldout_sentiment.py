"""Evaluate sentiment analysis on the NaijaSenti test split.

The sentiment model is trained on the NaijaSenti train split (see
scripts/train_tfidf_sentiment.py), so the test split is held out.

Reports accuracy and per-class recall and precision, overall and for each
language. The test set is downloaded on first run and cached; it is not
redistributed with NaijaML. Results are saved to evaluation_reports/.

Usage:
    python scripts/eval_heldout_sentiment.py
    python scripts/eval_heldout_sentiment.py --model path/to/sentiment_model.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Windows pipes and older consoles default to a legacy code page; Yorùbá/Igbo text needs UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR = PROJECT_ROOT / "evaluation_reports"

LANGS = ["yor", "hau", "ibo", "pcm"]
LABELS = ["positive", "neutral", "negative"]


def score(pairs: List[Tuple[str, str]]) -> Dict:
    """Accuracy and per-class recall and precision for (gold, predicted) pairs."""
    confusion = defaultdict(Counter)  # type: Dict[str, Counter]
    for gold, predicted in pairs:
        confusion[gold][predicted] += 1

    per_class = {}
    for label in LABELS:
        support = sum(confusion[label].values())
        predicted_total = sum(confusion[gold][label] for gold in LABELS)
        per_class[label] = {
            "support": support,
            "recall": confusion[label][label] / support if support else None,
            "precision": confusion[label][label] / predicted_total if predicted_total else None,
        }

    recalls = [v["recall"] for v in per_class.values() if v["recall"] is not None]
    return {
        "samples": len(pairs),
        "accuracy": sum(confusion[label][label] for label in LABELS) / len(pairs),
        "macro_recall": sum(recalls) / len(recalls),
        "per_class": per_class,
        "confusion": {gold: dict(row) for gold, row in confusion.items()},
    }


def run_evaluation(model_path: Optional[Path] = None, limit: int = 0) -> Dict:
    import numpy as np

    from naijaml.data import load_dataset
    from naijaml.nlp import sentiment

    if model_path is not None:
        with open(model_path, encoding="utf-8") as f:
            model = json.load(f)
        for key in ("idf", "coef", "intercept"):
            model[key] = np.array(model[key])
        sentiment._model = model

    by_lang = {}
    for lang in LANGS:
        data = load_dataset("naijasenti", lang=lang, split="test")
        if limit:
            data = data[:limit]
        by_lang[lang] = [(item["label"], sentiment.get_sentiment(item["text"])) for item in data]

    results = {"all": score([pair for pairs in by_lang.values() for pair in pairs])}
    results.update({lang: score(pairs) for lang, pairs in by_lang.items()})
    return {
        "dataset": "NaijaSenti test",
        "model": str(model_path) if model_path else "bundled",
        "results": results,
    }


def print_report(result: Dict) -> None:
    def pct(value: Optional[float]) -> str:
        return "{:.1%}".format(value) if value is not None else "-"

    print("\n" + "=" * 78)
    print("SENTIMENT - HELD-OUT EVALUATION ({}, {} model)".format(result["dataset"], result["model"]))
    print("=" * 78)
    print("\n{:<5} {:>8} {:>9} {:>13}   {}".format("Lang", "Samples", "Accuracy", "Macro recall", "Recall / precision per class"))
    print("-" * 78)
    for name, r in result["results"].items():
        classes = "  ".join(
            "{} {}/{}".format(label[:3], pct(r["per_class"][label]["recall"]), pct(r["per_class"][label]["precision"]))
            for label in LABELS)
        print("{:<5} {:>8,} {:>8.1%} {:>12.1%}   {}".format(name, r["samples"], r["accuracy"], r["macro_recall"], classes))


def main() -> Dict:
    parser = argparse.ArgumentParser(description="Evaluate sentiment analysis on the NaijaSenti test split")
    parser.add_argument("--model", type=Path, default=None,
                        help="Evaluate this sentiment_model.json instead of the bundled one")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N tweets per language")
    args = parser.parse_args()

    result = run_evaluation(model_path=args.model, limit=args.limit)
    result["date"] = datetime.now().isoformat()
    print_report(result)

    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / "{}_heldout_sentiment.json".format(datetime.now().strftime("%Y%m%d"))
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\nReport saved to: {}".format(report_path))
    return result


if __name__ == "__main__":
    main()
