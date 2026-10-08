"""Evaluate the Igbo diacritizer on the MasakhaNER 2.0 Igbo test split.

The Igbo diacritizer is trained on JW300 and on all splits of MasakhaNEWS, so
neither can serve as a test set. MasakhaNER 2.0 (Adelani et al., 2022) is not
part of its training data.

The diacritizer restores dot-below vowels (ị, ọ, ụ) only, so tone marks and
the dot on ṅ are removed from the reference before comparing.

The test set is downloaded on first run and cached; it is not redistributed
with NaijaML. Results are saved to evaluation_reports/.

Usage:
    python scripts/eval_heldout_igbo.py
    python scripts/eval_heldout_igbo.py --limit 500
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from datetime import datetime
from typing import Dict

from eval_heldout import REPORTS_DIR, score

DOT_BELOW = "̣"


def keep_dot_below_only(text: str) -> str:
    """Remove every combining mark except dot-below."""
    decomposed = unicodedata.normalize("NFD", text)
    kept = [c for c in decomposed if unicodedata.category(c) != "Mn" or c == DOT_BELOW]
    return unicodedata.normalize("NFC", "".join(kept))


def run_evaluation(limit: int = 0) -> Dict:
    from naijaml.data import load_dataset
    from naijaml.nlp.igbo_diacritizer import diacritize_igbo, strip_diacritics

    sentences = [" ".join(item["tokens"]) for item in load_dataset("masakhaner", lang="ibo", split="test")]
    if limit:
        sentences = sentences[:limit]
    gold = [keep_dot_below_only(s) for s in sentences]
    inputs = [strip_diacritics(s) for s in sentences]

    systems = {
        "no restoration (baseline)": inputs,
        "diacritize_igbo": [diacritize_igbo(s) for s in inputs],
    }
    return {
        "dataset": "MasakhaNER 2.0 Igbo test",
        "sentences": len(gold),
        "results": {name: score(gold, predicted, strip_diacritics) for name, predicted in systems.items()},
    }


def print_report(result: Dict) -> None:
    print("\n" + "=" * 72)
    print("IGBO DIACRITIZER - HELD-OUT EVALUATION ({})".format(result["dataset"]))
    print("=" * 72)
    print("Sentences: {}".format(result["sentences"]))
    print("\n{:<28} {:>10} {:>10} {:>14} {:>9}".format("System", "Word acc", "Sent acc", "Marked words", "Words"))
    print("-" * 76)
    for name, r in result["results"].items():
        print("{:<28} {:>9.1%} {:>9.1%} {:>13.1%} {:>9,}".format(
            name, r["word_accuracy"], r["sentence_accuracy"], r["accuracy_on_marked_words"], r["total_words"]))


def main() -> Dict:
    parser = argparse.ArgumentParser(description="Evaluate the Igbo diacritizer on MasakhaNER 2.0 test")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N sentences")
    args = parser.parse_args()

    result = run_evaluation(limit=args.limit)
    result["date"] = datetime.now().isoformat()
    print_report(result)

    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / "{}_heldout_igbo.json".format(datetime.now().strftime("%Y%m%d"))
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\nReport saved to: {}".format(report_path))
    return result


if __name__ == "__main__":
    main()
