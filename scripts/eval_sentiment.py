"""Evaluate sentiment analysis on unseen and cross-domain data.

Tests on curated fixtures covering fintech, product reviews, healthcare,
entertainment, and utility domains. Per-language and per-class metrics.
Results saved to evaluation_reports/.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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


def compute_classification_metrics(
    y_true: List[str], y_pred: List[str], labels: List[str]
) -> Dict:
    """Compute accuracy, per-class precision/recall/F1."""
    confusion = defaultdict(lambda: defaultdict(int))
    for true, pred in zip(y_true, y_pred):
        confusion[true][pred] += 1

    per_class = {}
    total_correct = 0

    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
        total_correct += tp

    total = len(y_true)
    accuracy = total_correct / total if total > 0 else 0.0
    macro_f1 = sum(m["f1"] for m in per_class.values()) / len(labels) if labels else 0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "total_samples": total,
    }


def evaluate_sentiment() -> Dict:
    """Run sentiment evaluation on unseen fixtures."""
    from naijaml.nlp import analyze_sentiment

    sent_file = FIXTURES_DIR / "unseen_sentiment.json"
    if not sent_file.exists():
        logger.error("Missing fixture: %s", sent_file)
        return {}

    with open(sent_file, encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    logger.info("Evaluating sentiment on %d samples", len(samples))

    sentiment_labels = ["positive", "negative", "neutral"]
    y_true_all = []
    y_pred_all = []
    y_conf_all = []
    errors = []

    # Track by language and domain
    by_lang = defaultdict(lambda: {"true": [], "pred": []})
    by_domain = defaultdict(lambda: {"true": [], "pred": []})

    for sample in samples:
        text = sample["text"]
        true_label = sample["label"]
        lang = sample.get("lang", "unknown")
        domain = sample.get("domain", "unknown")

        result = analyze_sentiment(text)
        pred_label = result["label"]
        confidence = result["confidence"]

        y_true_all.append(true_label)
        y_pred_all.append(pred_label)
        y_conf_all.append(confidence)

        by_lang[lang]["true"].append(true_label)
        by_lang[lang]["pred"].append(pred_label)
        by_domain[domain]["true"].append(true_label)
        by_domain[domain]["pred"].append(pred_label)

        if pred_label != true_label:
            errors.append({
                "text": text[:100],
                "true": true_label,
                "predicted": pred_label,
                "confidence": round(confidence, 4),
                "lang": lang,
                "domain": domain,
            })

    # Overall metrics
    overall = compute_classification_metrics(y_true_all, y_pred_all, sentiment_labels)

    # Per-language
    lang_metrics = {}
    for lang, data in by_lang.items():
        lang_metrics[lang] = compute_classification_metrics(data["true"], data["pred"], sentiment_labels)

    # Per-domain
    domain_metrics = {}
    for domain, data in by_domain.items():
        domain_metrics[domain] = compute_classification_metrics(data["true"], data["pred"], sentiment_labels)

    # Confidence calibration
    conf_bins = defaultdict(lambda: {"correct": 0, "total": 0})
    for true, pred, conf in zip(y_true_all, y_pred_all, y_conf_all):
        bin_key = round(conf, 1)
        conf_bins[bin_key]["total"] += 1
        if true == pred:
            conf_bins[bin_key]["correct"] += 1

    calibration = {}
    for bin_key in sorted(conf_bins.keys()):
        d = conf_bins[bin_key]
        calibration[str(bin_key)] = {
            "accuracy": round(d["correct"] / d["total"], 4) if d["total"] > 0 else 0,
            "avg_confidence": bin_key,
            "count": d["total"],
        }

    return {
        "module": "sentiment",
        "date": datetime.now().isoformat(),
        "overall": overall,
        "by_language": lang_metrics,
        "by_domain": domain_metrics,
        "confidence_calibration": calibration,
        "sample_errors": errors[:30],
    }


def print_report(result: Dict) -> None:
    """Print human-readable report."""
    if not result:
        return

    overall = result["overall"]
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS EVALUATION")
    print("=" * 60)
    print(f"Total samples: {overall['total_samples']}")
    print(f"Overall accuracy: {overall['accuracy']:.1%}")
    print(f"Macro F1: {overall['macro_f1']:.1%}")

    print("\nPer-class metrics:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 58)
    for cls, metrics in overall["per_class"].items():
        print(f"{cls:<12} {metrics['precision']:<12.1%} {metrics['recall']:<12.1%} "
              f"{metrics['f1']:<12.1%} {metrics['support']:<10}")

    print("\nPer-language accuracy:")
    for lang, metrics in result.get("by_language", {}).items():
        print(f"  {lang}: {metrics['accuracy']:.1%} (n={metrics['total_samples']})")

    print("\nPer-domain accuracy:")
    for domain, metrics in result.get("by_domain", {}).items():
        print(f"  {domain}: {metrics['accuracy']:.1%} (n={metrics['total_samples']})")

    errors = result.get("sample_errors", [])
    if errors:
        print(f"\nSample errors ({len(errors)} shown):")
        for err in errors[:8]:
            print(f"  [{err['true']}→{err['predicted']}] ({err['lang']}/{err['domain']}) "
                  f"\"{err['text'][:50]}...\"")


def main():
    result = evaluate_sentiment()
    print_report(result)

    report_path = REPORTS_DIR / f"{datetime.now().strftime('%Y%m%d')}_sentiment.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Report saved to %s", report_path)

    return result


if __name__ == "__main__":
    main()
