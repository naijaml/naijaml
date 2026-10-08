"""Evaluate language detection on unseen data.

Produces per-language accuracy, precision, recall, F1, confusion matrix,
and short-text vs long-text breakdown. Results saved to evaluation_reports/.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Windows pipes and older consoles default to a legacy code page; Yorùbá/Igbo text and "→" need UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
REPORTS_DIR = PROJECT_ROOT / "evaluation_reports"
REPORTS_DIR.mkdir(exist_ok=True)


def load_fixture_data() -> List[Dict]:
    """Load evaluation data from fixtures."""
    samples = []

    # 1. Language samples fixture (10 per language)
    lang_file = FIXTURES_DIR / "language_samples.json"
    if lang_file.exists():
        with open(lang_file, encoding="utf-8") as f:
            lang_data = json.load(f)
        for lang_code, texts in lang_data.items():
            # Map fixture codes to our codes
            code_map = {"hau": "hau", "yor": "yor", "ibo": "ibo", "pcm": "pcm", "eng": "eng"}
            mapped = code_map.get(lang_code, lang_code)
            for text in texts:
                samples.append({"text": text, "lang": mapped, "source": "language_samples"})

    # 2. Code-mixed samples (for code-switching evaluation)
    mixed_file = FIXTURES_DIR / "eval" / "code_mixed_samples.json"
    if mixed_file.exists():
        with open(mixed_file, encoding="utf-8") as f:
            mixed_data = json.load(f)
        for item in mixed_data["samples"]:
            samples.append({
                "text": item["text"],
                "lang": item["primary"],
                "source": "code_mixed",
                "is_mixed": True,
            })

    # 3. Sentiment fixture texts (have lang labels)
    sent_file = FIXTURES_DIR / "eval" / "unseen_sentiment.json"
    if sent_file.exists():
        with open(sent_file, encoding="utf-8") as f:
            sent_data = json.load(f)
        for item in sent_data["samples"]:
            samples.append({
                "text": item["text"],
                "lang": item["lang"],
                "source": "sentiment_fixture",
            })

    # 4. Yoruba diacritizer sentences (should detect as yor)
    yor_file = FIXTURES_DIR / "eval" / "unseen_yoruba_sentences.json"
    if yor_file.exists():
        with open(yor_file, encoding="utf-8") as f:
            yor_data = json.load(f)
        for text in yor_data["sentences"][:20]:  # Use first 20
            samples.append({"text": text, "lang": "yor", "source": "yoruba_diacritizer"})

    # 5. Igbo diacritizer sentences (should detect as ibo)
    ibo_file = FIXTURES_DIR / "eval" / "unseen_igbo_sentences.json"
    if ibo_file.exists():
        with open(ibo_file, encoding="utf-8") as f:
            ibo_data = json.load(f)
        for text in ibo_data["sentences"][:20]:
            samples.append({"text": text, "lang": "ibo", "source": "igbo_diacritizer"})

    return samples


def compute_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    """Compute per-class and overall metrics."""
    # Confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    for true, pred in zip(y_true, y_pred):
        confusion[true][pred] += 1

    # Per-class metrics
    per_class = {}
    total_correct = 0
    total = len(y_true)

    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        total_correct += tp

    accuracy = total_correct / total if total > 0 else 0.0

    # Macro averages
    macro_precision = sum(m["precision"] for m in per_class.values()) / len(labels)
    macro_recall = sum(m["recall"] for m in per_class.values()) / len(labels)
    macro_f1 = sum(m["f1"] for m in per_class.values()) / len(labels)

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "total_samples": total,
    }


def evaluate_langdetect() -> Dict:
    """Run full language detection evaluation."""
    from naijaml.nlp import detect_language, detect_language_with_confidence

    samples = load_fixture_data()
    if not samples:
        logger.error("No evaluation data found!")
        return {}

    logger.info("Evaluating on %d samples", len(samples))

    labels = ["yor", "hau", "ibo", "pcm", "eng"]
    y_true = []
    y_pred = []
    y_conf = []
    errors = []

    # Track by source and text length
    by_source = defaultdict(lambda: {"true": [], "pred": []})
    by_length = {"short": {"true": [], "pred": []}, "long": {"true": [], "pred": []}}

    for sample in samples:
        text = sample["text"]
        true_lang = sample["lang"]
        if true_lang not in labels:
            continue

        pred_lang = detect_language(text)
        _, confidence = detect_language_with_confidence(text)

        y_true.append(true_lang)
        y_pred.append(pred_lang)
        y_conf.append(confidence)

        source = sample.get("source", "unknown")
        by_source[source]["true"].append(true_lang)
        by_source[source]["pred"].append(pred_lang)

        length_cat = "short" if len(text.split()) < 10 else "long"
        by_length[length_cat]["true"].append(true_lang)
        by_length[length_cat]["pred"].append(pred_lang)

        if pred_lang != true_lang:
            errors.append({
                "text": text[:100],
                "true": true_lang,
                "predicted": pred_lang,
                "confidence": round(confidence, 4),
                "source": source,
            })

    # Overall metrics
    overall = compute_metrics(y_true, y_pred, labels)

    # Per-source metrics
    source_metrics = {}
    for source, data in by_source.items():
        source_metrics[source] = compute_metrics(data["true"], data["pred"], labels)

    # Short vs long text
    length_metrics = {}
    for length_cat, data in by_length.items():
        if data["true"]:
            length_metrics[length_cat] = compute_metrics(data["true"], data["pred"], labels)

    # Confidence calibration (binned)
    conf_bins = defaultdict(lambda: {"correct": 0, "total": 0})
    for true, pred, conf in zip(y_true, y_pred, y_conf):
        bin_key = round(conf, 1)
        conf_bins[bin_key]["total"] += 1
        if true == pred:
            conf_bins[bin_key]["correct"] += 1

    calibration = {}
    for bin_key in sorted(conf_bins.keys()):
        data = conf_bins[bin_key]
        calibration[str(bin_key)] = {
            "accuracy": round(data["correct"] / data["total"], 4) if data["total"] > 0 else 0,
            "avg_confidence": bin_key,
            "count": data["total"],
        }

    result = {
        "module": "langdetect",
        "date": datetime.now().isoformat(),
        "overall": overall,
        "by_source": source_metrics,
        "by_text_length": length_metrics,
        "confidence_calibration": calibration,
        "sample_errors": errors[:30],
    }

    return result


def print_report(result: Dict) -> None:
    """Print human-readable report."""
    if not result:
        return

    overall = result["overall"]
    print("\n" + "=" * 60)
    print("LANGUAGE DETECTION EVALUATION")
    print("=" * 60)
    print(f"Total samples: {overall['total_samples']}")
    print(f"Overall accuracy: {overall['accuracy']:.1%}")
    print(f"Macro F1: {overall['macro_f1']:.1%}")

    print("\nPer-language metrics:")
    print(f"{'Language':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 56)
    for lang, metrics in overall["per_class"].items():
        print(f"{lang:<10} {metrics['precision']:<12.1%} {metrics['recall']:<12.1%} "
              f"{metrics['f1']:<12.1%} {metrics['support']:<10}")

    print("\nConfusion matrix:")
    labels = sorted(overall["confusion_matrix"].keys())
    corner = "True\\Pred"  # a backslash inside an f-string expression needs Python 3.12
    print(f"{corner:<10}", end="")
    for label in ["yor", "hau", "ibo", "pcm", "eng"]:
        if label in labels or any(label in v for v in overall["confusion_matrix"].values()):
            print(f"{label:<8}", end="")
    print()
    for true_label in ["yor", "hau", "ibo", "pcm", "eng"]:
        if true_label in overall["confusion_matrix"]:
            print(f"{true_label:<10}", end="")
            for pred_label in ["yor", "hau", "ibo", "pcm", "eng"]:
                count = overall["confusion_matrix"][true_label].get(pred_label, 0)
                print(f"{count:<8}", end="")
            print()

    if result.get("by_text_length"):
        print("\nBy text length:")
        for length, metrics in result["by_text_length"].items():
            print(f"  {length} (<10 words): accuracy={metrics['accuracy']:.1%}, n={metrics['total_samples']}")

    if result.get("sample_errors"):
        print(f"\nSample errors ({len(result['sample_errors'])} shown):")
        for err in result["sample_errors"][:10]:
            print(f"  [{err['true']}→{err['predicted']}] \"{err['text'][:60]}...\" (conf={err['confidence']:.2f})")


def main():
    result = evaluate_langdetect()
    print_report(result)

    # Save report
    report_path = REPORTS_DIR / f"{datetime.now().strftime('%Y%m%d')}_langdetect.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Report saved to %s", report_path)

    return result


if __name__ == "__main__":
    main()
