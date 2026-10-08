"""Evaluate tokenizers: fertility rate, coverage, roundtrip fidelity, diacritic preservation.

Results saved to evaluation_reports/.
"""
from __future__ import annotations

import json
import logging
import sys
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
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
REPORTS_DIR = PROJECT_ROOT / "evaluation_reports"
REPORTS_DIR.mkdir(exist_ok=True)


def load_test_texts() -> Dict[str, List[str]]:
    """Load test texts per language from fixtures."""
    texts = {}

    # Language samples
    lang_file = FIXTURES_DIR / "language_samples.json"
    if lang_file.exists():
        with open(lang_file, encoding="utf-8") as f:
            texts = json.load(f)

    # Add diacritized Yoruba
    yor_file = FIXTURES_DIR / "eval" / "unseen_yoruba_sentences.json"
    if yor_file.exists():
        with open(yor_file, encoding="utf-8") as f:
            data = json.load(f)
        texts.setdefault("yor", []).extend(data["sentences"][:20])

    # Add diacritized Igbo
    ibo_file = FIXTURES_DIR / "eval" / "unseen_igbo_sentences.json"
    if ibo_file.exists():
        with open(ibo_file, encoding="utf-8") as f:
            data = json.load(f)
        texts.setdefault("ibo", []).extend(data["sentences"][:20])

    return texts


def evaluate_tokenizer_for_lang(lang: str, texts: List[str]) -> Dict:
    """Evaluate tokenizer for a specific language."""
    from naijaml.nlp.tokenizer import Tokenizer

    # Map fixture lang codes to tokenizer lang codes
    lang_map = {"yor": "yoruba", "hau": "hausa", "ibo": "igbo", "pcm": "pidgin", "eng": "naija"}
    tok_lang = lang_map.get(lang)
    if not tok_lang:
        return {"error": f"No tokenizer for {lang}"}

    try:
        tok = Tokenizer(tok_lang)
    except Exception as e:
        return {"error": str(e)}

    total_words = 0
    total_tokens = 0
    roundtrip_pass = 0
    roundtrip_fail = 0
    diacritic_preserved = 0
    diacritic_total = 0
    roundtrip_errors = []

    diacritic_chars = set("ọẹṣáàéèíìóòúùọ́ọ̀ẹ́ẹ̀ịụñ")

    for text in texts:
        words = text.split()
        total_words += len(words)

        # Encode
        token_ids = tok.encode(text)
        total_tokens += len(token_ids)

        # Roundtrip fidelity
        decoded = tok.decode(token_ids)
        if decoded.strip() == text.strip():
            roundtrip_pass += 1
        else:
            roundtrip_fail += 1
            if len(roundtrip_errors) < 10:
                roundtrip_errors.append({
                    "original": text[:80],
                    "decoded": decoded[:80],
                })

        # Diacritic preservation check
        for char in text:
            if char.lower() in diacritic_chars:
                diacritic_total += 1
                if char in decoded:
                    diacritic_preserved += 1

    fertility = total_tokens / total_words if total_words > 0 else 0
    roundtrip_rate = roundtrip_pass / (roundtrip_pass + roundtrip_fail) if (roundtrip_pass + roundtrip_fail) > 0 else 0
    diacritic_rate = diacritic_preserved / diacritic_total if diacritic_total > 0 else 1.0

    return {
        "language": lang,
        "total_texts": len(texts),
        "total_words": total_words,
        "total_tokens": total_tokens,
        "fertility_rate": round(fertility, 4),
        "roundtrip_fidelity": round(roundtrip_rate, 4),
        "roundtrip_pass": roundtrip_pass,
        "roundtrip_fail": roundtrip_fail,
        "diacritic_preservation": round(diacritic_rate, 4),
        "diacritic_chars_tested": diacritic_total,
        "roundtrip_errors": roundtrip_errors,
    }


def evaluate_tokenizers() -> Dict:
    """Run tokenizer evaluation across all languages."""
    texts_by_lang = load_test_texts()
    if not texts_by_lang:
        logger.error("No test texts found!")
        return {}

    results = {}
    for lang, texts in texts_by_lang.items():
        logger.info("Evaluating tokenizer for %s (%d texts)", lang, len(texts))
        results[lang] = evaluate_tokenizer_for_lang(lang, texts)

    return {
        "module": "tokenizer",
        "date": datetime.now().isoformat(),
        "per_language": results,
    }


def print_report(result: Dict) -> None:
    """Print human-readable report."""
    if not result:
        return

    print("\n" + "=" * 60)
    print("TOKENIZER EVALUATION")
    print("=" * 60)

    print(f"\n{'Language':<10} {'Fertility':<12} {'Roundtrip':<12} {'Diacritics':<12} {'Texts':<8}")
    print("-" * 54)

    for lang, metrics in result.get("per_language", {}).items():
        if "error" in metrics:
            print(f"{lang:<10} ERROR: {metrics['error']}")
            continue
        print(f"{lang:<10} {metrics['fertility_rate']:<12.2f} "
              f"{metrics['roundtrip_fidelity']:<12.1%} "
              f"{metrics['diacritic_preservation']:<12.1%} "
              f"{metrics['total_texts']:<8}")

    # Show roundtrip errors if any
    for lang, metrics in result.get("per_language", {}).items():
        errors = metrics.get("roundtrip_errors", [])
        if errors:
            print(f"\nRoundtrip errors for {lang}:")
            for err in errors[:3]:
                print(f"  Original: {err['original'][:60]}")
                print(f"  Decoded:  {err['decoded'][:60]}")
                print()


def main():
    result = evaluate_tokenizers()
    print_report(result)

    report_path = REPORTS_DIR / f"{datetime.now().strftime('%Y%m%d')}_tokenizer.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Report saved to %s", report_path)

    return result


if __name__ == "__main__":
    main()
