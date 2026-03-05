"""Master evaluation script for all NaijaML models and modules.

Runs all evaluation scripts and produces a unified JSON report.

Usage:
    uv run python scripts/evaluate_all.py
    uv run python scripts/evaluate_all.py --module langdetect
    uv run python scripts/evaluate_all.py --module sentiment,diacritizer
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR = PROJECT_ROOT / "evaluation_reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Add project to path
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def run_module(name: str, run_fn) -> dict:
    """Run an evaluation module with error handling and timing."""
    logger.info("=" * 40)
    logger.info("Running %s evaluation...", name)
    logger.info("=" * 40)

    start = time.time()
    try:
        result = run_fn()
        elapsed = time.time() - start
        result["_elapsed_seconds"] = round(elapsed, 2)
        result["_status"] = "success"
        logger.info("%s completed in %.1fs", name, elapsed)
        return result
    except Exception as e:
        elapsed = time.time() - start
        logger.error("%s failed: %s", name, e)
        traceback.print_exc()
        return {
            "_status": "error",
            "_error": str(e),
            "_elapsed_seconds": round(elapsed, 2),
        }


def main():
    parser = argparse.ArgumentParser(description="Run NaijaML evaluations")
    parser.add_argument(
        "--module", "-m",
        help="Comma-separated list of modules to evaluate (default: all). "
             "Options: langdetect, diacritizer, sentiment, tokenizer, preprocessing",
    )
    args = parser.parse_args()

    # Define all modules
    modules = {}

    def _langdetect():
        from eval_langdetect import evaluate_langdetect, print_report
        result = evaluate_langdetect()
        print_report(result)
        return result

    def _diacritizer():
        from eval_diacritizer import main as diac_main
        return diac_main()

    def _sentiment():
        from eval_sentiment import evaluate_sentiment, print_report
        result = evaluate_sentiment()
        print_report(result)
        return result

    def _tokenizer():
        from eval_tokenizer import evaluate_tokenizers, print_report
        result = evaluate_tokenizers()
        print_report(result)
        return result

    def _preprocessing():
        from eval_preprocessing import main as preproc_main
        return preproc_main()

    modules = {
        "langdetect": _langdetect,
        "diacritizer": _diacritizer,
        "sentiment": _sentiment,
        "tokenizer": _tokenizer,
        "preprocessing": _preprocessing,
    }

    # Filter modules if specified
    if args.module:
        selected = [m.strip() for m in args.module.split(",")]
        modules = {k: v for k, v in modules.items() if k in selected}
        if not modules:
            logger.error("No valid modules selected. Available: %s",
                         ", ".join(["langdetect", "diacritizer", "sentiment", "tokenizer", "preprocessing"]))
            sys.exit(1)

    # Run evaluations
    print("\n" + "=" * 60)
    print("NAIJAML FULL EVALUATION SUITE")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Modules: {', '.join(modules.keys())}")
    print("=" * 60)

    full_report = {
        "date": datetime.now().isoformat(),
        "modules_run": list(modules.keys()),
        "results": {},
    }

    total_start = time.time()
    for name, run_fn in modules.items():
        full_report["results"][name] = run_module(name, run_fn)

    total_elapsed = time.time() - total_start
    full_report["total_elapsed_seconds"] = round(total_elapsed, 2)

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for name, result in full_report["results"].items():
        status = result.get("_status", "unknown")
        elapsed = result.get("_elapsed_seconds", 0)
        if status == "success":
            # Extract key metric
            key_metric = ""
            if name == "langdetect" and "overall" in result:
                key_metric = f" (accuracy={result['overall'].get('accuracy', 0):.1%})"
            elif name == "sentiment" and "overall" in result:
                key_metric = f" (accuracy={result['overall'].get('accuracy', 0):.1%})"
            elif name == "diacritizer":
                yor = result.get("yoruba", {}).get("unseen_accuracy", {})
                if yor:
                    key_metric = f" (yor_word_acc={yor.get('word_accuracy', 0):.1%})"
            print(f"  {name:<20} OK  ({elapsed:.1f}s){key_metric}")
        else:
            print(f"  {name:<20} FAIL ({elapsed:.1f}s) - {result.get('_error', 'unknown error')}")

    print(f"\nTotal time: {total_elapsed:.1f}s")

    # Save full report
    report_path = REPORTS_DIR / f"{datetime.now().strftime('%Y%m%d')}_full_report.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False, default=str)
    print(f"Full report saved to: {report_path}")

    return full_report


if __name__ == "__main__":
    main()
