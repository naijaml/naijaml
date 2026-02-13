#!/usr/bin/env python3
"""Tokenizer benchmark: measures how major AI tokenizers handle Nigerian languages vs English.

Compares GPT-4, GPT-4o, Llama 3, Gemma 2, Mistral, BERT multilingual, and XLM-RoBERTa
tokenizers across Yoruba, Hausa, Igbo, Nigerian Pidgin, and English.

Usage:
    python benchmarks/tokenizer_benchmark.py           # full benchmark (100 entries)
    python benchmarks/tokenizer_benchmark.py --quick    # quick mode (10 entries)

Outputs:
    benchmarks/results/benchmark_results.json    - raw data
    benchmarks/results/benchmark_summary.csv     - aggregate stats
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# --- Constants ---

LANGUAGES = ["eng", "yor", "hau", "ibo", "pcm"]
LANG_NAMES = {
    "eng": "English",
    "yor": "Yoruba",
    "hau": "Hausa",
    "ibo": "Igbo",
    "pcm": "Pidgin",
}

# GPT-4o pricing: $2.50 per 1M input tokens
GPT4O_USD_PER_1M_TOKENS = 2.50
USD_TO_NGN = 1350.0  # CBN official rate, Feb 2026

TOKENIZER_IDS = [
    "gpt4_cl100k",
    "gpt4o_o200k",
    "llama3",
    "gemma",
    "mistral",
    "bert_multilingual",
    "xlm_roberta",
]

TOKENIZER_NAMES = {
    "gpt4_cl100k": "GPT-4 (cl100k_base)",
    "gpt4o_o200k": "GPT-4o (o200k_base)",
    "llama3": "Llama 3",
    "gemma": "Gemma 2",
    "mistral": "Mistral v0.3",
    "bert_multilingual": "BERT Multilingual",
    "xlm_roberta": "XLM-RoBERTa",
}


# --- Tokenizer Loading ---


def load_tiktoken(encoding_name: str):
    """Load a tiktoken encoding."""
    import tiktoken

    return tiktoken.get_encoding(encoding_name)


def load_hf_tokenizer(model_name: str):
    """Load a HuggingFace tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


LLAMA3_CANDIDATES = [
    "unsloth/llama-3-8b",
    "NousResearch/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B",
]


def _load_llama3_tokenizer():
    """Try multiple Llama 3 tokenizer sources (gated model has mirrors)."""
    from transformers import AutoTokenizer

    for model_id in LLAMA3_CANDIDATES:
        try:
            return AutoTokenizer.from_pretrained(model_id)
        except (OSError, Exception) as e:
            logger.info(f"  Could not load from {model_id}: {type(e).__name__}")
            continue
    raise RuntimeError(
        "Could not load Llama 3 tokenizer. Try: huggingface-cli login "
        "(requires access to meta-llama/Meta-Llama-3-8B)"
    )


def get_tokenizers() -> Dict[str, Any]:
    """Load all tokenizers. Returns dict mapping tokenizer_id -> tokenizer object.

    Skips tokenizers that fail to load (with a warning) rather than crashing.
    """
    tokenizers = {}

    specs = [
        ("gpt4_cl100k", "tiktoken", "cl100k_base"),
        ("gpt4o_o200k", "tiktoken", "o200k_base"),
        ("llama3", "llama3", None),
        ("gemma", "hf", "google/gemma-2b"),
        ("mistral", "hf", "mistralai/Mistral-7B-v0.3"),
        ("bert_multilingual", "hf", "bert-base-multilingual-cased"),
        ("xlm_roberta", "hf", "xlm-roberta-base"),
    ]

    for tok_id, tok_type, model_id in specs:
        name = TOKENIZER_NAMES[tok_id]
        logger.info(f"Loading {name}...")
        try:
            if tok_type == "tiktoken":
                tokenizers[tok_id] = ("tiktoken", load_tiktoken(model_id))
            elif tok_type == "llama3":
                tokenizers[tok_id] = ("hf", _load_llama3_tokenizer())
            else:
                tokenizers[tok_id] = ("hf", load_hf_tokenizer(model_id))
        except Exception as e:
            logger.warning(f"  SKIPPED {name}: {e}\n")
            continue

    loaded = [TOKENIZER_NAMES[t] for t in tokenizers]
    logger.info(f"Loaded {len(tokenizers)} tokenizers: {', '.join(loaded)}\n")
    return tokenizers


def tokenize(text: str, tokenizer_type: str, tokenizer: Any) -> Tuple[List[int], List[str]]:
    """Tokenize text and return (token_ids, token_strings)."""
    if tokenizer_type == "tiktoken":
        token_ids = tokenizer.encode(text)
        token_strings = [tokenizer.decode([tid]) for tid in token_ids]
    else:
        encoded = tokenizer(text, add_special_tokens=False)
        token_ids = encoded["input_ids"]
        token_strings = tokenizer.convert_ids_to_tokens(token_ids)
    return token_ids, token_strings


# --- Data Loading ---


def load_benchmark_words(data_path: Path, quick: bool = False) -> List[Dict[str, Any]]:
    """Load benchmark words from JSON file."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data["entries"]

    if quick:
        # Take first 10 entries for quick mode
        entries = entries[:10]
        logger.info(f"Quick mode: using {len(entries)} entries per language\n")
    else:
        logger.info(f"Full mode: using {len(entries)} entries per language\n")

    return entries


# --- Benchmarking ---


def run_benchmark(
    entries: List[Dict[str, Any]], tokenizers: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Run the benchmark across all entries and tokenizers.

    Returns a list of result dicts, one per (entry, language, tokenizer) combo.
    """
    results = []

    for entry in entries:
        entry_id = entry["id"]
        category = entry["category"]

        # Get English token counts first (for ratio calculation)
        eng_text = entry["eng"]
        eng_token_counts = {}
        for tok_id, (tok_type, tok_obj) in tokenizers.items():
            ids, _ = tokenize(eng_text, tok_type, tok_obj)
            eng_token_counts[tok_id] = len(ids)

        for lang in LANGUAGES:
            text = entry[lang]

            for tok_id, (tok_type, tok_obj) in tokenizers.items():
                token_ids, token_strings = tokenize(text, tok_type, tok_obj)
                n_tokens = len(token_ids)
                eng_count = eng_token_counts[tok_id]
                ratio = n_tokens / eng_count if eng_count > 0 else float("inf")

                results.append(
                    {
                        "entry_id": entry_id,
                        "category": category,
                        "language": lang,
                        "language_name": LANG_NAMES[lang],
                        "text": text,
                        "tokenizer": tok_id,
                        "tokenizer_name": TOKENIZER_NAMES[tok_id],
                        "n_tokens": n_tokens,
                        "token_splits": token_strings,
                        "english_equivalent": eng_text,
                        "english_tokens": eng_count,
                        "token_ratio_vs_english": round(ratio, 3),
                    }
                )

    return results


def compute_aggregate_stats(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute aggregate statistics per language per tokenizer."""
    import numpy as np

    # Group results by (language, tokenizer)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in results:
        key = (r["language"], r["tokenizer"])
        groups.setdefault(key, []).append(r)

    stats = []
    for (lang, tok_id), group in sorted(groups.items()):
        token_counts = [r["n_tokens"] for r in group]
        ratios = [r["token_ratio_vs_english"] for r in group]

        mean_tokens = float(np.mean(token_counts))
        mean_ratio = float(np.mean(ratios))

        # Cost multiplier: ratio of mean tokens to English mean tokens
        eng_key = ("eng", tok_id)
        if eng_key in groups:
            eng_mean = float(np.mean([r["n_tokens"] for r in groups[eng_key]]))
            cost_multiplier = mean_tokens / eng_mean if eng_mean > 0 else 1.0
        else:
            cost_multiplier = mean_ratio

        # Worst offender: highest token ratio in this group
        worst = max(group, key=lambda r: r["token_ratio_vs_english"])

        # Naira cost per 1M tokens equivalent
        naira_per_1m = GPT4O_USD_PER_1M_TOKENS * USD_TO_NGN  # base cost in NGN
        estimated_naira_cost = naira_per_1m * cost_multiplier

        stats.append(
            {
                "language": lang,
                "language_name": LANG_NAMES[lang],
                "tokenizer": tok_id,
                "tokenizer_name": TOKENIZER_NAMES[tok_id],
                "mean_tokens_per_entry": round(mean_tokens, 2),
                "mean_token_ratio_vs_english": round(mean_ratio, 3),
                "cost_multiplier_vs_english": round(cost_multiplier, 3),
                "estimated_naira_per_1m_equiv": round(estimated_naira_cost, 0),
                "worst_word": worst["text"],
                "worst_word_tokens": worst["n_tokens"],
                "worst_word_ratio": worst["token_ratio_vs_english"],
            }
        )

    return stats


def find_top_worst_tokenized(results: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
    """Find the top N worst tokenized words across all languages (excluding English)."""
    non_eng = [r for r in results if r["language"] != "eng"]
    sorted_results = sorted(non_eng, key=lambda r: r["token_ratio_vs_english"], reverse=True)

    seen = set()
    top = []
    for r in sorted_results:
        # Deduplicate by (text, tokenizer)
        key = (r["text"], r["tokenizer"])
        if key in seen:
            continue
        seen.add(key)
        top.append(
            {
                "text": r["text"],
                "language": r["language_name"],
                "tokenizer": r["tokenizer_name"],
                "n_tokens": r["n_tokens"],
                "english_equivalent": r["english_equivalent"],
                "english_tokens": r["english_tokens"],
                "ratio": r["token_ratio_vs_english"],
                "token_splits": r["token_splits"],
            }
        )
        if len(top) >= top_n:
            break

    return top


# --- Output ---


def save_results(
    results: List[Dict[str, Any]],
    stats: List[Dict[str, Any]],
    top_worst: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save all results to JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Raw results JSON
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "tokenizers": TOKENIZER_NAMES,
                    "languages": LANG_NAMES,
                    "gpt4o_pricing_usd_per_1m": GPT4O_USD_PER_1M_TOKENS,
                    "usd_to_ngn_rate": USD_TO_NGN,
                },
                "results": results,
                "aggregate_stats": stats,
                "top_worst_tokenized": top_worst,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"Saved raw results to {results_path}")

    # Summary CSV
    csv_path = output_dir / "benchmark_summary.csv"
    fieldnames = [
        "language",
        "language_name",
        "tokenizer",
        "tokenizer_name",
        "mean_tokens_per_entry",
        "mean_token_ratio_vs_english",
        "cost_multiplier_vs_english",
        "estimated_naira_per_1m_equiv",
        "worst_word",
        "worst_word_tokens",
        "worst_word_ratio",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)
    logger.info(f"Saved summary CSV to {csv_path}")


def print_summary_table(stats: List[Dict[str, Any]], top_worst: List[Dict[str, Any]]) -> None:
    """Print a clean summary table to stdout."""
    print("\n" + "=" * 90)
    print("  TOKENIZER BENCHMARK: Nigerian Languages vs English")
    print("=" * 90)

    # Group by tokenizer (use TOKENIZER_IDS order for those present)
    present_tok_ids = [t for t in TOKENIZER_IDS if any(s["tokenizer"] == t for s in stats)]
    for tok_id in present_tok_ids:
        tok_name = TOKENIZER_NAMES[tok_id]
        tok_stats = [s for s in stats if s["tokenizer"] == tok_id]
        if not tok_stats:
            continue

        print(f"\n{'─' * 90}")
        print(f"  {tok_name}")
        print(f"{'─' * 90}")
        print(f"  {'Language':<12} {'Mean Tokens':>12} {'Ratio vs Eng':>14} {'Cost Mult':>11} {'Est. ₦/1M':>12}")
        print(f"  {'─' * 10:<12} {'─' * 10:>12} {'─' * 12:>14} {'─' * 9:>11} {'─' * 10:>12}")

        for s in tok_stats:
            lang_name = s["language_name"]
            mean_tok = s["mean_tokens_per_entry"]
            ratio = s["mean_token_ratio_vs_english"]
            cost_mult = s["cost_multiplier_vs_english"]
            naira = s["estimated_naira_per_1m_equiv"]
            print(
                f"  {lang_name:<12} {mean_tok:>12.1f} {ratio:>14.2f}x {cost_mult:>10.2f}x ₦{naira:>10,.0f}"
            )

    # Top worst
    print(f"\n{'=' * 90}")
    print("  TOP 10 WORST TOKENIZED (highest token explosion vs English)")
    print(f"{'=' * 90}")
    print(f"  {'#':<3} {'Text':<30} {'Lang':<8} {'Tokenizer':<22} {'Tokens':>7} {'Ratio':>7}")
    print(f"  {'─' * 2:<3} {'─' * 28:<30} {'─' * 6:<8} {'─' * 20:<22} {'─' * 5:>7} {'─' * 5:>7}")

    for i, w in enumerate(top_worst, 1):
        text = w["text"][:28]
        print(
            f"  {i:<3} {text:<30} {w['language']:<8} {w['tokenizer']:<22} {w['n_tokens']:>7} {w['ratio']:>6.1f}x"
        )

    # Key takeaway
    print(f"\n{'=' * 90}")
    print("  KEY INSIGHT")
    print(f"{'=' * 90}")

    # Find language with highest average cost across tokenizers
    lang_avg_cost: Dict[str, List[float]] = {}
    for s in stats:
        if s["language"] != "eng":
            lang_avg_cost.setdefault(s["language_name"], []).append(
                s["cost_multiplier_vs_english"]
            )

    if lang_avg_cost:
        import numpy as np

        most_expensive = max(lang_avg_cost.items(), key=lambda x: np.mean(x[1]))
        avg_mult = np.mean(most_expensive[1])
        print(
            f"  {most_expensive[0]} is the most expensive language to process,"
        )
        print(
            f"  costing on average {avg_mult:.2f}x more tokens than English."
        )
        base_naira = GPT4O_USD_PER_1M_TOKENS * USD_TO_NGN
        print(
            f"  If English costs ₦{base_naira:,.0f} per 1M tokens (GPT-4o),"
        )
        print(
            f"  {most_expensive[0]} costs ~₦{base_naira * avg_mult:,.0f} for equivalent content."
        )

    print(f"\n{'=' * 90}\n")


# --- Main ---


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark AI tokenizers on Nigerian languages vs English"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run on 10 words per language for fast testing",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to benchmark_words.json (auto-detected if not set)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: results/)",
    )
    args = parser.parse_args()

    # Find paths
    script_dir = Path(__file__).resolve().parent

    if args.data:
        data_path = Path(args.data)
    else:
        data_path = script_dir / "benchmark_words.json"

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = script_dir / "results"

    if not data_path.exists():
        logger.error(f"Benchmark data not found at {data_path}")
        sys.exit(1)

    # Load data
    entries = load_benchmark_words(data_path, quick=args.quick)

    # Load tokenizers
    tokenizers = get_tokenizers()

    # Run benchmark
    logger.info("Running benchmark...")
    results = run_benchmark(entries, tokenizers)
    logger.info(f"Generated {len(results)} result entries.\n")

    # Compute stats
    stats = compute_aggregate_stats(results)
    top_worst = find_top_worst_tokenized(results, top_n=10)

    # Save outputs
    save_results(results, stats, top_worst, output_dir)

    # Print summary
    print_summary_table(stats, top_worst)

    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main()
