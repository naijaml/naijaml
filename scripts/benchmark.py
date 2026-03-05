"""Performance benchmarking for NaijaML functions.

Measures latency (p50/p95/p99), throughput, memory usage, and cold start time.

Usage:
    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --function detect_language
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR = PROJECT_ROOT / "evaluation_reports"
REPORTS_DIR.mkdir(exist_ok=True)


def get_memory_mb() -> float:
    """Get current RSS memory in MB (macOS/Linux)."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # macOS reports in bytes, Linux in KB
        import platform
        if platform.system() == "Darwin":
            return usage.ru_maxrss / (1024 * 1024)
        return usage.ru_maxrss / 1024
    except Exception:
        return 0.0


def benchmark_function(
    name: str,
    fn: Callable,
    inputs: List,
    warmup: int = 3,
    iterations: int = 50,
) -> Dict:
    """Benchmark a single function."""
    logger.info("Benchmarking %s (%d iterations)...", name, iterations)

    # Warmup
    for inp in inputs[:warmup]:
        fn(inp)

    # Force GC before benchmarking
    gc.collect()

    # Measure latency
    latencies = []
    for i in range(iterations):
        inp = inputs[i % len(inputs)]
        start = time.perf_counter()
        fn(inp)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    latencies.sort()

    # Throughput (texts/second)
    batch_start = time.perf_counter()
    count = 0
    while time.perf_counter() - batch_start < 2.0:  # 2 second window
        fn(inputs[count % len(inputs)])
        count += 1
    throughput = count / (time.perf_counter() - batch_start)

    return {
        "function": name,
        "iterations": iterations,
        "latency_ms": {
            "p50": round(latencies[len(latencies) // 2], 3),
            "p95": round(latencies[int(len(latencies) * 0.95)], 3),
            "p99": round(latencies[int(len(latencies) * 0.99)], 3),
            "min": round(min(latencies), 3),
            "max": round(max(latencies), 3),
            "mean": round(statistics.mean(latencies), 3),
        },
        "throughput_per_sec": round(throughput, 1),
    }


def benchmark_cold_start(name: str, import_and_call: str) -> Dict:
    """Measure cold start time (import + first call)."""
    import subprocess
    import sys

    cmd = [
        sys.executable, "-c",
        f"import time; start=time.perf_counter(); {import_and_call}; "
        f"print(round((time.perf_counter()-start)*1000, 1))"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            ms = float(result.stdout.strip())
            return {"function": name, "cold_start_ms": ms}
    except Exception as e:
        logger.warning("Cold start benchmark failed for %s: %s", name, e)

    return {"function": name, "cold_start_ms": -1}


def run_benchmarks() -> Dict:
    """Run all benchmarks."""
    # Sample inputs
    short_texts = [
        "Bawo ni o se wa?",
        "How far na?",
        "Ina kwana?",
        "Kedu ka i mere?",
        "This is a test",
    ]
    medium_texts = [
        "Ojo lo si oja lana, o ra onje fun awon omo re ni ile",
        "Wetin dey happen for this country, things no dey balance at all",
        "Ina kwana yaya aiki, mun tafi kasuwa da safe jiya",
        "Kedu ka i mere, o noro n'ulo akwukwo ugbu a",
        "The quick brown fox jumps over the lazy dog near the river",
    ]

    results = {}
    mem_before = get_memory_mb()

    # 1. Language detection
    from naijaml.nlp import detect_language
    results["detect_language"] = benchmark_function(
        "detect_language", detect_language, medium_texts
    )

    # 2. Sentiment analysis
    from naijaml.nlp import analyze_sentiment
    results["analyze_sentiment"] = benchmark_function(
        "analyze_sentiment", analyze_sentiment, medium_texts
    )

    # 3. PII masking
    from naijaml.nlp import mask_pii
    pii_texts = [
        "Call me on 08012345678 or email me@test.com",
        "No PII here, just regular text about Lagos",
        "My BVN is 22123456789 and phone is +2349012345678",
    ]
    results["mask_pii"] = benchmark_function(
        "mask_pii", mask_pii, pii_texts
    )

    # 4. Tokenizer
    try:
        from naijaml.nlp.tokenizer import Tokenizer
        tok = Tokenizer("yoruba")
        results["tokenize_yoruba"] = benchmark_function(
            "tokenize_yoruba", tok.encode, medium_texts
        )
    except Exception as e:
        logger.warning("Tokenizer benchmark skipped: %s", e)

    # 5. Diacritizer (slower, fewer iterations)
    try:
        from naijaml.nlp import diacritize
        diac_texts = [
            "Ojo lo si oja lana",
            "Bawo ni o se wa",
            "Omo mi dara pupo",
        ]
        results["diacritize"] = benchmark_function(
            "diacritize", diacritize, diac_texts, iterations=20
        )
    except Exception as e:
        logger.warning("Diacritizer benchmark skipped: %s", e)

    mem_after = get_memory_mb()

    # Cold start benchmarks
    cold_starts = [
        ("detect_language", "from naijaml.nlp import detect_language; detect_language('hello')"),
        ("analyze_sentiment", "from naijaml.nlp import analyze_sentiment; analyze_sentiment('hello')"),
        ("mask_pii", "from naijaml.nlp import mask_pii; mask_pii('hello')"),
    ]
    cold_start_results = {}
    for name, code in cold_starts:
        cold_start_results[name] = benchmark_cold_start(name, code)

    return {
        "module": "benchmark",
        "date": datetime.now().isoformat(),
        "benchmarks": results,
        "cold_starts": cold_start_results,
        "memory_mb": {
            "before": round(mem_before, 1),
            "after": round(mem_after, 1),
            "delta": round(mem_after - mem_before, 1),
        },
    }


def print_report(result: Dict) -> None:
    """Print human-readable report."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 60)

    print(f"\n{'Function':<25} {'p50 (ms)':<12} {'p95 (ms)':<12} {'p99 (ms)':<12} {'Throughput':<12}")
    print("-" * 73)

    for name, metrics in result.get("benchmarks", {}).items():
        lat = metrics.get("latency_ms", {})
        print(f"{name:<25} {lat.get('p50', 0):<12.1f} {lat.get('p95', 0):<12.1f} "
              f"{lat.get('p99', 0):<12.1f} {metrics.get('throughput_per_sec', 0):<12.0f}/s")

    print("\nCold start times:")
    for name, metrics in result.get("cold_starts", {}).items():
        ms = metrics.get("cold_start_ms", -1)
        print(f"  {name:<25} {ms:.0f}ms")

    mem = result.get("memory_mb", {})
    print(f"\nMemory: {mem.get('before', 0):.0f}MB → {mem.get('after', 0):.0f}MB "
          f"(+{mem.get('delta', 0):.0f}MB)")


def main():
    result = run_benchmarks()
    print_report(result)

    report_path = REPORTS_DIR / f"{datetime.now().strftime('%Y%m%d')}_benchmark.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Report saved to %s", report_path)

    return result


if __name__ == "__main__":
    main()
