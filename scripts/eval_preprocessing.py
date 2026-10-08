"""Evaluate PII masking and validate Nigerian constants.

Tests PII masking precision/recall, validates states/LGAs/banks/telcos.
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
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "eval"
REPORTS_DIR = PROJECT_ROOT / "evaluation_reports"
REPORTS_DIR.mkdir(exist_ok=True)


def evaluate_pii_masking() -> Dict:
    """Evaluate PII masking on test cases."""
    from naijaml.nlp import mask_pii

    pii_file = FIXTURES_DIR / "pii_test_cases.json"
    if not pii_file.exists():
        logger.error("Missing fixture: %s", pii_file)
        return {}

    with open(pii_file, encoding="utf-8") as f:
        data = json.load(f)

    test_cases = data["test_cases"]
    total = len(test_cases)
    correct = 0
    errors = []

    for case in test_cases:
        input_text = case["input"]
        expected = case["expected"]
        actual = mask_pii(input_text)

        if actual == expected:
            correct += 1
        else:
            errors.append({
                "input": input_text,
                "expected": expected,
                "actual": actual,
                "note": case.get("note", ""),
            })

    return {
        "total_cases": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total > 0 else 0,
        "errors": errors,
    }


def validate_constants() -> Dict:
    """Validate Nigerian constants for completeness and correctness."""
    from naijaml.utils.constants import (
        STATES, LGAS, BANKS, TELCOS,
        is_valid_phone, normalize_phone, get_telco,
        format_naira, parse_naira,
    )

    results = {}

    # States validation
    expected_state_count = 37  # 36 states + FCT
    state_count = len(STATES)
    results["states"] = {
        "count": state_count,
        "expected": expected_state_count,
        "complete": state_count == expected_state_count,
        "has_fct": "FCT" in STATES,
    }

    # Check key states exist
    key_states = ["Lagos", "Kano", "Rivers", "Oyo", "Kaduna", "Anambra", "FCT"]
    missing_states = [s for s in key_states if s not in STATES]
    results["states"]["missing_key_states"] = missing_states

    # LGA validation
    lga_states_covered = len(LGAS)
    total_lgas = sum(len(v) for v in LGAS.values())
    results["lgas"] = {
        "states_covered": lga_states_covered,
        "total_states": expected_state_count,
        "total_lgas": total_lgas,
        "expected_total": 774,
        "completeness": round(total_lgas / 774, 4),
        "missing_states": [s for s in STATES if s not in LGAS],
    }

    # Banks validation
    results["banks"] = {
        "count": len(BANKS),
        "has_traditional": all(
            name in BANKS for name in [
                "Access Bank", "Guaranty Trust Bank", "Zenith Bank",
                "First Bank", "United Bank for Africa",
            ]
        ),
        "has_digital": all(
            name in BANKS for name in ["Kuda Bank", "OPay", "PalmPay", "Moniepoint"]
        ),
    }

    # Telcos validation
    results["telcos"] = {
        "count": len(TELCOS),
        "operators": list(TELCOS.keys()),
        "expected": ["MTN", "Airtel", "Glo", "9mobile"],
        "complete": set(TELCOS.keys()) == {"MTN", "Airtel", "Glo", "9mobile"},
    }

    # Phone validation tests
    phone_tests = [
        ("08031234567", True, "MTN"),
        ("09031234567", True, "MTN"),
        ("+2348031234567", True, "MTN"),
        ("08021234567", True, "Airtel"),
        ("08051234567", True, "Glo"),
        ("08091234567", True, "9mobile"),
        ("12345", False, None),
        ("000000000000", False, None),
    ]
    phone_results = []
    for phone, expected_valid, expected_telco in phone_tests:
        actual_valid = is_valid_phone(phone)
        actual_telco = get_telco(phone) if actual_valid else None
        phone_results.append({
            "phone": phone,
            "valid_correct": actual_valid == expected_valid,
            "telco_correct": actual_telco == expected_telco if expected_valid else True,
        })

    results["phone_validation"] = {
        "tests": len(phone_results),
        "all_correct": all(r["valid_correct"] and r["telco_correct"] for r in phone_results),
        "details": phone_results,
    }

    # Naira formatting tests
    naira_tests = [
        (1500000, "₦1,500,000.00"),
        (0, "₦0.00"),
        (99.99, "₦99.99"),
    ]
    naira_results = []
    for amount, expected in naira_tests:
        actual = format_naira(amount)
        naira_results.append({
            "amount": amount,
            "expected": expected,
            "actual": actual,
            "correct": actual == expected,
        })

    results["naira_formatting"] = {
        "tests": len(naira_results),
        "all_correct": all(r["correct"] for r in naira_results),
        "details": naira_results,
    }

    return results


def print_report(result: Dict) -> None:
    """Print human-readable report."""
    print("\n" + "=" * 60)
    print("PREPROCESSING & CONSTANTS EVALUATION")
    print("=" * 60)

    # PII masking
    pii = result.get("pii_masking", {})
    if pii:
        print(f"\nPII Masking:")
        print(f"  Accuracy: {pii.get('accuracy', 0):.1%} ({pii.get('correct', 0)}/{pii.get('total_cases', 0)})")
        errors = pii.get("errors", [])
        if errors:
            print(f"  Errors ({len(errors)}):")
            for err in errors[:5]:
                print(f"    Input:    {err['input'][:60]}")
                print(f"    Expected: {err['expected'][:60]}")
                print(f"    Actual:   {err['actual'][:60]}")
                if err.get("note"):
                    print(f"    Note:     {err['note']}")
                print()

    # Constants
    constants = result.get("constants", {})
    if constants:
        states = constants.get("states", {})
        print(f"\nStates: {states.get('count', 0)}/{states.get('expected', 37)} "
              f"({'COMPLETE' if states.get('complete') else 'INCOMPLETE'})")

        lgas = constants.get("lgas", {})
        print(f"LGAs: {lgas.get('total_lgas', 0)}/{lgas.get('expected_total', 774)} "
              f"({lgas.get('completeness', 0):.1%} complete, {lgas.get('states_covered', 0)} states covered)")
        missing = lgas.get("missing_states", [])
        if missing:
            print(f"  Missing LGA data for: {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}")

        banks = constants.get("banks", {})
        print(f"Banks: {banks.get('count', 0)} "
              f"(traditional={'OK' if banks.get('has_traditional') else 'MISSING'}, "
              f"digital={'OK' if banks.get('has_digital') else 'MISSING'})")

        telcos = constants.get("telcos", {})
        print(f"Telcos: {'COMPLETE' if telcos.get('complete') else 'INCOMPLETE'} "
              f"({', '.join(telcos.get('operators', []))})")

        phone = constants.get("phone_validation", {})
        print(f"Phone validation: {'ALL PASS' if phone.get('all_correct') else 'SOME FAIL'} "
              f"({phone.get('tests', 0)} tests)")

        naira = constants.get("naira_formatting", {})
        print(f"Naira formatting: {'ALL PASS' if naira.get('all_correct') else 'SOME FAIL'} "
              f"({naira.get('tests', 0)} tests)")


def main():
    result = {
        "module": "preprocessing",
        "date": datetime.now().isoformat(),
        "pii_masking": evaluate_pii_masking(),
        "constants": validate_constants(),
    }

    print_report(result)

    report_path = REPORTS_DIR / f"{datetime.now().strftime('%Y%m%d')}_preprocessing.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Report saved to %s", report_path)

    return result


if __name__ == "__main__":
    main()
