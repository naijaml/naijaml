"""Robustness tests for all NaijaML public functions.

Tests edge cases: empty/None inputs, very long inputs, Unicode edge cases,
adversarial inputs, encoding issues, repeated characters, special chars.
"""
from __future__ import annotations

import pytest


# =============================================================================
# Edge case inputs
# =============================================================================

EMPTY_INPUTS = ["", " ", "  \t\n  "]
SPECIAL_CHARS = ["@#$%^&*()", "!!!???...", "123 456 789", "..."]
REPEATED = ["a" * 10000, "!" * 5000, "ọ" * 1000]
UNICODE_EDGE_CASES = [
    "\u200b\u200b\u200b",         # Zero-width spaces
    "test\u200dtest",             # Zero-width joiner
    "\ufeff BOM marker",          # BOM
    "Hello\u200eWorld",           # LTR mark
    "test\u0000null",             # Null character
    "\U0001f600\U0001f1f3\U0001f1ec",  # Emoji + flag
    "café",                       # Latin with diacritics (not Yoruba)
    "中文测试",                      # CJK characters
    "مرحبا",                       # Arabic
]
ADVERSARIAL = [
    "<script>alert('xss')</script>",
    "'; DROP TABLE users; --",
    "%s%s%s%s%s",
    "${7*7}",
    "{{constructor.constructor('return process')()}}",
]


# =============================================================================
# Language Detection Robustness
# =============================================================================

class TestLangdetectRobustness:
    """Robustness tests for detect_language."""

    def test_empty_string(self):
        from naijaml.nlp import detect_language
        result = detect_language("")
        # None is acceptable for empty input
        assert result is None or isinstance(result, str)

    def test_whitespace_only(self):
        from naijaml.nlp import detect_language
        result = detect_language("   \t\n  ")
        assert result is None or isinstance(result, str)

    @pytest.mark.parametrize("text", SPECIAL_CHARS)
    def test_special_characters(self, text):
        from naijaml.nlp import detect_language
        result = detect_language(text)
        assert isinstance(result, str)

    @pytest.mark.parametrize("text", UNICODE_EDGE_CASES)
    def test_unicode_edge_cases(self, text):
        from naijaml.nlp import detect_language
        result = detect_language(text)
        assert isinstance(result, str)

    @pytest.mark.parametrize("text", ADVERSARIAL)
    def test_adversarial_inputs(self, text):
        from naijaml.nlp import detect_language
        result = detect_language(text)
        assert isinstance(result, str)

    def test_very_long_input(self):
        from naijaml.nlp import detect_language
        text = "Bawo ni o se wa " * 5000  # ~80K chars
        result = detect_language(text)
        assert isinstance(result, str)

    def test_repeated_characters(self):
        from naijaml.nlp import detect_language
        result = detect_language("a" * 10000)
        assert isinstance(result, str)

    def test_single_character(self):
        from naijaml.nlp import detect_language
        result = detect_language("a")
        assert isinstance(result, str)

    def test_confidence_returns_tuple(self):
        from naijaml.nlp import detect_language_with_confidence
        lang, conf = detect_language_with_confidence("test")
        assert isinstance(lang, str)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0


# =============================================================================
# Sentiment Robustness
# =============================================================================

class TestSentimentRobustness:
    """Robustness tests for analyze_sentiment."""

    def test_empty_string(self):
        from naijaml.nlp import analyze_sentiment
        result = analyze_sentiment("")
        assert result["label"] in {"positive", "negative", "neutral"}
        assert "confidence" in result

    def test_whitespace_only(self):
        from naijaml.nlp import analyze_sentiment
        result = analyze_sentiment("   ")
        assert result["label"] in {"positive", "negative", "neutral"}

    @pytest.mark.parametrize("text", SPECIAL_CHARS)
    def test_special_characters(self, text):
        from naijaml.nlp import analyze_sentiment
        result = analyze_sentiment(text)
        assert result["label"] in {"positive", "negative", "neutral"}

    @pytest.mark.parametrize("text", UNICODE_EDGE_CASES)
    def test_unicode_edge_cases(self, text):
        from naijaml.nlp import analyze_sentiment
        result = analyze_sentiment(text)
        assert result["label"] in {"positive", "negative", "neutral"}

    @pytest.mark.parametrize("text", ADVERSARIAL)
    def test_adversarial_inputs(self, text):
        from naijaml.nlp import analyze_sentiment
        result = analyze_sentiment(text)
        assert result["label"] in {"positive", "negative", "neutral"}

    def test_very_long_input(self):
        from naijaml.nlp import analyze_sentiment
        text = "This is great " * 5000
        result = analyze_sentiment(text)
        assert result["label"] in {"positive", "negative", "neutral"}

    def test_batch_empty(self):
        from naijaml.nlp import analyze_sentiment_batch
        result = analyze_sentiment_batch([])
        assert result == []

    def test_batch_mixed(self):
        from naijaml.nlp import analyze_sentiment_batch
        result = analyze_sentiment_batch(["good", "", "bad"])
        assert len(result) == 3


# =============================================================================
# PII Masking Robustness
# =============================================================================

class TestPIIMaskingRobustness:
    """Robustness tests for mask_pii."""

    def test_empty_string(self):
        from naijaml.nlp import mask_pii
        result = mask_pii("")
        assert result == ""

    @pytest.mark.parametrize("text", SPECIAL_CHARS)
    def test_special_characters(self, text):
        from naijaml.nlp import mask_pii
        result = mask_pii(text)
        assert isinstance(result, str)

    @pytest.mark.parametrize("text", UNICODE_EDGE_CASES)
    def test_unicode_edge_cases(self, text):
        from naijaml.nlp import mask_pii
        result = mask_pii(text)
        assert isinstance(result, str)

    @pytest.mark.parametrize("text", ADVERSARIAL)
    def test_adversarial_no_code_execution(self, text):
        from naijaml.nlp import mask_pii
        result = mask_pii(text)
        assert isinstance(result, str)

    def test_no_false_positive_short_numbers(self):
        """Regular short numbers should not be masked."""
        from naijaml.nlp import mask_pii
        result = mask_pii("The year 2024 was great")
        assert "2024" in result

    def test_preserves_regular_text(self):
        """Text without PII should be unchanged."""
        from naijaml.nlp import mask_pii
        text = "Lagos is a beautiful city in Nigeria"
        assert mask_pii(text) == text


# =============================================================================
# Preprocessing Robustness
# =============================================================================

class TestPreprocessingRobustness:
    """Robustness tests for preprocessing functions."""

    def test_normalize_unicode_empty(self):
        from naijaml.nlp import normalize_unicode
        assert normalize_unicode("") == ""

    def test_strip_diacritics_empty(self):
        from naijaml.nlp import strip_diacritics
        assert strip_diacritics("") == ""

    def test_clean_social_media_empty(self):
        from naijaml.nlp import clean_social_media
        assert clean_social_media("") == ""

    @pytest.mark.parametrize("text", UNICODE_EDGE_CASES)
    def test_normalize_unicode_edge_cases(self, text):
        from naijaml.nlp import normalize_unicode
        result = normalize_unicode(text)
        assert isinstance(result, str)

    @pytest.mark.parametrize("text", UNICODE_EDGE_CASES)
    def test_clean_nigerian_text_edge_cases(self, text):
        from naijaml.nlp import clean_nigerian_text
        result = clean_nigerian_text(text)
        assert isinstance(result, str)

    def test_clean_social_media_urls(self):
        from naijaml.nlp import clean_social_media
        result = clean_social_media("Check https://evil.com/hack?x=1&y=2")
        assert "evil.com" not in result

    def test_extract_hashtags_empty(self):
        from naijaml.nlp import extract_hashtags
        assert extract_hashtags("") == []

    def test_extract_mentions_empty(self):
        from naijaml.nlp import extract_mentions
        assert extract_mentions("") == []


# =============================================================================
# Constants Robustness
# =============================================================================

class TestConstantsRobustness:
    """Robustness tests for constants utilities."""

    def test_format_naira_zero(self):
        from naijaml.utils.constants import format_naira
        assert format_naira(0) == "₦0.00"

    def test_format_naira_large(self):
        from naijaml.utils.constants import format_naira
        result = format_naira(999999999999.99)
        assert result.startswith("₦")

    def test_format_naira_negative(self):
        from naijaml.utils.constants import format_naira
        result = format_naira(-1000)
        assert isinstance(result, str)

    def test_parse_naira_invalid(self):
        from naijaml.utils.constants import parse_naira
        assert parse_naira("not a number") is None

    def test_parse_naira_empty(self):
        from naijaml.utils.constants import parse_naira
        assert parse_naira("") is None

    def test_is_valid_phone_empty(self):
        from naijaml.utils.constants import is_valid_phone
        assert is_valid_phone("") is False

    def test_is_valid_phone_garbage(self):
        from naijaml.utils.constants import is_valid_phone
        assert is_valid_phone("not a phone") is False

    def test_normalize_phone_invalid(self):
        from naijaml.utils.constants import normalize_phone
        assert normalize_phone("abc") is None

    def test_get_telco_invalid(self):
        from naijaml.utils.constants import get_telco
        assert get_telco("12345") is None

    def test_is_valid_bvn_empty(self):
        from naijaml.utils.constants import is_valid_bvn
        assert is_valid_bvn("") is False

    def test_is_valid_nin_empty(self):
        from naijaml.utils.constants import is_valid_nin
        assert is_valid_nin("") is False


# =============================================================================
# Tokenizer Robustness
# =============================================================================

class TestTokenizerRobustness:
    """Robustness tests for tokenizer."""

    def test_empty_string(self):
        from naijaml.nlp import tokenize
        result = tokenize("", lang="yoruba")
        assert isinstance(result, list)

    def test_single_char(self):
        from naijaml.nlp import tokenize
        result = tokenize("a", lang="yoruba")
        assert isinstance(result, list)

    @pytest.mark.parametrize("text", UNICODE_EDGE_CASES[:5])
    def test_unicode_edge_cases(self, text):
        from naijaml.nlp import tokenize
        result = tokenize(text, lang="yoruba")
        assert isinstance(result, list)

    def test_invalid_language(self):
        from naijaml.nlp.tokenizer import Tokenizer
        with pytest.raises((ValueError, KeyError)):
            Tokenizer("klingon")
