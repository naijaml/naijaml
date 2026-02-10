"""Tests for Nigerian language detection."""
from __future__ import annotations

import pytest


class TestDetectLanguage:
    """Tests for the main detect_language function."""

    def test_detects_yoruba(self, yoruba_samples):
        """Should correctly identify Yorùbá text."""
        from naijaml.nlp.langdetect import detect_language

        for text in yoruba_samples:
            result = detect_language(text)
            assert result == "yor", f"Failed on: {text}"

    def test_detects_hausa(self, hausa_samples):
        """Should correctly identify Hausa text."""
        from naijaml.nlp.langdetect import detect_language

        for text in hausa_samples:
            result = detect_language(text)
            assert result == "hau", f"Failed on: {text}"

    def test_detects_igbo(self, igbo_samples):
        """Should correctly identify Igbo text."""
        from naijaml.nlp.langdetect import detect_language

        for text in igbo_samples:
            result = detect_language(text)
            assert result == "ibo", f"Failed on: {text}"

    def test_detects_pidgin(self, pidgin_samples):
        """Should correctly identify Nigerian Pidgin text."""
        from naijaml.nlp.langdetect import detect_language

        for text in pidgin_samples:
            result = detect_language(text)
            assert result == "pcm", f"Failed on: {text}"

    def test_detects_english(self, english_samples):
        """Should correctly identify English text."""
        from naijaml.nlp.langdetect import detect_language

        for text in english_samples:
            result = detect_language(text)
            assert result == "eng", f"Failed on: {text}"

    def test_returns_valid_language_code(self):
        """Should always return a valid language code."""
        from naijaml.nlp.langdetect import detect_language, SUPPORTED_LANGUAGES

        result = detect_language("Some random text here")
        assert result in SUPPORTED_LANGUAGES

    def test_handles_empty_string(self):
        """Should handle empty string gracefully."""
        from naijaml.nlp.langdetect import detect_language

        result = detect_language("")
        assert result is None or result in ["yor", "hau", "ibo", "pcm", "eng"]

    def test_handles_short_text(self):
        """Should handle very short text (may be less accurate)."""
        from naijaml.nlp.langdetect import detect_language

        # Single word tests - these may be ambiguous
        result = detect_language("hello")
        assert result in ["yor", "hau", "ibo", "pcm", "eng"]

    def test_handles_mixed_case(self):
        """Should handle text in different cases."""
        from naijaml.nlp.langdetect import detect_language

        # Uppercase Pidgin
        result = detect_language("WETIN DEY HAPPEN")
        assert result == "pcm"

        # Mixed case Yorùbá
        result = detect_language("Ẹ KÚ IṢẸ́ O")
        assert result == "yor"


class TestDetectLanguageWithConfidence:
    """Tests for detection with confidence scores."""

    def test_returns_confidence_score(self):
        """Should return confidence along with language."""
        from naijaml.nlp.langdetect import detect_language_with_confidence

        lang, confidence = detect_language_with_confidence("Wetin dey happen?")
        assert lang == "pcm"
        assert 0.0 <= confidence <= 1.0

    def test_high_confidence_for_clear_text(self):
        """Should have high confidence for text with distinctive features."""
        from naijaml.nlp.langdetect import detect_language_with_confidence

        # Yorùbá with many diacritics - very distinctive
        lang, confidence = detect_language_with_confidence(
            "Ọjọ́ náà dára púpọ̀, ẹ kú iṣẹ́ o"
        )
        assert lang == "yor"
        assert confidence > 0.7

    def test_lower_confidence_for_ambiguous_text(self):
        """Should have lower confidence for ambiguous text."""
        from naijaml.nlp.langdetect import detect_language_with_confidence

        # "Na" could be Pidgin or Hausa
        lang, confidence = detect_language_with_confidence("Na")
        assert confidence < 0.9  # Should be less certain


class TestDetectAllLanguages:
    """Tests for getting all language probabilities."""

    def test_returns_all_languages(self):
        """Should return scores for all supported languages."""
        from naijaml.nlp.langdetect import detect_all_languages, SUPPORTED_LANGUAGES

        scores = detect_all_languages("Some text here")
        assert set(scores.keys()) == set(SUPPORTED_LANGUAGES)

    def test_scores_sum_to_one(self):
        """Probability scores should sum to approximately 1."""
        from naijaml.nlp.langdetect import detect_all_languages

        scores = detect_all_languages("Wetin dey happen for here?")
        total = sum(scores.values())
        assert 0.99 <= total <= 1.01  # Allow small floating point error

    def test_highest_score_matches_detect(self):
        """Highest scoring language should match detect_language result."""
        from naijaml.nlp.langdetect import detect_language, detect_all_languages

        text = "Ọjọ́ dára púpọ̀"
        detected = detect_language(text)
        scores = detect_all_languages(text)
        highest = max(scores, key=scores.get)
        assert detected == highest


class TestYorubaSpecificFeatures:
    """Tests for Yorùbá-specific detection features."""

    def test_detects_undiacritized_yoruba(self):
        """Should detect Yorùbá even without diacritics (harder case)."""
        from naijaml.nlp.langdetect import detect_language

        # Common Yorùbá phrases without diacritics
        # This is a harder case - may need word-level features
        result = detect_language("E ku ise o, bawo ni?")
        # Accept either yor or eng since undiacritized Yorùbá looks like English
        assert result in ["yor", "eng"]

    def test_detects_by_diacritics(self):
        """Should use Yorùbá diacritics as strong signal."""
        from naijaml.nlp.langdetect import detect_language

        # Text with ọ, ẹ, ṣ should strongly indicate Yorùbá
        result = detect_language("ọmọ ẹ̀gbọ́n mi ṣe dára")
        assert result == "yor"


class TestHausaSpecificFeatures:
    """Tests for Hausa-specific detection features."""

    def test_detects_hausa_patterns(self):
        """Should detect common Hausa patterns."""
        from naijaml.nlp.langdetect import detect_language

        result = detect_language("Yaya kake? Ina lafiya lau")
        assert result == "hau"


class TestPidginSpecificFeatures:
    """Tests for Nigerian Pidgin-specific detection features."""

    def test_detects_pidgin_markers(self):
        """Should detect distinctive Pidgin words."""
        from naijaml.nlp.langdetect import detect_language

        # "dey", "abeg", "wetin", "wahala" are Pidgin markers
        result = detect_language("I dey go market, abeg wait for me")
        assert result == "pcm"

    def test_distinguishes_pidgin_from_english(self):
        """Should distinguish Pidgin from standard English."""
        from naijaml.nlp.langdetect import detect_language

        assert detect_language("I am going to the market") == "eng"
        assert detect_language("I dey go market") == "pcm"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_numbers_only(self):
        """Should handle text with only numbers."""
        from naijaml.nlp.langdetect import detect_language

        result = detect_language("12345 67890")
        # Should return something, not crash
        assert result in ["yor", "hau", "ibo", "pcm", "eng", None]

    def test_handles_special_characters(self):
        """Should handle special characters and punctuation."""
        from naijaml.nlp.langdetect import detect_language

        result = detect_language("!!! ??? ... @#$%")
        assert result in ["yor", "hau", "ibo", "pcm", "eng", None]

    def test_handles_unicode_normalization(self):
        """Should handle different Unicode normalization forms."""
        from naijaml.nlp.langdetect import detect_language

        # Same text, different Unicode representations
        text_nfc = "ọjọ́"  # precomposed
        text_nfd = "ọjọ́"  # with combining characters

        result1 = detect_language(text_nfc)
        result2 = detect_language(text_nfd)
        # Both should detect as Yorùbá (or at least same result)
        assert result1 == result2
