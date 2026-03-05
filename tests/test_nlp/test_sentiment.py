"""Tests for sentiment analysis module."""
from __future__ import annotations

import pytest


class TestAnalyzeSentiment:
    """Test analyze_sentiment function."""

    def test_positive_english(self):
        from naijaml.nlp import analyze_sentiment
        result = analyze_sentiment("This is great, I love it!")
        assert result["label"] in {"positive", "negative", "neutral"}
        assert "confidence" in result
        assert "scores" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_negative_pidgin(self):
        from naijaml.nlp import analyze_sentiment
        result = analyze_sentiment("I no like am at all, e bad")
        assert result["label"] in {"positive", "negative", "neutral"}

    def test_result_structure(self):
        from naijaml.nlp import analyze_sentiment
        result = analyze_sentiment("Test text")
        assert "label" in result
        assert "confidence" in result
        assert "scores" in result
        assert isinstance(result["scores"], dict)
        # Scores should sum to ~1.0
        total = sum(result["scores"].values())
        assert 0.99 <= total <= 1.01

    def test_empty_returns_neutral(self):
        from naijaml.nlp import analyze_sentiment
        result = analyze_sentiment("")
        assert result["label"] == "neutral"


class TestGetSentiment:
    """Test get_sentiment simplified API."""

    def test_returns_string(self):
        from naijaml.nlp import get_sentiment
        result = get_sentiment("This is good")
        assert isinstance(result, str)
        assert result in {"positive", "negative", "neutral"}


class TestGetSentimentWithConfidence:
    """Test get_sentiment_with_confidence."""

    def test_returns_tuple(self):
        from naijaml.nlp import get_sentiment_with_confidence
        label, conf = get_sentiment_with_confidence("Test text")
        assert isinstance(label, str)
        assert isinstance(conf, float)
        assert label in {"positive", "negative", "neutral"}
        assert 0.0 <= conf <= 1.0


class TestAnalyzeBatch:
    """Test batch sentiment analysis."""

    def test_batch_returns_list(self):
        from naijaml.nlp import analyze_sentiment_batch
        texts = ["good", "bad", "okay"]
        results = analyze_sentiment_batch(texts)
        assert len(results) == 3
        for r in results:
            assert "label" in r

    def test_empty_batch(self):
        from naijaml.nlp import analyze_sentiment_batch
        assert analyze_sentiment_batch([]) == []

    def test_single_item_batch(self):
        from naijaml.nlp import analyze_sentiment_batch
        results = analyze_sentiment_batch(["hello"])
        assert len(results) == 1


class TestIsAvailable:
    """Test is_available check."""

    def test_returns_bool(self):
        from naijaml.nlp import is_sentiment_available
        result = is_sentiment_available()
        assert isinstance(result, bool)
