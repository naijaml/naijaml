"""Tests for dot-below-only Yorùbá diacritizer."""
from __future__ import annotations

import pytest


class TestStripDiacriticsTonelOnly:
    """Tests for strip_diacritics with tones_only option."""

    def test_strip_all_diacritics(self):
        """Default behavior should strip all diacritics."""
        from naijaml.nlp.diacritizer import strip_diacritics

        assert strip_diacritics("Ọjọ́ dára") == "Ojo dara"
        assert strip_diacritics("púpọ̀") == "pupo"
        assert strip_diacritics("ẹ̀ṣẹ̀") == "ese"

    def test_strip_tones_only_keeps_dot_below(self):
        """tones_only=True should keep ọ, ẹ, ṣ but remove tones."""
        from naijaml.nlp.diacritizer import strip_diacritics

        assert strip_diacritics("Ọjọ́ dára", tones_only=True) == "Ọjọ dara"
        assert strip_diacritics("púpọ̀", tones_only=True) == "pupọ"
        assert strip_diacritics("ẹ̀ṣẹ̀", tones_only=True) == "ẹṣẹ"
        assert strip_diacritics("iṣẹ́", tones_only=True) == "iṣẹ"

    def test_preserves_case_with_tones_only(self):
        """Should preserve case when stripping tones only."""
        from naijaml.nlp.diacritizer import strip_diacritics

        assert strip_diacritics("Ẹ KÚ IṢẸ́", tones_only=True) == "Ẹ KU IṢẸ"
        assert strip_diacritics("Ọ̀GÁ", tones_only=True) == "ỌGA"


class TestDotBelowDiacritizer:
    """Tests for the DotBelowDiacritizer model."""

    def test_train_and_predict(self):
        """Should train on data and make predictions."""
        from naijaml.nlp.diacritizer import DotBelowDiacritizer

        model = DotBelowDiacritizer()
        model.train([
            "Ọjọ́ dára",
            "Ẹ kú iṣẹ́",
            "Mo fẹ́ràn rẹ",
        ])

        assert model.total_syllables > 0
        assert len(model.syllable_freq) > 0

        # Should produce output (exact accuracy not guaranteed)
        result = model.diacritize("Ojo dara")
        assert result is not None
        assert len(result) > 0

    def test_restores_dot_below_only(self):
        """Should restore ọ, ẹ, ṣ without adding tones."""
        from naijaml.nlp.diacritizer import DotBelowDiacritizer

        model = DotBelowDiacritizer()
        model.train([
            "Ọjọ́ dára púpọ̀",  # Has tones
            "Ẹ kú iṣẹ́ o",     # Has tones
            "ọmọ mi dára",     # Has tones
        ])

        result = model.diacritize("Ojo")
        # Should have dot-below but NO tonal marks
        assert "ọ" in result.lower() or "Ọ" in result
        # Should NOT have acute/grave accents
        assert "́" not in result  # acute
        assert "̀" not in result  # grave

    def test_save_and_load(self, tmp_path):
        """Should save and load model correctly."""
        from naijaml.nlp.diacritizer import DotBelowDiacritizer

        # Train
        model = DotBelowDiacritizer()
        model.train([
            "Ọjọ́ dára púpọ̀",
            "Ẹ kú iṣẹ́ o",
        ])

        # Save
        model_path = tmp_path / "test_dot_below.json"
        model.save(model_path)
        assert model_path.exists()

        # Load
        loaded = DotBelowDiacritizer.load(model_path)
        assert loaded.total_syllables == model.total_syllables

        # Predictions should match
        test_text = "Ojo dara"
        assert model.diacritize(test_text) == loaded.diacritize(test_text)


class TestDiacritizeDotBelow:
    """Tests for the public diacritize_dot_below function."""

    def test_basic_dot_below_diacritization(self):
        """Should restore dot-below characters."""
        from naijaml.nlp.diacritizer import diacritize_dot_below

        result = diacritize_dot_below("Ojo")
        # Should at least get the ọ
        assert "ọ" in result.lower() or "Ọ" in result

    def test_handles_empty_string(self):
        """Should handle empty string gracefully."""
        from naijaml.nlp.diacritizer import diacritize_dot_below

        assert diacritize_dot_below("") == ""
        assert diacritize_dot_below("   ") == "   "

    def test_common_words(self):
        """Should handle common Yorùbá words."""
        from naijaml.nlp.diacritizer import diacritize_dot_below

        # These should get dot-below correct
        result = diacritize_dot_below("omo")
        assert "ọ" in result  # ọmọ

        result = diacritize_dot_below("ise")
        assert "ṣ" in result or "ẹ" in result  # iṣẹ́


class TestEvaluationFunctions:
    """Tests for the evaluation functions."""

    def test_evaluate_dot_below_returns_dict(self):
        """Should return dict with accuracy metrics."""
        from naijaml.nlp.diacritizer import evaluate_dot_below_accuracy

        # Use a small test set
        test_texts = [
            "Ọjọ́ dára púpọ̀",
            "Ẹ kú iṣẹ́ o",
        ]

        results = evaluate_dot_below_accuracy(test_texts, verbose=False)

        assert "char_accuracy" in results
        assert "word_accuracy" in results
        assert "dot_below_precision" in results
        assert "dot_below_recall" in results
        assert "dot_below_f1" in results

        assert 0.0 <= results["char_accuracy"] <= 1.0
        assert 0.0 <= results["word_accuracy"] <= 1.0

    def test_evaluate_full_returns_dict(self):
        """Should return dict with accuracy metrics."""
        from naijaml.nlp.diacritizer import evaluate_full_diacritization_accuracy

        test_texts = [
            "Ọjọ́ dára púpọ̀",
            "Ẹ kú iṣẹ́ o",
        ]

        results = evaluate_full_diacritization_accuracy(test_texts, verbose=False)

        assert "char_accuracy" in results
        assert "word_accuracy" in results
        assert 0.0 <= results["char_accuracy"] <= 1.0

    def test_compare_methods_returns_both(self):
        """Should return results for both methods."""
        from naijaml.nlp.diacritizer import compare_diacritization_methods

        test_texts = [
            "Ọjọ́ dára púpọ̀",
        ]

        results = compare_diacritization_methods(test_texts)

        assert "dot_below_only" in results
        assert "full_diacritization" in results
        assert "char_accuracy" in results["dot_below_only"]
        assert "char_accuracy" in results["full_diacritization"]


class TestImports:
    """Test that new functions are properly exported."""

    def test_import_from_nlp(self):
        """Should be importable from naijaml.nlp."""
        from naijaml.nlp import (
            diacritize_dot_below,
            diacritize_yoruba_dot_below,
            train_and_save_dot_below_model,
            evaluate_dot_below_accuracy,
            evaluate_full_diacritization_accuracy,
            compare_diacritization_methods,
            strip_yoruba_diacritics,
        )

        assert callable(diacritize_dot_below)
        assert callable(diacritize_yoruba_dot_below)
        assert callable(train_and_save_dot_below_model)
        assert callable(evaluate_dot_below_accuracy)
        assert callable(evaluate_full_diacritization_accuracy)
        assert callable(compare_diacritization_methods)
        assert callable(strip_yoruba_diacritics)

    def test_strip_yoruba_diacritics_has_tones_only(self):
        """strip_yoruba_diacritics should accept tones_only parameter."""
        from naijaml.nlp import strip_yoruba_diacritics

        assert strip_yoruba_diacritics("Ọjọ́", tones_only=True) == "Ọjọ"
        assert strip_yoruba_diacritics("Ọjọ́", tones_only=False) == "Ojo"
