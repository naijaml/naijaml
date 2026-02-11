"""Tests for Igbo diacritizer."""
from __future__ import annotations

import pytest


class TestSyllabify:
    """Tests for Igbo syllable segmentation."""

    def test_cv_syllables(self):
        """Should segment consonant-vowel patterns."""
        from naijaml.nlp.igbo_diacritizer import syllabify

        assert syllabify("biko") == ["bi", "ko"]
        # "nwanne" can be ['nwa', 'n', 'ne'] or ['nwa', 'nne'] depending on analysis
        # Our syllabifier treats the second 'n' as syllabic
        result = syllabify("nwanne")
        assert result[0] == "nwa"
        assert "ne" in result[-1]  # ends with 'ne'
        assert syllabify("ala") == ["a", "la"]

    def test_v_syllables(self):
        """Should segment vowel-only syllables."""
        from naijaml.nlp.igbo_diacritizer import syllabify

        assert syllabify("ala") == ["a", "la"]
        assert syllabify("ego") == ["e", "go"]
        assert syllabify("ike") == ["i", "ke"]

    def test_dotted_vowels(self):
        """Should handle ị, ọ, ụ correctly."""
        from naijaml.nlp.igbo_diacritizer import syllabify

        assert syllabify("ọbụla") == ["ọ", "bụ", "la"]
        assert syllabify("ịhụ") == ["ị", "hụ"]
        assert syllabify("ụlọ") == ["ụ", "lọ"]

    def test_digraphs(self):
        """Should handle Igbo digraphs (gb, kp, nw, etc.)."""
        from naijaml.nlp.igbo_diacritizer import syllabify

        assert syllabify("agba") == ["a", "gba"]
        assert syllabify("okpoko") == ["o", "kpo", "ko"]
        assert syllabify("nwoke") == ["nwo", "ke"]

    def test_nasal_syllables(self):
        """Should handle syllabic nasals."""
        from naijaml.nlp.igbo_diacritizer import syllabify

        # 'n' or 'm' before consonant can be syllabic
        result = syllabify("nke")
        assert len(result) >= 1

    def test_empty_string(self):
        """Should handle empty string."""
        from naijaml.nlp.igbo_diacritizer import syllabify

        assert syllabify("") == []


class TestStripDiacritics:
    """Tests for diacritic removal."""

    def test_removes_dot_below(self):
        """Should convert ị→i, ọ→o, ụ→u."""
        from naijaml.nlp.igbo_diacritizer import strip_diacritics

        assert strip_diacritics("ịhụ") == "ihu"
        assert strip_diacritics("ọbụla") == "obula"
        assert strip_diacritics("ụlọ") == "ulo"

    def test_removes_tonal_marks(self):
        """Should remove acute and grave accents."""
        from naijaml.nlp.igbo_diacritizer import strip_diacritics

        assert strip_diacritics("ńdị") == "ndi"
        assert strip_diacritics("ànyị") == "anyi"

    def test_preserves_case(self):
        """Should preserve uppercase/lowercase."""
        from naijaml.nlp.igbo_diacritizer import strip_diacritics

        assert strip_diacritics("Ọnụ") == "Onu"
        assert strip_diacritics("ỊHỤ") == "IHU"

    def test_preserves_spaces_punctuation(self):
        """Should preserve spaces and punctuation."""
        from naijaml.nlp.igbo_diacritizer import strip_diacritics

        assert strip_diacritics("Ọ dị mma!") == "O di mma!"
        assert strip_diacritics("Kedụ?") == "Kedu?"


class TestIgboDiacritizer:
    """Tests for the diacritizer model."""

    def test_train_and_predict(self):
        """Should train on data and make predictions."""
        from naijaml.nlp.igbo_diacritizer import IgboDiacritizer

        model = IgboDiacritizer()
        model.train([
            "Ọ bụ ezie",
            "Kedụ ka ị mere?",
            "Anyị na-agụ akwụkwọ",
        ])

        assert model.total_syllables > 0
        assert len(model.syllable_freq) > 0

        # Should produce some output
        result = model.diacritize("O bu ezie")
        assert result is not None
        assert len(result) > 0

    def test_preserves_non_igbo_chars(self):
        """Should preserve numbers, punctuation, etc."""
        from naijaml.nlp.igbo_diacritizer import IgboDiacritizer

        model = IgboDiacritizer()
        model.train(["Ọ bụ 2024!"])

        result = model.diacritize("O bu 2024!")
        assert "2024" in result
        assert "!" in result

    def test_save_and_load(self, tmp_path):
        """Should save and load model correctly."""
        from naijaml.nlp.igbo_diacritizer import IgboDiacritizer

        # Train
        model = IgboDiacritizer()
        model.train([
            "Ọ bụ ezie na ọ dị mma",
            "Anyị na-anatakwa ihe ọbụla",
        ])

        # Save
        model_path = tmp_path / "test_igbo_diacritizer.json"
        model.save(model_path)
        assert model_path.exists()

        # Load
        loaded = IgboDiacritizer.load(model_path)
        assert loaded.total_syllables == model.total_syllables

        # Predictions should match
        test_text = "O bu ezie"
        assert model.diacritize(test_text) == loaded.diacritize(test_text)


class TestDiacritizeIgbo:
    """Tests for the public diacritize_igbo function."""

    def test_basic_diacritization(self):
        """Should restore basic diacritics."""
        from naijaml.nlp.igbo_diacritizer import diacritize_igbo

        result = diacritize_igbo("O bu ezie")
        # Should get the ọ and ụ
        assert "ọ" in result.lower() or "Ọ" in result
        assert "ụ" in result.lower()

    def test_handles_empty_string(self):
        """Should handle empty string gracefully."""
        from naijaml.nlp.igbo_diacritizer import diacritize_igbo

        assert diacritize_igbo("") == ""
        assert diacritize_igbo("   ") == "   "

    def test_common_words(self):
        """Should handle common Igbo words."""
        from naijaml.nlp.igbo_diacritizer import diacritize_igbo

        # These should get the dot-below correct
        result = diacritize_igbo("ulo")
        assert "ụ" in result or "ọ" in result  # ụlọ

        result = diacritize_igbo("anyi")
        assert "ị" in result  # anyị

    def test_sentence_diacritization(self):
        """Should handle full sentences."""
        from naijaml.nlp.igbo_diacritizer import diacritize_igbo

        result = diacritize_igbo("O di mma, daalu")

        # Should contain some diacritized characters
        has_diacritics = any(c in result for c in "ịọụỊỌỤ")
        assert has_diacritics, f"Expected diacritics in: {result}"

    def test_round_trip(self):
        """Stripping then diacritizing should approximately restore original."""
        from naijaml.nlp.igbo_diacritizer import diacritize_igbo, strip_diacritics

        original = "Ọ bụ ezie na anyị na-anatakwa"
        stripped = strip_diacritics(original)
        restored = diacritize_igbo(stripped)

        # Count diacritized characters
        orig_diacritics = sum(1 for c in original if c in "ịọụỊỌỤ")
        restored_diacritics = sum(1 for c in restored if c in "ịọụỊỌỤ")

        # Should restore most diacritics (allow some error)
        assert restored_diacritics >= orig_diacritics * 0.7, \
            f"Expected more diacritics. Original: {original}, Restored: {restored}"


class TestIntegration:
    """Integration tests for the Igbo diacritizer."""

    def test_import_from_package(self):
        """Should be importable from naijaml.nlp."""
        from naijaml.nlp import diacritize_igbo, syllabify_igbo

        assert callable(diacritize_igbo)
        assert callable(syllabify_igbo)

    def test_both_diacritizers_available(self):
        """Both Yorùbá and Igbo diacritizers should be importable."""
        from naijaml.nlp import diacritize, diacritize_yoruba, diacritize_igbo

        # Yorùbá
        assert callable(diacritize)
        assert callable(diacritize_yoruba)

        # Igbo
        assert callable(diacritize_igbo)

        # They should produce different results for language-specific text
        yoruba_result = diacritize("ojo dara")
        igbo_result = diacritize_igbo("o di mma")

        # Both should produce output
        assert len(yoruba_result) > 0
        assert len(igbo_result) > 0

    def test_igbo_specific_patterns(self):
        """Should handle Igbo-specific patterns."""
        from naijaml.nlp import diacritize_igbo

        # Common Igbo words/phrases
        tests = [
            ("ndi", "ị"),      # ndị (people)
            ("ulo", "ụ"),      # ụlọ (house)
            ("anyi", "ị"),     # anyị (we)
        ]

        for undiac, expected_char in tests:
            result = diacritize_igbo(undiac)
            assert expected_char in result.lower(), \
                f"Expected '{expected_char}' in diacritization of '{undiac}', got '{result}'"
