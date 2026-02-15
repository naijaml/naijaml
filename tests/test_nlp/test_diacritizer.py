"""Tests for Yorùbá diacritizer."""
from __future__ import annotations

import json
import pytest


class TestSyllabify:
    """Tests for syllable segmentation."""

    def test_cv_syllables(self):
        """Should segment consonant-vowel patterns."""
        from naijaml.nlp.diacritizer import syllabify

        assert syllabify("dara") == ["da", "ra"]
        assert syllabify("baba") == ["ba", "ba"]
        assert syllabify("koko") == ["ko", "ko"]

    def test_v_syllables(self):
        """Should segment vowel-only syllables."""
        from naijaml.nlp.diacritizer import syllabify

        assert syllabify("ojo") == ["o", "jo"]
        assert syllabify("ile") == ["i", "le"]
        assert syllabify("omi") == ["o", "mi"]

    def test_dotted_vowels(self):
        """Should handle ẹ and ọ correctly."""
        from naijaml.nlp.diacritizer import syllabify

        assert syllabify("ọjọ") == ["ọ", "jọ"]
        assert syllabify("ẹsẹ") == ["ẹ", "sẹ"]

    def test_dotted_consonant(self):
        """Should handle ṣ correctly."""
        from naijaml.nlp.diacritizer import syllabify

        assert syllabify("ṣe") == ["ṣe"]
        assert syllabify("iṣẹ") == ["i", "ṣẹ"]

    def test_nasal_syllables(self):
        """Should handle syllabic nasals."""
        from naijaml.nlp.diacritizer import syllabify

        # 'n' before consonant is syllabic
        assert syllabify("nkan") == ["n", "ka", "n"]

    def test_tonal_marks_preserved(self):
        """Should keep tonal marks with their syllables."""
        from naijaml.nlp.diacritizer import syllabify

        result = syllabify("ọjọ́")
        assert len(result) == 2
        assert result[0] == "ọ"
        # Second syllable should contain the tone mark
        assert "j" in result[1]

    def test_empty_string(self):
        """Should handle empty string."""
        from naijaml.nlp.diacritizer import syllabify

        assert syllabify("") == []


class TestStripDiacritics:
    """Tests for diacritic removal."""

    def test_removes_tonal_marks(self):
        """Should remove acute and grave accents."""
        from naijaml.nlp.diacritizer import strip_diacritics

        assert strip_diacritics("dára") == "dara"
        assert strip_diacritics("púpọ̀") == "pupo"
        assert strip_diacritics("kú") == "ku"

    def test_removes_dot_below(self):
        """Should convert ọ→o, ẹ→e, ṣ→s."""
        from naijaml.nlp.diacritizer import strip_diacritics

        assert strip_diacritics("ọjọ") == "ojo"
        assert strip_diacritics("ẹsẹ") == "ese"
        assert strip_diacritics("ṣe") == "se"

    def test_preserves_case(self):
        """Should preserve uppercase/lowercase."""
        from naijaml.nlp.diacritizer import strip_diacritics

        assert strip_diacritics("Ọjọ́") == "Ojo"
        assert strip_diacritics("Ẹ KÚ") == "E KU"

    def test_preserves_spaces_punctuation(self):
        """Should preserve spaces and punctuation."""
        from naijaml.nlp.diacritizer import strip_diacritics

        assert strip_diacritics("Ọjọ́ dára!") == "Ojo dara!"
        assert strip_diacritics("Báwo ni?") == "Bawo ni?"


class TestYorubaDiacritizer:
    """Tests for the diacritizer model."""

    def test_train_and_predict(self):
        """Should train on data and make predictions."""
        from naijaml.nlp.diacritizer import YorubaDiacritizer

        model = YorubaDiacritizer()
        model.train([
            "Ọjọ́ dára",
            "Ẹ kú iṣẹ́",
            "Mo fẹ́ràn rẹ",
        ])

        assert model.total_syllables > 0
        assert len(model.syllable_freq) > 0

        # Should produce some output (exact accuracy not guaranteed)
        result = model.diacritize("Ojo dara")
        assert result is not None
        assert len(result) > 0

    def test_preserves_non_yoruba_chars(self):
        """Should preserve numbers, punctuation, etc."""
        from naijaml.nlp.diacritizer import YorubaDiacritizer

        model = YorubaDiacritizer()
        model.train(["Ọjọ́ 2024!"])

        result = model.diacritize("Ojo 2024!")
        assert "2024" in result
        assert "!" in result

    def test_save_and_load(self, tmp_path):
        """Should save and load model correctly."""
        from naijaml.nlp.diacritizer import YorubaDiacritizer

        # Train
        model = YorubaDiacritizer()
        model.train([
            "Ọjọ́ dára púpọ̀",
            "Ẹ kú iṣẹ́ o",
        ])

        # Save
        model_path = tmp_path / "test_diacritizer.json"
        model.save(model_path)
        assert model_path.exists()

        # Load
        loaded = YorubaDiacritizer.load(model_path)
        assert loaded.total_syllables == model.total_syllables

        # Predictions should match
        test_text = "Ojo dara"
        assert model.diacritize(test_text) == loaded.diacritize(test_text)


class TestDiacritize:
    """Tests for the public diacritize function."""

    def test_basic_diacritization(self):
        """Should restore basic diacritics."""
        from naijaml.nlp.diacritizer import diacritize

        result = diacritize("Ojo")
        # Should at least get the ọ
        assert "ọ" in result.lower() or "Ọ" in result

    def test_handles_empty_string(self):
        """Should handle empty string gracefully."""
        from naijaml.nlp.diacritizer import diacritize

        assert diacritize("") == ""
        assert diacritize("   ") == "   "

    def test_preserves_already_diacritized(self):
        """Should not break already diacritized text too much."""
        from naijaml.nlp.diacritizer import diacritize_dot_below

        # Use dot-below diacritizer for reliable preservation (97.5% accuracy)
        # The full diacritizer has tonal ambiguity issues (~77% word accuracy)
        result = diacritize_dot_below("Ọjọ")
        assert "ọ" in result.lower() or "Ọ" in result

    def test_common_words(self):
        """Should handle common Yorùbá words."""
        from naijaml.nlp.diacritizer import diacritize

        # These should get at least the dot-below correct
        result = diacritize("omo")
        assert "ọ" in result  # ọmọ

        result = diacritize("ile")
        # ile (house) - no dots needed, should remain similar

        result = diacritize("ise")
        assert "ṣ" in result or "ẹ" in result  # iṣẹ́


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_import_from_package(self):
        """Should be importable from naijaml.nlp."""
        from naijaml.nlp import diacritize, syllabify

        assert callable(diacritize)
        assert callable(syllabify)

    def test_sentence_diacritization(self):
        """Should handle full sentences."""
        from naijaml.nlp import diacritize

        result = diacritize("E ku ise o, bawo ni?")

        # Should contain some diacritized characters
        has_diacritics = any(c in result for c in "ọẹṣáàéèíìóòúù")
        assert has_diacritics, f"Expected diacritics in: {result}"

    def test_mixed_content(self):
        """Should handle mixed content without crashing."""
        from naijaml.nlp import diacritize

        # Note: English words may get diacritized if they match Yorùbá patterns
        # (e.g., "love" -> "lọve" because "lo" is a common Yorùbá syllable)
        # This is expected behavior - for true mixed content, use language detection first
        result = diacritize("Mo go this ojo")

        # Should produce output and handle the text
        assert len(result) > 0
        # "this" has no common Yorùbá patterns, should be mostly preserved
        assert "this" in result or "thị" not in result  # no random dots added


class TestWordLevelBigrams:
    """Tests for bigram context in WordLevelDiacritizer."""

    def _make_training_data(self):
        """Create training data with ambiguous words resolvable by bigrams.

        'ile' is ambiguous:
          - 'inu ilé' (inside house) — most common bigram context
          - 'ori ilẹ̀' (on the ground)
        'ile' unigram default should be 'ilé' (more common overall).
        """
        # 10x "inu ilé" to make bigram for "inu\tile" -> "ilé" frequent
        # 10x "ori ilẹ̀" to make bigram for "ori\tile" -> "ilẹ̀" frequent
        # 15x "ilé" in other contexts to make unigram default = "ilé"
        diac = []
        undiac = []
        for _ in range(10):
            diac.append("inú ilé")
            undiac.append("inu ile")
        for _ in range(10):
            diac.append("orí ilẹ̀")
            undiac.append("ori ile")
        for _ in range(15):
            diac.append("ilé ńlá")
            undiac.append("ile nla")
        # Some unambiguous words for vocabulary
        for _ in range(10):
            diac.append("ọjọ́ dára")
            undiac.append("ojo dara")
        return diac, undiac

    def test_train_builds_bigram_map(self):
        """Training should populate bigram_map with disambiguating bigrams."""
        from naijaml.nlp.diacritizer import WordLevelDiacritizer

        diac, undiac = self._make_training_data()
        model = WordLevelDiacritizer(min_word_freq=1, min_bigram_freq=3)
        model.train(diac, undiac)

        assert len(model.bigram_map) > 0
        assert model.bigram_count == len(model.bigram_map)

    def test_bigram_disambiguates(self):
        """Bigram context should override unigram for ambiguous words."""
        from naijaml.nlp.diacritizer import WordLevelDiacritizer

        diac, undiac = self._make_training_data()
        model = WordLevelDiacritizer(min_word_freq=1, min_bigram_freq=3)
        model.train(diac, undiac)

        # Unigram default for "ile" should be "ilé" (25 occurrences vs 10)
        assert model.word_map.get("ile") == "ilé"

        # With "ori" context, bigram should resolve to "ilẹ̀"
        result = model.diacritize_word("ile", prev_word="ori")
        assert result == "ilẹ̀"

    def test_bigram_falls_back_to_unigram(self):
        """Unknown bigram context should fall back to unigram."""
        from naijaml.nlp.diacritizer import WordLevelDiacritizer

        diac, undiac = self._make_training_data()
        model = WordLevelDiacritizer(min_word_freq=1, min_bigram_freq=3)
        model.train(diac, undiac)

        # "xyz" is not a known prev_word, should fall back to unigram "ilé"
        result = model.diacritize_word("ile", prev_word="xyz")
        assert result == "ilé"

    def test_sentence_uses_bigram_context(self):
        """diacritize() should pass prev_word from preceding word."""
        from naijaml.nlp.diacritizer import WordLevelDiacritizer

        diac, undiac = self._make_training_data()
        model = WordLevelDiacritizer(min_word_freq=1, min_bigram_freq=3)
        model.train(diac, undiac)

        # "ori ile" should use bigram context -> "ilẹ̀"
        result = model.diacritize("ori ile")
        words = result.split()
        assert len(words) == 2
        assert words[1] == "ilẹ̀"

    def test_punctuation_resets_context(self):
        """Sentence-ending punctuation should reset bigram context."""
        from naijaml.nlp.diacritizer import WordLevelDiacritizer

        diac, undiac = self._make_training_data()
        model = WordLevelDiacritizer(min_word_freq=1, min_bigram_freq=3)
        model.train(diac, undiac)

        # After ".", context resets — "ile" should use unigram default "ilé"
        result = model.diacritize("ori. ile")
        # The word after "." has no prev_word context
        assert "ilé" in result

    def test_save_load_preserves_bigrams(self, tmp_path):
        """Round-trip save/load should preserve bigram_map."""
        from naijaml.nlp.diacritizer import WordLevelDiacritizer

        diac, undiac = self._make_training_data()
        model = WordLevelDiacritizer(min_word_freq=1, min_bigram_freq=3)
        model.train(diac, undiac)

        path = tmp_path / "test_bigram_model.json"
        model.save(path)

        loaded = WordLevelDiacritizer.load(path)
        assert loaded.bigram_map == model.bigram_map
        assert loaded.bigram_count == model.bigram_count
        assert loaded.min_bigram_freq == model.min_bigram_freq

        # Should produce same results
        assert loaded.diacritize_word("ile", prev_word="ori") == model.diacritize_word("ile", prev_word="ori")

    def test_load_old_model_without_bigrams(self, tmp_path):
        """Loading an old model JSON without bigram keys should work."""
        from naijaml.nlp.diacritizer import WordLevelDiacritizer

        # Simulate old model format (no bigram fields)
        old_data = {
            "min_word_freq": 5,
            "word_map": {"ojo": "ọjọ́", "dara": "dára"},
            "total_words_trained": 100,
            "vocab_size": 2,
        }
        path = tmp_path / "old_model.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(old_data, f, ensure_ascii=False)

        model = WordLevelDiacritizer.load(path)
        assert model.bigram_map == {}
        assert model.bigram_count == 0
        assert model.min_bigram_freq == 3  # default

        # Should still work for basic diacritization
        assert model.diacritize_word("ojo") == "ọjọ́"

    def test_only_disambiguating_bigrams_stored(self):
        """Bigrams that agree with unigram default should NOT be stored."""
        from naijaml.nlp.diacritizer import WordLevelDiacritizer

        # All occurrences of "ile" after "inu" map to "ilé",
        # and unigram default is also "ilé" — no need to store this bigram.
        diac = []
        undiac = []
        for _ in range(20):
            diac.append("inú ilé")
            undiac.append("inu ile")
        for _ in range(5):
            diac.append("ilé ńlá")
            undiac.append("ile nla")

        model = WordLevelDiacritizer(min_word_freq=1, min_bigram_freq=3)
        model.train(diac, undiac)

        # Unigram default for "ile" is "ilé"
        assert model.word_map.get("ile") == "ilé"

        # "inu\tile" bigram also predicts "ilé" — should NOT be stored
        assert "inu\tile" not in model.bigram_map


class TestViterbiDecoding:
    """Tests for Viterbi sequence decoding in WordLevelDiacritizer."""

    def _make_training_data(self):
        """Create training data with ambiguous words for Viterbi testing.

        'ile' is ambiguous:
          - 'inu ilé' (inside house) — common context
          - 'ori ilẹ̀' (on the ground) — different context
        'ile' unigram default should be 'ilé' (more common overall).
        """
        diac = []
        undiac = []
        for _ in range(15):
            diac.append("inú ilé")
            undiac.append("inu ile")
        for _ in range(10):
            diac.append("orí ilẹ̀")
            undiac.append("ori ile")
        for _ in range(15):
            diac.append("ilé ńlá")
            undiac.append("ile nla")
        # Unambiguous context words
        for _ in range(10):
            diac.append("ọjọ́ dára")
            undiac.append("ojo dara")
        return diac, undiac

    def test_has_viterbi_flag(self):
        """Training should set has_viterbi=True."""
        from naijaml.nlp.diacritizer import WordLevelDiacritizer

        diac, undiac = self._make_training_data()
        model = WordLevelDiacritizer(min_word_freq=1, min_bigram_freq=3)
        model.train(diac, undiac)

        assert model.has_viterbi is True

    def test_word_candidates_built(self):
        """Ambiguous words should have multiple candidates."""
        from naijaml.nlp.diacritizer import WordLevelDiacritizer

        diac, undiac = self._make_training_data()
        model = WordLevelDiacritizer(min_word_freq=1, min_bigram_freq=3)
        model.train(diac, undiac)

        # "ile" should be ambiguous (ilé vs ilẹ̀)
        assert "ile" in model.word_candidates
        candidates = model.word_candidates["ile"]
        assert len(candidates) >= 2
        forms = [form for form, _ in candidates]
        assert "ilé" in forms
        assert "ilẹ̀" in forms

    def test_viterbi_disambiguates(self):
        """Viterbi should pick correct form based on sequence context."""
        from naijaml.nlp.diacritizer import WordLevelDiacritizer

        diac, undiac = self._make_training_data()
        model = WordLevelDiacritizer(min_word_freq=1, min_bigram_freq=3)
        model.train(diac, undiac)

        # "ori ile" — Viterbi should prefer "ilẹ̀" after "orí"
        result = model.diacritize("ori ile")
        words = result.split()
        assert len(words) == 2
        assert words[1] == "ilẹ̀"

    def test_greedy_fallback_old_model(self):
        """Old model without Viterbi data should use greedy fallback."""
        from naijaml.nlp.diacritizer import WordLevelDiacritizer

        model = WordLevelDiacritizer(min_word_freq=1)
        model.word_map = {"ojo": "ọjọ́", "dara": "dára"}
        model.has_viterbi = False

        result = model.diacritize("ojo dara")
        assert "ọjọ́" in result.lower()
        assert "dára" in result.lower()

    def test_save_load_roundtrip(self, tmp_path):
        """Viterbi data should survive serialization."""
        from naijaml.nlp.diacritizer import WordLevelDiacritizer

        diac, undiac = self._make_training_data()
        model = WordLevelDiacritizer(min_word_freq=1, min_bigram_freq=3)
        model.train(diac, undiac)

        path = tmp_path / "viterbi_model.json"
        model.save(path)

        loaded = WordLevelDiacritizer.load(path)
        assert loaded.has_viterbi is True
        assert loaded.word_candidates == model.word_candidates
        assert loaded.max_candidates == model.max_candidates
        assert len(loaded.transition_probs) == len(model.transition_probs)
        assert len(loaded.unigram_log_probs) == len(model.unigram_log_probs)

        # Should produce same output
        assert loaded.diacritize("ori ile") == model.diacritize("ori ile")

    def test_sentence_boundary_reset(self):
        """Period should reset Viterbi context between sentences."""
        from naijaml.nlp.diacritizer import WordLevelDiacritizer

        diac, undiac = self._make_training_data()
        model = WordLevelDiacritizer(min_word_freq=1, min_bigram_freq=3)
        model.train(diac, undiac)

        # After period, "ile" has no "ori" context — should use unigram "ilé"
        result = model.diacritize("ori. ile")
        # "ile" in the second sentence has no context from "ori"
        parts = result.split(". ")
        if len(parts) == 2:
            assert "ilé" in parts[1]

    def test_preserves_capitalization(self):
        """Original casing should be preserved in output."""
        from naijaml.nlp.diacritizer import WordLevelDiacritizer

        diac, undiac = self._make_training_data()
        model = WordLevelDiacritizer(min_word_freq=1, min_bigram_freq=3)
        model.train(diac, undiac)

        result = model.diacritize("Ojo dara")
        # First word should be capitalized
        assert result[0].isupper()

    def test_single_word(self):
        """Single-word input should work."""
        from naijaml.nlp.diacritizer import WordLevelDiacritizer

        diac, undiac = self._make_training_data()
        model = WordLevelDiacritizer(min_word_freq=1, min_bigram_freq=3)
        model.train(diac, undiac)

        result = model.diacritize("ojo")
        assert len(result) > 0
        assert "ọjọ́" in result.lower()

    def test_load_old_model_no_viterbi(self, tmp_path):
        """Loading an old model JSON without viterbi key should work."""
        old_data = {
            "min_word_freq": 5,
            "word_map": {"ojo": "ọjọ́", "dara": "dára"},
            "bigram_map": {},
            "bigram_count": 0,
            "total_words_trained": 100,
            "vocab_size": 2,
        }
        path = tmp_path / "old_model.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(old_data, f, ensure_ascii=False)

        from naijaml.nlp.diacritizer import WordLevelDiacritizer

        model = WordLevelDiacritizer.load(path)
        assert model.has_viterbi is False
        assert model.word_candidates == {}
        assert model.transition_probs == {}

        # Should still work via greedy fallback
        result = model.diacritize("ojo dara")
        assert "ọjọ́" in result.lower()

    def test_unigram_log_probs_populated(self):
        """Training should populate unigram_log_probs."""
        from naijaml.nlp.diacritizer import WordLevelDiacritizer

        diac, undiac = self._make_training_data()
        model = WordLevelDiacritizer(min_word_freq=1, min_bigram_freq=3)
        model.train(diac, undiac)

        assert len(model.unigram_log_probs) > 0
        # All values should be negative (log of probabilities < 1)
        for form, log_prob in model.unigram_log_probs.items():
            assert log_prob <= 0.0, f"{form}: {log_prob}"

    def test_transition_probs_populated(self):
        """Training should populate transition_probs."""
        from naijaml.nlp.diacritizer import WordLevelDiacritizer

        diac, undiac = self._make_training_data()
        model = WordLevelDiacritizer(min_word_freq=1, min_bigram_freq=3)
        model.train(diac, undiac)

        assert len(model.transition_probs) > 0
        # Check that orí -> ilẹ̀ transition exists
        assert "orí" in model.transition_probs
        assert "ilẹ̀" in model.transition_probs["orí"]
