"""Tests for NaijaML Tokenizer."""

import pytest
from naijaml.nlp import Tokenizer, tokenize, count_tokens, TOKENIZER_LANGUAGES


class TestTokenizerInit:
    """Test tokenizer initialization."""

    def test_init_yoruba(self):
        tok = Tokenizer("yoruba")
        assert tok.language == "yoruba"
        assert tok.vocab_size > 0

    def test_init_short_code(self):
        tok = Tokenizer("yor")
        assert tok.language == "yor"

    def test_init_all_languages(self):
        for lang in ["yoruba", "igbo", "hausa", "pidgin", "naija"]:
            tok = Tokenizer(lang)
            assert tok.vocab_size > 0

    def test_init_invalid_language(self):
        with pytest.raises(ValueError, match="Unknown language"):
            Tokenizer("french")

    def test_case_insensitive(self):
        tok1 = Tokenizer("YORUBA")
        tok2 = Tokenizer("Yoruba")
        tok3 = Tokenizer("yoruba")
        assert tok1.vocab_size == tok2.vocab_size == tok3.vocab_size


class TestEncodeDecode:
    """Test encoding and decoding."""

    @pytest.fixture
    def yoruba_tok(self):
        return Tokenizer("yoruba")

    @pytest.fixture
    def naija_tok(self):
        return Tokenizer("naija")

    def test_encode_basic(self, yoruba_tok):
        ids = yoruba_tok.encode("Ẹ kú àbọ̀")
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_decode_basic(self, yoruba_tok):
        ids = yoruba_tok.encode("Ẹ kú àbọ̀")
        text = yoruba_tok.decode(ids)
        assert isinstance(text, str)

    def test_roundtrip_yoruba(self, yoruba_tok):
        """Critical test: encode then decode should give back original."""
        original = "Ẹ kú àbọ̀"
        ids = yoruba_tok.encode(original)
        decoded = yoruba_tok.decode(ids)
        assert decoded == original

    def test_roundtrip_with_diacritics(self, yoruba_tok):
        """Test that compound diacritics survive roundtrip."""
        test_cases = [
            "ọ́",  # o + dot-below + acute
            "ẹ̀",  # e + dot-below + grave
            "Ọ̀rọ̀ àròsọ ni yìí",
            "Mo fẹ́ràn ẹ̀rọ ìbánisọ̀rọ̀",
        ]
        for text in test_cases:
            ids = yoruba_tok.encode(text)
            decoded = yoruba_tok.decode(ids)
            assert decoded == text, f"Roundtrip failed for: {text}"

    def test_roundtrip_igbo(self):
        tok = Tokenizer("igbo")
        text = "Kedụ ka ị mere"
        assert tok.decode(tok.encode(text)) == text

    def test_roundtrip_hausa(self):
        tok = Tokenizer("hausa")
        text = "Ina kwana?"
        assert tok.decode(tok.encode(text)) == text

    def test_roundtrip_pidgin(self):
        tok = Tokenizer("pidgin")
        text = "How far, wetin dey happen?"
        assert tok.decode(tok.encode(text)) == text

    def test_roundtrip_naija_unified(self, naija_tok):
        """Test unified tokenizer on all languages."""
        texts = [
            "Ẹ kú àbọ̀",  # Yoruba
            "Kedụ ka ị mere",  # Igbo
            "Ina kwana?",  # Hausa
            "How far, wetin dey happen?",  # Pidgin
        ]
        for text in texts:
            assert naija_tok.decode(naija_tok.encode(text)) == text


class TestTokenize:
    """Test tokenize function."""

    def test_tokenize_returns_strings(self):
        tok = Tokenizer("yoruba")
        tokens = tok.tokenize("Ẹ kú àbọ̀")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_tokenize_convenience_function(self):
        tokens = tokenize("Ẹ kú àbọ̀", lang="yoruba")
        assert isinstance(tokens, list)
        assert len(tokens) > 0


class TestBatchOperations:
    """Test batch encoding/decoding."""

    def test_encode_batch(self):
        tok = Tokenizer("yoruba")
        texts = ["Ẹ kú àbọ̀", "Báwo ni?", "Mo wá"]
        ids_batch = tok.encode_batch(texts)
        assert len(ids_batch) == 3
        assert all(isinstance(ids, list) for ids in ids_batch)

    def test_decode_batch(self):
        tok = Tokenizer("yoruba")
        texts = ["Ẹ kú àbọ̀", "Báwo ni?", "Mo wá"]
        ids_batch = tok.encode_batch(texts)
        decoded = tok.decode_batch(ids_batch)
        assert decoded == texts


class TestCountTokens:
    """Test token counting."""

    def test_count_tokens(self):
        count = count_tokens("Ẹ kú àbọ̀", lang="yoruba")
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_consistency(self):
        tok = Tokenizer("yoruba")
        text = "Ọjọ́ àìkú"
        assert count_tokens(text, lang="yoruba") == len(tok.encode(text))


class TestEfficiency:
    """Test that our tokenizer is more efficient than general-purpose ones."""

    def test_diacritics_not_split(self):
        """Compound diacritics should be single tokens, not split."""
        tok = Tokenizer("yoruba")

        # Single compound diacritic should be 1 token
        tokens = tok.tokenize("ọ́")
        # Should be 1 token (with possible space prefix)
        assert len(tokens) <= 2, f"Compound diacritic split into {len(tokens)} tokens: {tokens}"


class TestCaching:
    """Test tokenizer caching."""

    def test_same_language_reuses_tokenizer(self):
        tok1 = Tokenizer("yoruba")
        tok2 = Tokenizer("yoruba")
        # Should use cached tokenizer (same underlying object)
        assert tok1._tokenizer is tok2._tokenizer

    def test_different_language_different_tokenizer(self):
        tok1 = Tokenizer("yoruba")
        tok2 = Tokenizer("igbo")
        assert tok1._tokenizer is not tok2._tokenizer


class TestSupportedLanguages:
    """Test supported languages list."""

    def test_supported_languages_constant(self):
        assert "yoruba" in TOKENIZER_LANGUAGES
        assert "igbo" in TOKENIZER_LANGUAGES
        assert "hausa" in TOKENIZER_LANGUAGES
        assert "pidgin" in TOKENIZER_LANGUAGES
        assert "naija" in TOKENIZER_LANGUAGES
