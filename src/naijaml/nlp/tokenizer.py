"""
NaijaML Tokenizer - Efficient tokenizers for Nigerian languages.

Tokenizers trained on dedicated Nigerian language data, achieving:
- 63% fewer tokens than GPT-4, 45% fewer than AfriBERTa for Yoruba
- 50% fewer tokens than GPT-4, 40% fewer than AfriBERTa for Igbo
- 31% fewer tokens than GPT-4, 18% fewer than AfriBERTa for Hausa
- 14% fewer tokens than GPT-4 for Pidgin
- 100% diacritic preservation (compound diacritics stay as single tokens)
- 2.5x faster than GPT-4's tiktoken in batch mode

Unlike GPT-4/AfriBERTa/AfroXLMR which split compound diacritics like ọ́ into
multiple tokens, NaijaML tokenizers keep them as single tokens because they
were trained on dedicated Nigerian language corpora.

Example:
    >>> from naijaml.nlp import Tokenizer
    >>> tok = Tokenizer("yoruba")
    >>> tok.encode("Ọjọ́ àìkú")
    [234, 567, 89]
    >>> tok.decode([234, 567, 89])
    'Ọjọ́ àìkú'
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

# Language name mapping - users can use full names or ISO codes
LANGUAGE_MAP = {
    # Yoruba
    "yoruba": "yor-8k-rust.json",
    "yor": "yor-8k-rust.json",
    "yo": "yor-8k-rust.json",
    # Igbo
    "igbo": "ibo-8k-rust.json",
    "ibo": "ibo-8k-rust.json",
    "ig": "ibo-8k-rust.json",
    # Hausa
    "hausa": "hau-8k-rust.json",
    "hau": "hau-8k-rust.json",
    "ha": "hau-8k-rust.json",
    # Pidgin
    "pidgin": "pcm-8k-rust.json",
    "pcm": "pcm-8k-rust.json",
    "naija-pidgin": "pcm-8k-rust.json",
    # Unified (all 4 languages)
    "naija": "naija-8k-rust.json",
    "unified": "naija-8k-rust.json",
    "all": "naija-8k-rust.json",
}

SUPPORTED_LANGUAGES = ["yoruba", "igbo", "hausa", "pidgin", "naija"]

# Models directory
_MODELS_DIR = Path(__file__).parent / "models" / "tokenizer"

# Cache loaded tokenizers
_TOKENIZER_CACHE: dict = {}


def _get_tokenizers_module():
    """Lazy import of tokenizers library."""
    try:
        import tokenizers
        return tokenizers
    except ImportError:
        raise ImportError(
            "The 'tokenizers' library is required for NaijaML tokenizers.\n"
            "Install it with: pip install tokenizers"
        )


class Tokenizer:
    """
    Tokenizer for Nigerian languages.

    Trained on dedicated Nigerian language corpora, these tokenizers preserve
    diacritics correctly and achieve significant token reduction compared to
    general-purpose tokenizers like GPT-4's cl100k_base.

    Args:
        language: Language to tokenize. Options:
            - "yoruba" / "yor" / "yo"
            - "igbo" / "ibo" / "ig"
            - "hausa" / "hau" / "ha"
            - "pidgin" / "pcm"
            - "naija" / "unified" / "all" (handles all 4 languages)

    Example:
        >>> tok = Tokenizer("yoruba")
        >>> tokens = tok.encode("Ẹ kú àbọ̀")
        >>> tok.decode(tokens)
        'Ẹ kú àbọ̀'
    """

    def __init__(self, language: str = "naija"):
        language = language.lower().strip()

        if language not in LANGUAGE_MAP:
            raise ValueError(
                f"Unknown language: '{language}'. "
                f"Supported: {', '.join(SUPPORTED_LANGUAGES)}"
            )

        self.language = language
        self._model_name = LANGUAGE_MAP[language]
        self._tokenizer = self._load_tokenizer()

    def _load_tokenizer(self):
        """Load the tokenizer from bundled model file."""
        # Check cache first
        if self._model_name in _TOKENIZER_CACHE:
            return _TOKENIZER_CACHE[self._model_name]

        tokenizers = _get_tokenizers_module()

        model_path = _MODELS_DIR / self._model_name
        if not model_path.exists():
            raise FileNotFoundError(
                f"Tokenizer model not found: {model_path}. "
                "This is a bug in NaijaML - please report it."
            )

        tokenizer = tokenizers.Tokenizer.from_file(str(model_path))
        _TOKENIZER_CACHE[self._model_name] = tokenizer
        return tokenizer

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode.

        Returns:
            List of token IDs.
        """
        encoding = self._tokenizer.encode(text)
        return encoding.ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded text.
        """
        return self._tokenizer.decode(ids)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into token strings.

        Args:
            text: Text to tokenize.

        Returns:
            List of token strings.
        """
        encoding = self._tokenizer.encode(text)
        return encoding.tokens

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode multiple texts in parallel (faster than encoding one by one).

        Args:
            texts: List of texts to encode.

        Returns:
            List of token ID lists.
        """
        encodings = self._tokenizer.encode_batch(texts)
        return [enc.ids for enc in encodings]

    def decode_batch(self, ids_batch: List[List[int]]) -> List[str]:
        """
        Decode multiple token ID lists in parallel.

        Args:
            ids_batch: List of token ID lists.

        Returns:
            List of decoded texts.
        """
        return self._tokenizer.decode_batch(ids_batch)

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self._tokenizer.get_vocab_size()

    def __repr__(self) -> str:
        return f"Tokenizer(language='{self.language}', vocab_size={self.vocab_size})"


def tokenize(text: str, lang: str = "naija") -> List[str]:
    """
    Convenience function to tokenize text without creating a Tokenizer instance.

    Args:
        text: Text to tokenize.
        lang: Language (default: "naija" for unified tokenizer).

    Returns:
        List of token strings.

    Example:
        >>> tokenize("Ẹ kú àbọ̀", lang="yoruba")
        ['▁Ẹ', '▁kú', '▁àbọ̀']
    """
    tok = Tokenizer(lang)
    return tok.tokenize(text)


def count_tokens(text: str, lang: str = "naija") -> int:
    """
    Count tokens in text.

    Args:
        text: Text to count tokens in.
        lang: Language (default: "naija" for unified tokenizer).

    Returns:
        Number of tokens.

    Example:
        >>> count_tokens("Ẹ kú àbọ̀", lang="yoruba")
        3
    """
    tok = Tokenizer(lang)
    return len(tok.encode(text))


def get_supported_languages() -> List[str]:
    """Return list of supported languages."""
    return SUPPORTED_LANGUAGES.copy()
