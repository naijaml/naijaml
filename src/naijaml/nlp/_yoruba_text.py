"""Yorùbá character tables, diacritic stripping and syllable segmentation."""
from __future__ import annotations

import unicodedata
from typing import List, Tuple

# =============================================================================
# Yorùbá character mappings
# =============================================================================

# Vowels that can carry diacritics
YORUBA_VOWELS = set("aeiouAEIOU")

# Vowels with dot-below (these are distinct letters in Yorùbá)
DOTTED_VOWELS = {"ẹ": "e", "ọ": "o", "Ẹ": "E", "Ọ": "O"}
UNDOTTED_TO_DOTTED = {"e": "ẹ", "o": "ọ", "E": "Ẹ", "O": "Ọ"}

# Consonant with dot-below
DOTTED_CONSONANTS = {"ṣ": "s", "Ṣ": "S"}
UNDOTTED_CONS_TO_DOTTED = {"s": "ṣ", "S": "Ṣ"}

# Tonal marks (combining characters)
TONE_ACUTE = "\u0301"  # ́ high tone
TONE_GRAVE = "\u0300"  # ̀ low tone
TONE_MACRON = "\u0304"  # ̄ mid tone (rarely written)

# All combining diacritical marks to strip
COMBINING_MARKS = {TONE_ACUTE, TONE_GRAVE, TONE_MACRON, "\u0323"}  # 0323 = dot below

# Yorùbá consonants (including ṣ)
YORUBA_CONSONANTS = set("bdfghjklmnprstvwyBDFGHJKLMNPRSTVWYṣṢ")

# Valid Yorùbá characters
YORUBA_CHARS = YORUBA_VOWELS | set("ẹọṣẸỌṢ") | YORUBA_CONSONANTS | {"n", "N", "'"}


# =============================================================================
# Text normalization
# =============================================================================

def normalize_yoruba(text: str) -> str:
    """Normalize Yorùbá text to NFC form."""
    return unicodedata.normalize("NFC", text)


def strip_diacritics(text: str, tones_only: bool = False) -> str:
    """Remove diacritics from Yorùbá text.

    By default, removes ALL diacritics (tones + dot-below).
    With tones_only=True, removes only tonal marks but keeps dot-below (ọ, ẹ, ṣ).

    Args:
        text: Diacritized Yorùbá text.
        tones_only: If True, only remove tonal marks, keep dot-below characters.

    Returns:
        Text with specified diacritics removed.

    Example:
        >>> strip_diacritics("Ọjọ́ dára")
        'Ojo dara'
        >>> strip_diacritics("Ọjọ́ dára", tones_only=True)
        'Ọjọ dara'
    """
    text = normalize_yoruba(text)

    # Decompose to separate base characters from combining marks
    text = unicodedata.normalize("NFD", text)

    # Remove combining marks (selectively if tones_only)
    result = []
    for char in text:
        if unicodedata.category(char) == "Mn":  # Mark, Nonspacing
            if tones_only:
                # Only skip tonal marks (acute, grave, macron), keep dot-below
                if char in {TONE_ACUTE, TONE_GRAVE, TONE_MACRON}:
                    continue
                # Keep dot-below (0323)
                result.append(char)
            else:
                # Skip all combining marks
                continue
        else:
            result.append(char)

    text = "".join(result)

    # Recompose characters with their combining marks
    text = unicodedata.normalize("NFC", text)

    if not tones_only:
        # Convert dotted vowels/consonants to plain
        for dotted, plain in DOTTED_VOWELS.items():
            text = text.replace(dotted, plain)
        for dotted, plain in DOTTED_CONSONANTS.items():
            text = text.replace(dotted, plain)

    return text


def _is_vowel(char: str) -> bool:
    """Check if character is a Yorùbá vowel."""
    return char.lower() in "aeiouẹọ"


def _is_consonant(char: str) -> bool:
    """Check if character is a Yorùbá consonant."""
    return char.lower() in "bdfghjklmnprstvwyṣ"


def _is_nasal(char: str) -> bool:
    """Check if character is a syllabic nasal."""
    return char.lower() in "nm"


# =============================================================================
# Syllable segmentation
# =============================================================================

def syllabify(word: str) -> List[str]:
    """Segment a Yorùbá word into syllables.

    Yorùbá syllable structure:
    - V (vowel alone): a, o, e
    - CV (consonant + vowel): ba, lo, ṣe
    - N (syllabic nasal): n, m (when not followed by vowel)

    Args:
        word: A single Yorùbá word.

    Returns:
        List of syllables.

    Example:
        >>> syllabify("ọjọ")
        ['ọ', 'jọ']
        >>> syllabify("dara")
        ['da', 'ra']
        >>> syllabify("nkan")
        ['n', 'ka', 'n']
    """
    word = normalize_yoruba(word)

    if not word:
        return []

    syllables = []
    i = 0

    while i < len(word):
        char = word[i]

        # Handle non-Yorùbá characters (punctuation, numbers, etc.)
        if not (_is_vowel(char) or _is_consonant(char) or _is_nasal(char)):
            # Include as-is (could be apostrophe, hyphen, etc.)
            if syllables:
                syllables[-1] += char
            else:
                syllables.append(char)
            i += 1
            continue

        # Case 1: Vowel (possibly with tone marks following in NFD)
        if _is_vowel(char):
            syl = char
            i += 1
            # Collect any combining marks
            while i < len(word) and unicodedata.category(word[i]) == "Mn":
                syl += word[i]
                i += 1
            syllables.append(syl)
            continue

        # Case 2: Consonant
        if _is_consonant(char):
            # Check if followed by vowel (CV pattern)
            if i + 1 < len(word) and _is_vowel(word[i + 1]):
                syl = char + word[i + 1]
                i += 2
                # Collect any combining marks
                while i < len(word) and unicodedata.category(word[i]) == "Mn":
                    syl += word[i]
                    i += 1
                syllables.append(syl)
            else:
                # Consonant alone (rare, might be word boundary issue)
                syllables.append(char)
                i += 1
            continue

        # Case 3: Nasal (n, m)
        if _is_nasal(char):
            # Check if it's syllabic (not followed by vowel) or part of CV
            if i + 1 < len(word) and _is_vowel(word[i + 1]):
                # It's a consonant in CV
                syl = char + word[i + 1]
                i += 2
                while i < len(word) and unicodedata.category(word[i]) == "Mn":
                    syl += word[i]
                    i += 1
                syllables.append(syl)
            else:
                # Syllabic nasal
                syl = char
                i += 1
                # Collect any combining marks (ń, ǹ)
                while i < len(word) and unicodedata.category(word[i]) == "Mn":
                    syl += word[i]
                    i += 1
                syllables.append(syl)
            continue

        # Fallback: just add the character
        syllables.append(char)
        i += 1

    return syllables


def syllabify_text(text: str) -> List[Tuple[str, bool]]:
    """Segment text into tokens, marking which are words vs separators.

    Args:
        text: Full text string.

    Returns:
        List of (token, is_word) tuples.

    Example:
        >>> syllabify_text("Ọjọ dara!")
        [('Ọjọ', True), (' ', False), ('dara', True), ('!', False)]
    """
    tokens = []
    current_word = []

    for char in normalize_yoruba(text):
        if char.isalpha() or char in "ẹọṣẸỌṢ'":
            current_word.append(char)
        else:
            if current_word:
                tokens.append(("".join(current_word), True))
                current_word = []
            tokens.append((char, False))

    if current_word:
        tokens.append(("".join(current_word), True))

    return tokens
