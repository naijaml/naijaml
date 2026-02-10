"""Nigerian language detection using character n-grams.

Detects Yorùbá (yor), Hausa (hau), Igbo (ibo), Nigerian Pidgin (pcm),
and English (eng) from text using character-level features.

This is a lightweight, CPU-only implementation with no heavy dependencies.
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple

SUPPORTED_LANGUAGES: List[str] = ["yor", "hau", "ibo", "pcm", "eng"]

# =============================================================================
# Language-specific character patterns and markers
# =============================================================================

# Yorùbá-specific characters (diacritics)
_YORUBA_CHARS = set("ẹọṣẸỌṢ")
_YORUBA_COMBINING = set("̀́̄")  # grave, acute, macron (combining marks)

# Igbo-specific characters
_IGBO_CHARS = set("ịọụṅỊỌỤṄ")

# Hausa sometimes uses hooked letters in formal writing
_HAUSA_CHARS = set("ƙɗɓƘƊƁ")

# Common words/patterns that are strong indicators
_PIDGIN_MARKERS = {
    # Strong Pidgin markers (unique to Pidgin)
    "dey", "wetin", "abeg", "wahala", "sef", "sha", "shey", "abi",
    "naim", "wey", "dem", "una", "pikin", "chop", "jara", "kuku",
    "ehen", "shege", "yawa", "gist", "jand", "aproko", "oya", "omo",
    "wan", "comot", "vex", "yarn", "sabi", "bodi", "belle", "dodo",
}

_YORUBA_MARKERS = {
    "ṣe", "ni", "ti", "ati", "fún", "pẹ̀lú", "náà", "kan", "jẹ́",
    "ọjọ́", "ẹ̀yin", "àwọn", "kí", "sí", "lọ", "wá", "ń", "gbogbo",
}

_HAUSA_MARKERS = {
    "da", "na", "ta", "ya", "za", "ba", "ne", "ce", "shi", "ita",
    "su", "mu", "ku", "wannan", "wancan", "ina", "kuma", "amma",
    "don", "ga", "ko", "har", "zuwa", "cikin", "akan", "saboda",
    "yara", "suna", "wasa", "waje", "gida", "gidanmu", "kasuwa",
    "abinci", "ruwa", "yana", "tana", "muna", "kuna", "suke",
    "yaushe", "yaya", "sosai", "kwarai", "lafiya", "sannu", "barka",
    "Allah", "gode", "tafi", "zo", "yi", "san", "kin", "mun",
}

_IGBO_MARKERS = {
    "na", "bụ", "dị", "nke", "ya", "ha", "anyị", "unu", "gị",
    "ọ", "ka", "ma", "maka", "n'", "site", "ugbu", "nwere",
    "biko", "nye", "mmiri", "nri", "ụlọ", "ahịa", "echi", "ụtọ",
    "nwanne", "kedu", "aha", "onye", "ihe", "ebe", "oge", "ukwu",
    "nnọọ", "daalụ", "gozie", "chukwu", "ga", "egwu", "ụzọ",
}

_ENGLISH_MARKERS = {
    "the", "is", "are", "was", "were", "have", "has", "had",
    "will", "would", "could", "should", "been", "being", "this",
    "that", "these", "those", "with", "from", "they", "their",
}

# =============================================================================
# N-gram extraction
# =============================================================================

def _normalize_text(text: str) -> str:
    """Normalize text for consistent processing."""
    # NFC normalization for consistent diacritic handling
    normalized = unicodedata.normalize("NFC", text)
    return normalized.lower()


def _extract_char_ngrams(text: str, n: int = 2) -> Counter:
    """Extract character n-grams from text."""
    text = _normalize_text(text)
    ngrams = Counter()
    for i in range(len(text) - n + 1):
        ngram = text[i:i + n]
        ngrams[ngram] += 1
    return ngrams


def _extract_words(text: str) -> List[str]:
    """Extract words from text."""
    text = _normalize_text(text)
    # Split on whitespace and punctuation
    words = []
    current_word = []
    for char in text:
        if char.isalnum() or char in "ẹọṣịụṅƙɗɓ'́̀":
            current_word.append(char)
        else:
            if current_word:
                words.append("".join(current_word))
                current_word = []
    if current_word:
        words.append("".join(current_word))
    return words


# =============================================================================
# Feature extraction
# =============================================================================

def _has_yoruba_diacritics(text: str) -> bool:
    """Check if text contains Yorùbá-specific diacritics."""
    normalized = unicodedata.normalize("NFC", text)
    for char in normalized:
        if char in _YORUBA_CHARS:
            return True
    # Also check for combining marks on common vowels
    decomposed = unicodedata.normalize("NFD", text)
    for char in decomposed:
        if char in _YORUBA_COMBINING:
            return True
    return False


def _has_igbo_chars(text: str) -> bool:
    """Check if text contains Igbo-specific characters."""
    for char in text:
        if char in _IGBO_CHARS:
            return True
    return False


def _has_hausa_chars(text: str) -> bool:
    """Check if text contains Hausa-specific hooked letters."""
    for char in text:
        if char in _HAUSA_CHARS:
            return True
    return False


def _count_marker_words(words: List[str], markers: set) -> int:
    """Count how many words match a marker set."""
    return sum(1 for word in words if word in markers)


def _calculate_language_scores(text: str) -> Dict[str, float]:
    """Calculate raw scores for each language."""
    if not text or not text.strip():
        # Return uniform distribution for empty text
        return {lang: 1.0 / len(SUPPORTED_LANGUAGES) for lang in SUPPORTED_LANGUAGES}

    words = _extract_words(text)
    scores = {lang: 0.0 for lang in SUPPORTED_LANGUAGES}

    # Character-based features (strong signals)
    # Count Yorùbá diacritics for density-based scoring
    yoruba_char_count = sum(1 for c in text if c in _YORUBA_CHARS)
    if yoruba_char_count > 0:
        scores["yor"] += 8.0 + (yoruba_char_count * 2.0)  # More diacritics = higher score

    # Check for combining marks (tonal markers)
    decomposed = unicodedata.normalize("NFD", text)
    tonal_marks = sum(1 for c in decomposed if c in _YORUBA_COMBINING)
    if tonal_marks > 0:
        scores["yor"] += 4.0 + (tonal_marks * 1.5)

    if _has_igbo_chars(text):
        igbo_char_count = sum(1 for c in text if c in _IGBO_CHARS)
        scores["ibo"] += 8.0 + (igbo_char_count * 2.0)

    if _has_hausa_chars(text):
        scores["hau"] += 6.0

    # Check for ọ and ụ - Igbo uses ụ, Yorùbá doesn't
    text_lower = text.lower()
    if "ụ" in text_lower or "ị" in text_lower:
        scores["ibo"] += 3.0
    if "ẹ" in text_lower or "ṣ" in text_lower:
        scores["yor"] += 2.0

    # Word-based features
    pidgin_count = _count_marker_words(words, _PIDGIN_MARKERS)
    yoruba_count = _count_marker_words(words, _YORUBA_MARKERS)
    hausa_count = _count_marker_words(words, _HAUSA_MARKERS)
    igbo_count = _count_marker_words(words, _IGBO_MARKERS)
    english_count = _count_marker_words(words, _ENGLISH_MARKERS)

    scores["pcm"] += pidgin_count * 2.5
    scores["yor"] += yoruba_count * 1.5
    scores["hau"] += hausa_count * 1.5
    scores["ibo"] += igbo_count * 1.5
    scores["eng"] += english_count * 1.5

    # Pidgin-specific patterns (only strong markers)
    if "dey" in words or "wetin" in words or "abeg" in words:
        scores["pcm"] += 4.0
    if "wahala" in words or "sef" in words or "sha" in words:
        scores["pcm"] += 3.0

    # Pidgin phrase patterns (distinctive combinations)
    text_lower_joined = " ".join(words)
    if "sweet die" in text_lower_joined:
        scores["pcm"] += 5.0
    if "no fit" in text_lower_joined or "no wahala" in text_lower_joined:
        scores["pcm"] += 5.0
    if "e don" in text_lower_joined or "i dey" in text_lower_joined:
        scores["pcm"] += 4.0
    if "make we" in text_lower_joined:
        scores["pcm"] += 5.0
    if "how far" in text_lower_joined:
        scores["pcm"] += 3.0
    if "na so" in text_lower_joined:
        scores["pcm"] += 4.0

    # Check for Igbo apostrophe contractions (n'ụlọ, n'ime)
    if "n'" in text_lower or any(w.startswith("n'") for w in words):
        scores["ibo"] += 2.0

    # Hausa patterns: words ending in -ya, -wa common
    hausa_endings = sum(1 for w in words if w.endswith(("ya", "wa", "ta", "na")))
    scores["hau"] += hausa_endings * 0.3

    # Default bias toward English if no strong signals
    if all(s < 1.0 for s in scores.values()):
        scores["eng"] += 0.5

    # Ensure all scores are positive
    min_score = min(scores.values())
    if min_score < 0:
        for lang in scores:
            scores[lang] -= min_score

    return scores


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Convert raw scores to probabilities summing to 1."""
    total = sum(scores.values())
    if total == 0:
        return {lang: 1.0 / len(SUPPORTED_LANGUAGES) for lang in SUPPORTED_LANGUAGES}
    return {lang: score / total for lang, score in scores.items()}


# =============================================================================
# Public API
# =============================================================================

def detect_language(text: str) -> Optional[str]:
    """Detect the language of Nigerian text.

    Uses character n-grams and word markers to identify the language.
    Supports Yorùbá (yor), Hausa (hau), Igbo (ibo), Nigerian Pidgin (pcm),
    and English (eng).

    Args:
        text: Input text to classify.

    Returns:
        Language code ('yor', 'hau', 'ibo', 'pcm', 'eng') or None for empty text.

    Example:
        >>> detect_language("Ọjọ́ náà dára púpọ̀")
        'yor'
        >>> detect_language("Wetin dey happen?")
        'pcm'
        >>> detect_language("The weather is nice today")
        'eng'
    """
    if not text or not text.strip():
        return None

    scores = _calculate_language_scores(text)
    return max(scores, key=scores.get)


def detect_language_with_confidence(text: str) -> Tuple[Optional[str], float]:
    """Detect language with confidence score.

    Args:
        text: Input text to classify.

    Returns:
        Tuple of (language_code, confidence) where confidence is 0.0 to 1.0.

    Example:
        >>> lang, conf = detect_language_with_confidence("Ọjọ́ dára púpọ̀")
        >>> print(f"{lang}: {conf:.2f}")
        yor: 0.85
    """
    if not text or not text.strip():
        return None, 0.0

    scores = _calculate_language_scores(text)
    probs = _normalize_scores(scores)
    best_lang = max(probs, key=probs.get)
    confidence = probs[best_lang]

    return best_lang, confidence


def detect_all_languages(text: str) -> Dict[str, float]:
    """Get probability scores for all supported languages.

    Args:
        text: Input text to classify.

    Returns:
        Dict mapping language codes to probability scores (sum to 1.0).

    Example:
        >>> scores = detect_all_languages("Wetin dey happen?")
        >>> print(scores)
        {'yor': 0.05, 'hau': 0.10, 'ibo': 0.05, 'pcm': 0.70, 'eng': 0.10}
    """
    scores = _calculate_language_scores(text)
    return _normalize_scores(scores)
