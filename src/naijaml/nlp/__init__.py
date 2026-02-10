"""NaijaML NLP utilities for Nigerian languages."""
from naijaml.nlp.preprocess import (
    # Unicode
    normalize_unicode,
    strip_diacritics,
    # Social media
    clean_social_media,
    extract_hashtags,
    extract_mentions,
    # PII
    mask_pii,
    find_phones,
    find_naira_amounts,
    # Nigerian-specific
    normalize_naira_symbol,
    clean_nigerian_text,
)
from naijaml.nlp.langdetect import (
    detect_language,
    detect_language_with_confidence,
    detect_all_languages,
    SUPPORTED_LANGUAGES,
)

__all__ = [
    # Unicode
    "normalize_unicode",
    "strip_diacritics",
    # Social media
    "clean_social_media",
    "extract_hashtags",
    "extract_mentions",
    # PII
    "mask_pii",
    "find_phones",
    "find_naira_amounts",
    # Nigerian-specific
    "normalize_naira_symbol",
    "clean_nigerian_text",
    # Language detection
    "detect_language",
    "detect_language_with_confidence",
    "detect_all_languages",
    "SUPPORTED_LANGUAGES",
]
