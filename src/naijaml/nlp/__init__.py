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
    train_and_save_model,
    train_and_save_profiles,  # backwards compatibility alias
)
from naijaml.nlp.diacritizer import (
    diacritize,
    diacritize as diacritize_yoruba,  # explicit alias
    syllabify,
    syllabify as syllabify_yoruba,  # explicit alias
    train_and_save_model as train_diacritizer,
    train_and_save_model as train_yoruba_diacritizer,  # explicit alias
)
from naijaml.nlp.igbo_diacritizer import (
    diacritize_igbo,
    syllabify as syllabify_igbo,
    train_and_save_model as train_igbo_diacritizer,
)
from naijaml.nlp.evaluation import (
    evaluate_diacritizer,
    DiacritizerMetrics,
    run_all_evaluations,
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
    "train_and_save_model",
    "train_and_save_profiles",  # backwards compatibility alias
    # Yorùbá diacritization
    "diacritize",
    "diacritize_yoruba",
    "syllabify",
    "syllabify_yoruba",
    "train_diacritizer",
    "train_yoruba_diacritizer",
    # Igbo diacritization
    "diacritize_igbo",
    "syllabify_igbo",
    "train_igbo_diacritizer",
    # Evaluation
    "evaluate_diacritizer",
    "DiacritizerMetrics",
    "run_all_evaluations",
]
