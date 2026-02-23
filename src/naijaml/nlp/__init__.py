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
    # Pidgin handling
    is_pidgin_particle,
    preserve_pidgin_particles,
    get_pidgin_particles,
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
    diacritize_dot_below,
    diacritize_dot_below as diacritize_yoruba_dot_below,  # explicit alias
    syllabify,
    syllabify as syllabify_yoruba,  # explicit alias
    train_and_save_model as train_diacritizer,
    train_and_save_model as train_yoruba_diacritizer,  # explicit alias
    train_and_save_dot_below_model,
    evaluate_dot_below_accuracy,
    evaluate_full_diacritization_accuracy,
    compare_diacritization_methods,
    strip_diacritics as strip_yoruba_diacritics,  # expose tones_only option
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
from naijaml.nlp.sentiment import (
    analyze_sentiment,
    get_sentiment,
    get_sentiment_with_confidence,
    analyze_batch as analyze_sentiment_batch,
    is_available as is_sentiment_available,
)
from naijaml.nlp.tokenizer import (
    Tokenizer,
    tokenize,
    count_tokens,
    get_supported_languages as get_tokenizer_languages,
    SUPPORTED_LANGUAGES as TOKENIZER_LANGUAGES,
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
    # Pidgin handling
    "is_pidgin_particle",
    "preserve_pidgin_particles",
    "get_pidgin_particles",
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
    "diacritize_dot_below",
    "diacritize_yoruba_dot_below",
    "syllabify",
    "syllabify_yoruba",
    "train_diacritizer",
    "train_yoruba_diacritizer",
    "train_and_save_dot_below_model",
    "strip_yoruba_diacritics",
    # Diacritizer evaluation
    "evaluate_dot_below_accuracy",
    "evaluate_full_diacritization_accuracy",
    "compare_diacritization_methods",
    # Igbo diacritization
    "diacritize_igbo",
    "syllabify_igbo",
    "train_igbo_diacritizer",
    # Evaluation
    "evaluate_diacritizer",
    "DiacritizerMetrics",
    "run_all_evaluations",
    # Sentiment analysis
    "analyze_sentiment",
    "get_sentiment",
    "get_sentiment_with_confidence",
    "analyze_sentiment_batch",
    "is_sentiment_available",
    # Tokenizer
    "Tokenizer",
    "tokenize",
    "count_tokens",
    "get_tokenizer_languages",
    "TOKENIZER_LANGUAGES",
]
