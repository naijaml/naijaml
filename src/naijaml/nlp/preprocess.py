"""Nigerian text preprocessing utilities.

Provides functions to clean and normalize Nigerian text data, including:
- Unicode normalization (especially for Yorùbá diacritics)
- Social media text cleaning
- PII masking (phone numbers, BVN, NIN)
- Naira amount handling
"""
from __future__ import annotations

import re
import unicodedata
from typing import List

from naijaml.utils.constants import (
    PHONE_PATTERN_LOOSE,
    NAIRA_PATTERN,
    PIDGIN_PARTICLES,
)

# =============================================================================
# Unicode Normalization
# =============================================================================

def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Normalize Unicode text to a standard form.

    Important for Yorùbá text where diacritics can be represented
    as combining characters or precomposed characters.

    Args:
        text: Input text to normalize.
        form: Unicode normalization form ('NFC', 'NFD', 'NFKC', 'NFKD').
            NFC (default) composes characters (recommended for Yorùbá).

    Returns:
        Normalized text.

    Example:
        >>> # These look the same but have different byte representations
        >>> text1 = "ọjọ́"  # precomposed
        >>> text2 = "ọjọ́"   # with combining characters
        >>> normalize_unicode(text1) == normalize_unicode(text2)
        True
    """
    return unicodedata.normalize(form, text)


def strip_diacritics(text: str) -> str:
    """Remove all diacritical marks from text.

    Useful for creating search-friendly versions of Yorùbá text.

    Args:
        text: Input text with diacritics.

    Returns:
        Text with diacritics removed.

    Example:
        >>> strip_diacritics("Ojó lo sí ọjà lánà")
        'Ojo lo si oja lana'
    """
    # NFD decomposes characters, then we filter out combining marks
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(
        char for char in decomposed
        if unicodedata.category(char) != "Mn"  # Mn = Mark, Nonspacing
    )
    return unicodedata.normalize("NFC", stripped)


# =============================================================================
# Social Media Text Cleaning
# =============================================================================

# Common patterns in social media text
_URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+"
)

_MENTION_PATTERN = re.compile(
    r"@[A-Za-z0-9_]+"
)

_HASHTAG_PATTERN = re.compile(
    r"#[A-Za-z0-9_]+"
)

_EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)

_REPEATED_CHARS = re.compile(
    r"(.)\1{3,}"  # 4+ repeated characters
)

_MULTIPLE_SPACES = re.compile(
    r"\s+"
)


def clean_social_media(
    text: str,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    remove_hashtags: bool = False,
    lowercase: bool = False,
    reduce_repeated: bool = True,
) -> str:
    """Clean social media text (tweets, comments, etc.).

    Args:
        text: Input text from social media.
        remove_urls: Remove URLs.
        remove_mentions: Remove @mentions.
        remove_hashtags: Remove #hashtags (default False, keeps them).
        lowercase: Convert to lowercase.
        reduce_repeated: Reduce repeated characters (e.g., "loooool" -> "lool").

    Returns:
        Cleaned text.

    Example:
        >>> clean_social_media("@user This film too sweet!!! https://t.co/abc #Nollywood")
        'This film too sweet!! #Nollywood'
    """
    result = text

    if remove_urls:
        result = _URL_PATTERN.sub("", result)

    if remove_mentions:
        result = _MENTION_PATTERN.sub("", result)

    if remove_hashtags:
        result = _HASHTAG_PATTERN.sub("", result)

    if reduce_repeated:
        # Reduce 4+ repeated chars to 2
        result = _REPEATED_CHARS.sub(r"\1\1", result)

    if lowercase:
        result = result.lower()

    # Clean up whitespace
    result = _MULTIPLE_SPACES.sub(" ", result).strip()

    return result


def extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from text.

    Args:
        text: Input text.

    Returns:
        List of hashtags (without # symbol).

    Example:
        >>> extract_hashtags("Great movie! #Nollywood #NigerianFilm")
        ['Nollywood', 'NigerianFilm']
    """
    return [tag[1:] for tag in _HASHTAG_PATTERN.findall(text)]


def extract_mentions(text: str) -> List[str]:
    """Extract @mentions from text.

    Args:
        text: Input text.

    Returns:
        List of usernames (without @ symbol).

    Example:
        >>> extract_mentions("cc @funke_akindele @NigeriaFilms")
        ['funke_akindele', 'NigeriaFilms']
    """
    return [mention[1:] for mention in _MENTION_PATTERN.findall(text)]


# =============================================================================
# PII Masking
# =============================================================================

def mask_pii(
    text: str,
    mask_phones: bool = True,
    mask_emails: bool = True,
    mask_bvn: bool = True,
    mask_nin: bool = True,
    mask_naira: bool = False,
    phone_mask: str = "[PHONE]",
    email_mask: str = "[EMAIL]",
    bvn_mask: str = "[BVN]",
    nin_mask: str = "[NIN]",
    naira_mask: str = "[AMOUNT]",
) -> str:
    """Mask personally identifiable information (PII) in text.

    Args:
        text: Input text potentially containing PII.
        mask_phones: Mask Nigerian phone numbers.
        mask_emails: Mask email addresses.
        mask_bvn: Mask Bank Verification Numbers.
        mask_nin: Mask National Identification Numbers.
        mask_naira: Mask Naira amounts (default False).
        phone_mask: Replacement string for phone numbers.
        email_mask: Replacement string for emails.
        bvn_mask: Replacement string for BVN.
        nin_mask: Replacement string for NIN.
        naira_mask: Replacement string for Naira amounts.

    Returns:
        Text with PII masked.

    Example:
        >>> mask_pii("Call me on 08012345678 or email me@example.com")
        'Call me on [PHONE] or [EMAIL]'
    """
    result = text

    # Order matters: mask specific patterns first (phones, BVN) before
    # the broad NIN pattern, so NIN doesn't swallow them.
    if mask_phones:
        result = PHONE_PATTERN_LOOSE.sub(phone_mask, result)

    if mask_emails:
        result = _EMAIL_PATTERN.sub(email_mask, result)

    if mask_bvn:
        # BVN is 11 digits starting with 22
        result = re.sub(r"\b22\d{9}\b", bvn_mask, result)

    if mask_nin:
        # NIN is 11 digits — but we must NOT match numbers that are:
        #   - Already masked (contain [ from mask tokens)
        #   - Phone numbers (start with 0, 234, +234)
        #   - BVN numbers (start with 22) — only skip if mask_bvn is also on
        # We look for standalone 11-digit sequences preceded by "NIN" context
        # or use word-boundary matching, excluding phone/BVN prefixes.
        def _nin_replacer(match: re.Match) -> str:
            num = match.group(0)
            # Skip phone-like numbers (start with 0)
            if num.startswith("0"):
                return num
            # Skip BVN-like numbers (start with 22) when BVN masking is active
            if mask_bvn and num.startswith("22"):
                return num
            return nin_mask

        result = re.sub(r"(?<!\d)(\d{11})(?!\d)", _nin_replacer, result)

    if mask_naira:
        result = NAIRA_PATTERN.sub(naira_mask, result)

    return result


def find_phones(text: str) -> List[str]:
    """Find all Nigerian phone numbers in text.

    Args:
        text: Input text.

    Returns:
        List of phone numbers found.

    Example:
        >>> find_phones("Call 0801-234-5678 or +234 902 123 4567")
        ['0801-234-5678', '+234 902 123 4567']
    """
    return PHONE_PATTERN_LOOSE.findall(text)


def find_naira_amounts(text: str) -> List[str]:
    """Find all Naira amounts in text.

    Args:
        text: Input text.

    Returns:
        List of Naira amount strings found.

    Example:
        >>> find_naira_amounts("Price is ₦5,000 or NGN 10,000")
        ['₦5,000', 'NGN 10,000']
    """
    return NAIRA_PATTERN.findall(text)


# =============================================================================
# Nigerian-Specific Normalization
# =============================================================================

def normalize_naira_symbol(text: str) -> str:
    """Normalize various Naira representations to ₦.

    Args:
        text: Input text with Naira amounts.

    Returns:
        Text with normalized Naira symbol.

    Example:
        >>> normalize_naira_symbol("NGN 5000 or N5,000")
        '₦5000 or ₦5,000'
    """
    # Replace NGN followed by optional space and number
    result = re.sub(r"NGN\s?(?=\d)", "₦", text)
    # Replace standalone N before numbers (be careful not to replace regular N)
    result = re.sub(r"(?<![A-Za-z])N(?=\d)", "₦", result)
    return result


def clean_nigerian_text(
    text: str,
    normalize: bool = True,
    clean_social: bool = True,
    mask_pii_data: bool = False,
    normalize_negation: bool = False,
    lowercase: bool = False,
) -> str:
    """All-in-one text cleaning for Nigerian text data.

    Applies sensible defaults for cleaning Nigerian text from social media
    and other sources.

    Args:
        text: Input text.
        normalize: Apply Unicode normalization (NFC).
        clean_social: Clean social media artifacts (URLs, mentions).
        mask_pii_data: Mask phone numbers, emails, BVN, NIN.
        normalize_negation: Normalize Nigerian Pidgin negation patterns.
        lowercase: Convert to lowercase.

    Returns:
        Cleaned text.

    Example:
        >>> text = "@user Check https://t.co/abc Ọjọ́ is great! Call 08012345678"
        >>> clean_nigerian_text(text, mask_pii_data=True)
        'Check Ọjọ́ is great! Call [PHONE]'
    """
    result = text

    if normalize:
        result = normalize_unicode(result)

    if clean_social:
        result = clean_social_media(result)

    if mask_pii_data:
        result = mask_pii(result)

    if lowercase:
        result = result.lower()

    if normalize_negation:
        result = normalize_pidgin_negation(result)

    return result


# =============================================================================
# Nigerian Pidgin Handling
# =============================================================================

def is_pidgin_particle(word: str) -> bool:
    """Check if a word is a Nigerian Pidgin particle or discourse marker.

    These are words like 'sha', 'sef', 'abeg' that standard NLP tools often
    strip as noise or classify as errors, but are meaningful in Pidgin.

    Args:
        word: Word to check.

    Returns:
        True if the word is a known Pidgin particle.

    Example:
        >>> is_pidgin_particle("sha")
        True
        >>> is_pidgin_particle("hello")
        False
    """
    return word.lower() in PIDGIN_PARTICLES


def preserve_pidgin_particles(text: str, words_to_remove: List[str]) -> str:
    """Remove words from text while preserving Pidgin particles.

    Useful when cleaning text with a stopword list that might incorrectly
    include Pidgin discourse markers.

    Args:
        text: Input text.
        words_to_remove: List of words to remove (stopwords).

    Returns:
        Text with non-Pidgin stopwords removed but Pidgin particles kept.

    Example:
        >>> preserve_pidgin_particles("The film sha too sweet", ["the", "sha", "too"])
        'film sha sweet'
    """
    words = text.split()
    result = []
    for word in words:
        # Keep if not in removal list OR if it's a Pidgin particle
        word_lower = word.lower().strip(".,!?;:'\"")
        if word_lower not in words_to_remove or is_pidgin_particle(word_lower):
            result.append(word)
    return " ".join(result)


def get_pidgin_particles() -> set:
    """Get the set of known Nigerian Pidgin particles.

    Returns:
        Set of Pidgin particle strings.

    Example:
        >>> particles = get_pidgin_particles()
        >>> "abeg" in particles
        True
    """
    return PIDGIN_PARTICLES.copy()


# Ordered: specific patterns first, generic last.
_PIDGIN_NEGATION_PATTERNS = [
    # Idioms must be caught before generic "no go" patterns fire.
    (re.compile(r"\bno go lie\b", re.IGNORECASE), "honestly"),

    # Stacked modifiers: "e no too bad" / "no too bad" = mildly positive.
    (re.compile(r"\be no too bad\b", re.IGNORECASE), "it is good"),
    (re.compile(r"\bno too bad\b", re.IGNORECASE), "good"),

    # Compound "at all" intensifiers must come before shorter phrase matches.
    (re.compile(r"\bno bad at all\b", re.IGNORECASE), "very good"),
    (re.compile(r"\bno good at all\b", re.IGNORECASE), "very bad"),
    (re.compile(r"\bno sweet at all\b", re.IGNORECASE), "very bad"),

    # Relative clause negation: "no be ... wey bad" is a positive signal.
    (re.compile(r"\bno be \w+ wey bad\b", re.IGNORECASE), "good"),
    (re.compile(r"\bwey bad\b", re.IGNORECASE), "that is bad"),

    # Double negatives resolve to positive sentiment.
    (re.compile(r"\bno be bad\b", re.IGNORECASE), "good"),
    (re.compile(r"\bno bad\b", re.IGNORECASE), "good"),
    (re.compile(r"\bnot bad\b", re.IGNORECASE), "good"),

    # "no + positive word" resolves to in-vocabulary negative words.
    (re.compile(r"\be no sweet\b", re.IGNORECASE), "it is bad"),
    (re.compile(r"\bno sweet\b", re.IGNORECASE), "bad"),
    (re.compile(r"\bno good\b", re.IGNORECASE), "bad"),
    (re.compile(r"\bno fine\b", re.IGNORECASE), "ugly"),

    # Generic negation patterns.
    (re.compile(r"\bno be\b", re.IGNORECASE), "not"),
    (re.compile(r"\be no\b", re.IGNORECASE), "it is not"),
    (re.compile(r"\bno go\b", re.IGNORECASE), "will not"),
    (re.compile(r"\bno like\b", re.IGNORECASE), "hate"),
    (re.compile(r"\bno want\b", re.IGNORECASE), "reject"),
]


def normalize_pidgin_negation(text: str) -> str:
    """Normalize Nigerian Pidgin negation patterns before sentiment analysis.

    Pidgin uses double negatives and negation constructions that standard
    sentiment models misclassify because they're trained on English syntax.
    "no bad" is positive. "no sweet" is negative. This function normalizes
    those patterns to English equivalents the classifier understands.

    Args:
        text: Input Pidgin or code-mixed text.

    Returns:
        Text with negation patterns normalized.

    Example:
        >>> normalize_pidgin_negation("This thing no bad at all")
        'This thing very good'
        >>> normalize_pidgin_negation("E no sweet me")
        'it is bad me'
        >>> normalize_pidgin_negation("I no go lie this thing good")
        'I honestly this thing good'
    """
    result = text
    for pattern, replacement in _PIDGIN_NEGATION_PATTERNS:
        result = pattern.sub(replacement, result)
    return _MULTIPLE_SPACES.sub(" ", result).strip()
