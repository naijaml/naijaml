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
from typing import List, Optional

from naijaml.utils.constants import (
    PHONE_PATTERN_LOOSE,
    BVN_PATTERN,
    NIN_PATTERN,
    NAIRA_PATTERN,
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

    if mask_phones:
        result = PHONE_PATTERN_LOOSE.sub(phone_mask, result)

    if mask_emails:
        result = _EMAIL_PATTERN.sub(email_mask, result)

    if mask_bvn:
        # BVN is 11 digits starting with 22
        result = re.sub(r"\b22\d{9}\b", bvn_mask, result)

    if mask_nin:
        # NIN is any 11-digit number (more careful matching to avoid false positives)
        # Only match standalone 11-digit numbers that aren't phone numbers
        result = re.sub(r"(?<!\d)\d{11}(?!\d)", nin_mask, result)

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

    return result
