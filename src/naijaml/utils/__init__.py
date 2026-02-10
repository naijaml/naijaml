"""NaijaML utility functions."""
from naijaml.utils.constants import (
    # Data
    STATES,
    STATE_NAMES,
    LGAS,
    BANKS,
    BANK_NAMES,
    TELCOS,
    TELCO_NAMES,
    # Patterns
    PHONE_PATTERN,
    PHONE_PATTERN_LOOSE,
    BVN_PATTERN,
    NIN_PATTERN,
    NUBAN_PATTERN,
    NAIRA_PATTERN,
    # Functions
    format_naira,
    parse_naira,
    is_valid_phone,
    normalize_phone,
    get_telco,
    is_valid_bvn,
    is_valid_nin,
)

__all__ = [
    # Data
    "STATES",
    "STATE_NAMES",
    "LGAS",
    "BANKS",
    "BANK_NAMES",
    "TELCOS",
    "TELCO_NAMES",
    # Patterns
    "PHONE_PATTERN",
    "PHONE_PATTERN_LOOSE",
    "BVN_PATTERN",
    "NIN_PATTERN",
    "NUBAN_PATTERN",
    "NAIRA_PATTERN",
    # Functions
    "format_naira",
    "parse_naira",
    "is_valid_phone",
    "normalize_phone",
    "get_telco",
    "is_valid_bvn",
    "is_valid_nin",
]
